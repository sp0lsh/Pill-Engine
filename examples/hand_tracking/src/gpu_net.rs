//! Runs an ONNX graph (`onnx::Graph`) on the GPU as WGSL compute dispatches, one per node, on a
//! caller's wgpu device. Covers the operators of the two MediaPipe hand models: Conv (grouped,
//! padded, strided), PRelu, Clip, Add, Sigmoid, MaxPool, Pad (constant), Resize (linear,
//! half_pixel), Transpose, Reshape, Squeeze, Concat, GlobalAveragePool, Gemm. Tensors are f32,
//! one storage buffer each, in ONNX (row-major) layout; Reshape and Squeeze are views. Conv and
//! Gemm weights are stored as f16 (half the memory and bandwidth); arithmetic stays f32, since
//! WebGPU's shader-f16 is optional.

// Used by the web build and the tests
#![cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]

use crate::onnx::{Graph, Initializer, Node, TensorData};
use pill_engine::game::{bytemuck, wgpu};
use std::collections::HashMap;
use std::rc::Rc;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 64;
const MAX_WORKGROUPS_X: u32 = 65535;
/// u32 slots of each dispatch's parameter block (`Params` in the shader).
const PARAM_COUNT: usize = 16;

const SHADER: &str = r#"
struct Params { p: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> c: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
// Conv / Gemm weights as f16 pairs (low half first)
@group(0) @binding(5) var<storage, read> weights: array<u32>;

fn p(i: u32) -> u32 { return params.p[i / 4u][i % 4u]; }
fn pf(i: u32) -> f32 { return bitcast<f32>(p(i)); }
// p(0) is always the output element count
fn index(gid: vec3<u32>, groups: vec3<u32>) -> u32 { return gid.y * groups.x * 64u + gid.x; }
fn weight(i: u32) -> f32 {
    let pair = unpack2x16float(weights[i / 2u]);
    return select(pair.y, pair.x, i % 2u == 0u);
}

// p: 1 C_in, 2 H, 3 W, 4 C_out, 5 OH, 6 OW, 7 KH, 8 KW, 9 SH, 10 SW, 11 pad top, 12 pad left, 13 group
@compute @workgroup_size(64) fn conv(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let c_in = p(1); let h = p(2); let w = p(3); let c_out = p(4); let oh = p(5); let ow = p(6);
    let kh = p(7); let kw = p(8); let group = p(13);
    let ox = i % ow; let oy = (i / ow) % oh; let co = i / (ow * oh);
    let in_per_group = c_in / group; let out_per_group = c_out / group;
    let ci0 = (co / out_per_group) * in_per_group;
    var sum = c[co];
    for (var ci = 0u; ci < in_per_group; ci++) {
        for (var ky = 0u; ky < kh; ky++) {
            let y = i32(oy * p(9) + ky) - i32(p(11));
            if (y < 0 || y >= i32(h)) { continue; }
            for (var kx = 0u; kx < kw; kx++) {
                let x = i32(ox * p(10) + kx) - i32(p(12));
                if (x < 0 || x >= i32(w)) { continue; }
                sum += a[((ci0 + ci) * h + u32(y)) * w + u32(x)] * weight(((co * in_per_group + ci) * kh + ky) * kw + kx);
            }
        }
    }
    out[i] = sum;
}

// 1×1 convolution as a tiled matrix product out[co, px] = Σ_ci weight[co, ci] a[ci, px] + c[co],
// 16×16 outputs per workgroup. p: 1 C_in, 2 pixels, 3 C_out
var<workgroup> tile_w: array<array<f32, 16>, 16>;
var<workgroup> tile_x: array<array<f32, 16>, 16>;
@compute @workgroup_size(16, 16) fn pointwise(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
    let c_in = p(1); let pixels = p(2); let c_out = p(3);
    let px = group.x * 16u + local.x; let co = group.y * 16u + local.y;
    var sum = 0.0;
    for (var k0 = 0u; k0 < c_in; k0 += 16u) {
        let wk = k0 + local.x; let xk = k0 + local.y;
        var w = 0.0; var x = 0.0;
        if (co < c_out && wk < c_in) { w = weight(co * c_in + wk); }
        if (px < pixels && xk < c_in) { x = a[xk * pixels + px]; }
        tile_w[local.y][local.x] = w;
        tile_x[local.y][local.x] = x;
        workgroupBarrier();
        for (var k = 0u; k < 16u; k++) { sum += tile_w[local.y][k] * tile_x[k][local.x]; }
        workgroupBarrier();
    }
    if (px < pixels && co < c_out) { out[co * pixels + px] = sum + c[co]; }
}

// p: 1 channel size (H·W), 2 channels
@compute @workgroup_size(64) fn prelu(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let x = a[i];
    out[i] = select(x * b[(i / p(1)) % p(2)], x, x >= 0.0);
}

// p: 1 min, 2 max (f32 bits)
@compute @workgroup_size(64) fn clip(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    out[i] = clamp(a[i], pf(1), pf(2));
}

@compute @workgroup_size(64) fn add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    out[i] = a[i] + b[i];
}

@compute @workgroup_size(64) fn sigmoid(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    out[i] = 1.0 / (1.0 + exp(-a[i]));
}

// p: 1 H, 2 W, 3 OH, 4 OW, 5 KH, 6 KW, 7 SH, 8 SW, 9 pad top, 10 pad left
@compute @workgroup_size(64) fn max_pool(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let h = p(1); let w = p(2); let oh = p(3); let ow = p(4);
    let ox = i % ow; let oy = (i / ow) % oh; let ch = i / (ow * oh);
    var best = -3.4e38;
    for (var ky = 0u; ky < p(5); ky++) {
        let y = i32(oy * p(7) + ky) - i32(p(9));
        if (y < 0 || y >= i32(h)) { continue; }
        for (var kx = 0u; kx < p(6); kx++) {
            let x = i32(ox * p(8) + kx) - i32(p(10));
            if (x < 0 || x >= i32(w)) { continue; }
            best = max(best, a[(ch * h + u32(y)) * w + u32(x)]);
        }
    }
    out[i] = best;
}

// 4D: p 1..4 output dims, 5..8 input dims, 9..12 begin pads; zero outside the input
@compute @workgroup_size(64) fn pad(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    var rest = i; var offset = 0u; var stride = 1u; var inside = true;
    for (var axis = 3; axis >= 0; axis--) {
        let ua = u32(axis);
        let coord = i32(rest % p(1u + ua)) - i32(p(9u + ua)); rest /= p(1u + ua);
        if (coord < 0 || coord >= i32(p(5u + ua))) { inside = false; }
        offset += u32(max(coord, 0)) * stride; stride *= p(5u + ua);
    }
    out[i] = select(0.0, a[offset], inside);
}

// Linear, half_pixel, per channel plane. p: 1 H, 2 W, 3 OH, 4 OW
@compute @workgroup_size(64) fn resize_linear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let h = p(1); let w = p(2); let oh = p(3); let ow = p(4);
    let ox = i % ow; let oy = (i / ow) % oh; let ch = i / (ow * oh);
    let sy = clamp((f32(oy) + 0.5) * f32(h) / f32(oh) - 0.5, 0.0, f32(h - 1u));
    let sx = clamp((f32(ox) + 0.5) * f32(w) / f32(ow) - 0.5, 0.0, f32(w - 1u));
    let y0 = u32(floor(sy)); let x0 = u32(floor(sx));
    let y1 = min(y0 + 1u, h - 1u); let x1 = min(x0 + 1u, w - 1u);
    let fy = sy - f32(y0); let fx = sx - f32(x0);
    let base = ch * h * w;
    let top = mix(a[base + y0 * w + x0], a[base + y0 * w + x1], fx);
    let bottom = mix(a[base + y1 * w + x0], a[base + y1 * w + x1], fx);
    out[i] = mix(top, bottom, fy);
}

// 4D: p 1..4 output dims, 5..8 input strides of each output axis
@compute @workgroup_size(64) fn transpose(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    var rest = i; var offset = 0u;
    for (var axis = 3; axis >= 0; axis--) {
        let ua = u32(axis);
        offset += (rest % p(1u + ua)) * p(5u + ua); rest /= p(1u + ua);
    }
    out[i] = a[offset];
}

// Two inputs joined along an axis: p 1 inner size, 2 a's axis length, 3 b's axis length
@compute @workgroup_size(64) fn concat(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let inner = p(1); let la = p(2); let lb = p(3);
    let k = i % inner; let j = (i / inner) % (la + lb); let o = i / (inner * (la + lb));
    out[i] = select(b[(o * lb + j - la) * inner + k], a[(o * la + j) * inner + k], j < la);
}

// p: 1 plane size
@compute @workgroup_size(64) fn global_average_pool(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    var sum = 0.0;
    for (var k = 0u; k < p(1); k++) { sum += a[i * p(1) + k]; }
    out[i] = sum / f32(p(1));
}

// out[m, n] = Σ_k a[m, k] weight[k, n] + c[n]. p: 1 K, 2 N
@compute @workgroup_size(64) fn gemm(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = index(gid, groups); if (i >= p(0)) { return; }
    let k_len = p(1); let n_len = p(2);
    let n = i % n_len; let m = i / n_len;
    var sum = c[n];
    for (var k = 0u; k < k_len; k++) { sum += a[m * k_len + k] * weight(k * n_len + n); }
    out[i] = sum;
}
"#;

/// IEEE half-precision bits of `value`, rounded to nearest even.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Inf or NaN
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        // Subnormal half (or zero): shift the implicit-1 mantissa into place
        if half_exponent < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let half = full >> shift;
        let rest = full & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let round = rest > halfway || (rest == halfway && half & 1 == 1);
        return sign | (half + u32::from(round)) as u16;
    }
    let half = ((half_exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    let round = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa correctly bumps the exponent (up to infinity)
    sign | (half + u32::from(round)) as u16
}

/// A tensor on the GPU: its buffer (shared by views) and shape.
#[derive(Clone)]
struct Tensor {
    buffer: Rc<wgpu::Buffer>,
    shape: Vec<usize>,
}

fn elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

struct Step {
    pipeline: usize,
    bind_group: wgpu::BindGroup,
    workgroups: (u32, u32),
}

const ENTRY_POINTS: [&str; 13] = [
    "conv",
    "pointwise",
    "prelu",
    "clip",
    "add",
    "sigmoid",
    "max_pool",
    "pad",
    "resize_linear",
    "transpose",
    "concat",
    "global_average_pool",
    "gemm",
];

fn entry(name: &str) -> usize {
    ENTRY_POINTS
        .iter()
        .position(|entry| *entry == name)
        .expect("known entry point")
}

pub struct GpuNet {
    input: Rc<wgpu::Buffer>,
    outputs: Vec<Tensor>,
    pipelines: Vec<wgpu::ComputePipeline>,
    steps: Vec<Step>,
}

struct Builder<'a> {
    device: &'a wgpu::Device,
    graph: &'a Graph,
    layout: wgpu::BindGroupLayout,
    tensors: HashMap<String, Tensor>,
    steps: Vec<Step>,
    dummy: Rc<wgpu::Buffer>,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl<'a> Builder<'a> {
    fn initializer(&self, name: &str) -> Result<&'a Initializer, String> {
        self.graph
            .initializers
            .get(name)
            .ok_or_else(|| format!("gpu_net: {name} is not an initializer"))
    }

    fn ints(&self, name: &str) -> Result<Vec<i64>, String> {
        match &self.initializer(name)?.data {
            TensorData::I64(values) => Ok(values.clone()),
            TensorData::F32(_) => Err(format!("gpu_net: {name} is not int64")),
        }
    }

    fn scalar(&self, name: &str) -> Result<f32, String> {
        match &self.initializer(name)?.data {
            TensorData::F32(values) if values.len() == 1 => Ok(values[0]),
            _ => Err(format!("gpu_net: {name} is not an f32 scalar")),
        }
    }

    /// The tensor `name`: an earlier output, or an initializer uploaded on first use.
    fn tensor(&mut self, name: &str) -> Result<Tensor, String> {
        if let Some(tensor) = self.tensors.get(name) {
            return Ok(tensor.clone());
        }
        let init = self.initializer(name)?;
        let TensorData::F32(values) = &init.data else {
            return Err(format!("gpu_net: {name} is not f32"));
        };
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let tensor = Tensor {
            buffer: Rc::new(buffer),
            shape: init.dims.clone(),
        };
        self.tensors.insert(name.to_string(), tensor.clone());
        Ok(tensor)
    }

    /// The f32 initializer `name` as an f16-pair buffer (`weights` in the shader), with its shape.
    fn half_weights(&self, name: &str) -> Result<Tensor, String> {
        let init = self.initializer(name)?;
        let TensorData::F32(values) = &init.data else {
            return Err(format!("gpu_net: {name} is not f32"));
        };
        let packed: Vec<u32> = values
            .chunks(2)
            .map(|pair| {
                u32::from(f32_to_f16(pair[0])) | (u32::from(f32_to_f16(*pair.get(1).unwrap_or(&0.0))) << 16)
            })
            .collect();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(Tensor {
            buffer: Rc::new(buffer),
            shape: init.dims.clone(),
        })
    }

    fn output(&mut self, node: &Node, shape: Vec<usize>) -> Tensor {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&node.outputs[0]),
            size: (elements(&shape).max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tensor = Tensor {
            buffer: Rc::new(buffer),
            shape,
        };
        self.tensors.insert(node.outputs[0].clone(), tensor.clone());
        tensor
    }

    /// Placeholder for a binding a kernel doesn't read.
    fn unused(&self) -> Tensor {
        Tensor {
            buffer: self.dummy.clone(),
            shape: vec![1],
        }
    }

    fn view(&mut self, node: &Node, input: Tensor, shape: Vec<usize>) {
        self.tensors.insert(
            node.outputs[0].clone(),
            Tensor {
                buffer: input.buffer,
                shape,
            },
        );
    }

    /// One dispatch of `entry` over `out`'s elements; `params[0]` is set to the element count.
    fn dispatch(&mut self, entry_point: &str, inputs: &[&Tensor], out: &Tensor, params: &[u32]) {
        let groups = (elements(&out.shape) as u32).div_ceil(WORKGROUP_SIZE);
        let workgroups = (groups.min(MAX_WORKGROUPS_X), groups.div_ceil(MAX_WORKGROUPS_X));
        self.dispatch_groups(entry_point, inputs, out, params, workgroups);
    }

    /// `dispatch` with an explicit workgroup grid.
    fn dispatch_groups(
        &mut self,
        entry_point: &str,
        inputs: &[&Tensor],
        out: &Tensor,
        params: &[u32],
        workgroups: (u32, u32),
    ) {
        let count = elements(&out.shape) as u32;
        let mut block = [0u32; PARAM_COUNT];
        block[0] = count;
        block[1..1 + params.len()].copy_from_slice(params);
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_net_params"),
                contents: bytemuck::cast_slice(&block),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffer = |i: usize| -> &wgpu::Buffer {
            inputs.get(i).map_or(&self.dummy, |tensor| &tensor.buffer)
        };
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(entry_point),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer(0).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer(1).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer(2).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffer(3).as_entire_binding(),
                },
            ],
        });
        self.steps.push(Step {
            pipeline: entry(entry_point),
            bind_group,
            workgroups,
        });
    }

    fn node(&mut self, node: &Node) -> Result<(), String> {
        let x = self.tensor(&node.inputs[0])?;
        let u = |v: usize| v as u32;
        match node.op.as_str() {
            "Conv" => {
                let weight = self.half_weights(&node.inputs[1])?;
                let bias = self.tensor(node.inputs.get(2).ok_or("gpu_net: Conv without bias")?)?;
                let [_, c_in, h, w] = x.shape[..] else {
                    return Err("gpu_net: Conv input is not 4D".to_string());
                };
                let [c_out, _, kh, kw] = weight.shape[..] else {
                    return Err("gpu_net: Conv weight is not 4D".to_string());
                };
                let strides = node.ints("strides").unwrap_or(&[1, 1]).to_vec();
                let pads = node.ints("pads").unwrap_or(&[0, 0, 0, 0]).to_vec();
                if node.ints("dilations").is_some_and(|d| d.iter().any(|&v| v != 1)) {
                    return Err("gpu_net: Conv dilation is not supported".to_string());
                }
                let (sh, sw) = (strides[0] as usize, strides[1] as usize);
                let oh = (h + pads[0] as usize + pads[2] as usize - kh) / sh + 1;
                let ow = (w + pads[1] as usize + pads[3] as usize - kw) / sw + 1;
                let out = self.output(node, vec![1, c_out, oh, ow]);
                let group = node.int("group", 1) as u32;
                if (kh, kw, sh, sw, group) == (1, 1, 1, 1, 1) && pads.iter().all(|&p| p == 0) {
                    let pixels = h * w;
                    self.dispatch_groups(
                        "pointwise",
                        &[&x, &self.unused(), &bias, &weight],
                        &out,
                        &[u(c_in), u(pixels), u(c_out)],
                        (u(pixels).div_ceil(16), u(c_out).div_ceil(16)),
                    );
                    return Ok(());
                }
                self.dispatch(
                    "conv",
                    &[&x, &self.unused(), &bias, &weight],
                    &out,
                    &[
                        u(c_in),
                        u(h),
                        u(w),
                        u(c_out),
                        u(oh),
                        u(ow),
                        u(kh),
                        u(kw),
                        u(sh),
                        u(sw),
                        pads[0] as u32,
                        pads[1] as u32,
                        group,
                    ],
                );
            }
            "PRelu" => {
                let slope = self.tensor(&node.inputs[1])?;
                let channels = elements(&slope.shape);
                let plane = x.shape.get(2..).map_or(1, elements);
                let out = self.output(node, x.shape.clone());
                self.dispatch("prelu", &[&x, &slope], &out, &[u(plane), u(channels)]);
            }
            "Clip" => {
                let min = self.scalar(&node.inputs[1])?;
                let max = self.scalar(&node.inputs[2])?;
                let out = self.output(node, x.shape.clone());
                self.dispatch("clip", &[&x], &out, &[min.to_bits(), max.to_bits()]);
            }
            "Add" => {
                let y = self.tensor(&node.inputs[1])?;
                if elements(&y.shape) != elements(&x.shape) {
                    return Err("gpu_net: Add with broadcasting is not supported".to_string());
                }
                let out = self.output(node, x.shape.clone());
                self.dispatch("add", &[&x, &y], &out, &[]);
            }
            "Sigmoid" => {
                let out = self.output(node, x.shape.clone());
                self.dispatch("sigmoid", &[&x], &out, &[]);
            }
            "MaxPool" => {
                let [n, c, h, w] = x.shape[..] else {
                    return Err("gpu_net: MaxPool input is not 4D".to_string());
                };
                let kernel = node.ints("kernel_shape").ok_or("gpu_net: MaxPool kernel")?.to_vec();
                let strides = node.ints("strides").unwrap_or(&[1, 1]).to_vec();
                let pads = node.ints("pads").unwrap_or(&[0, 0, 0, 0]).to_vec();
                let oh = (h + (pads[0] + pads[2]) as usize - kernel[0] as usize) / strides[0] as usize + 1;
                let ow = (w + (pads[1] + pads[3]) as usize - kernel[1] as usize) / strides[1] as usize + 1;
                let out = self.output(node, vec![n, c, oh, ow]);
                self.dispatch(
                    "max_pool",
                    &[&x],
                    &out,
                    &[
                        u(h),
                        u(w),
                        u(oh),
                        u(ow),
                        kernel[0] as u32,
                        kernel[1] as u32,
                        strides[0] as u32,
                        strides[1] as u32,
                        pads[0] as u32,
                        pads[1] as u32,
                    ],
                );
            }
            "Pad" => {
                let pads = self.ints(&node.inputs[1])?;
                if node.string("mode").is_some_and(|mode| mode != "constant") || pads.len() != 8 {
                    return Err("gpu_net: only 4D constant Pad is supported".to_string());
                }
                let shape: Vec<usize> = (0..4)
                    .map(|axis| x.shape[axis] + (pads[axis] + pads[axis + 4]) as usize)
                    .collect();
                let out = self.output(node, shape.clone());
                let mut params: Vec<u32> = shape.iter().map(|&d| u(d)).collect();
                params.extend(x.shape.iter().map(|&d| u(d)));
                params.extend(pads[..4].iter().map(|&p| p as u32));
                self.dispatch("pad", &[&x], &out, &params);
            }
            "Resize" => {
                let sizes = self.ints(node.inputs.get(3).ok_or("gpu_net: Resize without sizes")?)?;
                if node.string("mode") != Some("linear")
                    || node.string("coordinate_transformation_mode") != Some("half_pixel")
                {
                    return Err("gpu_net: only linear half_pixel Resize is supported".to_string());
                }
                let shape: Vec<usize> = sizes.iter().map(|&d| d as usize).collect();
                let out = self.output(node, shape.clone());
                self.dispatch(
                    "resize_linear",
                    &[&x],
                    &out,
                    &[u(x.shape[2]), u(x.shape[3]), u(shape[2]), u(shape[3])],
                );
            }
            "Transpose" => {
                let perm = node.ints("perm").ok_or("gpu_net: Transpose perm")?.to_vec();
                if x.shape.len() != 4 {
                    return Err("gpu_net: Transpose input is not 4D".to_string());
                }
                let strides = [x.shape[1] * x.shape[2] * x.shape[3], x.shape[2] * x.shape[3], x.shape[3], 1];
                let shape: Vec<usize> = perm.iter().map(|&axis| x.shape[axis as usize]).collect();
                let out = self.output(node, shape.clone());
                let mut params: Vec<u32> = shape.iter().map(|&d| u(d)).collect();
                params.extend(perm.iter().map(|&axis| u(strides[axis as usize])));
                self.dispatch("transpose", &[&x], &out, &params);
            }
            "Reshape" => {
                let target = self.ints(&node.inputs[1])?;
                let known: usize = target
                    .iter()
                    .enumerate()
                    .map(|(axis, &d)| match d {
                        0 => x.shape[axis],
                        -1 => 1,
                        d => d as usize,
                    })
                    .product();
                let shape = target
                    .iter()
                    .enumerate()
                    .map(|(axis, &d)| match d {
                        0 => x.shape[axis],
                        -1 => elements(&x.shape) / known,
                        d => d as usize,
                    })
                    .collect();
                self.view(node, x, shape);
            }
            "Squeeze" => {
                let axes = self.ints(&node.inputs[1])?;
                let shape = x
                    .shape
                    .iter()
                    .enumerate()
                    .filter(|(axis, _)| !axes.contains(&(*axis as i64)))
                    .map(|(_, &d)| d)
                    .collect();
                self.view(node, x, shape);
            }
            "Concat" => {
                let y = self.tensor(&node.inputs[1])?;
                if node.inputs.len() != 2 {
                    return Err("gpu_net: Concat of more than 2 inputs".to_string());
                }
                let axis = node.int("axis", 0) as usize;
                let inner = elements(&x.shape[axis + 1..]);
                let mut shape = x.shape.clone();
                shape[axis] += y.shape[axis];
                let out = self.output(node, shape);
                self.dispatch("concat", &[&x, &y], &out, &[u(inner), u(x.shape[axis]), u(y.shape[axis])]);
            }
            "GlobalAveragePool" => {
                let plane = elements(&x.shape[2..]);
                let out = self.output(node, vec![x.shape[0], x.shape[1], 1, 1]);
                self.dispatch("global_average_pool", &[&x], &out, &[u(plane)]);
            }
            "Gemm" => {
                let weight = self.half_weights(&node.inputs[1])?;
                let bias = self.tensor(node.inputs.get(2).ok_or("gpu_net: Gemm without bias")?)?;
                if node.int("transA", 0) != 0 || node.int("transB", 0) != 0 {
                    return Err("gpu_net: transposed Gemm is not supported".to_string());
                }
                let (m, k) = (x.shape[0], x.shape[1]);
                let n = weight.shape[1];
                let out = self.output(node, vec![m, n]);
                self.dispatch("gemm", &[&x, &self.unused(), &bias, &weight], &out, &[u(k), u(n)]);
            }
            op => return Err(format!("gpu_net: unsupported operator {op}")),
        }
        Ok(())
    }
}

impl GpuNet {
    /// Builds the dispatches of `graph` for an input of `input_shape`.
    pub fn new(device: &wgpu::Device, graph: &Graph, input_shape: &[usize]) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_net"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, true),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_net"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_net"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipelines = ENTRY_POINTS
            .iter()
            .map(|entry_point| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        let input_name = graph.inputs.first().ok_or("gpu_net: graph without input")?;
        let input = Rc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_net_input"),
            size: (elements(input_shape) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let dummy = Rc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_net_unused"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        let mut builder = Builder {
            device,
            graph,
            layout,
            tensors: HashMap::new(),
            steps: Vec::new(),
            dummy,
        };
        builder.tensors.insert(
            input_name.clone(),
            Tensor {
                buffer: input.clone(),
                shape: input_shape.to_vec(),
            },
        );
        for node in &graph.nodes {
            builder
                .node(node)
                .map_err(|error| format!("{error} (node {})", node.outputs[0]))?;
        }
        let outputs = graph
            .outputs
            .iter()
            .map(|name| builder.tensor(name))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            input,
            outputs,
            pipelines,
            steps: builder.steps,
        })
    }

    /// Input buffer (f32, `input_shape` row-major); write it before `encode`.
    pub fn input(&self) -> &wgpu::Buffer {
        &self.input
    }

    /// Output buffers and their element counts, in graph output order.
    pub fn outputs(&self) -> impl Iterator<Item = (&wgpu::Buffer, usize)> {
        self.outputs
            .iter()
            .map(|tensor| (&*tensor.buffer, elements(&tensor.shape)))
    }

    /// Records the whole graph into `encoder`.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_net"),
            timestamp_writes: None,
        });
        for step in &self.steps {
            pass.set_pipeline(&self.pipelines[step.pipeline]);
            pass.set_bind_group(0, &step.bind_group, &[]);
            pass.dispatch_workgroups(step.workgroups.0, step.workgroups.1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::{self, HandRoi, Image};
    use rten_tensor::prelude::*;

    const PALM_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/palm_detection_mediapipe_2023feb.onnx"
    );
    const HAND_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/handpose_estimation_mediapipe_2023feb.onnx"
    );

    fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("device")
    }

    /// Runs `onnx` on the GPU and with RTen for `input`; returns the largest output difference
    /// relative to the largest RTen output magnitude.
    fn relative_error(onnx: &str, input_shape: [usize; 4], input: Vec<f32>) -> f32 {
        let bytes = std::fs::read(onnx).expect("model");
        let (device, queue) = device();
        let net = GpuNet::new(&device, &crate::onnx::parse(&bytes).expect("parse"), &input_shape)
            .expect("gpu net");
        queue.write_buffer(net.input(), 0, bytemuck::cast_slice(&input));
        let mut encoder = device.create_command_encoder(&Default::default());
        net.encode(&mut encoder);
        let readbacks: Vec<(wgpu::Buffer, usize)> = net
            .outputs()
            .map(|(buffer, len)| {
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (len * 4) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, (len * 4) as u64);
                (readback, len)
            })
            .collect();
        queue.submit([encoder.finish()]);
        for (readback, _) in &readbacks {
            readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        }
        device.poll(wgpu::PollType::Wait).expect("poll");

        let model = rten::Model::load(bytes).expect("rten model");
        let tensor = rten_tensor::NdTensor::from_data(input_shape, input);
        let expected = model
            .run(vec![(model.input_ids()[0], tensor.view().into())], model.output_ids(), None)
            .expect("rten run");
        let mut worst = 0.0f32;
        for ((readback, len), expected) in readbacks.iter().zip(expected) {
            let expected: rten_tensor::Tensor<f32> = expected.try_into().expect("f32 output");
            let expected = expected.to_vec();
            assert_eq!(expected.len(), *len);
            let gpu: Vec<f32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
            let scale = expected.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
            let error = gpu.iter().zip(&expected).fold(0.0f32, |m, (g, e)| m.max((g - e).abs()));
            worst = worst.max(error / scale);
        }
        worst
    }

    fn victory() -> Image {
        let decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/victory.png")).expect("png"),
        ));
        let mut reader = decoder.read_info().expect("png header");
        let mut rgba = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut rgba).expect("png frame");
        Image {
            width: info.width as usize,
            height: info.height as usize,
            rgba,
        }
    }

    #[test]
    fn palm_model_victory_input_gpu_matches_rten() {
        let input = hands::palm_input(&victory());
        let error = relative_error(PALM_ONNX, [1, 192, 192, 3], input);
        assert!(error < 1e-3, "relative error {error}");
    }

    #[test]
    fn hand_model_victory_crop_gpu_matches_rten() {
        let image = victory();
        let roi = HandRoi::from_palm(
            pill_engine::game::Vector2f::new(110.0, 150.0),
            pill_engine::game::Vector2f::new(220.0, 290.0),
            &[pill_engine::game::Vector2f::new(165.0, 280.0); 7],
        );
        let error = relative_error(HAND_ONNX, [1, 224, 224, 3], roi.input(&image));
        assert!(error < 1e-3, "relative error {error}");
    }
}

