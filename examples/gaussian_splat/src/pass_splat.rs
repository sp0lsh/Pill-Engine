use crate::gaussian_cloud::{GaussianCloud, GaussianCloudSource};
use pill_engine::game::{Pass, PillSlotMapKey};
use pill_engine::internal::{PillRenderer, WorldQuery};

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
#[cfg(target_arch = "wasm32")]
use std::io::BufReader;
use wgpu_3dgs_viewer::core::{Gaussians, GaussiansSource, IterGaussian};

// ── GPU-side types ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplat {
    position: [f32; 3],
    _pad0: f32,
    scale: [f32; 3],
    _pad1: f32,
    color: [f32; 4],
    quat: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

// ── Shader ────────────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct CameraUniform {
    view     : mat4x4<f32>,
    proj     : mat4x4<f32>,
    viewport : vec2<f32>,
    _pad     : vec2<f32>,
}

struct Splat {
    position : vec3<f32>,
    _pad0    : f32,
    scale    : vec3<f32>,
    _pad1    : f32,
    color    : vec4<f32>,
    quat     : vec4<f32>,
}

@group(0) @binding(0) var<uniform>         cam    : CameraUniform;
@group(0) @binding(1) var<storage, read>   splats : array<Splat>;

struct VOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       uv    : vec2<f32>,
    @location(1)       color : vec4<f32>,
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+z*w),      2.0*(x*z-y*w)),
        vec3<f32>(2.0*(x*y-z*w),         1.0-2.0*(x*x+z*z),  2.0*(y*z+x*w)),
        vec3<f32>(2.0*(x*z+y*w),         2.0*(y*z-x*w),      1.0-2.0*(x*x+y*y)),
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    let iid = vid / 6u;
    let cid = vid % 6u;
    var out: VOut;

    let s = splats[iid];

    let cp4  = cam.view * vec4<f32>(s.position, 1.0);
    let t    = cp4.xyz;

    if t.z > -0.1 {
        out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let R      = quat_to_mat3(s.quat);
    let M      = mat3x3<f32>(R[0]*s.scale.x, R[1]*s.scale.y, R[2]*s.scale.z);
    let sigma3 = M * transpose(M);

    let W = mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);

    let fx  = cam.proj[0][0] * cam.viewport.x * 0.5;
    let fy  = cam.proj[1][1] * cam.viewport.y * 0.5;
    let iz  = 1.0 / t.z;
    let iz2 = iz * iz;

    let J = mat3x2<f32>(
        vec2<f32>(-fx * iz,        0.0        ),
        vec2<f32>( 0.0,           -fy * iz    ),
        vec2<f32>( fx * t.x*iz2,  fy * t.y*iz2),
    );

    let T      = J * W;
    let sigma2 = (T * sigma3) * transpose(T);

    let a = sigma2[0][0];
    let b = sigma2[0][1];
    let c = sigma2[1][1];

    let mid = 0.5 * (a + c);
    let rr  = sqrt(max(0.0, mid*mid - (a*c - b*b)));
    let l1  = mid + rr;
    let l2  = mid - rr;

    var ev1: vec2<f32>;
    if abs(b) > 1e-4 { ev1 = normalize(vec2<f32>(b, l1 - a)); }
    else              { ev1 = vec2<f32>(1.0, 0.0); }
    let ev2 = vec2<f32>(-ev1.y, ev1.x);

    let to_ndc = vec2<f32>(2.0 / cam.viewport.x, 2.0 / cam.viewport.y);
    let ax1    = ev1 * sqrt(abs(l1)) * 3.0 * to_ndc;
    let ax2    = ev2 * sqrt(abs(l2)) * 3.0 * to_ndc;

    let dc = array<vec2<f32>, 6>(
        -ax1-ax2,  ax1-ax2, -ax1+ax2,
        -ax1+ax2,  ax1-ax2,  ax1+ax2,
    );
    let uvs = array<vec2<f32>, 6>(
        vec2(-1.,-1.), vec2(1.,-1.), vec2(-1., 1.),
        vec2(-1., 1.), vec2(1.,-1.), vec2( 1., 1.),
    );

    let clip = cam.proj * cp4;
    let ndc  = clip.xy / clip.w;

    out.pos   = vec4<f32>(ndc + dc[cid], clip.z / clip.w, 1.0);
    out.uv    = uvs[cid] * 3.0;
    out.color = s.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let alpha = in.color.a * exp(-0.5 * dot(in.uv, in.uv));
    if alpha < 1.0 / 255.0 { discard; }
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

// ── Pipeline state ────────────────────────────────────────────────────────────

struct SplatState {
    pipeline:   wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    splat_buf:  wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    splats:     Vec<GpuSplat>,
}

// ── Public pass ───────────────────────────────────────────────────────────────

pub struct PassSplat {
    cloud_name: String,
    state:      Option<SplatState>,
}

impl PassSplat {
    pub fn new(cloud_name: &str) -> Self {
        Self { cloud_name: cloud_name.to_string(), state: None }
    }

    fn load_splats(source: &GaussianCloudSource) -> Result<Vec<GpuSplat>> {
        let gaussians = match source {
            GaussianCloudSource::Path(path) => {
                [GaussiansSource::Ply, GaussiansSource::Spz]
                    .into_iter()
                    .find_map(|fmt| Gaussians::read_from_file(path, fmt).ok())
                    .ok_or_else(|| anyhow!("cannot load {:?} as PLY or SPZ", path))?
            }
            #[cfg(target_arch = "wasm32")]
            GaussianCloudSource::Bytes(bytes) => {
                let mut r = BufReader::new(bytes.as_slice());
                Gaussians::read_from(&mut r, GaussiansSource::Ply)
                    .or_else(|_| {
                        let mut r = BufReader::new(bytes.as_slice());
                        Gaussians::read_from(&mut r, GaussiansSource::Spz)
                    })
                    .map_err(|e| anyhow!("cannot load gaussian bytes: {e}"))?
            }
        };

        Ok(gaussians
            .iter_gaussian()
            .map(|g| {
                let rgba = g.color.as_vec4() / 255.0;
                // Y-axis flip: negate position.y; conjugate quaternion through S_y → (-qx, qy, -qz, qw)
                GpuSplat {
                    position: [g.pos.x, -g.pos.y, g.pos.z],
                    _pad0: 0.0,
                    scale: g.scale.to_array(),
                    _pad1: 0.0,
                    color: rgba.to_array(),
                    quat: [-g.rot.x, g.rot.y, -g.rot.z, g.rot.w],
                }
            })
            .collect())
    }

    fn init_pipeline(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        splats:         Vec<GpuSplat>,
    ) -> Result<SplatState> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("pass_splat_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("pass_splat_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("pass_splat_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let linear_format = surface_format.remove_srgb_suffix();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("pass_splat_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: linear_format,
                    blend:  Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pass_splat_camera"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let splat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pass_splat_data"),
            size:               (splats.len() * std::mem::size_of::<GpuSplat>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("pass_splat_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buf.as_entire_binding() },
            ],
        });

        Ok(SplatState { pipeline, camera_buf, splat_buf, bind_group, splats })
    }
}

impl Pass for PassSplat {
    fn get_label(&self) -> &str { "pass_splat" }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> { Ok(()) }

    fn draw(
        &mut self,
        encoder:  &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame:    &wgpu::SurfaceTexture,
        _view:    &wgpu::TextureView,
        world:    &WorldQuery<'_>,
    ) -> Result<()> {
        if self.state.is_none() {
            let cloud = world
                .resources
                .get_resource_by_name::<GaussianCloud>(&self.cloud_name)
                .map_err(|_| anyhow!("GaussianCloud '{}' not found", self.cloud_name))?;

            let splats = Self::load_splats(&cloud.source)?;
            self.state = Some(Self::init_pipeline(
                renderer.get_device(),
                renderer.get_surface_format(),
                splats,
            )?);
        }

        let state = self.state.as_mut().unwrap();
        let queue = renderer.get_queue();

        let entity_idx = world.active_camera.data().index as usize;

        let cam = world
            .camera_components
            .data
            .get(entity_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| anyhow!("active camera component not found"))?;

        let xform = world
            .transform_components
            .data
            .get(entity_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| anyhow!("active camera transform not found"))?;

        let tex_size = frame.texture.size();
        let aspect   = tex_size.width as f32 / tex_size.height as f32;

        let pitch   = Mat3::from_rotation_x(xform.rotation.x.to_radians());
        let yaw     = Mat3::from_rotation_y(xform.rotation.y.to_radians());
        let forward = (yaw * pitch) * Vec3::Z;
        let view    = Mat4::look_to_rh(xform.position, forward, Vec3::Y);
        let proj    = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.range.start, cam.range.end);

        let view_z = -forward;
        state.splats.sort_unstable_by(|a, b| {
            let da = Vec3::from(a.position).dot(view_z);
            let db = Vec3::from(b.position).dot(view_z);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        queue.write_buffer(&state.splat_buf, 0, bytemuck::cast_slice(&state.splats));

        let uniform = CameraUniform {
            view:     view.to_cols_array_2d(),
            proj:     proj.to_cols_array_2d(),
            viewport: [tex_size.width as f32, tex_size.height as f32],
            _pad:     [0.0; 2],
        };
        queue.write_buffer(&state.camera_buf, 0, bytemuck::bytes_of(&uniform));

        let linear_fmt  = renderer.get_surface_format().remove_srgb_suffix();
        let render_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(linear_fmt),
            ..Default::default()
        });

        let count = state.splats.len() as u32;
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_splat"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           &render_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.07, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set:      None,
            timestamp_writes:         None,
        });

        rpass.set_pipeline(&state.pipeline);
        rpass.set_bind_group(0, &state.bind_group, &[]);
        rpass.draw(0..count * 6, 0..1);

        Ok(())
    }
}
