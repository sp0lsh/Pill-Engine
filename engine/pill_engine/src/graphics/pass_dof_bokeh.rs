//! Depth of Field (DoF) post-processing pass (single-pass bokeh, disk kernel)

use crate::graphics::Pass;
use crate::graphics::PillRenderer;
use crate::graphics::RendererTextureHandle;
use crate::renderer::Renderer;
use crate::resources::ResourceManager;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::TextureFormat;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DofParams {
    focus_distance: f32,
    aperture: f32,
    focal_length: f32,
    sensor_size: f32,
    time_s: f32,
    focus_point: f32,
    focus_scale: f32,
}

pub struct PassDofBokeh {
    pub name: String,
    pub input_color: RendererTextureHandle,
    pub input_depth: RendererTextureHandle,
    pub output: RendererTextureHandle,
    pub format: TextureFormat,
    // Global DoF params (hardcoded for now)
    pub focus_distance: f32,
    pub aperture: f32,
    pub focal_length: f32,
    pub sensor_size: f32,
    // Pipeline and bind group
    pipeline: Option<crate::graphics::PipelineV2>,
    bind_group: Option<wgpu::BindGroup>,
    params_buffer: Option<wgpu::Buffer>,
}

impl PassDofBokeh {
    pub fn new(
        name: &str,
        input_color: RendererTextureHandle,
        input_depth: RendererTextureHandle,
        output: RendererTextureHandle,
        format: TextureFormat,
    ) -> Self {
        Self {
            name: name.to_string(),
            input_color,
            input_depth,
            output,
            format,
            focus_distance: 5.0, // meters
            aperture: 2.8,       // f-stop
            focal_length: 50.0,  // mm
            sensor_size: 36.0,   // mm (full-frame)
            pipeline: None,
            bind_group: None,
            params_buffer: None,
        }
    }

    // Optionally: methods to update params, toggle debug, etc.
}

impl Pass for PassDofBokeh {
    fn get_label(&self) -> &str {
        &self.name
    }

    fn init(
        &mut self,
        renderer: &mut dyn PillRenderer,
        resources: &mut crate::resources::ResourceManager,
    ) -> Result<()> {
        let device = renderer.get_device();
        eprintln!("[DoF] Initializing PassDofBokeh: {}", self.name);
        // Only this inlined WGSL shader is used for DoF (vertex + fragment)
        let vs = r#"
        struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
        @vertex fn main(@builtin(vertex_index) vi: u32) -> VSOut {
            var pos = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -3.0),
                vec2<f32>( 3.0,  1.0),
                vec2<f32>(-1.0,  1.0)
            );
            var tuv = array<vec2<f32>, 3>(
                vec2<f32>(0.0, 2.0),
                vec2<f32>(2.0, 0.0),
                vec2<f32>(0.0, 0.0)
            );
            var o: VSOut;
            o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
            o.uv = tuv[vi];
            return o;
        }
        "#;

        // WGSL fragment shader for step-by-step DoF debug and effect
        // Set debug mode here: 0 = pass-through, 1 = depth/CoC debug, 2 = DoF effect
        // To change mode, edit the value below and rebuild.
        let fs = r#"
        struct Params {
            focus_distance: f32,
            aperture: f32,
            focal_length: f32,
            sensor_size: f32,
            time_s: f32,
            focus_point: f32, // meters
            focus_scale: f32,
        };
        @group(0) @binding(0) var t_color: texture_2d<f32>;
        // Linear view-space depth in meters (R16Float) from PassLinearizeDepth
        @group(0) @binding(1) var t_depth: texture_2d<f32>;
        @group(0) @binding(2) var<uniform> params: Params;
        @group(0) @binding(3) var s_color: sampler;

        // NOTE: Split view for proof:
        // - uv.y > 0.5 : debug visualization
        // - uv.y <= 0.5: DoF result
        const DEBUG_VIEW: i32 = 2; // 0=None, 1=pass-through, 2=abs(CoC), 3=linearDepthNormalized

        const GOLDEN_ANGLE: f32 = 2.39996323;
        const MAX_BLUR_SIZE: f32 = 20.0;
        const RAD_SCALE: f32 = 0.5; // Smaller = nicer blur, larger = faster
        const FAR: f32 = 100.0;
        const NEAR: f32 = 0.1;
        // Artistic CoC remap: <1 boosts small CoC (bigger bokeh for far objects) without increasing MAX_BLUR_SIZE.
        const COC_CURVE: f32 = 0.5; // sqrt

        fn load_linear_depth(uv: vec2<f32>, dims: vec2<u32>) -> f32 {
            let coord0 = vec2<i32>(uv * vec2<f32>(dims));
            let maxc = vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1);
            let coord = clamp(coord0, vec2<i32>(0, 0), maxc);
            return textureLoad(t_depth, coord, 0).r;
        }

        // Reference:
        // - `https://web.archive.org/web/20201215123940/https://blog.tuxedolabs.com/2018/05/04/bokeh-depth-of-field-in-single-pass.html`
        // - `https://github.com/bkaradzic/bgfx/tree/master/examples/45-bokeh`.
        fn coc(depthLinear: f32, focusPoint: f32, focusScale: f32) -> f32 {
            return clamp((1.0 / focusPoint - 1.0 / depthLinear) * focusScale, -1.0, 1.0);
        }

        @fragment fn main(
            @builtin(position) pos: vec4<f32>,
            @location(0) uv: vec2<f32>
        ) -> @location(0) vec4<f32> {
            let dims = textureDimensions(t_depth, 0u);
            let color4 = textureSample(t_color, s_color, uv);
            let color = color4.rgb;
            let center_depth = max(load_linear_depth(uv, dims), 1e-4);
            let focusPointFromUi = max(params.focus_point, 1e-4);
            let focusPoint = focusPointFromUi;
            let coc = coc(center_depth, max(focusPoint, 1e-4), params.focus_scale);
            let coc_abs = pow(abs(coc), COC_CURVE);

            // Proof view in bottom half (split by real screen-space Y, not UVs).
            let y01 = pos.y / f32(dims.y);
            if (y01 < 0.25 && DEBUG_VIEW > 0) {
                if (DEBUG_VIEW == 1) {
                    return color4;
                } else if (DEBUG_VIEW == 2) {
                    return vec4<f32>(coc_abs, coc_abs, coc_abs, 1.0);
                } else if (DEBUG_VIEW == 3) {
                    let linear_depth = clamp(center_depth / FAR, 0.0, 1.0);
                    return vec4<f32>(linear_depth, linear_depth, linear_depth, 1.0);
                }
            }

            // DoF result in top half
            let center_size = coc_abs * MAX_BLUR_SIZE;
            let pixel_size = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));

            var acc = color;
            var tot = 1.0;
            var radius = RAD_SCALE;
            var ang = 0.0;
            loop {
                if (radius >= MAX_BLUR_SIZE) { break; }

                let dir = vec2<f32>(cos(ang), sin(ang));
                let tc = uv + dir * pixel_size * radius;
                let sample_color = textureSample(t_color, s_color, tc).rgb;
                let sample_depth = max(load_linear_depth(tc, dims), 1e-4);

                let sample_coc = coc(sample_depth, max(focusPoint, 1e-4), params.focus_scale);
                var sample_size = pow(abs(sample_coc), COC_CURVE) * MAX_BLUR_SIZE;

                // Prevent background blur bleeding over in-focus foreground (Voxagon clamp trick)
                if (sample_depth > center_depth) {
                    sample_size = clamp(sample_size, 0.0, center_size * 2.0);
                }

                let m = smoothstep(radius - 0.5, radius + 0.5, sample_size);
                acc = acc + mix(acc / tot, sample_color, m);
                tot = tot + 1.0;

                ang = ang + GOLDEN_ANGLE;
                radius = radius + RAD_SCALE / radius;
            }

            acc = acc / tot;
            return vec4<f32>(acc, 1.0);
        }
        "#;

        // Bind group layout: color, depth, params
        let bgl = match device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dof_bokeh_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        }) {
            bgl => bgl,
        };
        eprintln!("[DoF] Created bind group layout ");

        let pl = match device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dof_bokeh_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        }) {
            pl => pl,
        };
        eprintln!("[DoF] Created pipeline layout ");

        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_bokeh_vs"),
            source: wgpu::ShaderSource::Wgsl(vs.into()),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dof_bokeh_fs"),
            source: wgpu::ShaderSource::Wgsl(fs.into()),
        });

        let pipeline = match device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("dof_bokeh_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &vs_mod,
                entry_point: "main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_mod,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format, // Back to offscreen format
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        }) {
            p => p,
        };
        eprintln!("[DoF] Created render pipeline ");

        // Prepare uniform buffer for params (updated per-frame)
        let params = DofParams {
            focus_distance: self.focus_distance,
            aperture: self.aperture,
            focal_length: self.focal_length,
            sensor_size: self.sensor_size,
            time_s: 0.0,
            focus_point: 5.0,
            focus_scale: 3.0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dof_bokeh_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Get texture views
        let color_view = match resources.gpu().textures.get(self.input_color) {
            Some(tex) => {
                eprintln!("[DoF] Found input_color texture ");
                tex.texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            }
            None => {
                eprintln!("[DoF][ERROR] input_color texture not found!");
                return Err(anyhow::anyhow!("input_color texture not found "));
            }
        };
        let depth_view = match resources.gpu().textures.get(self.input_depth) {
            Some(tex) => {
                eprintln!("[DoF] Found input_depth texture ");
                tex.texture
                    .create_view(&wgpu::TextureViewDescriptor::default())
            }
            None => {
                eprintln!("[DoF][ERROR] input_depth texture not found!");
                return Err(anyhow::anyhow!("input_depth texture not found "));
            }
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        eprintln!("[DoF] Created sampler ");
        let bind_group = match device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dof_bokeh_bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        }) {
            bg => bg,
        };
        eprintln!("[DoF] Created bind group ");

        self.pipeline = Some(crate::graphics::PipelineV2 {
            pipeline,
            bind_group_layouts: vec![bgl],
        });
        self.bind_group = Some(bind_group);
        self.params_buffer = Some(params_buf);
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        resources: &mut crate::resources::ResourceManager,
        _frame: &wgpu::SurfaceTexture,
        _view: &wgpu::TextureView,
        _world: &crate::graphics::WorldQuery,
    ) -> Result<()> {
        // Update params from UI/shared state
        if let Ok(pp) = resources.post_process.lock() {
            let params = DofParams {
                focus_distance: self.focus_distance,
                aperture: self.aperture,
                focal_length: self.focal_length,
                sensor_size: self.sensor_size,
                time_s: pp.time_s,
                focus_point: pp.focus_point,
                focus_scale: pp.focus_scale,
            };
            renderer.get_queue().write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&params),
            );
        }

        // Render to output texture (which gets read by Compose pass)
        let out_view = match resources.gpu().textures.get(self.output) {
            Some(tex) => tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            None => {
                eprintln!("[DoF][ERROR] output texture not found!");
                return Err(anyhow::anyhow!("output texture not found "));
            }
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&self.name),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline.as_ref().unwrap().pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
