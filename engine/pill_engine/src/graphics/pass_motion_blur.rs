use crate::graphics::Pass;
use crate::graphics::PillRenderer;
use crate::graphics::RendererTextureHandle;
use crate::resources::ResourceManager;
use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use wgpu::TextureFormat;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionBlurParams {
    velocity_scale: f32,
    min_speed: f32,
    depth_softness: f32,
    enabled: u32,
    max_samples: u32,
    debug_velocity: u32,
    _pad: [u32; 2],
}

pub struct PassMotionBlur {
    pub name: String,
    pub input_color: RendererTextureHandle,
    pub input_velocity: RendererTextureHandle,
    pub input_depth: RendererTextureHandle,
    pub output: RendererTextureHandle,
    pub format: TextureFormat,
    pipeline: Option<crate::graphics::PipelineV2>,
    bind_group: Option<wgpu::BindGroup>,
    params_buffer: Option<wgpu::Buffer>,
}

impl PassMotionBlur {
    pub fn new(
        name: &str,
        input_color: RendererTextureHandle,
        input_velocity: RendererTextureHandle,
        input_depth: RendererTextureHandle,
        output: RendererTextureHandle,
        format: TextureFormat,
    ) -> Self {
        Self {
            name: name.to_string(),
            input_color,
            input_velocity,
            input_depth,
            output,
            format,
            pipeline: None,
            bind_group: None,
            params_buffer: None,
        }
    }
}

impl Pass for PassMotionBlur {
    fn get_label(&self) -> &str {
        &self.name
    }

    fn init(
        &mut self,
        renderer: &mut dyn PillRenderer,
        resources: &mut ResourceManager,
    ) -> Result<()> {
        let device = renderer.get_device();
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

        let fs = r#"
        struct Params {
            velocity_scale: f32,
            min_speed: f32,
            depth_softness: f32,
            enabled: u32,
            max_samples: u32,
            debug_velocity: u32,
            _pad: vec2<u32>,
        };

        @group(0) @binding(0) var t_color: texture_2d<f32>;
        @group(0) @binding(1) var t_velocity: texture_2d<f32>;
        @group(0) @binding(2) var t_depth: texture_2d<f32>; // linear depth
        @group(0) @binding(3) var<uniform> params: Params;
        @group(0) @binding(4) var s_linear: sampler;

        const MAX_SAMPLES: i32 = 64;

        fn sample_depth(uv: vec2<f32>) -> f32 {
            let dims = textureDimensions(t_depth, 0u);
            let coord = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0,0), vec2<i32>(i32(dims.x)-1, i32(dims.y)-1));
            return textureLoad(t_depth, coord, 0).r;
        }

        const ARROW_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 0.2);

        // xyz = arrow color, w = mask (coverage)
        fn compute_arrow_overlay(
            uv: vec2<f32>,
            spacing_px: f32,
            thickness_px: f32,
            scale: f32,
            dims: vec2<u32>
        ) -> vec4<f32> {
            let spacing = spacing_px / vec2<f32>(f32(dims.x), f32(dims.y));
            let cell = floor(uv / spacing);
            let cell_center = (cell + vec2<f32>(0.5, 0.5)) * spacing;
            let v_sample = textureSample(t_velocity, s_linear, cell_center).xy * scale;
            let v_len = length(v_sample);
            if (v_len <= 1e-4) {
                return vec4<f32>(ARROW_COLOR, 0.0);
            }

            let dir = v_sample / v_len;
            let tangent = vec2<f32>(-dir.y, dir.x);
            let rel = uv - cell_center;
            let proj = dot(rel, dir);
            let ortho = dot(rel, tangent);
            let arrow_len = clamp(v_len * 0.5, spacing.x * 0.25, spacing.x * 1.5);
            let half_thick = (thickness_px / vec2<f32>(f32(dims.x), f32(dims.y))).x;
            let line_mask = smoothstep(half_thick, 0.0, abs(ortho)) * step(-spacing.x, proj) * step(proj, arrow_len);
            let head_size = arrow_len * 0.2;
            let head_mask = step(arrow_len - head_size, proj) * smoothstep(half_thick * 2.0, 0.0, abs(ortho + (proj - arrow_len) * 0.5));
            let mask = clamp(max(line_mask, head_mask), 0.0, 1.0);
            return vec4<f32>(ARROW_COLOR, mask);
        }

        @fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let dims = textureDimensions(t_color, 0u);
            let texel = 1.0 / vec2<f32>(f32(dims.x), f32(dims.y));
            let spacing_px = 24.0;
            let thickness_px = 1.25;

            var vel = textureSample(t_velocity, s_linear, uv).xy * params.velocity_scale;
            let center_speed = length(vel / texel);
            let enabled = params.enabled != 0u;
            if (!enabled || center_speed < params.min_speed) {
                let base = textureSample(t_color, s_linear, uv).rgb;
                if (params.debug_velocity != 0u) {
                    let dbg = vec3<f32>(0.5, 0.5, 0.5) + vec3<f32>(vel, 0.0) * 0.5;
                    let overlay = compute_arrow_overlay(uv, spacing_px, thickness_px, params.velocity_scale, dims);
                    let out_rgb = mix(dbg, overlay.xyz, overlay.w);
                    return vec4<f32>(out_rgb, 1.0);
                }
                return vec4<f32>(base, 1.0);
            }

            let center_depth = sample_depth(uv);
            let maxSamples = min(i32(params.max_samples), MAX_SAMPLES);
            let mut_samples = max(1, maxSamples);
            let nSamples = clamp(i32(center_speed), 1, mut_samples);

            var acc = textureSample(t_color, s_linear, uv).rgb;
            var w_acc = 1.0;
            // guard for single sample
            if (nSamples > 1) {
                for (var i: i32 = 1; i < nSamples; i = i + 1) {
                    let t = (f32(i) / f32(nSamples - 1)) - 0.5;
                    let offset = vel * t;
                    let sample_uv = uv + offset;
                    let sample_color = textureSample(t_color, s_linear, sample_uv).rgb;
                    let sample_depth = sample_depth(sample_uv);
                    let dz = abs(sample_depth - center_depth);
                    let depth_w = 1.0 / (1.0 + dz * params.depth_softness);
                    acc = acc + sample_color * depth_w;
                    w_acc = w_acc + depth_w;
                }
            }
            var color_out = acc / w_acc;
            if (params.debug_velocity != 0u) {
                let overlay = compute_arrow_overlay(uv, spacing_px, thickness_px, params.velocity_scale, dims);
                let out_rgb = mix(color_out, overlay.xyz, overlay.w);
                return vec4<f32>(out_rgb, 1.0);
            }
            return vec4<f32>(color_out, 1.0);
        }
        "#;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion_blur_bgl"),
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("motion_blur_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur_vs"),
            source: wgpu::ShaderSource::Wgsl(vs.into()),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion_blur_fs"),
            source: wgpu::ShaderSource::Wgsl(fs.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion_blur_pipeline"),
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
                    format: self.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let params = MotionBlurParams {
            velocity_scale: 1.0,
            min_speed: 0.001,
            depth_softness: 50.0,
            enabled: 1,
            max_samples: 16,
            debug_velocity: 0,
            _pad: [0, 0],
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("motion_blur_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let color_view = resources
            .gpu()
            .textures
            .get(self.input_color)
            .expect("motion blur input color")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let velocity_view = resources
            .gpu()
            .textures
            .get(self.input_velocity)
            .expect("motion blur velocity")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = resources
            .gpu()
            .textures
            .get(self.input_depth)
            .expect("motion blur depth")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion_blur_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&velocity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        self.pipeline = Some(crate::graphics::PipelineV2 {
            pipeline,
            bind_group_layouts: vec![bgl],
        });
        self.bind_group = Some(bind_group);
        self.params_buffer = Some(params_buffer);
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
        if let Ok(pp) = resources.post_process.lock() {
            let params = MotionBlurParams {
                velocity_scale: pp.mb_strength,
                min_speed: pp.mb_min_speed,
                depth_softness: pp.mb_depth_softness,
                enabled: if pp.mb_enabled { 1 } else { 0 },
                max_samples: pp.mb_max_samples,
                debug_velocity: if pp.mb_debug_velocity { 1 } else { 0 },
                _pad: [0, 0],
            };
            renderer.get_queue().write_buffer(
                self.params_buffer.as_ref().unwrap(),
                0,
                bytemuck::bytes_of(&params),
            );
        }

        let out_view = resources
            .gpu()
            .textures
            .get(self.output)
            .expect("motion blur output")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

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
