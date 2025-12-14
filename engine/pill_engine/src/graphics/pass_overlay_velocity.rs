use anyhow::Result;
use wgpu::CommandEncoder;

use crate::graphics::renderer::{
    Pass, PillRenderer as EnginePillRenderer, PipelineV2, RendererTextureHandle, WorldQuery,
};

pub struct PassOverlayVelocity {
    label: String,
    pipeline: Option<PipelineV2>,
    bind_group_rect: Option<wgpu::BindGroup>,
    bind_group_material: Option<wgpu::BindGroup>,
    params_buffer: Option<wgpu::Buffer>,
    rect: [f32; 4],
    params: [f32; 4], // x: scale, y: mode (0=signed rgb, 1=magnitude), z: spacing_px, w: thickness_px
    target_format: wgpu::TextureFormat,
    velocity_texture: RendererTextureHandle,
}

impl PassOverlayVelocity {
    pub fn new(
        label: &str,
        rect: [f32; 4],
        params: [f32; 4],
        target_format: wgpu::TextureFormat,
        velocity_texture: RendererTextureHandle,
    ) -> Self {
        Self {
            label: label.to_string(),
            pipeline: None,
            bind_group_rect: None,
            bind_group_material: None,
            params_buffer: None,
            rect,
            params,
            target_format,
            velocity_texture,
        }
    }
}

impl Pass for PassOverlayVelocity {
    fn get_label(&self) -> &str {
        &self.label
    }

    fn init(
        &mut self,
        renderer: &mut dyn EnginePillRenderer,
        resources: &mut crate::resources::ResourceManager,
    ) -> Result<()> {
        let device = renderer.get_device();
        // Rect UBO
        let rect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_velocity_rect_ubo"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut view = rect_buffer.slice(..).get_mapped_range_mut();
            let src = bytemuck::bytes_of(&self.rect);
            view[..src.len()].copy_from_slice(src);
        }
        rect_buffer.unmap();

        // Params UBO
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay_velocity_params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        {
            let mut view = params_buffer.slice(..).get_mapped_range_mut();
            let src = bytemuck::bytes_of(&self.params);
            view[..src.len()].copy_from_slice(src);
        }
        params_buffer.unmap();

        let vs = r#"
          @group(1) @binding(0) var<uniform> URect: vec4<f32>; // bottom-left, top-right in [0,1]

          struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
          @vertex fn main(@builtin(vertex_index) vi: u32) -> VSOut {
            var unit = array<vec2<f32>, 6>(
              vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
              vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0), vec2<f32>(-1.0, -1.0)
            );
            let p = unit[vi];  // [-1,1] NDC space
            let s = p*0.5+0.5; // [0,1] screen space
            let r = URect.xy + s * (URect.zw - URect.xy); // move by rect [0,1]
            let ndc = r*2.0-1.0; // [-1,1] NDC space
            var out: VSOut;
            out.pos = vec4<f32>(ndc.x, ndc.y, 0.0, 1.0);
            out.uv = vec2<f32>(s.x, 1.0 - s.y);
            return out;
          }
        "#;

        let fs = r#"
          @group(0) @binding(0) var tex: texture_2d<f32>;
          @group(0) @binding(1) var smp: sampler;
          @group(0) @binding(2) var<uniform> UParams: vec4<f32>; // x: scale, y: mode, z: spacing_px, w: thickness_px

          const MODE_SIGNED: f32 = 0.0;
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
            let v_sample = textureSample(tex, smp, cell_center).xy * scale;
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
            let dims = textureDimensions(tex, 0u);
            let scale = UParams.x;
            let mode = UParams.y;
            let spacing_px = max(UParams.z, 4.0);
            let thickness_px = max(UParams.w, 1.0);

            // Sample velocity at the current pixel (for background visualization).
            let coord = vec2<i32>(uv * vec2<f32>(dims));
            let vel_px = textureLoad(tex, coord, 0).xy;

            // Base visualization (signed or magnitude).
            var base_rgb: vec3<f32>;
            if (mode <= MODE_SIGNED + 0.5) {
              base_rgb = vec3<f32>(0.5, 0.5, 0.5) + vec3<f32>(vel_px, 0.0) * scale;
            } else {
              let m = length(vel_px) * scale;
              base_rgb = vec3<f32>(m, m, m);
            }

            let overlay = compute_arrow_overlay(uv, spacing_px, thickness_px, scale, dims);
            let arrow_rgb = mix(base_rgb, overlay.xyz, overlay.w);

            return vec4<f32>(arrow_rgb, 1.0);
          }
        "#;

        // Bind group layouts
        let bgl_material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_velocity_material_bgl"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                    },
                    count: None,
                },
            ],
        });
        let bgl_rect = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_velocity_rect_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(16).unwrap()),
                },
                count: None,
            }],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_velocity_pl"),
            bind_group_layouts: &[&bgl_material, &bgl_rect],
            push_constant_ranges: &[],
        });

        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_velocity_vs"),
            source: wgpu::ShaderSource::Wgsl(vs.into()),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay_velocity_fs"),
            source: wgpu::ShaderSource::Wgsl(fs.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_velocity_pipeline"),
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
                    format: self.target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let bind_group_rect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_velocity_rect_bg"),
            layout: &bgl_rect,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rect_buffer.as_entire_binding(),
            }],
        });

        let velocity_view = resources
            .gpu()
            .textures
            .get(self.velocity_texture)
            .expect("velocity texture")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let bind_group_material = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_velocity_material_bg"),
            layout: &bgl_material,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&velocity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        self.pipeline = Some(PipelineV2 {
            pipeline,
            bind_group_layouts: vec![bgl_material, bgl_rect],
        });
        self.bind_group_rect = Some(bind_group_rect);
        self.bind_group_material = Some(bind_group_material);
        self.params_buffer = Some(params_buffer);
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut CommandEncoder,
        _renderer: &mut dyn EnginePillRenderer,
        resources: &mut crate::resources::ResourceManager,
        _frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
        _world: &WorldQuery,
    ) -> Result<()> {
        if let Ok(pp) = resources.post_process.lock() {
            if !pp.mb_debug_velocity {
                return Ok(());
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&self.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline.as_ref().unwrap().pipeline);
        pass.set_bind_group(0, &self.bind_group_material.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, &self.bind_group_rect.as_ref().unwrap(), &[]);
        pass.draw(0..6, 0..1);
        Ok(())
    }
}
