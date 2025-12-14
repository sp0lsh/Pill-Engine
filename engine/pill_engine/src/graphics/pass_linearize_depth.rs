use anyhow::Result;
use wgpu::CommandEncoder;

use crate::graphics::projection::{depth_unpack_consts_from_proj, perspective_rh_no};
use crate::graphics::renderer::{
    Pass, PillRenderer as EnginePillRenderer, PipelineV2, RendererTextureHandle, WorldQuery,
};
use bytemuck::{Pod, Zeroable};
use pill_core::PillSlotMapKey;
use wgpu::util::DeviceExt;

pub struct PassLinearizeDepth {
    label: String,
    pipeline: Option<PipelineV2>,
    bind_group_material: Option<wgpu::BindGroup>,
    params_buffer: Option<wgpu::Buffer>,
    target_format: wgpu::TextureFormat,
    // Inputs/outputs
    depth_texture: RendererTextureHandle,
    depth_copy_target: RendererTextureHandle,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DepthUnpackParams {
    depth_mul: f32,
    depth_add: f32,
    _pad: [f32; 2],
}

impl PassLinearizeDepth {
    pub fn new(
        label: &str,
        depth_texture: RendererTextureHandle,
        depth_copy_target: RendererTextureHandle,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            label: label.to_string(),
            pipeline: None,
            bind_group_material: None,
            params_buffer: None,
            target_format,
            depth_texture,
            depth_copy_target,
        }
    }
}

impl Pass for PassLinearizeDepth {
    fn get_label(&self) -> &str {
        &self.label
    }

    fn init(
        &mut self,
        renderer: &mut dyn EnginePillRenderer,
        resources: &mut crate::resources::ResourceManager,
    ) -> Result<()> {
        let device = renderer.get_device();

        // Fullscreen triangle (no vertex buffers)
        let vs = r#"
        struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32>, };
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
        @group(0) @binding(0) var texDepth: texture_depth_2d;
        @group(0) @binding(1) var<uniform> u_depthUnpack: vec4<f32>; // (mul, add, _, _)

        fn linearize_depth(screenDepth: f32) -> f32 {
          // See projection.rs reference comment.
          // linear = depthMul / (depthAdd - screenDepth)
          let depthMul = u_depthUnpack.x;
          let depthAdd = u_depthUnpack.y;
          return depthMul / (depthAdd - screenDepth);
        }

        @fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
          // IMPORTANT: textureLoad requires in-bounds coords; fullscreen-triangle UVs can go slightly out of range,
          // so clamp to [0..dims-1] to avoid undefined results (often "two flat values").
          let dims = textureDimensions(texDepth, 0u);
          let w = i32(dims.x);
          let h = i32(dims.y);
          let px = vec2<f32>(f32(dims.x), f32(dims.y));
          let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0) - vec2<f32>(1.0) / px);
          let coord_f = uv_clamped * px;
          let cx = clamp(i32(coord_f.x), 0, w - 1);
          let cy = clamp(i32(coord_f.y), 0, h - 1);
          let raw = textureLoad(texDepth, vec2<i32>(cx, cy), 0);
          let lin = linearize_depth(raw);
          // Linear view-space depth in meters in R (render target is R16Float)
          return vec4<f32>(lin, 0.0, 0.0, 1.0);
        }
        "#;

        // Bind group layout: depth texture + depth unpack params
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy_depth_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(
                                std::mem::size_of::<DepthUnpackParams>() as u64
                            )
                            .unwrap(),
                        ),
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("linearize_depth_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("linearize_depth_vs"),
            source: wgpu::ShaderSource::Wgsl(vs.into()),
        });
        let fs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("linearize_depth_fs"),
            source: wgpu::ShaderSource::Wgsl(fs.into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("linearize_depth_pipeline"),
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

        // Params buffer (updated per-frame from active camera projection)
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("depth_unpack_params"),
            contents: bytemuck::bytes_of(&DepthUnpackParams {
                depth_mul: 0.0,
                depth_add: 1.0,
                _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Material bind group: depth view + params buffer
        let depth_view = resources
            .gpu()
            .textures
            .get(self.depth_texture)
            .expect("scene depth")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group_material = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_depth_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.pipeline = Some(PipelineV2 {
            pipeline,
            bind_group_layouts: vec![bgl],
        });
        self.bind_group_material = Some(bind_group_material);
        self.params_buffer = Some(params_buf);

        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut CommandEncoder,
        renderer: &mut dyn EnginePillRenderer,
        resources: &mut crate::resources::ResourceManager,
        _frame: &wgpu::SurfaceTexture,
        _view: &wgpu::TextureView,
        world: &WorldQuery,
    ) -> Result<()> {
        // Depth unpack consts derived from a [-1..1] depth projection.
        let camera_storage = world
            .camera_components
            .data
            .get(world.active_camera.data().index as usize)
            .unwrap();
        let cam = camera_storage.as_ref().unwrap();
        let fov_y = cam.fov.to_radians();
        let aspect = cam.aspect.get_value();
        let z_near = cam.range.start;
        let z_far = cam.range.end;
        let proj2 = perspective_rh_no(fov_y, aspect, z_near, z_far);
        let (depth_mul, depth_add) = depth_unpack_consts_from_proj(proj2);
        renderer.get_queue().write_buffer(
            self.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&DepthUnpackParams {
                depth_mul,
                depth_add,
                _pad: [0.0; 2],
            }),
        );

        // Target view: write to the depth copy render target
        let out_view = resources
            .gpu()
            .textures
            .get(self.depth_copy_target)
            .expect("depth copy target")
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&self.label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline.as_ref().unwrap().pipeline);
        pass.set_bind_group(0, &self.bind_group_material.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
