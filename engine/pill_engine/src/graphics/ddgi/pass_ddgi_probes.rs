use crate::config::{
    DDGI_PROBES_DEBUG_FRAGMENT_SHADER, DDGI_PROBES_DEBUG_VERTEX_SHADER, DEPTH_TARGET_FORMAT,
};
use crate::ecs::RenderStateComponent;
use crate::graphics::ddgi::DdgiFrame;
use crate::graphics::shader_bindings::{
    ddgi_probes_debug_fragment, ddgi_probes_debug_vertex, merge_layouts, structs::ProbeDebugCamera,
};
use crate::graphics::{
    camera_view_projection, BufferDesc, FrameTarget, Pass, PassTextureHandle, PillRenderer,
    Pipeline, PipelineDesc, ShaderDesc, WorldQuery,
};

use pill_core::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// Vertices of one probe sphere: `RINGS` × `SEGMENTS` quads in `ddgi_probes_debug.hlsl`.
const SPHERE_VERTICES: u32 = 8 * 12 * 6;

/// Debug view of the DDGI probes (`Gi::debug_probes`): a small sphere per probe at its relocated
/// position, shaded with its irradiance, drawn over the scene with depth testing.
pub(crate) struct PassDdgiProbes {
    frame: Rc<RefCell<DdgiFrame>>,
    color_target: PassTextureHandle,
    depth_target: PassTextureHandle,
    pipeline: Option<Pipeline>,
    camera: Option<wgpu::Buffer>,
    /// Bind group and the `DdgiFrame::generation` it was built for.
    group: Option<(u64, wgpu::BindGroup)>,
}

impl PassDdgiProbes {
    pub(crate) fn new(
        frame: Rc<RefCell<DdgiFrame>>,
        color_target: PassTextureHandle,
        depth_target: PassTextureHandle,
    ) -> Self {
        Self {
            frame,
            color_target,
            depth_target,
            pipeline: None,
            camera: None,
            group: None,
        }
    }
}

impl Pass for PassDdgiProbes {
    fn get_label(&self) -> &str {
        "pass_ddgi_probes"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let hdr_format = renderer.get_hdr_format();
        let pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some("pass_ddgi_probes_pipeline"),
            vs: ShaderDesc {
                source: DDGI_PROBES_DEBUG_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: DDGI_PROBES_DEBUG_FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            fragment_constants: &[],
            vertex_buffers: &[],
            bind_groups: merge_layouts(&[
                ddgi_probes_debug_vertex::layout(),
                ddgi_probes_debug_fragment::layout(),
            ]),
            targets: &[Some(wgpu::ColorTargetState {
                format: hdr_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TARGET_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
        })?;
        self.camera = Some(renderer.create_buffer(BufferDesc {
            label: Some("pass_ddgi_probes_camera"),
            byte_size: std::mem::size_of::<ProbeDebugCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let settings = world.get_global::<RenderStateComponent>()?.gi;
        let frame = self.frame.borrow();
        if !settings.enabled || !settings.debug_probes || frame.probe_count == 0 {
            return Ok(());
        }
        let (Some(pipeline), Some(camera_buffer)) = (&self.pipeline, &self.camera) else {
            return Ok(());
        };
        let (camera, transform) = world.active_camera()?;
        let camera_uniform = ProbeDebugCamera {
            view_projection: camera_view_projection(camera, transform).to_cols_array_2d(),
        };
        renderer
            .get_queue()
            .write_buffer(camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
        if self
            .group
            .as_ref()
            .is_none_or(|(generation, _)| *generation != frame.generation)
        {
            let group = ddgi_probes_debug_vertex::Group0 {
                camera: camera_buffer,
                ddgi: &frame.params,
                ddgi_probes: &frame.probes,
            }
            .create(renderer.get_device(), &pipeline.bind_group_layouts[0]);
            self.group = Some((frame.generation, group));
        }
        let (_, group) = self
            .group
            .as_ref()
            .expect("PassDdgiProbes: bind group built above");
        let color_view = renderer
            .get_render_target_view(self.color_target)
            .expect("PassDdgiProbes: color target missing");
        let depth_view = renderer
            .get_render_target_view(self.depth_target)
            .expect("PassDdgiProbes: depth target missing");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_ddgi_probes"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..SPHERE_VERTICES, 0..frame.probe_count);
        Ok(())
    }
}
