use super::{Pass, PassContext};
use crate::{config::MAX_INSTANCE_PER_DRAWCALL_COUNT, drawers::mesh_drawer::MeshDrawer};
use anyhow::Result;

pub struct PassScene {
    mesh_drawer: MeshDrawer,
}

impl PassScene {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            mesh_drawer: MeshDrawer::new(device, MAX_INSTANCE_PER_DRAWCALL_COUNT as u32),
        }
    }
}

impl Pass for PassScene {
    fn record(&mut self, encoder: &mut wgpu::CommandEncoder, ctx: &mut PassContext<'_>) -> Result<()> {
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: ctx.output_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(ctx.clear_color),
                store: wgpu::StoreOp::Store,
            },
        };
        let depth_stencil_attachment = wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        };

        self.mesh_drawer.record_draw_commands(
            ctx.queue,
            encoder,
            ctx.storage,
            color_attachment,
            depth_stencil_attachment,
            ctx.camera,
            ctx.render_queue,
            ctx.transform_storage,
            ctx.timer,
        )
    }
}
