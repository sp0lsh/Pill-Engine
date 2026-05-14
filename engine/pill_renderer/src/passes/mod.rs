mod pass_scene;
pub use pass_scene::PassScene;

use crate::resources::{RendererCamera, RendererResourceStorage};
use pill_core::Timer;
use pill_engine::internal::{ComponentStorage, RenderQueueItem, TransformComponent};
use anyhow::Result;

pub struct PassContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub storage: &'a RendererResourceStorage,
    pub camera: &'a RendererCamera,
    pub render_queue: &'a [RenderQueueItem],
    pub transform_storage: &'a ComponentStorage<TransformComponent>,
    pub output_view: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub clear_color: wgpu::Color,
    pub screen_size: [u32; 2],
    pub timer: &'a mut Timer,
}

pub trait Pass {
    fn record(&mut self, encoder: &mut wgpu::CommandEncoder, ctx: &mut PassContext<'_>) -> Result<()>;
}
