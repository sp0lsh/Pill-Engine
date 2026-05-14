use crate::graphics::{Pass, PillRenderer, WorldQuery};
use anyhow::Result;

pub struct PassScene;

impl PassScene {
    pub fn new() -> Self {
        Self
    }
}

impl Pass for PassScene {
    fn get_label(&self) -> &str {
        "pass_scene"
    }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> {
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        renderer.record_scene_pass(encoder, view, world)
    }
}
