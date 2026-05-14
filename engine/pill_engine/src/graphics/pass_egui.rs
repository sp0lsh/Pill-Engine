use crate::{
    ecs::EguiClient,
    graphics::{Pass, PillRenderer, WorldQuery},
    renderer::drawers::egui_drawer::EguiDrawer,
};
use anyhow::Result;
use std::sync::Arc;

pub struct PassEgui {
    window: Arc<winit::window::Window>,
    client: Arc<EguiClient>,
    drawer: Option<EguiDrawer>,
}

impl PassEgui {
    pub fn new(window: Arc<winit::window::Window>, client: Arc<EguiClient>) -> Self {
        Self {
            window,
            client,
            drawer: None,
        }
    }
}

impl Pass for PassEgui {
    fn get_label(&self) -> &str {
        "pass_egui"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.drawer = Some(EguiDrawer::new(
            renderer.get_device(),
            renderer.get_surface_format(),
            None,
            1,
            self.window.clone(),
        ));
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
        _world: &WorldQuery<'_>,
    ) -> Result<()> {
        let drawer = self.drawer.as_mut().expect("PassEgui not initialized");

        for ev in self.client.take_events() {
            drawer.handle_input(&ev);
        }

        let size = self.window.inner_size();
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: drawer.window_scale_factor,
        };

        let ui_fn = self.client.take_ui();
        let run_ui: Box<dyn FnMut(&egui::Context)> = match ui_fn {
            Some(f) => Box::new(move |ctx: &egui::Context| f(ctx)),
            None => Box::new(|_ctx: &egui::Context| {}),
        };

        drawer.record_draw_commands(
            renderer.get_device(),
            renderer.get_queue(),
            encoder,
            view,
            screen_desc,
            run_ui,
            &mut pill_core::Timer::new(),
        )
    }
}
