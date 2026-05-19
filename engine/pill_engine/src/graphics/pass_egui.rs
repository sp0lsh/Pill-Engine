use crate::{
    ecs::EguiClient,
    graphics::{Pass, PillRenderer, WorldQuery},
};
use anyhow::Result;
use std::sync::Arc;

pub struct PassEgui {
    client:   Arc<EguiClient>,
    renderer: Option<EguiRenderer>,
}

/// Minimal egui renderer that uses only wgpu — no egui_winit, no NSWindow calls.
/// Safe to create and use from within macOS drawRect callbacks.
/// The overlay is read-only; keyboard/mouse input is not forwarded.
struct EguiRenderer {
    context:          egui::Context,
    renderer:         egui_wgpu::Renderer,
    pixels_per_point: f32,
}

impl EguiRenderer {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions {
                depth_stencil_format: None,
                msaa_samples: 1,
                dithering: false,
                ..Default::default()
            },
        );
        // Start at 2.0 for HiDPI/Retina; updated when ScaleFactorChanged events arrive.
        Self { context: egui::Context::default(), renderer, pixels_per_point: 2.0 }
    }

    fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size_px: [u32; 2],
        mut run_ui: impl FnMut(&egui::Context),
    ) -> Result<()> {
        let ppp = self.pixels_per_point;
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point: ppp,
        };
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(size_px[0] as f32 / ppp, size_px[1] as f32 / ppp),
            )),
            ..Default::default()
        };

        let full_output = self.context.run(raw_input, |ctx| run_ui(ctx));

        let tris = self.context.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer.update_buffers(device, queue, encoder, &tris, &screen_desc);

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // SAFETY: rpass borrows encoder, which outlives this function.
        let rpass: &mut wgpu::RenderPass<'static> =
            unsafe { std::mem::transmute(&mut rpass) };
        self.renderer.render(rpass, &tris, &screen_desc);

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        Ok(())
    }
}

impl PassEgui {
    pub fn new(_window: Arc<winit::window::Window>, client: Arc<EguiClient>) -> Self {
        Self { client, renderer: None }
    }
}

impl Pass for PassEgui {
    fn get_label(&self) -> &str { "pass_egui" }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> { Ok(()) }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame:    &wgpu::SurfaceTexture,
        view:     &wgpu::TextureView,
        _world:   &WorldQuery<'_>,
    ) -> Result<()> {
        // Lazy-init: EguiRenderer uses only wgpu, safe to create from within drawRect.
        // Use the raw surface format (SRGB) to match the view parameter provided by the renderer.
        if self.renderer.is_none() {
            self.renderer = Some(EguiRenderer::new(
                renderer.get_device(),
                renderer.get_surface_format(),
            ));
        }
        let egui_renderer = self.renderer.as_mut().unwrap();

        // Update scale factor from window events without touching NSWindow APIs.
        for ev in self.client.take_events() {
            if let winit::event::WindowEvent::ScaleFactorChanged { scale_factor, .. } = ev {
                egui_renderer.pixels_per_point = scale_factor as f32;
            }
        }

        let tex_size = frame.texture.size();
        let mut ui_fn = self.client.take_ui();

        egui_renderer.draw(
            renderer.get_device(),
            renderer.get_queue(),
            encoder,
            view,
            [tex_size.width, tex_size.height],
            |ctx| {
                if let Some(ref f) = ui_fn { f(ctx); }
                ui_fn = None;
            },
        )
    }
}
