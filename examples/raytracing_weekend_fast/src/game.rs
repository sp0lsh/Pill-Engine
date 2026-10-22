//! Apple/Renderer.mm's CPU branch: a frame loop that times `UpdateTest` + `DrawTest` and every
//! 10+ frames reports `"CPU: %.2fms (%.1f FPS) %.1fMrays/s %.2fMrays/frame frames %i"`. The
//! float backbuffer is shown through his `LinearToSRGB` (Shaders.metal), converted outside the
//! timed span; the UI only swaps buffers under the lock, so it never stalls the tracing threads.

use crate::{
    fast::Test,
    scene::{K_FLAG_ANIMATE, K_FLAG_PROGRESSIVE},
};
use pill_engine::game::*;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// Config.h `kBackbufferWidth` × `kBackbufferHeight`.
const K_BACKBUFFER_WIDTH: usize = 1280;
const K_BACKBUFFER_HEIGHT: usize = 720;
/// Renderer.mm's default `g_TestFlags`.
const TEST_FLAGS: u32 = K_FLAG_PROGRESSIVE | K_FLAG_ANIMATE;

struct Shared {
    /// RGBA8, sRGB, row 0 at the top.
    pixels: Mutex<Vec<u8>>,
    version: AtomicU64,
    label: Mutex<String>,
    thread_count: usize,
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("raytracing_weekend_fast")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        // The CPU image is the only output: skip the scene passes, run just the egui pass
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .pass_chain = PassChain::UiOnly;
        // The rendering system needs an active camera
        engine
            .build_entity(scene)
            .with_component(TransformComponent::new())
            .with_component(CameraComponent::builder().enabled(true).build())
            .build();

        let shared = Arc::new(Shared {
            pixels: Mutex::new(vec![255; K_BACKBUFFER_WIDTH * K_BACKBUFFER_HEIGHT * 4]),
            version: AtomicU64::new(0),
            label: Mutex::new(String::new()),
            thread_count: std::thread::available_parallelism().map_or(4, |n| n.get()),
        });
        let frame_shared = Arc::clone(&shared);
        std::thread::spawn(move || frame_loop(&frame_shared));

        let egui_component = engine.get_global_component_mut::<EguiComponent>()?;
        egui_component.show_engine_windows = false;
        let texture: Mutex<Option<(egui::TextureHandle, u64)>> = Mutex::new(None);
        egui_component.register_ui("raytracing_weekend_fast.image", move |context| {
            image_ui(context, &shared, &texture)
        });
        Ok(())
    }
}

/// Renderer.mm `_doRenderingWith`, CPU path.
fn frame_loop(shared: &Shared) {
    let mut test = Test::new();
    let mut backbuffer_pixels = vec![0.0f32; K_BACKBUFFER_WIDTH * K_BACKBUFFER_HEIGHT * 4];
    let mut display = vec![255u8; K_BACKBUFFER_WIDTH * K_BACKBUFFER_HEIGHT * 4];
    let clock_start = Instant::now();
    let mut frame_time = Duration::ZERO;
    let mut ray_counter: u64 = 0;
    let mut total_counter: i32 = 0;
    let mut frame_counter: u64 = 0;
    loop {
        let time1 = Instant::now();
        // His curT is mach_absolute_time in seconds; only the animation phase depends on it
        let cur_t = (time1 - clock_start).as_secs_f32();

        test.update_test(
            cur_t,
            total_counter,
            K_BACKBUFFER_WIDTH,
            K_BACKBUFFER_HEIGHT,
            TEST_FLAGS,
        );
        let ray_count = test.draw_test(
            total_counter,
            K_BACKBUFFER_WIDTH,
            K_BACKBUFFER_HEIGHT,
            &mut backbuffer_pixels,
            TEST_FLAGS,
            shared.thread_count,
        );
        ray_counter += ray_count as u64;

        let time2 = Instant::now();
        frame_counter += 1;
        total_counter += 1;
        frame_time += time2 - time1;
        if frame_counter > 10 {
            let s = frame_time.as_secs_f32() / frame_counter as f32;
            let label = format!(
                "CPU: {:.2}ms ({:.1} FPS) {:.1}Mrays/s {:.2}Mrays/frame frames {}",
                s * 1000.0,
                1.0 / s,
                (ray_counter / frame_counter) as f32 / s * 1.0e-6,
                (ray_counter / frame_counter) as f32 * 1.0e-6,
                total_counter
            );
            println!("{label}");
            *shared.label.lock().unwrap() = label;
            frame_counter = 0;
            frame_time = Duration::ZERO;
            ray_counter = 0;
        }

        // Shaders.metal LinearToSRGB; his backbuffer row 0 is the bottom of the image
        let linear_to_srgb = |c: f32| (1.055 * c.powf(0.416_666_67) - 0.055).max(0.0);
        for y in 0..K_BACKBUFFER_HEIGHT {
            let source = &backbuffer_pixels
                [(K_BACKBUFFER_HEIGHT - 1 - y) * K_BACKBUFFER_WIDTH * 4..]
                [..K_BACKBUFFER_WIDTH * 4];
            let target = &mut display[y * K_BACKBUFFER_WIDTH * 4..][..K_BACKBUFFER_WIDTH * 4];
            for (rgba, linear) in target.chunks_exact_mut(4).zip(source.chunks_exact(4)) {
                for channel in 0..3 {
                    rgba[channel] = (linear_to_srgb(linear[channel]).min(1.0) * 255.0 + 0.5) as u8;
                }
            }
        }
        std::mem::swap(&mut *shared.pixels.lock().unwrap(), &mut display);
        shared.version.fetch_add(1, Ordering::Relaxed);
    }
}

fn image_ui(
    context: &egui::Context,
    shared: &Shared,
    texture: &Mutex<Option<(egui::TextureHandle, u64)>>,
) {
    let version = shared.version.load(Ordering::Relaxed);
    let mut texture = texture.lock().unwrap();
    if texture
        .as_ref()
        .map_or(true, |(_, uploaded)| *uploaded != version)
    {
        let pixels = shared.pixels.lock().unwrap().clone();
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [K_BACKBUFFER_WIDTH, K_BACKBUFFER_HEIGHT],
            &pixels,
        );
        match texture.as_mut() {
            Some((handle, uploaded)) => {
                handle.set(image, egui::TextureOptions::LINEAR);
                *uploaded = version;
            }
            None => {
                let handle = context.load_texture(
                    "raytracing_weekend_fast",
                    image,
                    egui::TextureOptions::LINEAR,
                );
                *texture = Some((handle, version));
            }
        }
    }
    let Some((handle, _)) = texture.as_ref() else {
        return;
    };

    egui::CentralPanel::default()
        .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
        .show(context, |ui| {
            let available = ui.available_size();
            let aspect = K_BACKBUFFER_WIDTH as f32 / K_BACKBUFFER_HEIGHT as f32;
            let size = if available.x / available.y > aspect {
                egui::vec2(available.y * aspect, available.y)
            } else {
                egui::vec2(available.x, available.x / aspect)
            };
            ui.centered_and_justified(|ui| ui.image((handle.id(), size)));
        });

    let label = shared.label.lock().unwrap().clone();
    egui::Area::new(egui::Id::new("raytracing_weekend_fast.stats"))
        .fixed_pos(egui::pos2(8.0, 8.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_black_alpha(160))
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!(
                                "{label}  {} threads",
                                shared.thread_count
                            ))
                            .monospace()
                            .color(egui::Color32::WHITE),
                        )
                        .extend(),
                    );
                });
        });
}
