//! Progressive CPU path tracing of Ray Tracing in One Weekend's final scene, with the book's
//! code structure and no optimizations. Each pass maps every pixel to one more sample with rayon
//! (`par_iter_mut` over the accumulation buffer) and publishes the whole image as sRGB8; the egui
//! overlay uploads it as a texture whenever it changed and shows it letterboxed over the window.

use crate::{
    camera::Camera,
    color::{write_color, Color},
    hittable_list::HittableList,
    scene,
};
use pill_engine::game::*;
use rayon::prelude::*;
use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// The book's chapter width (400 × 225; its final render is 1200 × 675), 500 samples per pixel.
const IMAGE_WIDTH: usize = 400;
const SAMPLES_PER_PIXEL: u32 = 500;

struct Shared {
    width: usize,
    height: usize,
    /// RGBA8, sRGB-encoded.
    pixels: Mutex<Vec<u8>>,
    /// Bumped on every published pass.
    version: AtomicU64,
    /// Rays traced so far, all bounces included.
    rays: AtomicU64,
    /// Samples per pixel in the published image.
    samples: AtomicU32,
    thread_count: usize,
    started: Instant,
    finished: Mutex<Option<Duration>>,
}

impl Shared {
    fn samples(&self) -> u32 {
        self.samples.load(Ordering::Relaxed)
    }
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("raytracing_weekend")?;
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

        let shared = spawn_render();
        let egui_component = engine.get_global_component_mut::<EguiComponent>()?;
        egui_component.show_engine_windows = false;
        let texture: Mutex<Option<(egui::TextureHandle, u64)>> = Mutex::new(None);
        egui_component.register_ui("raytracing_weekend.image", move |context| {
            image_ui(context, &shared, &texture)
        });
        Ok(())
    }
}

fn spawn_render() -> Arc<Shared> {
    let world = scene::final_scene(&mut rand::rng());
    let camera = scene::final_camera(IMAGE_WIDTH, SAMPLES_PER_PIXEL);
    let shared = Arc::new(Shared {
        width: camera.image_width,
        height: camera.image_height,
        pixels: Mutex::new(vec![0; camera.image_width * camera.image_height * 4]),
        version: AtomicU64::new(0),
        rays: AtomicU64::new(0),
        samples: AtomicU32::new(0),
        thread_count: rayon::current_num_threads(),
        started: Instant::now(),
        finished: Mutex::new(None),
    });
    let render_shared = Arc::clone(&shared);
    // The render thread owns the world and camera; rayon's threads borrow them
    std::thread::spawn(move || render(&camera, &world, &render_shared));
    shared
}

/// The book's render loop, one sample per pixel per pass: every pixel is an independent
/// `pixel → sample` map over the accumulation buffer, with a thread-local RNG.
fn render(camera: &Camera, world: &HittableList, shared: &Shared) {
    let width = camera.image_width;
    let mut accumulation = vec![Color::default(); width * camera.image_height];
    for pass in 1..=camera.samples_per_pixel {
        let rays: u64 = accumulation
            .par_iter_mut()
            .enumerate()
            .map(|(index, sum)| {
                let mut rays = 0;
                *sum += camera.pixel_sample(
                    index % width,
                    index / width,
                    world,
                    &mut rand::rng(),
                    &mut rays,
                );
                rays
            })
            .sum();

        let pixel_samples_scale = 1.0 / pass as f64;
        let rgba: Vec<u8> = accumulation
            .par_iter()
            .flat_map_iter(|&sum| {
                let [r, g, b] = write_color(pixel_samples_scale * sum);
                [r, g, b, 255]
            })
            .collect();
        *shared.pixels.lock().unwrap() = rgba;
        shared.samples.store(pass, Ordering::Relaxed);
        shared.version.fetch_add(1, Ordering::Relaxed);
        shared.rays.fetch_add(rays, Ordering::Relaxed);
    }
    *shared.finished.lock().unwrap() = Some(shared.started.elapsed());
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
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [shared.width, shared.height],
            &shared.pixels.lock().unwrap(),
        );
        match texture.as_mut() {
            Some((handle, uploaded)) => {
                handle.set(image, egui::TextureOptions::LINEAR);
                *uploaded = version;
            }
            None => {
                let handle =
                    context.load_texture("raytracing_weekend", image, egui::TextureOptions::LINEAR);
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
            let aspect = shared.width as f32 / shared.height as f32;
            let size = if available.x / available.y > aspect {
                egui::vec2(available.y * aspect, available.y)
            } else {
                egui::vec2(available.x, available.x / aspect)
            };
            ui.centered_and_justified(|ui| ui.image((handle.id(), size)));
        });

    let samples = shared.samples();
    let elapsed = shared
        .finished
        .lock()
        .unwrap()
        .unwrap_or_else(|| shared.started.elapsed());
    egui::Area::new(egui::Id::new("raytracing_weekend.stats"))
        .fixed_pos(egui::pos2(8.0, 8.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_black_alpha(160))
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} × {}  {samples} / {SAMPLES_PER_PIXEL} spp  {:.1} s  {:.1} Mrays/s  {} threads",
                            shared.width,
                            shared.height,
                            elapsed.as_secs_f32(),
                            shared.rays.load(Ordering::Relaxed) as f32 / elapsed.as_secs_f32() / 1e6,
                            shared.thread_count,
                        ))
                        .monospace()
                        .color(egui::Color32::WHITE),
                    );
                });
        });
}
