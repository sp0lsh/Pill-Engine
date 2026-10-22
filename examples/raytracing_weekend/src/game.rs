//! Progressive CPU path tracing of Ray Tracing in One Weekend's final scene, with the book's
//! code structure and no optimizations. One worker thread per core owns every n-th image row,
//! adds one sample per pixel to its rows per pass and publishes each finished row as sRGB8 into a
//! shared image; the egui overlay uploads that image as a texture whenever it changed and shows
//! it letterboxed over the window.

use crate::{
    camera::Camera,
    color::{write_color, Color},
    hittable_list::HittableList,
    scene,
};
use pill_engine::game::*;
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
    /// Bumped on every published row.
    version: AtomicU64,
    /// Rays traced so far, all bounces included.
    rays: AtomicU64,
    /// Completed passes per worker; the image's sample count is their minimum.
    worker_samples: Vec<AtomicU32>,
    started: Instant,
    finished: Mutex<Option<Duration>>,
}

impl Shared {
    fn samples(&self) -> u32 {
        self.worker_samples
            .iter()
            .map(|samples| samples.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0)
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

        let shared = spawn_workers();
        let egui_component = engine.get_global_component_mut::<EguiComponent>()?;
        egui_component.show_engine_windows = false;
        let texture: Mutex<Option<(egui::TextureHandle, u64)>> = Mutex::new(None);
        egui_component.register_ui("raytracing_weekend.image", move |context| {
            image_ui(context, &shared, &texture)
        });
        Ok(())
    }
}

fn spawn_workers() -> Arc<Shared> {
    let world = Arc::new(scene::final_scene());
    let camera = Arc::new(scene::final_camera(IMAGE_WIDTH, SAMPLES_PER_PIXEL));
    let worker_count = std::thread::available_parallelism().map_or(4, |n| n.get());
    let shared = Arc::new(Shared {
        width: camera.image_width,
        height: camera.image_height,
        pixels: Mutex::new(vec![0; camera.image_width * camera.image_height * 4]),
        version: AtomicU64::new(0),
        rays: AtomicU64::new(0),
        worker_samples: (0..worker_count).map(|_| AtomicU32::new(0)).collect(),
        started: Instant::now(),
        finished: Mutex::new(None),
    });
    for worker in 0..worker_count {
        let camera = Arc::clone(&camera);
        let world = Arc::clone(&world);
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || render_rows(&camera, &world, &shared, worker, worker_count));
    }
    shared
}

/// Accumulates SAMPLES_PER_PIXEL passes over rows `worker, worker + worker_count, …`.
fn render_rows(
    camera: &Camera,
    world: &HittableList,
    shared: &Shared,
    worker: usize,
    worker_count: usize,
) {
    let width = camera.image_width;
    let rows: Vec<usize> = (worker..camera.image_height)
        .step_by(worker_count)
        .collect();
    let mut accumulation = vec![Color::default(); rows.len() * width];
    let mut row_rgba = vec![255u8; width * 4];
    for pass in 1..=camera.samples_per_pixel {
        let pixel_samples_scale = 1.0 / pass as f64;
        for (row_index, &y) in rows.iter().enumerate() {
            let row = &mut accumulation[row_index * width..(row_index + 1) * width];
            let mut rays = 0;
            for (x, sum) in row.iter_mut().enumerate() {
                *sum += camera.pixel_sample(x, y, world, &mut rays);
                let [r, g, b] = write_color(pixel_samples_scale * *sum);
                row_rgba[x * 4..x * 4 + 3].copy_from_slice(&[r, g, b]);
            }
            shared.pixels.lock().unwrap()[y * width * 4..(y + 1) * width * 4]
                .copy_from_slice(&row_rgba);
            shared.version.fetch_add(1, Ordering::Relaxed);
            shared.rays.fetch_add(rays, Ordering::Relaxed);
        }
        shared.worker_samples[worker].store(pass, Ordering::Relaxed);
    }
    let mut finished = shared.finished.lock().unwrap();
    if finished.is_none() && shared.samples() == SAMPLES_PER_PIXEL {
        *finished = Some(shared.started.elapsed());
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
                            shared.worker_samples.len(),
                        ))
                        .monospace()
                        .color(egui::Color32::WHITE),
                    );
                });
        });
}
