//! Hand tracking from the laptop camera, web build: the first camera's video fills the window
//! (mirrored), and each hand found by OpenCV Zoo's MediaPipe palm detector + hand landmark
//! models (res/models, run on the GPU by `gpu_net`) is drawn as 21 landmarks and its skeleton.
//! The native build only shows a notice.

use crate::hands::{LANDMARK_COUNT, SKELETON};
use pill_engine::game::*;
use std::sync::Mutex;

const POINT_RADIUS_PX: f32 = 5.0;
const BONE_WIDTH_PX: f32 = 3.0;
const BONE_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 255, 255);
const POINT_COLOR: egui::Color32 = egui::Color32::from_rgb(255, 64, 96);

/// What the overlay draws, written by the tracking system.
struct Overlay {
    status: String,
    /// Size of the frames the landmarks are in, pixels
    frame_size: Vector2f,
    hands: Vec<[Vector2f; LANDMARK_COUNT]>,
    /// GPU submit → read-back of the last inference, milliseconds
    latency_ms: f32,
}

static OVERLAY: Mutex<Overlay> = Mutex::new(Overlay {
    status: String::new(),
    frame_size: Vector2f::ZERO,
    hands: Vec::new(),
    latency_ms: 0.0,
});

fn with_overlay<T>(f: impl FnOnce(&mut Overlay) -> T) -> T {
    f(&mut OVERLAY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()))
}

fn set_status(status: impl Into<String>) {
    with_overlay(|overlay| overlay.status = status.into());
}

/// Screen uv → video uv scale around the centre for a `video` scaled to cover `screen` and
/// mirrored (selfie view): x is negative.
pub fn cover_uv_scale(video: Vector2f, screen: Vector2f) -> Vector2f {
    let scale = (screen.x / video.x).max(screen.y / video.y);
    screen / (video * scale) * Vector2f::new(-1.0, 1.0)
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("hand_tracking")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        // Queried by the engine's PBR pass, though this scene has no meshes
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(CameraComponent::builder().enabled(true).build())
            .build();

        // The video is display-referred: the tonemap pass only sRGB-encodes it
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .tonemap = Tonemap::None;
        let egui = engine.get_global_component_mut::<EguiComponent>()?;
        egui.show_engine_windows = false;
        egui.register_ui("hand_tracking.overlay", overlay_ui);

        #[cfg(target_arch = "wasm32")]
        web::start(engine)?;
        #[cfg(not(target_arch = "wasm32"))]
        set_status("Hand tracking runs in the web build (PillLauncher -a build -t web)");
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use super::{set_status, with_overlay};
    use crate::{camera, camera_pass::CameraPass, gpu_tracker::GpuTracker};
    use pill_engine::game::*;
    use std::cell::RefCell;

    const PALM_MODEL_URL: &str = "res/models/palm_detection_mediapipe_2023feb.onnx";
    const HAND_MODEL_URL: &str = "res/models/handpose_estimation_mediapipe_2023feb.onnx";

    thread_local! {
        // (palm, hand) ONNX bytes once downloaded; TrackingPass builds the GPU tracker from them
        static MODELS: RefCell<Option<(Vec<u8>, Vec<u8>)>> = const { RefCell::new(None) };
    }

    pub fn start(engine: &mut Engine) -> Result<()> {
        set_status("Waiting for camera permission…");
        wasm_bindgen_futures::spawn_local(async {
            match camera::open().await {
                Ok(()) => set_status("Loading hand models…"),
                Err(error) => set_status(format!("Camera: {error}")),
            }
        });
        wasm_bindgen_futures::spawn_local(async {
            match fetch_models().await {
                Ok(models) => MODELS.with(|slot| *slot.borrow_mut() = Some(models)),
                Err(error) => set_status(format!("Models: {error}")),
            }
        });
        engine.add_render_pass(Box::new(TrackingPass { tracker: None }))?;
        engine.add_render_pass(Box::new(CameraPass::new()))?;
        Ok(())
    }

    async fn fetch_models() -> std::result::Result<(Vec<u8>, Vec<u8>), String> {
        Ok((
            camera::fetch(PALM_MODEL_URL).await?,
            camera::fetch(HAND_MODEL_URL).await?,
        ))
    }

    /// Steps the GPU tracker once per frame on the renderer's device; draws nothing.
    struct TrackingPass {
        tracker: Option<GpuTracker>,
    }

    impl Pass for TrackingPass {
        fn get_label(&self) -> &str {
            "hand_tracking_pass"
        }

        fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> {
            Ok(())
        }

        fn draw(
            &mut self,
            _encoder: &mut wgpu::CommandEncoder,
            renderer: &mut dyn PillRenderer,
            _frame_target: &FrameTarget,
            _world: &WorldQuery<'_>,
        ) -> Result<()> {
            if self.tracker.is_none() {
                let Some((palm, hand)) = MODELS.with(|slot| slot.borrow_mut().take()) else {
                    return Ok(());
                };
                self.tracker = Some(GpuTracker::new(renderer.get_device(), &palm, &hand)?);
            }
            let tracker = self.tracker.as_mut().expect("tracker built above");
            let mut frame_size = None;
            let finished = tracker.step(renderer.get_device(), renderer.get_queue(), || {
                let frame = camera::frame()?;
                frame_size = Some(Vector2f::new(frame.width as f32, frame.height as f32));
                Some(frame)
            });
            if finished || frame_size.is_some() {
                with_overlay(|overlay| {
                    overlay.status.clear();
                    if let Some(size) = frame_size {
                        overlay.frame_size = size;
                    }
                    overlay.hands = tracker.hands().to_vec();
                    overlay.latency_ms = tracker.latency_ms;
                });
            }
            Ok(())
        }
    }
}

/// Landmarks and skeleton over the video (same cover + mirror mapping as `CameraPass`), the
/// status line and the inference times.
fn overlay_ui(context: &egui::Context) {
    let painter = context.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("hand_tracking.overlay"),
    ));
    let screen = context.screen_rect();
    let points = |pixels: f32| pixels / context.pixels_per_point();
    with_overlay(|overlay| {
        if overlay.frame_size.x > 0.0 {
            let screen_size = Vector2f::new(screen.width(), screen.height());
            let uv_scale = cover_uv_scale(overlay.frame_size, screen_size);
            let to_screen = |p: Vector2f| {
                let screen_uv = (p / overlay.frame_size - 0.5) / uv_scale + 0.5;
                egui::pos2(
                    screen.min.x + screen_uv.x * screen_size.x,
                    screen.min.y + screen_uv.y * screen_size.y,
                )
            };
            for landmarks in &overlay.hands {
                for (a, b) in SKELETON {
                    painter.line_segment(
                        [to_screen(landmarks[a]), to_screen(landmarks[b])],
                        egui::Stroke::new(points(BONE_WIDTH_PX), BONE_COLOR),
                    );
                }
                for &landmark in landmarks {
                    painter.circle_filled(to_screen(landmark), points(POINT_RADIUS_PX), POINT_COLOR);
                }
            }
        }
        let text = if overlay.status.is_empty() {
            format!(
                "hands: {}   inference: {:.1} ms",
                overlay.hands.len(),
                overlay.latency_ms
            )
        } else {
            overlay.status.clone()
        };
        painter.text(
            screen.min + egui::vec2(points(12.0), points(12.0)),
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::proportional(points(28.0)),
            egui::Color32::WHITE,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_uv_scale_wider_screen_crops_video_height_and_mirrors() {
        // 4:3 video on a 16:9 screen: full width, 3/4 · 16/9 of the height
        let scale = cover_uv_scale(Vector2f::new(640.0, 480.0), Vector2f::new(1280.0, 720.0));
        assert!((scale.x + 1.0).abs() < 1e-6);
        assert!((scale.y - 0.75).abs() < 1e-6);
    }
}
