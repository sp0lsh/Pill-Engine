//! Hand gestures from the laptop camera, web build: the first camera's video fills the window
//! (mirrored); each hand found by OpenCV Zoo's MediaPipe palm detector + hand landmark models
//! (res/models, run on the GPU by `gpu_net`) is classified into an AR interaction gesture
//! (`gestures`), listed on screen and drawn styled by it: the skeleton colored per gesture, a
//! ray along the index finger when pointing, a circle between thumb and index when pinching.
//! Two boxes in front of the camera prototype a gesture-driven 3D editor (`interaction`): a pinch
//! on a box picks it up; while the pinch holds, the box follows the hand in 3D: across the screen
//! with the pinch, in depth as the hand moves toward or away from the camera, and in rotation as
//! the hand turns.
//! The native build only shows a notice.

use crate::gestures::{pinch_point, Gesture, PinchPhase};
use crate::hands::{INDEX_MCP, INDEX_TIP, LANDMARK_COUNT, MIDDLE_MCP, SKELETON, WRIST};
use crate::interaction::{self, CameraFrame, Hold};
use pill_engine::{define_component, define_global_component, game::*};
use std::sync::Mutex;

/// Box centres and rotations (degrees), metres in front of the camera at the origin.
const BOXES: [(Vector3f, Vector3f); 2] = [
    (Vector3f::new(-0.8, 0.0, -3.0), Vector3f::new(20.0, 30.0, 0.0)),
    (Vector3f::new(0.8, 0.0, -3.0), Vector3f::new(-15.0, -25.0, 0.0)),
];

define_component!(EditorBox {
    index: usize,
    material: MaterialHandle,
    held_material: MaterialHandle,
});

/// A box held by a pinching hand (`Gestures::ids`).
struct Held {
    hand: u64,
    box_index: usize,
    hold: Hold,
}

define_global_component!(Editor { held: Vec<Held> });

const POINT_RADIUS_PX: f32 = 5.0;
const BONE_WIDTH_PX: f32 = 3.0;
const GESTURE_BONE_WIDTH_PX: f32 = 6.0;
const RAY_WIDTH_PX: f32 = 3.0;
/// Pinch circle radius, palm sizes
const PINCH_RADIUS: f32 = 0.3;

/// Skeleton color and width of a hand showing `gesture`.
fn style(gesture: Gesture) -> (egui::Color32, f32) {
    let color = match gesture {
        Gesture::None => return (egui::Color32::WHITE, BONE_WIDTH_PX),
        Gesture::Point => egui::Color32::from_rgb(0, 220, 255),
        Gesture::OpenPalm => egui::Color32::from_rgb(64, 230, 96),
        Gesture::Grab => egui::Color32::from_rgb(255, 150, 32),
        Gesture::Pinch(_) => egui::Color32::from_rgb(255, 64, 220),
    };
    (color, GESTURE_BONE_WIDTH_PX)
}

fn label(gesture: Gesture) -> String {
    match gesture {
        Gesture::None => "—".to_string(),
        Gesture::Point => "Point".to_string(),
        Gesture::OpenPalm => "Open palm".to_string(),
        Gesture::Grab => "Grab".to_string(),
        Gesture::Pinch(PinchPhase::Start) => "Pinch (start)".to_string(),
        Gesture::Pinch(PinchPhase::Hold { seconds }) => format!("Pinch (hold {seconds:.1} s)"),
        Gesture::Pinch(PinchPhase::Release) => "Pinch (release)".to_string(),
    }
}

/// The last tracker result (written by `TrackingPass`, read by `interaction_system` and the
/// overlay) and what only the overlay shows.
struct Overlay {
    status: String,
    /// Size of the frames the landmarks are in, pixels
    frame_size: Vector2f,
    hands: Vec<[Vector2f; LANDMARK_COUNT]>,
    /// Metric 3D landmarks of `hands`
    world_hands: Vec<[Vector3f; LANDMARK_COUNT]>,
    /// The gesture and the stable id of each of `hands`
    gestures: Vec<Gesture>,
    hand_ids: Vec<u64>,
    /// Held boxes, one line each
    editor: Vec<String>,
    /// GPU submit → read-back of the last inference, milliseconds
    latency_ms: f32,
}

static OVERLAY: Mutex<Overlay> = Mutex::new(Overlay {
    status: String::new(),
    frame_size: Vector2f::ZERO,
    hands: Vec::new(),
    world_hands: Vec::new(),
    gestures: Vec::new(),
    hand_ids: Vec::new(),
    editor: Vec::new(),
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
        let scene = engine.create_scene("hand_gestures")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<EditorBox>(scene)?;
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .range(0.1..100.0)
                    .build(),
            )
            .build();

        let cube = engine.add_resource(Mesh::from_cooked_mesh_bytes(
            "cube",
            include_bytes!("../res/models/cube.cooked_mesh"),
        )?)?;
        let held_material = engine.add_resource(
            Material::new("box_held")
                .albedo(Color::new(1.0, 0.85, 0.2))
                .emissive(Color::new(0.6, 0.45, 0.0)),
        )?;
        let colors = [Color::new(0.2, 0.45, 1.0), Color::new(1.0, 0.35, 0.25)];
        for (index, ((position, rotation), color)) in BOXES.into_iter().zip(colors).enumerate() {
            let material = engine.add_resource(
                Material::new(&format!("box_{index}"))
                    .albedo(color)
                    .roughness(0.5),
            )?;
            engine
                .build_entity(scene)
                .with_component(
                    TransformComponent::builder()
                        .position(position)
                        .rotation(rotation)
                        .build(),
                )
                .with_component(
                    MeshRenderingComponent::builder()
                        .mesh(&cube)
                        .material(&material)
                        .build(),
                )
                .with_component(EditorBox {
                    index,
                    material,
                    held_material,
                })
                .build();
        }
        engine.add_global_component(Editor { held: Vec::new() })?;
        engine.add_system("interaction", interaction_system)?;

        // The video is display-referred: the tonemap pass only sRGB-encodes it
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .tonemap = Tonemap::None;
        let egui = engine.get_global_component_mut::<EguiComponent>()?;
        egui.show_engine_windows = false;
        egui.register_ui("hand_gestures.overlay", overlay_ui);

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
    use crate::{camera, camera_pass::CameraPass, gestures::Gestures, gpu_tracker::GpuTracker};
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
        engine.add_render_pass(Box::new(TrackingPass {
            tracker: None,
            gestures: Gestures::default(),
        }))?;
        engine.add_render_pass(Box::new(CameraPass::new()))?;
        Ok(())
    }

    async fn fetch_models() -> std::result::Result<(Vec<u8>, Vec<u8>), String> {
        Ok((
            camera::fetch(PALM_MODEL_URL).await?,
            camera::fetch(HAND_MODEL_URL).await?,
        ))
    }

    /// Steps the GPU tracker once per frame on the renderer's device and classifies each result's
    /// hands; draws nothing.
    struct TrackingPass {
        tracker: Option<GpuTracker>,
        gestures: Gestures,
    }

    impl Pass for TrackingPass {
        fn get_label(&self) -> &str {
            "hand_gestures.tracking_pass"
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
            if finished {
                let gestures = self
                    .gestures
                    .update(tracker.hands(), crate::gpu_tracker::now_ms() / 1000.0);
                with_overlay(|overlay| {
                    overlay.status.clear();
                    overlay.hands = tracker.hands().to_vec();
                    overlay.world_hands = tracker.world_hands().to_vec();
                    overlay.gestures = gestures;
                    overlay.hand_ids = self.gestures.ids();
                    overlay.latency_ms = tracker.latency_ms;
                });
            }
            if let Some(size) = frame_size {
                with_overlay(|overlay| overlay.frame_size = size);
            }
            Ok(())
        }
    }
}

/// A pinch that starts on a box picks it up; while the pinch holds, the box follows the hand
/// (`Hold`); held boxes are highlighted.
fn interaction_system(engine: &mut Engine) -> Result<()> {
    let (hands, world_hands, gestures, hand_ids, frame_size) = with_overlay(|overlay| {
        (
            overlay.hands.clone(),
            overlay.world_hands.clone(),
            overlay.gestures.clone(),
            overlay.hand_ids.clone(),
            overlay.frame_size,
        )
    });
    if frame_size.x <= 0.0 {
        return Ok(());
    }
    let Some(camera) = engine
        .iterate_two_components_mut::<CameraComponent, TransformComponent>()?
        .find(|(_, camera, _)| camera.enabled)
        .map(|(_, camera, transform)| CameraFrame::new(camera, transform))
    else {
        return Ok(());
    };
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    let hand = |id: u64| hand_ids.iter().position(|&hand| hand == id);
    let pose = |i: usize| interaction::hand_pose(&hands[i], &world_hands[i], frame_size, &camera);

    let mut held = std::mem::take(&mut engine.get_global_component_mut::<Editor>()?.held);
    held.retain(|held| {
        hand(held.hand).is_some_and(|i| {
            matches!(
                gestures.get(i),
                Some(Gesture::Pinch(PinchPhase::Start | PinchPhase::Hold { .. }))
            )
        })
    });

    for (i, gesture) in gestures.iter().enumerate() {
        let id = hand_ids[i];
        if *gesture != Gesture::Pinch(PinchPhase::Start) || held.iter().any(|held| held.hand == id) {
            continue;
        }
        let pose = pose(i);
        let ray = camera.ray(pose.pinch_uv);
        let nearest = engine
            .iterate_two_components_mut::<TransformComponent, EditorBox>()?
            .filter(|(_, _, editor_box)| !held.iter().any(|held| held.box_index == editor_box.index))
            .filter_map(|(_, transform, editor_box)| {
                interaction::hit_box(&ray, transform).map(|distance| (distance, editor_box.index, *transform))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((distance, box_index, transform)) = nearest {
            let hit = ray.origin + ray.direction * distance;
            held.push(Held {
                hand: id,
                box_index,
                hold: Hold::new(hit, &transform, &camera, &pose),
            });
        }
    }

    for (_, transform, rendering, editor_box) in engine
        .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, EditorBox>()?
    {
        let held_box = held.iter_mut().find(|held| held.box_index == editor_box.index);
        let wanted = if held_box.is_some() {
            editor_box.held_material
        } else {
            editor_box.material
        };
        if rendering.material_handle != Some(wanted) {
            rendering.set_material(&wanted);
        }
        let Some(held_box) = held_box else {
            continue;
        };
        let Some(i) = hand(held_box.hand) else {
            continue;
        };
        let (position, rotation) = held_box.hold.update(&pose(i), delta_time, &camera);
        if let Some(position) = position {
            transform.set_position(position);
        }
        transform.set_rotation(rotation);
    }

    with_overlay(|overlay| {
        overlay.editor = held
            .iter()
            .filter_map(|held| {
                let i = hand(held.hand)?;
                Some(format!("box {}: held by hand {}", held.box_index + 1, i + 1))
            })
            .collect();
    });
    engine.get_global_component_mut::<Editor>()?.held = held;
    Ok(())
}

/// Each hand styled by its gesture over the video (same cover + mirror mapping as
/// `CameraPass`), its index-finger ray or pinch circle, and the gesture list.
fn overlay_ui(context: &egui::Context) {
    let painter = context.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("hand_gestures.overlay"),
    ));
    let screen = context.screen_rect();
    let points = |pixels: f32| pixels / context.pixels_per_point();
    with_overlay(|overlay| {
        if overlay.frame_size.x > 0.0 {
            let to_screen = |p: Vector2f| {
                let uv = interaction::screen_uv(p, overlay.frame_size, screen.aspect_ratio());
                screen.min + egui::vec2(uv.x * screen.width(), uv.y * screen.height())
            };
            for (i, landmarks) in overlay.hands.iter().enumerate() {
                let gesture = overlay.gestures.get(i).copied().unwrap_or(Gesture::None);
                let (color, width) = style(gesture);
                let stroke = egui::Stroke::new(points(width), color);
                for (a, b) in SKELETON {
                    painter.line_segment([to_screen(landmarks[a]), to_screen(landmarks[b])], stroke);
                }
                for &landmark in landmarks {
                    painter.circle_filled(to_screen(landmark), points(POINT_RADIUS_PX), color);
                }
                match gesture {
                    Gesture::Point => {
                        // From the index base through its tip to beyond the screen edge
                        let (base, tip) = (to_screen(landmarks[INDEX_MCP]), to_screen(landmarks[INDEX_TIP]));
                        let direction = (tip - base).normalized();
                        let far = tip + direction * screen.size().length();
                        painter.line_segment([tip, far], egui::Stroke::new(points(RAY_WIDTH_PX), color));
                    }
                    Gesture::Pinch(phase) => {
                        let centre = to_screen(pinch_point(landmarks));
                        let palm = to_screen(landmarks[WRIST]).distance(to_screen(landmarks[MIDDLE_MCP]));
                        let radius = PINCH_RADIUS * palm;
                        if matches!(phase, PinchPhase::Hold { .. }) {
                            painter.circle_filled(centre, radius, color.gamma_multiply(0.6));
                        } else {
                            painter.circle_stroke(centre, radius, egui::Stroke::new(points(RAY_WIDTH_PX), color));
                        }
                    }
                    _ => {}
                }
            }
        }
        let text = if overlay.status.is_empty() {
            let mut lines = vec![format!(
                "hands: {}   inference: {:.1} ms",
                overlay.hands.len(),
                overlay.latency_ms
            )];
            lines.extend(overlay.hands.iter().enumerate().map(|(i, _)| {
                let gesture = overlay.gestures.get(i).copied().unwrap_or(Gesture::None);
                format!("hand {}: {}", i + 1, label(gesture))
            }));
            lines.extend(overlay.editor.iter().cloned());
            lines.join("\n")
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
