use pill_engine::game::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::free_fly::{free_fly_system, FreeFlyComponent};
use crate::gaussian_cloud::GaussianCloud;
use crate::pass_splat::PassSplat;

pub static ACTIVE_SCENE: AtomicUsize = AtomicUsize::new(0);

pub struct ScenePose {
    pub position: [f32; 3],
    pub yaw:      f32,
    pub pitch:    f32,
}

pub const SCENES: &[(&str, ScenePose)] = &[
    ("splat",    ScenePose { position: [0.0, 1.0, 3.0], yaw: 180.0, pitch: 0.0 }),
    ("room",     ScenePose { position: [0.0, 1.0, 3.0], yaw: 180.0, pitch: 0.0 }),
    ("garden",   ScenePose { position: [0.0, 1.0, 3.0], yaw: 180.0, pitch: 0.0 }),
    ("playroom", ScenePose { position: [0.0, 1.0, 3.0], yaw: 180.0, pitch: 0.0 }),
    ("train",    ScenePose { position: [0.0, 1.0, 3.0], yaw: 180.0, pitch: 0.0 }),
];

pub const SCENE_NAMES: &[&str] = &["splat", "room", "garden", "playroom", "train"];

// Auto-computed camera poses filled by PassSplat after each scene loads
pub static AUTO_POSES: Mutex<[Option<ScenePose>; 5]> =
    Mutex::new([None, None, None, None, None]);

// Tracks whether the auto pose has been applied for each scene slot
static POSE_APPLIED: [AtomicBool; 5] = [
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
    AtomicBool::new(false),
];

pub struct GaussianGame {}

impl PillGame for GaussianGame {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("default")?;
        engine.set_active_scene(scene)?;

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FreeFlyComponent>(scene)?;

        engine.register_resource_type::<GaussianCloud>(16)?;

        for (name, _) in SCENES {
            engine.add_resource(GaussianCloud::from_path(name, &format!("{name}.ply")))?;
        }

        let initial = &SCENES[0].1;
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::from(initial.position))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .clear_color(Color::new(0.05, 0.05, 0.07))
                    .build(),
            )
            .with_component(FreeFlyComponent::new(initial.yaw, initial.pitch, 5.0))
            .build();

        engine.set_render_passes(vec![Box::new(PassSplat::new("splat"))])?;

        engine.add_system("free_fly",      free_fly_system)?;
        engine.add_system("scene_switch",  scene_switch_system)?;
        engine.add_system("camera_update", camera_update_system)?;

        Ok(())
    }
}

fn scene_switch_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component::<InputComponent>()?;
    let keys = [
        KeyboardKey::Digit1,
        KeyboardKey::Digit2,
        KeyboardKey::Digit3,
        KeyboardKey::Digit4,
        KeyboardKey::Digit5,
    ];

    let mut pressed = None;
    for (i, key) in keys.iter().enumerate() {
        if input.get_key_pressed(*key) {
            pressed = Some(i);
        }
    }

    if let Some(idx) = pressed {
        ACTIVE_SCENE.store(idx, Ordering::Relaxed);

        let guard = AUTO_POSES.lock().unwrap();
        if let Some(ref auto) = guard[idx] {
            // Previously loaded — apply computed pose immediately
            let pos   = auto.position;
            let yaw   = auto.yaw;
            let pitch = auto.pitch;
            drop(guard);
            POSE_APPLIED[idx].store(true, Ordering::Relaxed);
            for (_, transform, fly) in
                engine.iterate_two_components_mut::<TransformComponent, FreeFlyComponent>()?
            {
                transform.set_position(Vector3f::from(pos));
                fly.yaw   = yaw;
                fly.pitch = pitch;
            }
        } else {
            // First load — camera_update_system will position once data is ready
            drop(guard);
            POSE_APPLIED[idx].store(false, Ordering::Relaxed);
        }
    }

    Ok(())
}

fn camera_update_system(engine: &mut Engine) -> Result<()> {
    let idx = ACTIVE_SCENE.load(Ordering::Relaxed).min(SCENE_NAMES.len() - 1);
    if POSE_APPLIED[idx].load(Ordering::Relaxed) {
        return Ok(());
    }
    let guard = AUTO_POSES.lock().unwrap();
    if let Some(ref pose) = guard[idx] {
        let position = pose.position;
        let yaw      = pose.yaw;
        let pitch    = pose.pitch;
        drop(guard);

        for (_, transform, fly) in
            engine.iterate_two_components_mut::<TransformComponent, FreeFlyComponent>()?
        {
            transform.set_position(Vector3f::from(position));
            fly.yaw   = yaw;
            fly.pitch = pitch;
        }
        POSE_APPLIED[idx].store(true, Ordering::Relaxed);
    }
    Ok(())
}
