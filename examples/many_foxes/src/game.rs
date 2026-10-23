//! Bevy's `many_foxes` stress test: 1000 Khronos `Fox.glb` running in concentric rings (2 m apart,
//! foxes ≥ 2 m apart, alternate rings in opposite directions at 2 m/s). Each fox is an entity
//! with a `SkeletalAnimationComponent` (model, clip, time); the engine poses and skins it.

use std::f32::consts::PI;
use std::sync::{Arc, Mutex};

use pill_engine::define_component;
use pill_engine::game::*;
use web_time::Instant;

const FOX: &[u8] = include_bytes!("../../skinning/res/models/Fox.glb");
const FOX_COUNT: usize = 1000;
/// 0 = Survey, 1 = Walk, 2 = Run
const CLIP: usize = 2;
/// All foxes at the same animation time; otherwise fox `i` starts `i / 10` s in (as Bevy)
const SYNC: bool = false;
/// Fox units are centimetres
const FOX_SCALE: f32 = 0.01;
const RING_SPACING: f32 = 2.0;
const FOX_SPACING: f32 = 2.0;
/// Tangential ring speed, m/s
const RING_SPEED: f32 = 2.0;

// angle: around the ring at t = 0 (rad); base_yaw: heading on top of the ring orientation (rad);
// direction: +1 / −1 rotation sense of the ring about +Y
define_component!(FoxRing {
    radius: f32,
    angle: f32,
    base_yaw: f32,
    direction: f32,
});

/// Per-frame times summed over the current 1 s report window
#[derive(Default)]
struct Diagnostics {
    frames: u32,
    window_start: Option<Instant>,
    /// game animation update, engine skeletal animation, rings, whole frame (s)
    stages: [f64; 4],
}

const STAGE_NAMES: [&str; 4] = ["anim time", "skeletal", "rings", "frame"];

struct Foxes {
    start: Instant,
    last_frame: Instant,
    diagnostics: Diagnostics,
    label: Arc<Mutex<String>>,
}

static FOXES: Mutex<Option<Foxes>> = Mutex::new(None);

pub struct Game {}

fn foxes_state() -> Result<std::sync::MutexGuard<'static, Option<Foxes>>> {
    FOXES.lock().map_err(|_| PillError::from("foxes state poisoned"))
}

/// Advances every fox's clip time by the frame time
fn animation_system(engine: &mut Engine) -> Result<()> {
    let started = Instant::now();
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    for (_, animation) in engine.iterate_one_component_mut::<SkeletalAnimationComponent>()? {
        animation.time += delta_time;
    }
    if let Some(state) = foxes_state()?.as_mut() {
        state.diagnostics.stages[0] += started.elapsed().as_secs_f64();
    }
    Ok(())
}

fn ring_system(engine: &mut Engine) -> Result<()> {
    let started = Instant::now();
    let mut guard = foxes_state()?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let elapsed = state.start.elapsed().as_secs_f32();
    for (_, transform, ring) in engine.iterate_two_components_mut::<TransformComponent, FoxRing>()? {
        let theta = ring.direction * RING_SPEED / ring.radius * elapsed;
        // Ry(θ)·(r cos a, 0, r sin a)
        let (sin_a, cos_a) = (ring.angle - theta).sin_cos();
        transform.set_position(Vector3f::new(ring.radius * cos_a, 0.0, ring.radius * sin_a));
        transform.set_rotation(Vector3f::new(0.0, (theta + ring.base_yaw - ring.angle).to_degrees(), 0.0));
    }
    state.diagnostics.stages[2] += started.elapsed().as_secs_f64();
    Ok(())
}

/// Reports the previous frame (its skeletal animation update ran after the game systems)
fn diagnostics_system(engine: &mut Engine) -> Result<()> {
    let skeletal_ms = engine.get_skeletal_animation_cpu_ms()?;
    let mut guard = foxes_state()?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let now = Instant::now();
    let diagnostics = &mut state.diagnostics;
    diagnostics.stages[1] += f64::from(skeletal_ms) / 1e3;
    diagnostics.stages[3] += (now - state.last_frame).as_secs_f64();
    state.last_frame = now;
    diagnostics.frames += 1;
    let window_start = *diagnostics.window_start.get_or_insert(now);
    if (now - window_start).as_secs_f32() < 1.0 {
        return Ok(());
    }
    let frames = f64::from(diagnostics.frames);
    let mut report = format!("foxes {FOX_COUNT}  clip {CLIP}  sync {SYNC}  ({} frames avg)", diagnostics.frames);
    for (name, seconds) in STAGE_NAMES.iter().zip(diagnostics.stages) {
        report += &format!("\n{name:<10} {:7.2} ms", seconds / frames * 1e3);
    }
    report += &format!("\n{:<10} {:7.1}", "fps", frames / diagnostics.stages[3]);
    println!("[many_foxes] {}", report.replace('\n', " | "));
    if let Ok(mut label) = state.label.lock() {
        *label = report;
    }
    *diagnostics = Diagnostics::default();
    Ok(())
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("many_foxes")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<SkeletalAnimationComponent>(scene)?;
        engine.register_component::<FoxRing>(scene)?;

        // Texture + material from the scene loader; skeleton, mesh and clips from the skin loader
        let GltfSceneData {
            textures,
            materials,
            ..
        } = parse_gltf_scene("fox", FOX, &[], &mut [], &mut CookedImages::new())?;
        let data = parse_gltf_skinned(FOX, &[])?;
        let material_index = data.primitives[0].material;

        let mut texture_handles = Vec::with_capacity(textures.len());
        for texture in textures {
            texture_handles.push(engine.add_resource(Texture::from_bytes_owned(
                &texture.name,
                texture.kind,
                texture.rtex,
            ))?);
        }
        // `parse_gltf_scene` appends its default material last
        let gltf_material = &materials[material_index.unwrap_or(materials.len() - 1)];
        let [red, green, blue, _] = gltf_material.base_color;
        let mut material = Material::new(&gltf_material.name)
            .albedo(Color::new(red, green, blue))
            .metallic(gltf_material.metallic)
            .roughness(gltf_material.roughness);
        if let Some(i) = gltf_material.albedo {
            material = material.albedo_texture(texture_handles[i]);
        }
        let material_handle = engine.add_resource(material)?;
        let fox = engine.add_skeletal_model(data)?;

        // Concentric rings, as Bevy: even rings face −tangent (base yaw π) and turn +Y, odd rings
        // face +tangent and turn −Y, so every fox runs forward
        let mut spawned = 0;
        let mut ring_index = 0;
        let mut radius = RING_SPACING;
        while spawned < FOX_COUNT {
            let (base_yaw, direction) = if ring_index % 2 == 0 { (PI, 1.0) } else { (0.0, -1.0) };
            let circumference = 2.0 * PI * radius;
            let foxes_in_ring = ((circumference / FOX_SPACING) as usize).min(FOX_COUNT - spawned);
            let spacing_angle = circumference / (foxes_in_ring as f32 * radius);
            for fox_in_ring in 0..foxes_in_ring {
                let mut animation = SkeletalAnimationComponent::new(fox, CLIP);
                animation.time = if SYNC { 0.0 } else { spawned as f32 / 10.0 };
                engine
                    .build_entity(scene)
                    .with_component(
                        TransformComponent::builder()
                            .scale(Vector3f::splat(FOX_SCALE))
                            .build(),
                    )
                    .with_component(MeshRenderingComponent::builder().material(&material_handle).build())
                    .with_component(animation)
                    .with_component(FoxRing {
                        radius,
                        angle: fox_in_ring as f32 * spacing_angle,
                        base_yaw,
                        direction,
                    })
                    .build();
                spawned += 1;
            }
            radius += RING_SPACING;
            ring_index += 1;
        }

        // Bevy's camera: zoom 0.8 off the radius past the last ring
        let zoom = 0.8;
        let camera_position = Vector3f::new(radius * 1.25 * zoom, radius * 0.5 * zoom, radius * 1.5 * zoom);
        let mut camera_transform = TransformComponent::builder().position(camera_position).build();
        camera_transform.look_at(0.2 * Vector3f::new(camera_position.x, 0.0, camera_position.z));
        engine
            .build_entity(scene)
            .with_component(camera_transform)
            .with_component(CameraComponent::builder().enabled(true).build())
            .build();

        let label = Arc::new(Mutex::new(String::new()));
        {
            let egui_component = engine.get_global_component_mut::<EguiComponent>()?;
            egui_component.show_engine_windows = false;
            let label = Arc::clone(&label);
            egui_component.register_ui("many_foxes.stats", move |context| {
                let text = label.lock().map(|l| l.clone()).unwrap_or_default();
                egui::Area::new(egui::Id::new("many_foxes.stats"))
                    .fixed_pos(egui::pos2(8.0, 8.0))
                    .order(egui::Order::Foreground)
                    .show(context, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_black_alpha(160))
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                // Extend: the area is sized on the first frames, before any report
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(text)
                                            .monospace()
                                            .color(egui::Color32::WHITE),
                                    )
                                    .extend(),
                                );
                            });
                    });
            });
        }

        let now = Instant::now();
        *foxes_state()? = Some(Foxes {
            start: now,
            last_frame: now,
            diagnostics: Diagnostics::default(),
            label,
        });
        engine.add_system("diagnostics", diagnostics_system)?;
        engine.add_system("animation", animation_system)?;
        engine.add_system("ring", ring_system)?;
        Ok(())
    }
}
