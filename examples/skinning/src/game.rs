//! Khronos `Fox.glb` and `RecursiveSkeletons.glb` played with the engine's CPU skinning baseline
//! (`pill_engine::game::skinning`): every frame samples the clip, walks the hierarchy, builds the
//! joint matrices, skins every primitive on the CPU and overwrites its mesh's GPU vertex buffers.
//! Recursive Skeletons binds one 40-vertex mesh to 84 skins over 4 chained armatures.

use std::sync::{Arc, Mutex};
use web_time::Instant;

use pill_engine::define_component;
use pill_engine::game::skinning::{global_transforms, joint_matrices, pose, skin};
use pill_engine::game::*;

const FOX: &[u8] = include_bytes!("../res/models/Fox.glb");
const RECURSIVE_SKELETONS: &[u8] = include_bytes!("../res/models/RecursiveSkeletons.glb");
/// Fox: 0 = Survey, 1 = Walk, 2 = Run
const FOX_CLIP: usize = 0;
/// Both models are in centimetres
const MODEL_SCALE: f32 = 0.01;
/// Recursive Skeletons spans x -125..54, z -42..125 (cm); centred 2.5 m right of the fox
const RECURSIVE_SKELETONS_POSITION: Vector3f = Vector3f::new(2.85, 0.0, -0.41);

define_component!(OrbitCamera {
    yaw: f32,
    pitch: f32,
    radius: f32,
});

struct SkinnedModel {
    name: &'static str,
    data: GltfSkinnedData,
    clip: usize,
    /// One per `data.primitives`
    meshes: Vec<MeshHandle>,
    vertices: Vec<pill_engine::game::skinning::CookedVertex>,
}

struct Skinning {
    models: Vec<SkinnedModel>,
    start: Instant,
    label: Arc<Mutex<String>>,
}

static SKINNING: Mutex<Option<Skinning>> = Mutex::new(None);

pub struct Game {}

fn skin_system(engine: &mut Engine) -> Result<()> {
    let mut guard = SKINNING.lock().map_err(|_| PillError::from("skinning state poisoned"))?;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let elapsed = state.start.elapsed().as_secs_f32();
    let mut lines = Vec::with_capacity(state.models.len());
    for model in &mut state.models {
        let animation = &model.data.animations[model.clip];
        // A single-key clip has duration 0
        let time = elapsed % animation.duration.max(f32::EPSILON);

        let started = Instant::now();
        let globals = global_transforms(&model.data.nodes, &pose(&model.data.nodes, animation, time));
        let mut skin_seconds = started.elapsed().as_secs_f64();
        let mut vertex_count = 0;
        for (primitive, mesh) in model.data.primitives.iter().zip(&model.meshes) {
            let started = Instant::now();
            let joints = joint_matrices(&model.data.skins[primitive.skin], &globals);
            skin(primitive, &joints, &mut model.vertices);
            skin_seconds += started.elapsed().as_secs_f64();
            vertex_count += model.vertices.len();
            engine.update_mesh_vertices(mesh, bytemuck::cast_slice(&model.vertices))?;
        }
        lines.push(format!(
            "{} {} t={time:.2}/{:.2}s  skins {}  vertices {vertex_count}  CPU skin {:.3} ms",
            model.name,
            animation.name,
            animation.duration,
            model.data.skins.len(),
            skin_seconds * 1e3,
        ));
    }
    if let Ok(mut label) = state.label.lock() {
        *label = lines.join("\n");
    }
    Ok(())
}

fn orbit_camera_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component_mut::<InputComponent>()?;
    let mouse_delta = input.get_mouse_delta();
    let scroll_delta = input.get_mouse_scroll_delta();
    let left_mouse_button_held = input.get_mouse_button(MouseButton::Left);

    for (_, transform, orbit) in
        engine.iterate_two_components_mut::<TransformComponent, OrbitCamera>()?
    {
        if left_mouse_button_held {
            orbit.yaw -= mouse_delta.x * 0.3;
            orbit.pitch = (orbit.pitch + mouse_delta.y * 0.3).clamp(-70.0, 70.0);
        }
        orbit.radius = (orbit.radius - scroll_delta.y * 0.2).clamp(0.5, 10.0);

        let target = Vector3f::new(1.4, 0.6, 0.0);
        let pitch_radians = orbit.pitch.to_radians();
        let yaw_radians = orbit.yaw.to_radians();
        let radius_cos_pitch = orbit.radius * pitch_radians.cos();
        transform.set_position(
            target
                + Vector3f::new(
                    radius_cos_pitch * yaw_radians.sin(),
                    orbit.radius * pitch_radians.sin(),
                    radius_cos_pitch * yaw_radians.cos(),
                ),
        );
        transform.look_at(target);
    }
    Ok(())
}

/// Spawns one entity per skinned primitive, each with its own mesh (skins sharing a glTF mesh still
/// deform it differently). Texture + materials come from the scene loader; skins, rest pose and
/// clips from the skin loader.
fn load_model(
    engine: &mut Engine,
    scene: SceneHandle,
    name: &'static str,
    glb: &[u8],
    clip: usize,
    position: Vector3f,
) -> Result<SkinnedModel> {
    let GltfSceneData {
        textures,
        materials,
        ..
    } = parse_gltf_scene(name, glb, &[], &mut [], &mut CookedImages::new())?;
    let data = parse_gltf_skinned(glb, &[])?;

    let mut texture_handles = Vec::with_capacity(textures.len());
    for texture in textures {
        texture_handles.push(engine.add_resource(Texture::from_bytes_owned(
            &texture.name,
            texture.kind,
            texture.rtex,
        ))?);
    }
    let mut material_handles = Vec::with_capacity(materials.len());
    for gltf_material in &materials {
        let [red, green, blue, _] = gltf_material.base_color;
        let mut material = Material::new(&format!("{name}_{}", gltf_material.name))
            .albedo(Color::new(red, green, blue))
            .metallic(gltf_material.metallic)
            .roughness(gltf_material.roughness);
        if let Some(i) = gltf_material.albedo {
            material = material.albedo_texture(texture_handles[i]);
        }
        material_handles.push(engine.add_resource(material)?);
    }

    let mut meshes = Vec::with_capacity(data.primitives.len());
    for primitive in &data.primitives {
        let mesh = engine.add_resource(Mesh::from_data(
            &format!("{name}_{}", primitive.name),
            MeshData {
                vertices: bytemuck::cast_slice(&primitive.rest).to_vec(),
                indices: primitive.indices.clone(),
                trace_lod: None,
            },
        ))?;
        // `parse_gltf_scene` appends its default material last
        let material = material_handles[primitive.material.unwrap_or(materials.len() - 1)];
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(position)
                    .scale(Vector3f::splat(MODEL_SCALE))
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&mesh)
                    .material(&material)
                    .build(),
            )
            .build();
        meshes.push(mesh);
    }
    Ok(SkinnedModel {
        name,
        data,
        clip,
        meshes,
        vertices: Vec::new(),
    })
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("skinning")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<OrbitCamera>(scene)?;

        let models = vec![
            load_model(engine, scene, "fox", FOX, FOX_CLIP, Vector3f::ZERO)?,
            load_model(
                engine,
                scene,
                "recursive_skeletons",
                RECURSIVE_SKELETONS,
                0,
                RECURSIVE_SKELETONS_POSITION,
            )?,
        ];

        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(CameraComponent::builder().enabled(true).fov(50.0).build())
            .with_component(OrbitCamera {
                yaw: 20.0,
                pitch: 15.0,
                radius: 5.0,
            })
            .build();

        let label = Arc::new(Mutex::new(String::new()));
        {
            let egui_component = engine.get_global_component_mut::<EguiComponent>()?;
            egui_component.show_engine_windows = false;
            let label = Arc::clone(&label);
            egui_component.register_ui("skinning.stats", move |context| {
                let text = label.lock().map(|l| l.clone()).unwrap_or_default();
                egui::Area::new(egui::Id::new("skinning.stats"))
                    .fixed_pos(egui::pos2(8.0, 8.0))
                    .order(egui::Order::Foreground)
                    .show(context, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_black_alpha(160))
                            .inner_margin(6.0)
                            .show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(text)
                                        .monospace()
                                        .color(egui::Color32::WHITE),
                                );
                            });
                    });
            });
        }

        *SKINNING.lock().map_err(|_| PillError::from("skinning state poisoned"))? = Some(Skinning {
            models,
            start: Instant::now(),
            label,
        });
        engine.add_system("skin", skin_system)?;
        engine.add_system("orbit_camera", orbit_camera_system)?;
        Ok(())
    }
}
