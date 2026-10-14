use pill_engine::{define_component, game::*};

use crate::loader::{self, SponzaCpu};

// Fly camera: view angles in degrees; yaw 0 looks down -Z, positive pitch looks up
define_component!(FlyCamera {
    yaw: f32,
    pitch: f32,
});

// Point light drifting around its base position (light test)
define_component!(DriftingLight {
    base: Vector3f,
    phase: f32,
});

// Shadowed light sweeping along the court (see sweep_position); spots aim down into it
define_component!(SweepingLight {
    phase: f32,
    height: f32,
    aim: bool,
});

pub struct Game {}

// The background load (spawn_local task on WASM, thread on native) can't hold &mut Engine; it
// leaves its result here for `sponza_drain_system` to upload on the next frame.
type Loaded = std::result::Result<SponzaCpu, String>;
static PENDING: std::sync::Mutex<Option<Loaded>> = std::sync::Mutex::new(None);

// Meters per second; Shift multiplies
const FLY_SPEED: f32 = 4.0;
const FLY_FAST_MULTIPLIER: f32 = 3.0;
// Degrees per pixel of mouse movement
const LOOK_SENSITIVITY: f32 = 0.2;

// FPS controls: hold the left mouse button to look, WASD to move, E/Q up/down, Shift faster.
fn fly_camera_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    let input = engine.get_global_component::<InputComponent>()?;
    let mouse_delta = input.get_mouse_delta();
    let look_held = input.get_mouse_button(MouseButton::Left);
    let axis = |positive: KeyboardKey, negative: KeyboardKey| {
        input.get_key(positive) as i32 as f32 - input.get_key(negative) as i32 as f32
    };
    // x = right, y = up, z = forward
    let movement = Vector3f::new(
        axis(KeyboardKey::KeyD, KeyboardKey::KeyA),
        axis(KeyboardKey::KeyE, KeyboardKey::KeyQ),
        axis(KeyboardKey::KeyW, KeyboardKey::KeyS),
    );
    let fast = input.get_key(KeyboardKey::ShiftLeft) || input.get_key(KeyboardKey::ShiftRight);
    let speed = FLY_SPEED * if fast { FLY_FAST_MULTIPLIER } else { 1.0 };

    for (_, transform, fly) in
        engine.iterate_two_components_mut::<TransformComponent, FlyCamera>()?
    {
        if look_held {
            fly.yaw -= mouse_delta.x * LOOK_SENSITIVITY;
            fly.pitch = (fly.pitch - mouse_delta.y * LOOK_SENSITIVITY).clamp(-89.0, 89.0);
        }
        let (yaw, pitch) = (fly.yaw.to_radians(), fly.pitch.to_radians());
        let forward = Vector3f::new(
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        );
        let right = forward.cross(Vector3f::Y).normalize_or_zero();
        let step = (right * movement.x + Vector3f::Y * movement.y + forward * movement.z)
            .normalize_or_zero()
            * speed
            * delta_time;
        transform.set_position(transform.position + step);
        let target = transform.position + forward;
        transform.look_at(target);
    }
    Ok(())
}

// Shadow test: a few intense moving lights in the court and no fill light, so the columns' shadows
// are the only structure in the image. The 256 clustered lanterns are kept dark (J switches them
// on) to show many-light shading without washing the shadows out.
const LANTERN_COLUMNS: usize = 32;
const LANTERN_ROWS: [f32; 4] = [-6.5, -2.2, 2.2, 6.5]; // z
const LANTERN_HEIGHTS: [f32; 2] = [1.2, 7.2]; // ground floor, gallery
const LANTERN_INTENSITY: f32 = 3.0;
// Court centre line between the column rows (z ≈ −2.4 and +1.9)
const COURT_Z: f32 = -0.25;

/// Fraction in 0..1 from an integer (cheap hash, for jitter and hues).
fn hash(value: usize) -> f32 {
    ((value as f32 * 12.9898).sin() * 43_758.547).fract().abs()
}

/// Saturated color of a hue in 0..1.
fn hue_color(hue: f32) -> [f32; 3] {
    let channel =
        |offset: f32| (((hue + offset) * std::f32::consts::TAU).cos() * 0.5 + 0.5).powf(2.0);
    [channel(0.0), channel(1.0 / 3.0), channel(2.0 / 3.0)]
}

/// Path of a shadowed light: sweeps along the court at `height`, phase-shifted per light.
fn sweep_position(time: f32, phase: f32, height: f32) -> Vector3f {
    let t = time * 0.25 + phase;
    Vector3f::new(
        t.sin() * 7.0,
        height + (t * 1.7).sin() * 0.4,
        COURT_Z + (t * 1.3).cos() * 1.0,
    )
}

/// 3 shadowed point lights low in the court, 2 shadowed spot lights from above it, and 256
/// lanterns (dark until J).
fn spawn_lights(engine: &mut Engine, scene: SceneHandle) {
    let shadowed_points: [([f32; 3], f32); 3] = [
        ([1.0, 0.55, 0.25], 0.0),
        ([0.35, 0.6, 1.0], 2.1),
        ([1.0, 0.95, 0.85], 4.2),
    ];
    for (color, phase) in shadowed_points {
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(sweep_position(0.0, phase, 1.4))
                    .build(),
            )
            .with_component(
                LightComponent::builder()
                    .color(color)
                    .intensity(40.0)
                    .range(16.0)
                    .cast_shadows(true)
                    .build(),
            )
            .with_component(SweepingLight {
                phase,
                height: 1.4,
                aim: false,
            })
            .build();
    }
    // Spots looking down into the court from above the gallery roof line, aimed at a point that
    // sweeps with them
    for phase in [1.0, 3.6] {
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(sweep_position(0.0, phase, 11.0))
                    .build(),
            )
            .with_component(
                LightComponent::builder()
                    .light_type(LightType::Spot {
                        inner_angle: 18.0,
                        outer_angle: 28.0,
                    })
                    .color([1.0, 0.9, 0.75])
                    .intensity(250.0)
                    .range(16.0)
                    .cast_shadows(true)
                    .build(),
            )
            .with_component(SweepingLight {
                phase,
                height: 11.0,
                aim: true,
            })
            .build();
    }

    let mut index = 0;
    for &height in &LANTERN_HEIGHTS {
        for &row in &LANTERN_ROWS {
            for column in 0..LANTERN_COLUMNS {
                let x = -13.5 + 27.0 * column as f32 / (LANTERN_COLUMNS - 1) as f32;
                let base = Vector3f::new(
                    x + (hash(index) - 0.5) * 0.6,
                    height + hash(index + 1000) * 1.5,
                    row + (hash(index + 2000) - 0.5) * 0.8,
                );
                engine
                    .build_entity(scene)
                    .with_component(TransformComponent::builder().position(base).build())
                    .with_component(
                        LightComponent::builder()
                            .color(hue_color(hash(index + 3000)))
                            .intensity(0.0)
                            .range(3.5)
                            .build(),
                    )
                    .with_component(DriftingLight {
                        base,
                        phase: hash(index + 4000) * std::f32::consts::TAU,
                    })
                    .build();
                index += 1;
            }
        }
    }
}

/// glTF material of the ground and gallery floors (Khronos Sponza).
const FLOOR_MATERIAL: usize = 9;
/// Scales the floor's roughness texture: polished, wet stone.
const WET_FLOOR_ROUGHNESS: f32 = 0.1;
/// Scales the floor's albedo: wet stone is darker.
const WET_FLOOR_DARKENING: f32 = 0.5;

/// Moves the lights. L toggles the froxel light-count heatmap, K the point/spot shadows, J the
/// lanterns, G diffuse GI, H the indirect-light-only view, R screen-space reflections, T the
/// reflections-only view.
fn lights_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    let input = engine.get_global_component::<InputComponent>()?;
    let toggle_heatmap = input.get_key_pressed(KeyboardKey::KeyL);
    let toggle_shadows = input.get_key_pressed(KeyboardKey::KeyK);
    let toggle_lanterns = input.get_key_pressed(KeyboardKey::KeyJ);
    let toggle_gi = input.get_key_pressed(KeyboardKey::KeyG);
    let toggle_indirect_only = input.get_key_pressed(KeyboardKey::KeyH);
    let toggle_ssr = input.get_key_pressed(KeyboardKey::KeyR);
    let toggle_reflections_only = input.get_key_pressed(KeyboardKey::KeyT);
    let ssr = &mut engine
        .get_global_component_mut::<RenderStateComponent>()?
        .ssr;
    if toggle_ssr {
        ssr.enabled = !ssr.enabled;
    }
    if toggle_reflections_only {
        ssr.debug_reflections_only = !ssr.debug_reflections_only;
    }
    let gi = &mut engine
        .get_global_component_mut::<RenderStateComponent>()?
        .gi;
    if toggle_gi {
        gi.enabled = !gi.enabled;
    }
    if toggle_indirect_only {
        gi.debug_indirect_only = !gi.debug_indirect_only;
    }
    let lights = &mut engine
        .get_global_component_mut::<RenderStateComponent>()?
        .punctual_lights;
    if toggle_heatmap {
        lights.debug_heatmap = !lights.debug_heatmap;
    }
    if toggle_shadows {
        lights.shadows_enabled = !lights.shadows_enabled;
    }
    for (_, transform, sweep) in
        engine.iterate_two_components_mut::<TransformComponent, SweepingLight>()?
    {
        let position = sweep_position(time, sweep.phase, sweep.height);
        transform.set_position(position);
        if sweep.aim {
            // Down into the court, toward the column row on the far side
            transform.look_at(Vector3f::new(position.x * 0.8, 0.0, COURT_Z + 2.5));
        }
    }
    for (_, transform, drift) in
        engine.iterate_two_components_mut::<TransformComponent, DriftingLight>()?
    {
        let t = time * 0.7 + drift.phase;
        transform.set_position(
            drift.base + Vector3f::new(t.cos() * 0.4, (t * 1.3).sin() * 0.3, (t * 0.9).sin() * 0.4),
        );
    }
    if toggle_lanterns {
        for (_, light, _) in engine.iterate_two_components_mut::<LightComponent, DriftingLight>()? {
            light.intensity = if light.intensity > 0.0 {
                0.0
            } else {
                LANTERN_INTENSITY
            };
        }
    }
    Ok(())
}

// Uploads the loaded payload once it arrives; a failed load is logged and the empty scene keeps running.
fn sponza_drain_system(engine: &mut Engine) -> Result<()> {
    let loaded = PENDING
        .lock()
        .map_err(|_| PillError::from("Sponza load lock poisoned"))?
        .take();
    match loaded {
        Some(Ok(cpu)) => {
            let scene = engine.get_active_scene_handle()?;
            spawn_assets(engine, scene, cpu)
        }
        Some(Err(error)) => {
            log::error!("Sponza load failed: {error}");
            Ok(())
        }
        None => Ok(()),
    }
}

// Uploads the CPU payload to the GPU and builds one entity per primitive.
fn spawn_assets(engine: &mut Engine, scene: SceneHandle, cpu: SponzaCpu) -> Result<()> {
    // --- IBL (background + diffuse/specular; the BRDF LUT is engine-owned) ---
    let equirect = &cpu.equirect;
    let background = engine.create_gpu_texture_f32(
        "sponza_equirect",
        &equirect.pixels,
        equirect.width,
        equirect.height,
    )?;
    let (diffuse, specular_mips) = &cpu.ibl;
    let (diffuse_width, diffuse_height) = ibl::DIFFUSE_SIZE;
    let diffuse_handle = engine.create_gpu_texture_f32(
        "sponza_diffuse_ibl",
        diffuse,
        diffuse_width,
        diffuse_height,
    )?;
    let (specular_width, specular_height) = ibl::SPECULAR_SIZE;
    let specular_handle = engine.create_gpu_mipped_texture_f32(
        "sponza_specular_ibl",
        specular_mips,
        specular_width,
        specular_height,
    )?;
    {
        let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
        render_state.background = Some(background);
        render_state.ibl_diffuse = diffuse_handle;
        render_state.ibl_specular = specular_handle;
    }

    let GltfSceneData {
        textures,
        materials,
        meshes,
    } = cpu.scene;

    // --- Textures (moved, so each is freed once uploaded) ---
    let mut texture_handles: Vec<TextureHandle> = Vec::with_capacity(textures.len());
    for texture in textures {
        texture_handles.push(engine.add_resource(Texture::from_bytes_owned(
            &texture.name,
            texture.kind,
            texture.rtex,
        ))?);
    }

    // --- Materials ---
    let mut material_handles: Vec<MaterialHandle> = Vec::with_capacity(materials.len());
    for (index, material) in materials.iter().enumerate() {
        // Wet floor: darker, near-mirror stone that shows the screen-space reflections
        let (roughness, darkening) = if index == FLOOR_MATERIAL {
            (
                material.roughness * WET_FLOOR_ROUGHNESS,
                WET_FLOOR_DARKENING,
            )
        } else {
            (material.roughness, 1.0)
        };
        let [red, green, blue] = material.emissive_color;
        let mut pbr = Material::new(&material.name)
            .albedo(Color::new(
                material.base_color[0] * darkening,
                material.base_color[1] * darkening,
                material.base_color[2] * darkening,
            ))
            .metallic(material.metallic)
            .roughness(roughness)
            .emissive(Color::new(red, green, blue));
        if let Some(i) = material.albedo {
            pbr = pbr.albedo_texture(texture_handles[i]);
        }
        if let Some(i) = material.normal {
            pbr = pbr.normal_texture(texture_handles[i]);
        }
        if let Some(i) = material.metallic_roughness {
            pbr = pbr.metallic_roughness_texture(texture_handles[i]);
        }
        if let Some(i) = material.emissive {
            pbr = pbr.emissive_texture(texture_handles[i]);
        }
        material_handles.push(engine.add_resource(pbr)?);
    }

    // --- Meshes + entities (world transform already baked into the vertices) ---
    for mesh in &meshes {
        let mesh_handle =
            engine.add_resource(Mesh::from_cooked_mesh_bytes(&mesh.name, &mesh.rmsh)?)?;
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&mesh_handle)
                    .material(&material_handles[mesh.material])
                    .build(),
            )
            .build();
    }

    log::info!("Sponza: uploaded {} meshes to GPU", meshes.len());
    Ok(())
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("sponza")?;
        engine.set_active_scene(scene)?;
        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            render_state.background_color = [0.0, 0.0, 0.0];
            // Night, no fill: the shadowed point/spot lights are the only light
            for light in render_state.lights.iter_mut() {
                light.intensity = 0.0;
            }
            render_state.ibl_intensity = 0.0;
            // 3 point lights (6 cube faces each) + 2 spot lights (1 map each)
            render_state.punctual_lights.max_shadow_maps = 20;
            // Diffuse GI: light bouncing off the court floor onto columns and arches
            render_state.gi.enabled = true;
            // Post chain for the screen-space reflections in the wet floor; motion blur off
            render_state.pass_chain = PassChain::PostFx;
            render_state.motion_blur = 0.0;
            render_state.ssr.enabled = true;
        }

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FlyCamera>(scene)?;
        engine.register_component::<LightComponent>(scene)?;
        engine.register_component::<DriftingLight>(scene)?;
        engine.register_component::<SweepingLight>(scene)?;

        // Build the camera up front so the window renders (clear color + sky once IBL lands)
        // while the ~50 MB model downloads.
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(7.5, 0.7, COURT_Z))
                    .build(),
            )
            .with_component(CameraComponent::builder().enabled(true).fov(60.0).build())
            // Low, looking down the court toward -X between the column rows, at a grazing angle to
            // the wet floor that shows its reflections
            .with_component(FlyCamera {
                yaw: 90.0,
                pitch: -7.0,
            })
            .build();

        engine.add_system("fly_camera", fly_camera_system)?;
        spawn_lights(engine, scene);
        engine.add_system("lights", lights_system)?;

        // Load in the background; frames render meanwhile and the drain system uploads the result
        let store = |result| {
            if let Ok(mut pending) = PENDING.lock() {
                *pending = Some(result);
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move { store(loader::load().await) });
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || store(pollster::block_on(loader::load())));
        engine.add_system("sponza_drain", sponza_drain_system)?;

        Ok(())
    }
}
