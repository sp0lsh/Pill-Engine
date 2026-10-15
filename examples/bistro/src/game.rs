use pill_engine::{define_component, game::*};
use std::sync::Mutex;

use crate::loader::{self, BistroCpu};
use crate::ui;

// Fly camera: view angles in degrees; yaw 0 looks down -Z, positive pitch looks up
define_component!(FlyCamera {
    yaw: f32,
    pitch: f32,
});

pub struct Game {}

// Meters per second; Shift multiplies
const FLY_SPEED: f32 = 5.0;
const FLY_FAST_MULTIPLIER: f32 = 4.0;
// Degrees per pixel of mouse movement
const LOOK_SENSITIVITY: f32 = 0.2;

/// A camera placement: position, yaw, pitch (degrees, as `FlyCamera`).
type Viewpoint = (Vector3f, f32, f32);

/// Keys 1–4 (meters).
const VIEWPOINTS: [Viewpoint; 4] = [
    // Corner café down the street
    (Vector3f::new(-25.0, 1.7, 0.0), 270.0, 2.0),
    // The Vespa in front of the café
    (Vector3f::new(-12.0, 1.7, 3.0), 300.0, 0.0),
    // Terrace table under the red awnings
    (Vector3f::new(-4.0, 1.7, -7.0), 30.0, 0.0),
    // Above the street
    (Vector3f::new(0.0, 20.0, 0.0), 90.0, -40.0),
];

// The background load can't hold &mut Engine; it leaves its result here for `drain_system`
static LOADED: Mutex<Option<std::result::Result<BistroCpu, String>>> = Mutex::new(None);

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
    let viewpoint = [
        KeyboardKey::Digit1,
        KeyboardKey::Digit2,
        KeyboardKey::Digit3,
        KeyboardKey::Digit4,
    ]
    .iter()
    .position(|&key| input.get_key_pressed(key))
    .map(|index| VIEWPOINTS[index]);

    for (_, transform, fly) in
        engine.iterate_two_components_mut::<TransformComponent, FlyCamera>()?
    {
        if let Some((position, yaw, pitch)) = viewpoint {
            transform.set_position(position);
            (fly.yaw, fly.pitch) = (yaw, pitch);
        }
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

/// Uploads the scene once the background load finished.
fn drain_system(engine: &mut Engine) -> Result<()> {
    let Some(result) = LOADED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
    else {
        return Ok(());
    };
    let cpu = result.map_err(|error| -> PillError {
        ui::set_status(format!("Loading failed: {error}"));
        error.into()
    })?;
    let scene = engine.get_active_scene_handle()?;
    let summary = spawn_assets(engine, scene, cpu)?;
    ui::set_status(summary);
    Ok(())
}

/// Uploads the CPU payload: IBL, textures, materials, one entity per primitive. Returns a
/// one-line summary for the panel.
fn spawn_assets(engine: &mut Engine, scene: SceneHandle, cpu: BistroCpu) -> Result<String> {
    let name = loader::NAME;
    // --- IBL (background + diffuse/specular; the BRDF LUT is engine-owned) ---
    let equirect = &cpu.equirect;
    let background = engine.create_gpu_texture_f32(
        &format!("{name}_equirect"),
        &equirect.pixels,
        equirect.width,
        equirect.height,
    )?;
    let (diffuse, specular_mips) = &cpu.ibl;
    let (diffuse_width, diffuse_height) = ibl::DIFFUSE_SIZE;
    let diffuse_handle = engine.create_gpu_texture_f32(
        &format!("{name}_diffuse_ibl"),
        diffuse,
        diffuse_width,
        diffuse_height,
    )?;
    let (specular_width, specular_height) = ibl::SPECULAR_SIZE;
    let specular_handle = engine.create_gpu_mipped_texture_f32(
        &format!("{name}_specular_ibl"),
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
    let texture_count = textures.len();

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
    for material in &materials {
        material_handles.push(engine.add_resource(bistro_material(material, &texture_handles))?);
    }

    // --- Meshes + entities (world transform already baked into the vertices) ---
    let mut triangles = 0;
    let (mut min, mut max) = (Vector3f::splat(f32::MAX), Vector3f::splat(f32::MIN));
    for mesh in &meshes {
        let (mesh_min, mesh_max, mesh_triangles) = rmsh_bounds(&mesh.rmsh);
        (min, max) = (min.min(mesh_min), max.max(mesh_max));
        triangles += mesh_triangles;
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

    Ok(format!(
        "{} meshes, {:.2}M triangles, {} textures; bounds {min:.1} .. {max:.1} m",
        meshes.len(),
        triangles as f32 / 1e6,
        texture_count
    ))
}

/// Pill material of a Bistro glTF material. The conversion stores occlusion / roughness /
/// metalness in `KHR_materials_specular`'s texture (R / G / B) and DirectX normal maps.
fn bistro_material(material: &GltfMaterialData, textures: &[TextureHandle]) -> Material {
    let [red, green, blue, alpha] = material.base_color;
    let [emissive_red, emissive_green, emissive_blue] = material.emissive_color;
    // The roughness factor (0.55 everywhere) is an exporter default; the texture is authoritative.
    // Foliage (alpha-tested leaf cards) is fully rough instead: its vertex normals are edited to
    // light the cards as one volume, so seen from the side they are edge-on and would reflect the
    // sky at grazing Fresnel; at roughness 1 the Fresnel stays at F0
    let foliage = matches!(material.alpha_mode, GltfAlphaMode::Mask(_));
    let orm = material
        .specular
        .or(material.metallic_roughness)
        .filter(|_| !foliage);
    let (metallic, roughness) = if foliage {
        (0.0, 1.0)
    } else if orm.is_some() {
        (1.0, 1.0)
    } else {
        (material.metallic, material.roughness)
    };
    let opacity = match material.alpha_mode {
        GltfAlphaMode::Blend => alpha,
        _ => 1.0,
    }
    .min(1.0 - 0.7 * material.transmission);
    let mut pbr = Material::new(&material.name)
        .albedo(Color::new(red, green, blue))
        .metallic(metallic)
        .roughness(roughness)
        .emissive(Color::new(emissive_red, emissive_green, emissive_blue))
        .opacity(opacity)
        .double_sided(material.double_sided)
        .directx_normals(true);
    if let GltfAlphaMode::Mask(cutoff) = material.alpha_mode {
        pbr = pbr.alpha_cutoff(cutoff);
    }
    if let Some(index) = material.albedo {
        pbr = pbr.albedo_texture(textures[index]);
    }
    if let Some(index) = material.normal {
        pbr = pbr.normal_texture(textures[index]);
    }
    if let Some(index) = orm {
        pbr = pbr.metallic_roughness_texture(textures[index]);
    }
    if let Some(index) = material.emissive {
        pbr = pbr.emissive_texture(textures[index]);
    }
    pbr
}

/// Bounds and triangle count of a cooked mesh (RMSH: b"RMSH" | version | vertex count | index
/// count | 14-float vertices (position first) | u32 indices).
fn rmsh_bounds(rmsh: &[u8]) -> (Vector3f, Vector3f, usize) {
    let read_u32 = |offset: usize| u32::from_le_bytes(rmsh[offset..offset + 4].try_into().unwrap());
    let (vertex_count, index_count) = (read_u32(8) as usize, read_u32(12) as usize);
    let (mut min, mut max) = (Vector3f::splat(f32::MAX), Vector3f::splat(f32::MIN));
    for vertex in rmsh[16..16 + vertex_count * 56].chunks_exact(56) {
        let coordinate =
            |offset: usize| f32::from_le_bytes(vertex[offset..offset + 4].try_into().unwrap());
        let position = Vector3f::new(coordinate(0), coordinate(4), coordinate(8));
        (min, max) = (min.min(position), max.max(position));
    }
    (min, max, index_count / 3)
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            render_state.pass_chain = PassChain::PostFx;
            render_state.background_color = [0.55, 0.65, 0.8];
            // The sun (lights[0]) and the sky follow the panel's hour; no fill lights
            for light in render_state.lights[1..].iter_mut() {
                light.intensity = 0.0;
            }
            render_state.shadow.map_size = 4096;
            render_state.shadow.distance = 60.0;
            render_state.tonemap = Tonemap::lottes();
            // GI probes around the camera, every probe each frame with few rays: 64 rays from all
            // of them against the Exterior's 2.8M triangles cost ~150 ms
            render_state.gi.follow_camera = true;
            // A second grid at 6 m (±84–96 m: the whole Exterior) lights what lies past the fine
            // grid around the camera (spacing from the panel), which blends into it toward its
            // edge
            render_state.gi.cascades = 2;
            render_state.gi.cascade_spacings[1] = 6.0;
            render_state.gi.rays_per_probe = 4;
            // Averages ~1000 frames (~4000 rays per texel): 1/128 left visible jitter at 4 rays
            // (temporal std 0.84 vs 0.29 of 255); lighting changes blend fast
            render_state.gi.blend_speed = 1.0 / 512.0;
            render_state.lens.vignette = 0.15;
        }
        ui::register(engine)?;
        let scene = engine.create_scene(loader::NAME)?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FlyCamera>(scene)?;
        let (position, yaw, pitch) = VIEWPOINTS[0];
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().position(position).build())
            .with_component(CameraComponent::builder().enabled(true).fov(60.0).build())
            .with_component(FlyCamera { yaw, pitch })
            .build();
        ui::set_status("Loading (first run: download + BC7 cook)…".to_string());
        std::thread::spawn(|| {
            *LOADED
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(loader::load());
        });

        engine.add_system("fly_camera", fly_camera_system)?;
        engine.add_system("bistro_shortcuts", ui::shortcuts_system)?;
        engine.add_system("bistro_settings", ui::apply_settings_system)?;
        engine.add_system("bistro_drain", drain_system)?;
        Ok(())
    }
}
