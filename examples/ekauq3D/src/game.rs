use pill_engine::{
    define_component, define_global_component,
    game::{
        egui,
        quake_map::{light_style_brightness, QuakeCollision, QuakeLevel},
        quake_player::{PlayerInput, EYE_HEIGHT},
        *,
    },
};

use crate::{
    deathmatch::{self, BotModel, Deathmatch},
    loader::{self, LoadedLevel},
    player::{self, Mode, PlayerRig},
    sounds::{self, Sfx},
    tour,
    weapon::{self, Flash, MuzzleFlash, Particle, ViewModel, Weapon},
};

// Fly camera: view angles in degrees; yaw 0 looks down -Z, positive pitch looks up
define_component!(FlyCamera {
    yaw: f32,
    pitch: f32,
});

// Quake light style flicker: `base` intensity scaled by the style's brightness, 0 while off
define_component!(LightStyle {
    style: u32,
    base: f32,
    on: bool,
});

/// One animated texture's material and its (albedo, emissive) frames.
pub struct TextureAnimation {
    material: MaterialHandle,
    frames: Vec<(TextureHandle, Option<TextureHandle>)>,
    shown: usize,
}

/// Camera position and FlyCamera yaw/pitch of a spawn point.
#[derive(Clone, Copy)]
pub struct CameraSpot {
    pub position: Vector3f,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
}

/// Native screenshot run (`EKAUQ_CAPTURE=<dir>`): each of the first spawns is captured once GI
/// has settled, then the game exits.
pub struct Capture {
    directory: String,
    shot: usize,
    frames: u32,
    // `EKAUQ_CAPTURE_PLAYER=1`: a scripted player run (run, shoot) instead of the spawn views
    player_script: bool,
    // `EKAUQ_CAPTURE_TOUR=1`: the camera flies through the level, every 2nd frame captured
    tour: bool,
    tour_path: Vec<CameraSpot>,
}

define_global_component!(LevelState {
    map: String,
    loaded: bool,
    // The map's fog density, restored when F turns fog back on
    fog_density: f32,
    spawns: Vec<CameraSpot>,
    spawn_index: usize,
    animations: Vec<TextureAnimation>,
    capture: Option<Capture>,
    // Solid brushes of the level (player movement, shots)
    collision: Option<QuakeCollision>,
    // The player and the bots, once the level is loaded
    deathmatch: Option<Deathmatch>,
});

pub struct Game {}

// The background load (spawn_local task on WASM, thread on native) can't hold &mut Engine; it
// leaves its result here for `level_drain_system` to upload on the next frame.
type Loaded = std::result::Result<LoadedLevel, String>;
static PENDING: std::sync::Mutex<Option<Loaded>> = std::sync::Mutex::new(None);

// Meters per second (Quake runs at 320 units/s = 10 m/s); Shift multiplies
const FLY_SPEED: f32 = 6.0;
const FLY_FAST_MULTIPLIER: f32 = 3.0;
// Degrees per pixel of mouse movement
const LOOK_SENSITIVITY: f32 = 0.2;
/// Quake `_sunlight` units → directional light intensity.
const SUN_SCALE: f32 = 0.02;
/// Scales the map's fog density.
const FOG_SCALE: f32 = 0.5;
/// Animated textures change frame 5 times per second (Quake).
const TEXTURE_FRAMES_PER_SECOND: f32 = 5.0;
/// Capture run: frames each view renders before its screenshot (GI and SSR settle), and the
/// number of spawn points captured.
const CAPTURE_SETTLE_FRAMES: u32 = 180;
const CAPTURE_VIEWS: usize = 4;

// FPS controls: hold the left mouse button to look, WASD to move, E/Q up/down, Shift faster.
fn fly_camera_system(engine: &mut Engine) -> Result<()> {
    if engine.get_global_component::<PlayerRig>()?.mode == Mode::Player {
        return Ok(());
    }
    if capturing(engine)? {
        return Ok(());
    }
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

/// A capture run ignores input, so keys or mouse reaching its window don't change the shots.
fn capturing(engine: &Engine) -> Result<bool> {
    Ok(engine
        .get_global_component::<LevelState>()?
        .capture
        .is_some())
}

/// Puts the camera at spawn point `index` (wrapping).
fn move_camera_to_spawn(engine: &mut Engine, index: usize) -> Result<()> {
    let state = engine.get_global_component_mut::<LevelState>()?;
    if state.spawns.is_empty() {
        return Ok(());
    }
    state.spawn_index = index % state.spawns.len();
    let spot = state.spawns[state.spawn_index];
    for (_, transform, fly) in
        engine.iterate_two_components_mut::<TransformComponent, FlyCamera>()?
    {
        transform.set_position(spot.position);
        fly.yaw = spot.yaw;
        fly.pitch = spot.pitch;
    }
    engine
        .get_global_component_mut::<RenderStateComponent>()?
        .reset_motion_history();
    Ok(())
}

/// Tab: next spawn point. L light-count heatmap, K point/spot shadows, G diffuse GI, H
/// indirect-light-only view, R screen-space reflections, T reflections-only view, F fog, X auto
/// exposure, N spectator/player. Player mode: 1 axe, 2 shotgun (see `weapon`).
fn controls_system(engine: &mut Engine) -> Result<()> {
    if capturing(engine)? {
        return Ok(());
    }
    let input = engine.get_global_component::<InputComponent>()?;
    let next_spawn = input.get_key_pressed(KeyboardKey::Tab);
    let toggle_heatmap = input.get_key_pressed(KeyboardKey::KeyL);
    let toggle_shadows = input.get_key_pressed(KeyboardKey::KeyK);
    let toggle_gi = input.get_key_pressed(KeyboardKey::KeyG);
    let toggle_indirect_only = input.get_key_pressed(KeyboardKey::KeyH);
    let toggle_ssr = input.get_key_pressed(KeyboardKey::KeyR);
    let toggle_reflections_only = input.get_key_pressed(KeyboardKey::KeyT);
    let toggle_fog = input.get_key_pressed(KeyboardKey::KeyF);
    let toggle_auto_exposure = input.get_key_pressed(KeyboardKey::KeyX);
    let toggle_player = input.get_key_pressed(KeyboardKey::KeyN);
    {
        let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
        if toggle_heatmap {
            render_state.punctual_lights.debug_heatmap =
                !render_state.punctual_lights.debug_heatmap;
        }
        if toggle_shadows {
            render_state.punctual_lights.shadows_enabled =
                !render_state.punctual_lights.shadows_enabled;
        }
        if toggle_gi {
            render_state.gi.enabled = !render_state.gi.enabled;
        }
        if toggle_indirect_only {
            render_state.gi.debug_indirect_only = !render_state.gi.debug_indirect_only;
        }
        if toggle_ssr {
            render_state.ssr.enabled = !render_state.ssr.enabled;
        }
        if toggle_auto_exposure {
            render_state.auto_exposure.enabled = !render_state.auto_exposure.enabled;
        }
        if toggle_reflections_only {
            render_state.ssr.debug_reflections_only = !render_state.ssr.debug_reflections_only;
        }
    }
    if toggle_fog {
        let fog_density = engine.get_global_component::<LevelState>()?.fog_density;
        let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
        render_state.fog_density = if render_state.fog_density > 0.0 {
            0.0
        } else {
            fog_density
        };
    }
    if toggle_player {
        player::toggle_mode(engine)?;
    }
    if next_spawn {
        let index = engine.get_global_component::<LevelState>()?.spawn_index + 1;
        move_camera_to_spawn(engine, index)?;
    }
    Ok(())
}

// Light styles: Quake's flicker/pulse patterns at 10 frames per second
fn light_style_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    for (_, light, style) in engine.iterate_two_components_mut::<LightComponent, LightStyle>()? {
        light.intensity = if style.on {
            style.base * light_style_brightness(style.style, time)
        } else {
            0.0
        };
    }
    Ok(())
}

// Animated textures (`+0name`..): swaps the material's albedo/emissive on frame changes
fn texture_animation_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    let frame = (time * TEXTURE_FRAMES_PER_SECOND) as usize;
    let mut changes = Vec::new();
    for animation in &mut engine.get_global_component_mut::<LevelState>()?.animations {
        let shown = frame % animation.frames.len();
        if shown != animation.shown {
            animation.shown = shown;
            changes.push((animation.material, animation.frames[shown]));
        }
    }
    for (material, (albedo, emissive)) in changes {
        let material = engine.get_resource_mut::<Material>(&material)?;
        material.set_albedo_texture(Some(albedo));
        if emissive.is_some() {
            material.set_emissive_texture(emissive);
        }
    }
    Ok(())
}

// Capture run: settle, screenshot, next spawn; exit after the last one
fn capture_system(engine: &mut Engine) -> Result<()> {
    let (path, next) = {
        let state = engine.get_global_component_mut::<LevelState>()?;
        if state.capture.is_none() || !state.loaded {
            return Ok(());
        }
        let spawn_count = state.spawns.len().clamp(1, CAPTURE_VIEWS);
        let map = state.map.clone();
        let Some(capture) = state.capture.as_mut() else {
            return Ok(());
        };
        if capture.player_script {
            return capture_player_script(engine);
        }
        if capture.tour {
            return capture_tour(engine);
        }
        capture.frames += 1;
        let path = (capture.frames == CAPTURE_SETTLE_FRAMES)
            .then(|| format!("{}/{map}_{}.png", capture.directory, capture.shot));
        // The capture is written while the next frame starts
        let next = if capture.frames == CAPTURE_SETTLE_FRAMES + 3 {
            capture.shot += 1;
            capture.frames = 0;
            if capture.shot >= spawn_count {
                log::info!("ekauq3D: captures done");
                std::process::exit(0);
            }
            Some(capture.shot)
        } else {
            None
        };
        (path, next)
    };
    if let Some(path) = path {
        log::info!("ekauq3D: capturing {path}");
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .capture = Some(path);
    }
    if let Some(index) = next {
        move_camera_to_spawn(engine, index)?;
    }
    Ok(())
}

/// Camera tour for frame-to-frame comparisons (shadow popping): a collision-free fly-through
/// from spawn 0 past the level's lights (`tour::fly_through`), one path step per frame (not per second) so runs
/// match; every `EKAUQ_TOUR_STRIDE`-th frame (default 2) is captured as `{map}_tour_{frame}.png`.
/// `EKAUQ_TOUR_STEP`: path step per frame in Quake units (default 4, ≈ 4 m/s at 60 fps).
fn capture_tour(engine: &mut Engine) -> Result<()> {
    const SETTLE_FRAMES: u32 = 120;
    const LIGHTS_VISITED: usize = 16;
    const MAX_FRAMES: usize = 6000;
    let env_number = |name: &str, default: f32| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .unwrap_or(default)
    };
    let step = env_number("EKAUQ_TOUR_STEP", 4.0).max(0.1);
    let stride = env_number("EKAUQ_TOUR_STRIDE", 2.0).max(1.0) as u32;
    let (spot, path) = {
        let lights: Vec<Vector3f> = engine
            .iterate_two_components::<TransformComponent, LightComponent>()?
            .map(|(_, transform, _)| transform.position)
            .collect();
        let state = engine.get_global_component_mut::<LevelState>()?;
        let map = state.map.clone();
        if state
            .capture
            .as_ref()
            .is_some_and(|capture| capture.tour_path.is_empty())
        {
            let start = state
                .spawns
                .first()
                .map_or(Vector3f::ZERO, |spawn| spawn.position);
            let waypoints = tour::light_waypoints(start, &lights, LIGHTS_VISITED);
            let mut path = state
                .collision
                .as_ref()
                .map(|collision| tour::fly_through(collision, start, &waypoints, step))
                .unwrap_or_default();
            path.truncate(MAX_FRAMES);
            log::info!("ekauq3D: tour of {} frames", path.len());
            if path.is_empty() {
                std::process::exit(0);
            }
            if let Some(capture) = state.capture.as_mut() {
                capture.tour_path = path;
            }
        }
        let Some(capture) = state.capture.as_mut() else {
            return Ok(());
        };
        capture.frames += 1;
        let frame = capture.frames;
        let index = frame.saturating_sub(SETTLE_FRAMES) as usize;
        if index >= capture.tour_path.len() + 3 {
            log::info!("ekauq3D: tour done");
            std::process::exit(0);
        }
        let spot = capture.tour_path[index.min(capture.tour_path.len() - 1)];
        let path = (frame > SETTLE_FRAMES && frame % stride == 0)
            .then(|| format!("{}/{map}_tour_{frame:05}.png", capture.directory));
        (spot, path)
    };
    for (_, transform, fly) in
        engine.iterate_two_components_mut::<TransformComponent, FlyCamera>()?
    {
        transform.set_position(spot.position);
        fly.yaw = spot.yaw;
        fly.pitch = spot.pitch;
    }
    if let Some(path) = path {
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .capture = Some(path);
    }
    Ok(())
}

/// Scripted player run for screenshots: player mode at the first spawn, run forward, one shot,
/// switch to the axe (key 1), two swings. Frames are counted from the level upload.
fn capture_player_script(engine: &mut Engine) -> Result<()> {
    const SHOTS: [(u32, &str); 13] = [
        (60, "run_a"),
        (67, "run_b"),
        (121, "shot_1"),
        (123, "shot_3"),
        (128, "shot_8"),
        (170, "after"),
        (184, "axe_switch"),
        (200, "axe_rest"),
        (213, "axe_a1"),
        (219, "axe_a2"),
        (225, "axe_a3"),
        (231, "axe_a4"),
        (281, "axe_b2"),
    ];
    const FIRE_FRAME: u32 = 120;
    const AXE_SELECT_FRAME: u32 = 180;
    const AXE_SWING_FRAMES: [u32; 2] = [210, 270];
    let (frame, path) = {
        let state = engine.get_global_component_mut::<LevelState>()?;
        let map = state.map.clone();
        let Some(capture) = state.capture.as_mut() else {
            return Ok(());
        };
        capture.frames += 1;
        let frame = capture.frames;
        let path = SHOTS
            .iter()
            .find(|(at, _)| *at == frame)
            .map(|(_, label)| format!("{}/{map}_player_{label}.png", capture.directory));
        (frame, path)
    };
    if frame == 1 {
        player::toggle_mode(engine)?;
        // Run toward the most open of 16 directions
        let origin = engine.get_global_component::<PlayerRig>()?.state.origin;
        let eye = [origin[0], origin[1], origin[2] + EYE_HEIGHT];
        let yaw = {
            let state = engine.get_global_component::<LevelState>()?;
            let Some(collision) = state.collision.as_ref() else {
                return Ok(());
            };
            (0..16)
                .map(|step| step as f32 * 22.5)
                .max_by(|a, b| {
                    let reach = |yaw: f32| {
                        let (sin, cos) = yaw.to_radians().sin_cos();
                        let end = [eye[0] + cos * 2048.0, eye[1] + sin * 2048.0, eye[2]];
                        collision.trace_ray(eye, end).fraction
                    };
                    reach(*a).total_cmp(&reach(*b))
                })
                .unwrap_or(0.0)
        };
        let rig = engine.get_global_component_mut::<PlayerRig>()?;
        rig.yaw = yaw;
        rig.scripted = Some(PlayerInput {
            forward: 1.0,
            yaw,
            ..PlayerInput::default()
        });
    }
    if frame == FIRE_FRAME || AXE_SWING_FRAMES.contains(&frame) {
        engine.get_global_component_mut::<Weapon>()?.scripted_fire = true;
    }
    if frame == AXE_SELECT_FRAME {
        engine.get_global_component_mut::<Weapon>()?.scripted_select =
            Some(weapon::WeaponKind::Axe);
    }
    if let Some(path) = path {
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .capture = Some(path);
    }
    if frame > SHOTS[SHOTS.len() - 1].0 + 3 {
        std::process::exit(0);
    }
    Ok(())
}

/// Footsteps, jump and landing sounds from the player's moves this frame.
fn player_sounds_system(engine: &mut Engine) -> Result<()> {
    let rig = engine.get_global_component::<PlayerRig>()?;
    if rig.mode != Mode::Player {
        return Ok(());
    }
    let events = rig.events;
    if events.jumped {
        sounds::play(engine, Sfx::Jump, None)?;
    } else if events.footstep {
        sounds::play(engine, Sfx::Footstep, None)?;
    }
    if events.landed_speed > player::HARD_LANDING_SPEED {
        sounds::play(engine, Sfx::Land, None)?;
    } else if events.landed_speed > 0.0 {
        sounds::play(engine, Sfx::Footstep, None)?;
    }
    if events.head_landed {
        sounds::play(engine, Sfx::Land, None)?;
    }
    Ok(())
}

/// Watermark, the deathmatch HUD, and the crosshair in player mode.
fn overlay(ctx: &egui::Context) {
    watermark(ctx);
    deathmatch::hud(ctx);
    if player::CROSSHAIR.load(std::sync::atomic::Ordering::Relaxed) {
        player::crosshair(ctx);
    }
}

/// White "powered by (pill) Pill Engine" in the bottom-right corner.
fn watermark(ctx: &egui::Context) {
    egui::Area::new(egui::Id::new("ekauq3d.watermark"))
        .anchor(egui::Align2::RIGHT_BOTTOM, [-14.0, -12.0])
        .order(egui::Order::Foreground)
        .interactable(false)
        .show(ctx, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(
                    egui::RichText::new("powered by")
                        .color(egui::Color32::WHITE)
                        .size(13.0),
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(22.0, 22.0), egui::Sense::hover());
                paint_pill_logo(ui.painter(), rect.center());
                ui.label(
                    egui::RichText::new("Pill Engine")
                        .color(egui::Color32::WHITE)
                        .size(16.0)
                        .strong(),
                );
            });
        });
}

/// The Pill Engine logo's capsule: tilted 45°, outlined, its lower-left half filled.
fn paint_pill_logo(painter: &egui::Painter, center: egui::Pos2) {
    const RADIUS: f32 = 5.0;
    const HALF_LENGTH: f32 = 4.5; // center to each cap's center
    const ARC_POINTS: usize = 12;
    let along = egui::vec2(1.0, -1.0).normalized(); // toward the top-right cap
    let across = egui::vec2(-along.y, along.x);
    // Half circle around `cap`, from `+across` through `outward` to `-across`
    let arc = |cap: egui::Pos2, outward: egui::Vec2| {
        (0..=ARC_POINTS).map(move |i| {
            let angle = std::f32::consts::PI * i as f32 / ARC_POINTS as f32;
            let side = if outward == along { -1.0 } else { 1.0 };
            cap + (across * angle.cos() * side + outward * angle.sin()) * RADIUS
        })
    };
    let top = center + along * HALF_LENGTH;
    let bottom = center - along * HALF_LENGTH;
    let outline: Vec<egui::Pos2> = arc(top, along).chain(arc(bottom, -along)).collect();
    let filled: Vec<egui::Pos2> = arc(bottom, -along)
        .chain([center - across * RADIUS, center + across * RADIUS])
        .collect();
    painter.add(egui::Shape::convex_polygon(
        filled,
        egui::Color32::WHITE,
        egui::Stroke::NONE,
    ));
    painter.add(egui::Shape::closed_line(
        outline,
        egui::Stroke::new(1.6, egui::Color32::WHITE),
    ));
}

// Uploads the loaded level once it arrives; a failed load is logged and the empty scene keeps running.
fn level_drain_system(engine: &mut Engine) -> Result<()> {
    let loaded = PENDING
        .lock()
        .map_err(|_| PillError::from("Level load lock poisoned"))?
        .take();
    match loaded {
        Some(Ok(LoadedLevel {
            level,
            collision,
            axe,
            shotgun,
            arena,
            nav,
            knowledge,
            player,
        })) => {
            let scene = engine.get_active_scene_handle()?;
            engine.get_global_component_mut::<LevelState>()?.collision = Some(collision);
            weapon::spawn_weapons(engine, scene, axe, shotgun)?;
            let map = engine.get_global_component::<LevelState>()?.map.clone();
            let deathmatch = deathmatch::spawn_deathmatch(
                engine,
                scene,
                &map,
                arena,
                nav,
                knowledge,
                player,
                loader::requested_bots(),
                loader::requested_network(),
            )?;
            engine.get_global_component_mut::<LevelState>()?.deathmatch = Some(deathmatch);
            spawn_level(engine, scene, level)
        }
        Some(Err(error)) => {
            log::error!("ekauq3D: load failed: {error}");
            Ok(())
        }
        None => Ok(()),
    }
}

// Uploads the level to the GPU: environment, textures, materials, one entity per mesh and light
fn spawn_level(engine: &mut Engine, scene: SceneHandle, level: QuakeLevel) -> Result<()> {
    // --- Environment: sky panorama (background), sky-dome light (IBL), sun, fog ---
    let sky = match &level.sky {
        Some(sky) => Some(engine.create_gpu_texture_f32(
            "ekauq3d_sky",
            &sky.pixels,
            sky.width,
            sky.height,
        )?),
        None => None,
    };
    let dome = level.lighting_environment();
    let dome = engine.create_gpu_texture_f32(
        "ekauq3d_sky_light",
        &dome.pixels,
        dome.width,
        dome.height,
    )?;
    {
        let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
        render_state.background = sky;
        render_state.environment = Some(dome);
        render_state.ibl_intensity = 0.5;
        if let Some(sun) = &level.sun {
            render_state.lights[0] = DirectionalLight {
                direction: sun.direction,
                color: sun.color,
                intensity: sun.light * SUN_SCALE,
            };
            render_state.shadow.enabled = true;
        }
        if let Some(fog) = &level.fog {
            render_state.fog_color = fog.color;
            render_state.fog_density = fog.density * FOG_SCALE;
        }
    }

    let GltfSceneData {
        textures,
        materials,
        meshes,
    } = level.scene;

    // Moved, so each is freed once uploaded
    let mut texture_handles: Vec<TextureHandle> = Vec::with_capacity(textures.len());
    for texture in textures {
        texture_handles.push(engine.add_resource(Texture::from_bytes_owned(
            &texture.name,
            texture.kind,
            texture.rtex,
        ))?);
    }

    let mut material_handles: Vec<MaterialHandle> = Vec::with_capacity(materials.len());
    for material in &materials {
        let [red, green, blue, alpha] = material.base_color;
        let [emissive_red, emissive_green, emissive_blue] = material.emissive_color;
        let mut pbr = Material::new(&material.name)
            .albedo(Color::new(red, green, blue))
            .metallic(material.metallic)
            .roughness(material.roughness)
            .emissive(Color::new(emissive_red, emissive_green, emissive_blue))
            .double_sided(material.double_sided);
        match material.alpha_mode {
            GltfAlphaMode::Opaque => {}
            GltfAlphaMode::Mask(cutoff) => pbr = pbr.alpha_cutoff(cutoff),
            GltfAlphaMode::Blend => pbr = pbr.opacity(alpha),
        }
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

    // --- Meshes + entities (positions already in world space) ---
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

    for light in &level.lights {
        let position = Vector3f::from_array(light.position);
        let mut transform = TransformComponent::builder().position(position).build();
        let mut builder = LightComponent::builder()
            .color(light.color)
            .intensity(light.intensity)
            .range(light.range)
            .cast_shadows(true);
        if let Some(spot) = &light.spot {
            builder = builder.light_type(LightType::Spot {
                inner_angle: spot.inner_angle,
                outer_angle: spot.outer_angle,
            });
            transform.look_at(position + Vector3f::from_array(spot.direction));
        }
        engine
            .build_entity(scene)
            .with_component(transform)
            .with_component(builder.build())
            .with_component(LightStyle {
                style: light.style,
                base: light.intensity,
                on: !light.start_off,
            })
            .build();
    }

    // --- Animations, spawns, camera ---
    let animations = level
        .animations
        .iter()
        .map(|animation| TextureAnimation {
            material: material_handles[animation.material],
            frames: animation
                .frames
                .iter()
                .map(|frame| {
                    (
                        texture_handles[frame.albedo],
                        frame.emissive.map(|i| texture_handles[i]),
                    )
                })
                .collect(),
            shown: 0,
        })
        .collect();
    let spawns = level
        .spawns
        .iter()
        .map(|spawn| CameraSpot {
            position: Vector3f::from_array(spawn.position),
            yaw: player::fly_yaw_from_quake(spawn.yaw),
            pitch: spawn.pitch,
        })
        .collect();
    {
        let state = engine.get_global_component_mut::<LevelState>()?;
        state.animations = animations;
        state.spawns = spawns;
        state.fog_density = level
            .fog
            .as_ref()
            .map_or(0.0, |fog| fog.density * FOG_SCALE);
        state.loaded = true;
    }
    move_camera_to_spawn(engine, 0)?;

    log::info!(
        "ekauq3D: uploaded {} meshes, {} materials, {} lights",
        meshes.len(),
        materials.len(),
        level.lights.len()
    );
    Ok(())
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let map = loader::requested_map();
        let scene = engine.create_scene("ekauq3d")?;
        engine.set_active_scene(scene)?;
        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            render_state.pass_chain = PassChain::PostFx;
            render_state.background_color = [0.0, 0.0, 0.0];
            // The sun (lights[0]) is set from the map; no fill lights
            for light in render_state.lights.iter_mut() {
                light.intensity = 0.0;
            }
            render_state.shadow.map_size = 4096;
            render_state.shadow.distance = 60.0;
            render_state.motion_blur = 0.0;
            // Filmic: a fully lit surface (1.0) stays near its texture color; Reinhard would halve it
            render_state.tonemap = Tonemap::lottes();
            // Eye adaptation between dark interiors and sunlit courtyards (X toggles;
            // `EKAUQ_AUTO_EXPOSURE=0` starts with it off, natively)
            #[cfg(not(target_arch = "wasm32"))]
            let auto_exposure =
                std::env::var("EKAUQ_AUTO_EXPOSURE").map_or(true, |value| value != "0");
            #[cfg(target_arch = "wasm32")]
            let auto_exposure = true;
            render_state.auto_exposure = AutoExposure {
                enabled: auto_exposure,
                min_ev: -6.0,
                max_ev: 4.0,
                ..AutoExposure::default()
            };
            render_state.ssr.enabled = true;
            // Diffuse GI around the camera: a fine grid and a coarse one past it (as Bistro)
            render_state.gi.enabled = true;
            render_state.gi.follow_camera = true;
            render_state.gi.cascades = 2;
            render_state.gi.cascade_spacings[1] = 6.0;
            render_state.gi.rays_per_probe = 4;
            render_state.gi.blend_speed = 1.0 / 512.0;
        }

        engine
            .get_global_component_mut::<EguiComponent>()?
            .set_overlay(overlay);

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FlyCamera>(scene)?;
        engine.register_component::<LightComponent>(scene)?;
        engine.register_component::<LightStyle>(scene)?;
        engine.register_component::<ViewModel>(scene)?;
        engine.register_component::<Particle>(scene)?;
        engine.register_component::<Flash>(scene)?;
        engine.register_component::<MuzzleFlash>(scene)?;
        engine.register_component::<BotModel>(scene)?;
        engine.add_global_component(PlayerRig::new())?;
        engine.add_global_component(Weapon::new())?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            engine.register_component::<AudioSourceComponent>(scene)?;
            engine.register_component::<AudioListenerComponent>(scene)?;
            engine.register_component::<sounds::SfxSlot>(scene)?;
            sounds::spawn_sounds(engine, scene)?;
        }

        #[cfg(not(target_arch = "wasm32"))]
        let capture = std::env::var("EKAUQ_CAPTURE").ok().map(|directory| {
            let _ = std::fs::create_dir_all(&directory);
            Capture {
                directory,
                shot: 0,
                frames: 0,
                player_script: std::env::var("EKAUQ_CAPTURE_PLAYER")
                    .is_ok_and(|value| value == "1"),
                tour: std::env::var("EKAUQ_CAPTURE_TOUR").is_ok_and(|value| value == "1"),
                tour_path: Vec::new(),
            }
        });
        #[cfg(target_arch = "wasm32")]
        let capture = None;
        engine.add_global_component(LevelState {
            map: map.clone(),
            loaded: false,
            fog_density: 0.0,
            spawns: Vec::new(),
            spawn_index: 0,
            animations: Vec::new(),
            capture,
            collision: None,
            deathmatch: None,
        })?;

        // Build the camera up front so the window renders while the level downloads
        let camera = engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            // Near plane 5 cm: the scaled-down view models sit close to the eye (see `weapon`)
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(75.0)
                    .range(0.05..100.0)
                    .build(),
            )
            .with_component(FlyCamera {
                yaw: 0.0,
                pitch: 0.0,
            })
            .build();
        #[cfg(not(target_arch = "wasm32"))]
        engine.add_component_to_entity(
            scene,
            camera,
            AudioListenerComponent::builder().enabled(true).build(),
        )?;
        #[cfg(target_arch = "wasm32")]
        let _ = camera;

        engine.add_system("controls", controls_system)?;
        engine.add_system("player_input", player::player_input_system)?;
        engine.add_system("deathmatch", deathmatch::deathmatch_system)?;
        engine.add_system("player_view", player::player_view_system)?;
        engine.add_system("fly_camera", fly_camera_system)?;
        engine.add_system("weapon", weapon::weapon_system)?;
        engine.add_system("bot_render", deathmatch::bot_render_system)?;
        engine.add_system("effects", weapon::effects_system)?;
        engine.add_system("player_sounds", player_sounds_system)?;
        engine.add_system("light_styles", light_style_system)?;
        engine.add_system("texture_animation", texture_animation_system)?;
        engine.add_system("capture", capture_system)?;

        // Load in the background; frames render meanwhile and the drain system uploads the result
        let store = |result| {
            if let Ok(mut pending) = PENDING.lock() {
                *pending = Some(result);
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(async move { store(loader::load(&map).await) });
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || store(pollster::block_on(loader::load(&map))));
        engine.add_system("level_drain", level_drain_system)?;

        Ok(())
    }
}
