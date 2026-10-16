//! Bevy's `bevymark_3d` stress test (examples/stress_tests/bevymark_3d.rs) on Pill: hold the left
//! mouse button to spawn 10 000 cubes per second at the top of a 50 m box; they fall, bounce off
//! the floor and walls. Same constants, RNG seeds and streams, materials, camera, light and stats
//! overlay as Bevy's default run (no `--benchmark`, waves or per-instance materials), and Bevy's
//! view defaults: exposure EV100 9.7, ambient 80 cd/m², Tony McMapface tonemapping, deband dither
//! (`TonyMcMapfacePass`). The cubes show the Pill logo where Bevy's show its `branding/icon.png`.

use crate::tonemap_pass::TonyMcMapfacePass;
use chacha20::ChaCha8Rng;
use pill_engine::{define_component, game::*};
use rand::{seq::IndexedRandom, RngExt, SeedableRng};
use std::sync::Mutex;

const CUBES_PER_SECOND: u32 = 10000;
const GRAVITY: f32 = -9.8;
const MAX_VELOCITY: f32 = 10.;
const CUBE_SCALE: f32 = 1.0;
const HALF_CUBE_SIZE: f32 = CUBE_SCALE * 0.5;
const VOLUME_WIDTH: usize = 50;
const VOLUME_SIZE: Vector3f = Vector3f::splat(VOLUME_WIDTH as f32);
/// Bevy's `init_materials` capacity without `--benchmark`: max(texture count 1, waves 0, 256).
const MATERIAL_COUNT: usize = 256;

/// Bevy's `Exposure::default()` (EV100 9.7, Blender's): exp2(−ev100) / 1.2. Pill has no camera
/// exposure, so the light intensities are pre-multiplied by it.
const EXPOSURE: f32 = 0.001_001_908;
const SUN_ILLUMINANCE_LUX: f32 = 10000.0;
/// Bevy's `GlobalAmbientLight::default()` brightness, cd/m² (white).
const AMBIENT_BRIGHTNESS: f32 = 80.0;
/// Radiance of Pill's neutral IBL environment (`DEFAULT_IBL_FALLBACK_PIXEL` 77/255, linear).
const DEFAULT_ENVIRONMENT_RADIANCE: f32 = 77.0 / 255.0;
/// Bevy's `ClearColor::default()`, srgb(43, 44, 47), is written without tonemapping; this is the
/// HDR color Tony McMapface maps to it (solved offline against the LUT).
const BACKGROUND_HDR: [f32; 3] = [0.022_010_85, 0.022_966_22, 0.025_933_02];

/// Bevy's FrameTimeDiagnosticsPlugin: 120-frame history, EMA smoothing factor 2 / 21 s.
const FPS_HISTORY_LENGTH: usize = 120;
const FPS_EMA_SMOOTHING: f64 = 2.0 / 21.0;
const FONT_SIZE_PX: f32 = 40.0;
const FONT_FAMILY: &str = "fira_mono";
/// Bevy's default font (bevy_text/src/FiraMono-subset.ttf, OFL).
const FONT_BYTES: &[u8] = include_bytes!("../res/fonts/FiraMono-subset.ttf");

define_component!(Cube { velocity: Vector3f });

struct State {
    materials: Vec<MaterialHandle>,
    cube_mesh: MeshHandle,
    transform_rng: ChaCha8Rng,
    velocity_rng: ChaCha8Rng,
    wave: usize,
    count: usize,
    fps: Option<Fps>,
}

/// Bevy's FPS diagnostic: raw, simple moving average, exponential moving average.
struct Fps {
    raw: f64,
    history: std::collections::VecDeque<f64>,
    sum: f64,
    ema: f64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> T {
    let mut guard = STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(guard
        .as_mut()
        .expect("bevymark state initialized in start()"))
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("bevymark_3d")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<Cube>(scene)?;

        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            render_state.background_color = BACKGROUND_HDR;
            // TonyMcMapfacePass tonemaps; the engine pass only encodes sRGB
            render_state.tonemap = Tonemap::None;
            render_state.shadow.enabled = false;
            render_state.ibl_intensity =
                AMBIENT_BRIGHTNESS * EXPOSURE / DEFAULT_ENVIRONMENT_RADIANCE;
            // DirectionalLight at (1, 2, 3) looking at the origin, shadows off
            let direction = -Vector3f::new(1.0, 2.0, 3.0).normalize();
            let off = DirectionalLight {
                direction: [0.0, -1.0, 0.0],
                color: [0.0; 3],
                intensity: 0.0,
            };
            render_state.lights = [
                DirectionalLight {
                    direction: direction.to_array(),
                    color: [1.0; 3],
                    intensity: SUN_ILLUMINANCE_LUX * EXPOSURE,
                },
                off,
                off,
            ];
        }
        engine
            .get_global_component_mut::<EguiComponent>()?
            .show_engine_windows = false;

        let logo = engine.add_resource(Texture::from_bytes(
            "pill_logo",
            TextureType::Color,
            include_bytes!("../res/textures/pill_logo.cooked_tex"),
        ))?;
        let cube_mesh = engine.add_resource(Mesh::from_cooked_mesh_bytes(
            "cube",
            include_bytes!("../res/models/cube.cooked_mesh"),
        )?)?;
        let materials = init_materials(engine, logo)?;

        let mut camera_transform = TransformComponent::builder()
            .position(VOLUME_SIZE * 1.3)
            .build();
        camera_transform.look_at(Vector3f::ZERO);
        engine
            .build_entity(scene)
            .with_component(camera_transform)
            // Bevy's PerspectiveProjection::default(): 45° vertical, near 0.1, far 1000
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(45.0)
                    .range(0.1..1000.0)
                    .build(),
            )
            .build();

        *STATE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(State {
            materials,
            cube_mesh,
            transform_rng: ChaCha8Rng::seed_from_u64(26),
            velocity_rng: ChaCha8Rng::seed_from_u64(97),
            wave: 0,
            count: 0,
            fps: None,
        });

        engine
            .get_global_component_mut::<EguiComponent>()?
            .register_ui("bevymark.stats", stats_ui);

        // Bevy applies the spawn commands after Update, so new cubes first move the next frame
        engine.add_render_pass(Box::new(TonyMcMapfacePass::new()))?;
        engine.add_system("movement", movement_system)?;
        engine.add_system("collision", collision_system)?;
        engine.add_system("mouse_handler", mouse_handler)?;
        engine.add_system("counter", counter_system)?;
        Ok(())
    }
}

/// Bevy's `init_materials`: the logo on white, then 255 random linear colors (seed 42) on the
/// logo; opaque, roughness 0.5, metallic 0, reflectance 0.5 (F0 0.04, Pill's).
fn init_materials(engine: &mut Engine, logo: TextureHandle) -> Result<Vec<MaterialHandle>> {
    let textures = [logo];
    let mut materials = Vec::with_capacity(MATERIAL_COUNT);
    materials.push(engine.add_resource(standard_material("material_0", Color::ONE, logo))?);
    let mut color_rng = ChaCha8Rng::seed_from_u64(42);
    let mut texture_rng = ChaCha8Rng::seed_from_u64(42);
    while materials.len() < MATERIAL_COUNT {
        let color = Color::new(color_rng.random(), color_rng.random(), color_rng.random());
        let texture = *textures.choose(&mut texture_rng).expect("one texture");
        let name = format!("material_{}", materials.len());
        materials.push(engine.add_resource(standard_material(&name, color, texture))?);
    }
    Ok(materials)
}

fn standard_material(name: &str, base_color: Color, texture: TextureHandle) -> Material {
    Material::new(name)
        .albedo(base_color)
        .albedo_texture(texture)
        .roughness(0.5)
        .metallic(0.0)
}

fn mouse_handler(engine: &mut Engine) -> Result<()> {
    let pressed = engine
        .get_global_component::<InputComponent>()?
        .get_mouse_button(MouseButton::Left);
    if !pressed {
        return Ok(());
    }
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time as f64;
    let spawn_count = (CUBES_PER_SECOND as f64 * delta_time) as usize;
    spawn_cubes(engine, spawn_count)
}

/// Bevy's `spawn_cubes` without waves to pre-simulate: a batch with material `wave % 256` at
/// random x on the top back edge, moving +z at a random speed.
fn spawn_cubes(engine: &mut Engine, spawn_count: usize) -> Result<()> {
    let scene = engine.get_active_scene_handle()?;
    let (material, mesh, cubes) = with_state(|state| {
        let material = state.materials[state.wave % state.materials.len()];
        let spawn_y = VOLUME_SIZE.y / 2.0 - HALF_CUBE_SIZE;
        let spawn_z = -VOLUME_SIZE.z / 2.0 + HALF_CUBE_SIZE;
        let cubes: Vec<(Vector3f, Vector3f)> = (0..spawn_count)
            .map(|_| {
                let position = Vector3f::new(
                    (state.transform_rng.random::<f32>() - 0.5) * VOLUME_SIZE.x,
                    spawn_y,
                    spawn_z,
                );
                let velocity =
                    Vector3f::new(0., 0., MAX_VELOCITY * state.velocity_rng.random::<f32>());
                (position, velocity)
            })
            .collect();
        state.wave += 1;
        state.count += spawn_count;
        (material, state.cube_mesh, cubes)
    });
    for (position, velocity) in cubes {
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().position(position).build())
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&mesh)
                    .material(&material)
                    .build(),
            )
            .with_component(Cube { velocity })
            .build();
    }
    Ok(())
}

fn step_movement(translation: &mut Vector3f, velocity: &mut Vector3f, dt: f32) {
    translation.x += velocity.x * dt;
    translation.y += velocity.y * dt;
    translation.z += velocity.z * dt;
    velocity.y += GRAVITY * dt;
}

fn movement_system(engine: &mut Engine) -> Result<()> {
    let dt = engine.get_global_component::<TimeComponent>()?.delta_time;
    for (_entity, cube, transform) in
        engine.iterate_two_components_mut::<Cube, TransformComponent>()?
    {
        let mut translation = transform.position;
        step_movement(&mut translation, &mut cube.velocity, dt);
        transform.set_position(translation);
    }
    Ok(())
}

fn handle_collision(translation: &Vector3f, velocity: &mut Vector3f) {
    if (velocity.x > 0. && translation.x + HALF_CUBE_SIZE > VOLUME_SIZE.x / 2.0)
        || (velocity.x <= 0. && translation.x - HALF_CUBE_SIZE < -VOLUME_SIZE.x / 2.0)
    {
        velocity.x = -velocity.x;
    }
    if (velocity.z > 0. && translation.z + HALF_CUBE_SIZE > VOLUME_SIZE.z / 2.0)
        || (velocity.z <= 0. && translation.z - HALF_CUBE_SIZE < -VOLUME_SIZE.z / 2.0)
    {
        velocity.z = -velocity.z;
    }

    let velocity_y = velocity.y;
    if velocity_y < 0. && translation.y - HALF_CUBE_SIZE < -VOLUME_SIZE.y / 2.0 {
        velocity.y = -velocity_y;
    }
    if translation.y + HALF_CUBE_SIZE > VOLUME_SIZE.y / 2.0 && velocity_y > 0.0 {
        velocity.y = 0.0;
    }
}

fn collision_system(engine: &mut Engine) -> Result<()> {
    for (_entity, cube, transform) in
        engine.iterate_two_components_mut::<Cube, TransformComponent>()?
    {
        handle_collision(&transform.position, &mut cube.velocity);
    }
    Ok(())
}

/// Bevy's FPS diagnostic from this frame's delta time.
fn counter_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time as f64;
    if delta_time == 0.0 {
        return Ok(());
    }
    let raw = 1.0 / delta_time;
    with_state(|state| match &mut state.fps {
        None => {
            state.fps = Some(Fps {
                raw,
                history: std::collections::VecDeque::from([raw]),
                sum: raw,
                ema: raw,
            })
        }
        Some(fps) => {
            let alpha = (delta_time / FPS_EMA_SMOOTHING).clamp(0.0, 1.0);
            fps.ema += alpha * (raw - fps.ema);
            if fps.history.len() >= FPS_HISTORY_LENGTH {
                fps.sum -= fps.history.pop_front().unwrap_or(0.0);
            }
            fps.history.push_back(raw);
            fps.sum += raw;
            fps.raw = raw;
        }
    });
    Ok(())
}

/// Bevy's stats node: top-left, 5 px padding, black at 75 %, 40 px Fira Mono, lime labels and
/// aqua values.
fn stats_ui(context: &egui::Context) {
    // set_fonts applies from the next frame: install Fira Mono, draw once egui has bound it
    let family = egui::FontFamily::Name(FONT_FAMILY.into());
    if !context.fonts(|fonts| fonts.families().contains(&family)) {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            FONT_FAMILY.to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(FONT_BYTES)),
        );
        fonts.families.insert(family, vec![FONT_FAMILY.to_owned()]);
        context.set_fonts(fonts);
        return;
    }

    let (count, fps) = with_state(|state| {
        let fps = state
            .fps
            .as_ref()
            .map(|fps| (fps.raw, fps.sum / fps.history.len() as f64, fps.ema));
        (state.count, fps)
    });
    let value = |number: Option<f64>| number.map_or(String::new(), |n| format!("{n:.2}"));
    let lines = [
        ("Cube Count: ", count.to_string()),
        ("\nFPS (raw): ", value(fps.map(|fps| fps.0))),
        ("\nFPS (SMA): ", value(fps.map(|fps| fps.1))),
        ("\nFPS (EMA): ", value(fps.map(|fps| fps.2))),
    ];

    // Sizes in physical pixels, as Bevy's window has scale factor 1
    let points = |pixels: f32| pixels / context.pixels_per_point();
    let font = egui::FontId::new(
        points(FONT_SIZE_PX),
        egui::FontFamily::Name(FONT_FAMILY.into()),
    );
    let mut job = egui::text::LayoutJob::default();
    for (label, value) in lines {
        for (text, color) in [
            (label, egui::Color32::from_rgb(0, 255, 0)),
            (value.as_str(), egui::Color32::from_rgb(0, 255, 255)),
        ] {
            job.append(
                text,
                0.0,
                egui::TextFormat {
                    font_id: font.clone(),
                    color,
                    // Bevy's default line height: 1.2 × font size
                    line_height: Some(points(FONT_SIZE_PX * 1.2)),
                    ..Default::default()
                },
            );
        }
    }
    egui::Area::new(egui::Id::new("bevymark.stats"))
        .fixed_pos(egui::pos2(0.0, 0.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_black_alpha(191))
                .inner_margin(points(5.0))
                .show(ui, |ui| {
                    ui.add(egui::Label::new(job).extend());
                });
        });
}
