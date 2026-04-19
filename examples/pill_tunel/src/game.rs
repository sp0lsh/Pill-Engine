use pill_engine::{define_component, game::*};

// --- Scene constants ---------------------------------------------------------

// Camera
const CAMERA_POSITION_Z: f32 = 0.0;
const CAMERA_FOV: f32 = 60.0;
const CLEAR_COLOR_R: f32 = 0.02;
const CLEAR_COLOR_G: f32 = 0.02;
const CLEAR_COLOR_B: f32 = 0.05;

// Camera drift — gentle sine-wave floating to make the composition breathe.
const CAMERA_DRIFT_SPEED_X: f32 = 0.30;
const CAMERA_DRIFT_SPEED_Y: f32 = 0.22;
const CAMERA_DRIFT_AMPLITUDE_X: f32 = 0.35;
const CAMERA_DRIFT_AMPLITUDE_Y: f32 = 0.25;

// Tunnel geometry
// Concentric layers — each entry is one nested tube radius. Add/remove entries
// to add layers; pills-per-ring and ring count are shared across layers.
const TUNNEL_LAYER_RADII: &[f32] = &[6.0, 9.5, 13.0];
const TUNNEL_NEAR_Z: f32 = -3.0;
const TUNNEL_FAR_Z: f32 = 80.0;
const PILL_RINGS: usize = 30;
const PILLS_PER_RING: usize = 16;
const RING_ANGLE_STAGGER: f32 = 1.0;

// Tunnel pill appearance and motion
const PILL_SCALE: f32 = 0.5;
const PILL_FORWARD_SPEED: f32 = 8.0;
const PILL_SPIN_X: f32 = 0.4;
const PILL_SPIN_Y: f32 = 0.8;
const PILL_SPIN_Z: f32 = 1.2;
const PILL_SCALE_JITTER: f32 = 0.25; // +/- fraction of PILL_SCALE applied per pill
const PILL_SPIN_VARIANCE: f32 = 0.4; // +/- fraction of base spin applied per pill

// Hero pill — anchors the composition with a large, slowly rotating
// centerpiece. Camera looks down +Z (tunnel pills approach from high +Z and
// wrap past near-Z = -3); positive Z is "in front".
const HERO_POSITION_Z: f32 = 6.0;
const HERO_SCALE: f32 = 1.5;
const HERO_SPIN_X: f32 = 0.15;
const HERO_SPIN_Y: f32 = 0.25;
const HERO_SPIN_Z: f32 = 0.10;

// Brand-cohesive palette: 4 warm pinks/amber + 1 cool lavender for contrast.
// Each ring is tinted from this palette, giving the tunnel a chromatic rhythm
// without drifting off-brand.
const PALETTE: &[(f32, f32, f32)] = &[
    (1.00, 0.45, 0.60), // coral pink — brand primary
    (1.00, 0.72, 0.38), // amber
    (1.00, 0.32, 0.52), // rose
    (0.95, 0.40, 0.82), // magenta
    (0.70, 0.55, 1.00), // lavender (cool accent)
];
const HERO_TINT: (f32, f32, f32) = (1.00, 0.55, 0.65);

// --- Components --------------------------------------------------------------

define_component!(TunnelPillComponent { angle: f32 });
define_component!(HeroPillComponent {});

pub struct WebGame {}

// --- Systems -----------------------------------------------------------------

// Streams tunnel pills forward; wraps them back to the far end when they pass
// the near plane. Adds per-pill spin variance driven by the pill's `angle` seed
// so neighbors rotate at slightly different rates (organic, not grid-like).
fn tunnel_pills_system(engine: &mut Engine) -> Result<()> {
    let dt = engine.get_global_component::<TimeComponent>()?.delta_time;
    let tunnel_length = TUNNEL_FAR_Z - TUNNEL_NEAR_Z;

    for (_entity, transform, pill) in
        engine.iterate_two_components_mut::<TransformComponent, TunnelPillComponent>()?
    {
        let mut z = transform.position.z - PILL_FORWARD_SPEED * dt;
        while z < TUNNEL_NEAR_Z {
            z += tunnel_length;
        }
        // Preserve x/y (baked in at spawn from angle × layer_radius); only z advances.
        transform.set_position(Vector3f::new(transform.position.x, transform.position.y, z));

        let spin_variance = 1.0 + PILL_SPIN_VARIANCE * (pill.angle * 3.0).sin();
        let rot = transform.rotation;
        transform.set_rotation(Vector3f::new(
            rot.x + PILL_SPIN_X * dt * spin_variance,
            rot.y + PILL_SPIN_Y * dt * spin_variance,
            rot.z + PILL_SPIN_Z * dt,
        ));
    }

    Ok(())
}

// Slowly rotates the hero pill. Tagged separately so tunnel motion skips it.
fn hero_pill_system(engine: &mut Engine) -> Result<()> {
    let dt = engine.get_global_component::<TimeComponent>()?.delta_time;
    for (_entity, transform, _hero) in
        engine.iterate_two_components_mut::<TransformComponent, HeroPillComponent>()?
    {
        let rot = transform.rotation;
        transform.set_rotation(Vector3f::new(
            rot.x + HERO_SPIN_X * dt,
            rot.y + HERO_SPIN_Y * dt,
            rot.z + HERO_SPIN_Z * dt,
        ));
    }
    Ok(())
}

// Adds subtle sine-wave drift to the camera position — breathes life into an
// otherwise fixed viewpoint without disorienting the viewer.
fn camera_drift_system(engine: &mut Engine) -> Result<()> {
    let time = engine.get_global_component::<TimeComponent>()?.time;
    for (_entity, transform, _cam) in
        engine.iterate_two_components_mut::<TransformComponent, CameraComponent>()?
    {
        let drift_x = (time * CAMERA_DRIFT_SPEED_X).sin() * CAMERA_DRIFT_AMPLITUDE_X;
        let drift_y = (time * CAMERA_DRIFT_SPEED_Y).cos() * CAMERA_DRIFT_AMPLITUDE_Y;
        transform.set_position(Vector3f::new(drift_x, drift_y, CAMERA_POSITION_Z));
    }
    Ok(())
}

// --- Game --------------------------------------------------------------------

impl PillGame for WebGame {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let active_scene = engine.create_scene("default")?;
        engine.set_active_scene(active_scene)?;

        engine.register_component::<TransformComponent>(active_scene)?;
        engine.register_component::<CameraComponent>(active_scene)?;
        engine.register_component::<MeshRenderingComponent>(active_scene)?;
        engine.register_component::<TunnelPillComponent>(active_scene)?;
        engine.register_component::<HeroPillComponent>(active_scene)?;

        // Assets are embedded in the binary on every target. Paths in
        // include_bytes! are relative to this source file (src/game.rs →
        // ../res/…). Zero runtime I/O, single-file deploy.
        let pill_mesh_handle = engine.add_resource(Mesh::from_obj_bytes(
            "pill",
            include_bytes!("../res/models/pill.obj"),
        )?)?;
        let pill_color_handle = engine.add_resource::<Texture>(Texture::from_bytes(
            "pill_color",
            TextureType::Color,
            include_bytes!("../res/textures/pill_color.png"),
        ))?;
        let pill_normal_handle = engine.add_resource::<Texture>(Texture::from_bytes(
            "pill_normal",
            TextureType::Normal,
            include_bytes!("../res/textures/pill_normal.png"),
        ))?;

        // One material per palette tint; rings cycle through them.
        let mut tunnel_materials = Vec::with_capacity(PALETTE.len());
        for (i, (r, g, b)) in PALETTE.iter().enumerate() {
            let mat = engine.add_resource(
                Material::builder(&format!("tunnel_material_{i}"))
                    .texture("color", pill_color_handle)?
                    .texture("normal", pill_normal_handle)?
                    .color_parameter("tint", Color::new(*r, *g, *b))?
                    .build(),
            )?;
            tunnel_materials.push(mat);
        }

        let hero_material = engine.add_resource(
            Material::builder("hero_material")
                .texture("color", pill_color_handle)?
                .texture("normal", pill_normal_handle)?
                .color_parameter(
                    "tint",
                    Color::new(HERO_TINT.0, HERO_TINT.1, HERO_TINT.2),
                )?
                .build(),
        )?;

        // Camera.
        engine
            .build_entity(active_scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 0.0, CAMERA_POSITION_Z))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(CAMERA_FOV)
                    .clear_color(Color::new(
                        CLEAR_COLOR_R,
                        CLEAR_COLOR_G,
                        CLEAR_COLOR_B,
                    ))
                    .build(),
            )
            .build();

        // Hero pill — the composition's focal anchor.
        engine
            .build_entity(active_scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 0.0, HERO_POSITION_Z))
                    .scale(Vector3f::new(HERO_SCALE, HERO_SCALE, HERO_SCALE))
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&pill_mesh_handle)
                    .material(&hero_material)
                    .build(),
            )
            .with_component(HeroPillComponent {})
            .build();

        // Tunnel rings. Outer loop: concentric layers; middle: rings along Z;
        // inner: pills around the ring. Each layer shifts the palette by one
        // so adjacent layers are chromatically distinct.
        let ring_spacing = (TUNNEL_FAR_Z - TUNNEL_NEAR_Z) / PILL_RINGS as f32;
        let angle_step = std::f32::consts::TAU / PILLS_PER_RING as f32;

        for (layer_index, &layer_radius) in TUNNEL_LAYER_RADII.iter().enumerate() {
            for ring_index in 0..PILL_RINGS {
                let z = TUNNEL_NEAR_Z + ring_spacing * ring_index as f32;
                let stagger = if ring_index % 2 == 0 {
                    0.0
                } else {
                    angle_step * RING_ANGLE_STAGGER
                };
                let material_index =
                    (ring_index + layer_index) % tunnel_materials.len();
                let ring_material = tunnel_materials[material_index];

                for pill_index in 0..PILLS_PER_RING {
                    let angle = angle_step * pill_index as f32 + stagger;
                    let x = angle.cos() * layer_radius;
                    let y = angle.sin() * layer_radius;

                    // Scale jitter: deterministic per-pill variance so
                    // neighbors differ slightly, breaking the grid.
                    let scale_jitter = 1.0
                        + PILL_SCALE_JITTER
                            * ((angle * 3.0).sin()
                                + (ring_index as f32 * 1.3).cos()
                                + (layer_index as f32 * 0.7).sin())
                            * 0.5;
                    let scale = PILL_SCALE * scale_jitter;

                    engine
                        .build_entity(active_scene)
                        .with_component(
                            TransformComponent::builder()
                                .position(Vector3f::new(x, y, z))
                                .scale(Vector3f::new(scale, scale, scale))
                                .build(),
                        )
                        .with_component(
                            MeshRenderingComponent::builder()
                                .mesh(&pill_mesh_handle)
                                .material(&ring_material)
                                .build(),
                        )
                        .with_component(TunnelPillComponent { angle })
                        .build();
                }
            }
        }

        engine.add_system("tunnel_pills", tunnel_pills_system)?;
        engine.add_system("hero_pill", hero_pill_system)?;
        engine.add_system("camera_drift", camera_drift_system)?;

        Ok(())
    }
}
