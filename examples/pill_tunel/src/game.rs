use pill_engine::{define_component, game::*};

// --- Tweakable scene constants ---

// Camera
const CAMERA_POSITION_Z: f32 = 0.0;
const CAMERA_FOV: f32 = 75.0;
const CLEAR_COLOR_R: f32 = 0.02;
const CLEAR_COLOR_G: f32 = 0.02;
const CLEAR_COLOR_B: f32 = 0.05;

// Tunnel geometry
const TUNNEL_RADIUS: f32 = 6.0;
const TUNNEL_NEAR_Z: f32 = -3.0;
const TUNNEL_FAR_Z: f32 = 80.0;
const PILL_RINGS: usize = 30;
const PILLS_PER_RING: usize = 16;
const RING_ANGLE_STAGGER: f32 = 1.;

// Pill appearance and motion
const PILL_SCALE: f32 = 0.5;
const PILL_FORWARD_SPEED: f32 = 8.0;
const PILL_SPIN_X: f32 = 0.4;
const PILL_SPIN_Y: f32 = 0.8;
const PILL_SPIN_Z: f32 = 1.2;
const PILL_TINT_R: f32 = 1.0;
const PILL_TINT_G: f32 = 0.45;
const PILL_TINT_B: f32 = 0.6;

define_component!(TunnelPillComponent { angle: f32 });

pub struct WebGame {}

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
        let angle = pill.angle;
        let x = angle.cos() * TUNNEL_RADIUS;
        let y = angle.sin() * TUNNEL_RADIUS;
        transform.set_position(Vector3f::new(x, y, z));

        let rot = transform.rotation;
        transform.set_rotation(Vector3f::new(
            rot.x + PILL_SPIN_X * dt,
            rot.y + PILL_SPIN_Y * dt,
            rot.z + PILL_SPIN_Z * dt,
        ));
    }

    Ok(())
}

impl PillGame for WebGame {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let active_scene = engine.create_scene("default")?;
        engine.set_active_scene(active_scene)?;

        engine.register_component::<TransformComponent>(active_scene)?;
        engine.register_component::<CameraComponent>(active_scene)?;
        engine.register_component::<MeshRenderingComponent>(active_scene)?;
        engine.register_component::<TunnelPillComponent>(active_scene)?;

        // Assets are embedded in the binary on every target (include_bytes!
        // paths are relative to this source file, src/game.rs → ../res/…).
        // Zero runtime I/O, single-file deploy.
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

        let material_handle = engine.add_resource(
            Material::builder("pill_material")
                .texture("color", pill_color_handle)?
                .texture("normal", pill_normal_handle)?
                .color_parameter(
                    "tint",
                    Color::new(PILL_TINT_R, PILL_TINT_G, PILL_TINT_B),
                )?
                .build(),
        )?;

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

        let ring_spacing = (TUNNEL_FAR_Z - TUNNEL_NEAR_Z) / PILL_RINGS as f32;
        let angle_step = std::f32::consts::TAU / PILLS_PER_RING as f32;
        let pill_scale = Vector3f::new(PILL_SCALE, PILL_SCALE, PILL_SCALE);

        for ring_index in 0..PILL_RINGS {
            let z = TUNNEL_NEAR_Z + ring_spacing * ring_index as f32;
            let stagger = if ring_index % 2 == 0 {
                0.0
            } else {
                angle_step * RING_ANGLE_STAGGER
            };
            for pill_index in 0..PILLS_PER_RING {
                let angle = angle_step * pill_index as f32 + stagger;
                let x = angle.cos() * TUNNEL_RADIUS;
                let y = angle.sin() * TUNNEL_RADIUS;

                engine
                    .build_entity(active_scene)
                    .with_component(
                        TransformComponent::builder()
                            .position(Vector3f::new(x, y, z))
                            .scale(pill_scale)
                            .build(),
                    )
                    .with_component(
                        MeshRenderingComponent::builder()
                            .mesh(&pill_mesh_handle)
                            .material(&material_handle)
                            .build(),
                    )
                    .with_component(TunnelPillComponent { angle })
                    .build();
            }
        }

        engine.add_system("tunnel_pills", tunnel_pills_system)?;

        Ok(())
    }
}
