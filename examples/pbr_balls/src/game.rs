use pill_engine::{define_component, game::*};

const EQUIRECT: &[u8] = include_bytes!("../res/textures/studio_equirect.cooked_tex");
const DIFFUSE_IBL: &[u8] = include_bytes!("../res/textures/studio_diffuse_ibl.cooked_tex");
const SPECULAR_IBL: &[u8] = include_bytes!("../res/textures/studio_specular_ibl.cooked_tex");
const BRDF_LUT: &[u8] = include_bytes!("../res/textures/brdf_lut.cooked_tex");

define_component!(OrbitCamera {
    yaw: f32,
    pitch: f32,
    radius: f32,
});

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        engine.set_background_texture(EQUIRECT.to_vec())?;
        engine.set_ibl_textures(
            DIFFUSE_IBL.to_vec(),
            SPECULAR_IBL.to_vec(),
            BRDF_LUT.to_vec(),
        )?;

        let scene = engine.create_scene("pbr_balls")?;
        engine.set_active_scene(scene)?;

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<PbrRenderableComponent>(scene)?;
        engine.register_component::<OrbitCamera>(scene)?;

        engine.add_system("orbit_camera", orbit_camera_system)?;

        let mesh_handle = engine.add_resource(Mesh::new(
            "spheres_mesh",
            "models/MetalRoughSpheres.cooked_mesh".into(),
        ))?;

        let albedo_handle = engine.add_resource(Texture::new(
            "spheres_albedo",
            TextureType::Color,
            ResourceLoader::Path("models/MetalRoughSpheres_albedo.cooked_tex".into()),
        ))?;

        let mr_handle = engine.add_resource(Texture::new(
            "spheres_mr",
            TextureType::MetallicRoughness,
            ResourceLoader::Path("models/MetalRoughSpheres_metallic_roughness.cooked_tex".into()),
        ))?;

        let mat_handle = engine.add_resource(
            PBRMaterial::new("spheres_mat")
                .albedo_texture(albedo_handle)
                .metallic_roughness_texture(mr_handle)
                .metallic(1.0)
                .roughness(1.0),
        )?;

        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(
                PbrRenderableComponent::builder()
                    .mesh(&mesh_handle)
                    .pbr_material(&mat_handle)
                    .build(),
            )
            .build();

        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 0.0, 10.0))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .clear_color(Color::new(0.0, 0.0, 0.0))
                    .look_at(Some(Vector3f::new(0.0, 0.0, 0.0)))
                    .build(),
            )
            .with_component(OrbitCamera {
                yaw: 0.0,
                pitch: 0.0,
                radius: 10.0,
            })
            .build();

        Ok(())
    }
}

fn orbit_camera_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component_mut::<InputComponent>()?;
    let mouse_delta = input.get_mouse_delta();
    let scroll_delta = input.get_mouse_scroll_delta();
    let lmb = input.get_mouse_button(MouseButton::Left);

    for (_, tfm, orbit) in engine.iterate_two_components_mut::<TransformComponent, OrbitCamera>()? {
        if lmb {
            orbit.yaw -= mouse_delta.x * 0.3;
            orbit.pitch = (orbit.pitch + mouse_delta.y * 0.3).clamp(-70.0, 70.0);
        }
        orbit.radius = (orbit.radius - scroll_delta.y * 0.5).clamp(1.0, 50.0);

        let pr = orbit.pitch.to_radians();
        let yr = orbit.yaw.to_radians();
        tfm.set_position(Vector3f::new(
            orbit.radius * pr.cos() * yr.sin(),
            orbit.radius * pr.sin(),
            orbit.radius * pr.cos() * yr.cos(),
        ));
    }
    Ok(())
}
