use pill_engine::game::*;

pub struct Game {}

const TARGET_FPS: f32 = 60.0;

fn rotate_system(engine: &mut Engine) -> Result<()> {
    let dt = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .min(1.0 / TARGET_FPS);

    for (_entity, transform, _pbr) in
        engine.iterate_two_components_mut::<TransformComponent, PbrRenderableComponent>()?
    {
        transform.rotate_around_axis(2.0 * dt, Vector3f::new(0.0, 0.0, 1.0));
    }

    Ok(())
}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("helmet")?;
        engine.set_active_scene(scene)?;

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<PbrRenderableComponent>(scene)?;

        let mesh_handle = engine.add_resource(Mesh::new(
            "helmet_mesh",
            "models/DamagedHelmet.cooked_mesh".into(),
        ))?;

        let albedo_handle = engine.add_resource(Texture::new(
            "helmet_albedo",
            TextureType::Color,
            ResourceLoader::Path("models/DamagedHelmet_albedo.cooked_tex".into()),
        ))?;

        let normal_handle = engine.add_resource(Texture::new(
            "helmet_normal",
            TextureType::Normal,
            ResourceLoader::Path("models/DamagedHelmet_normal.cooked_tex".into()),
        ))?;

        let material_handle = engine.add_resource(
            PBRMaterial::new("helmet_material")
                .albedo_texture(albedo_handle)
                .normal_texture(normal_handle),
        )?;

        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 0.0, -3.0))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .clear_color(Color::new(0.05, 0.05, 0.06))
                    .build(),
            )
            .build();

        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .rotation(Vector3f::new(90.0, 0.0, 0.0))
                    .build(),
            )
            .with_component(
                PbrRenderableComponent::builder()
                    .mesh(&mesh_handle)
                    .pbr_material(&material_handle)
                    .build(),
            )
            .build();

        engine.add_system("rotate", rotate_system)?;

        Ok(())
    }
}
