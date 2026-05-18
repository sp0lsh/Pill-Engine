use pill_engine::game::*;

use crate::free_fly::{free_fly_system, FreeFlyComponent};
use crate::gaussian_cloud::GaussianCloud;
use crate::pass_splat::PassSplat;

pub struct GaussianGame {}

impl PillGame for GaussianGame {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("default")?;
        engine.set_active_scene(scene)?;

        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FreeFlyComponent>(scene)?;

        engine.register_resource_type::<GaussianCloud>(16)?;

        #[cfg(not(target_arch = "wasm32"))]
        engine.add_resource(GaussianCloud::from_path("scene", "splat.ply"))?;
        #[cfg(target_arch = "wasm32")]
        engine.add_resource(GaussianCloud::from_bytes("scene", include_bytes!("../res/splat.ply")))?;

        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 1.0, 3.0))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(60.0)
                    .clear_color(Color::new(0.05, 0.05, 0.07))
                    .build(),
            )
            .with_component(FreeFlyComponent::new(180.0, 0.0, 5.0))
            .build();

        engine.set_render_passes(vec![Box::new(PassSplat::new("scene"))])?;

        engine.add_system("free_fly", free_fly_system)?;

        Ok(())
    }
}
