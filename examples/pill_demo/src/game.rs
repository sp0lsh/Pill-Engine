use cgmath::InnerSpace;
use pill_engine::{define_component, game::*};
use rand::Rng;

// Draft import for renderer vNext API (as per DESIGN.md)
// use pill_engine::pill_renderer as pr;

// Define custom component
pub struct PillComponent {}

impl Component for PillComponent {}

impl PillTypeMapKey for PillComponent {
    type Storage = ComponentStorage<Self>;
}

// Component for motion blur test pill
define_component!(MovingPillComponent {
    target_position: Vector3f,
    velocity: f32,
    rotation_speed: f32,
});

// Game
pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        // let _scene_benchmark = create_scene_benchmark(engine, "Benchmark 60k pills")?;
        // let _scene_pills = create_scene_pills(engine, "Pills")?;
        // let _scene_dof_proof = create_scene_dof_proof(engine, "DoF Proof")?;
        // let _scene_mb_proof = create_scene_mb_proof(engine, "MB Proof")?;
        // let _scene_pbr_grid: SceneHandle = create_scene_pbr_grid(engine, "PBR Grid")?;
        // let _scene_gltf_glb: SceneHandle = create_scene_gltf_glb(engine, "GLTF glb")?;
        // let _scene_gltf_gltf: SceneHandle = create_scene_gltf_gltf(engine, "GLTF gltf")?;
        // let _scene_gltf_pbr: SceneHandle = create_scene_gltf_pbr(engine, "GLTF PBR Spheres")?; // TODO: fix
        let _scene_gltf_sponza: SceneHandle = create_scene_gltf_sponza(engine, "GLTF Sponza")?;
        engine.set_active_scene(_scene_gltf_sponza)?;
        Ok(())
    }
}

fn create_scene_dof_proof(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Add systems
    engine.add_system("PillRotation", pill_rotation_system)?;

    // Mesh
    let pill_mesh_handle = get_or_add_mesh(engine, "Pill", "models/pill.obj")?;

    // Textures (same as normal pills scene so PBR bindings are valid)
    let pill_color_texture = Texture::new(
        "PillColor",
        TextureType::Gamma,
        ResourceLoadType::Path("textures/pill_color.png".into()),
    );
    let pill_color_texture_handle = engine.add_resource::<Texture>(pill_color_texture)?;
    let pill_normal_texture = Texture::new(
        "PillNormal",
        TextureType::Linear,
        ResourceLoadType::Path("textures/pill_normal.png".into()),
    );
    let pill_normal_texture_handle = engine.add_resource::<Texture>(pill_normal_texture)?;

    // Create a few distinct materials so pills are easy to distinguish on-screen.
    let tints = [
        Color::new(1.0, 0.2, 0.2),
        Color::new(0.2, 1.0, 0.2),
        Color::new(0.2, 0.2, 1.0),
        Color::new(1.0, 1.0, 0.2),
        Color::new(1.0, 0.2, 1.0),
    ];
    let mut materials: Vec<PBRMaterialHandle> = Vec::new();
    for (i, tint) in tints.iter().enumerate() {
        let mut mat: PBRMaterial = PBRMaterial::new(&format!("DoFProofMat{}", i));
        mat.set_albedo_texture(pill_color_texture_handle.clone());
        mat.set_normal_texture(pill_normal_texture_handle.clone());
        mat.set_base_color_factor(*tint);
        mat.set_metallic_factor(0.0);
        mat.set_roughness_factor(0.4);
        let handle = engine.add_resource::<PBRMaterial>(mat)?;
        materials.push(handle);
    }

    // Camera: look straight down +Z from z=-16 so depth ordering is obvious.
    let camera = engine.create_entity(scene)?;
    let cam_xform = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -16.0))
        .rotation(Vector3f::new(0.0, 0.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, cam_xform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Five pills at fixed depths along view direction; sizes/positions make it easy to verify focus changes.
    // Distances from camera are approx: 6, 16, 26, 46, 86 units.
    let zs = [-10.0, 0.0, 10.0, 30.0, 70.0];
    let xs: [f32; 5] = [-5.0, -3.0, 0.0, 15.0, 50.0];
    for i in 0..zs.len() {
        let pill = engine.create_entity(scene)?;
        let transform_component = TransformComponent::builder()
            .scale(Vector3f::new(1.2, 1.8, 1.2))
            .rotation(Vector3f::new(0.0, 0.0, 0.0))
            .position(Vector3f::new(xs[i], 0.0, zs[i]))
            .build();
        engine.add_component_to_entity(scene, pill, transform_component)?;
        let mesh_rendering_component: MeshRenderingComponent = MeshRenderingComponent::builder()
            .mesh(&pill_mesh_handle)
            .material(&materials[i])
            .build();
        engine.add_component_to_entity(scene, pill, mesh_rendering_component)?;
        engine.add_component_to_entity(scene, pill, PillComponent {})?;
    }

    Ok(scene)
}

fn create_scene_mb_proof(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<MovingPillComponent>(scene)?;

    // Add systems
    engine.add_system("MotionBlurMovement", motion_blur_movement_system)?;

    // Mesh
    let pill_mesh_handle = get_or_add_mesh(engine, "Pill", "models/pill.obj")?;

    // Textures
    let pill_color_texture = Texture::new(
        "PillColor",
        TextureType::Gamma,
        ResourceLoadType::Path("textures/pill_color.png".into()),
    );
    let pill_color_texture_handle = engine.add_resource::<Texture>(pill_color_texture)?;
    let pill_normal_texture = Texture::new(
        "PillNormal",
        TextureType::Linear,
        ResourceLoadType::Path("textures/pill_normal.png".into()),
    );
    let pill_normal_texture_handle = engine.add_resource::<Texture>(pill_normal_texture)?;

    // Create material for the moving pill (bright color to make motion blur visible)
    let mut mat: PBRMaterial = PBRMaterial::new("MotionBlurMat");
    mat.set_albedo_texture(pill_color_texture_handle.clone());
    mat.set_normal_texture(pill_normal_texture_handle.clone());
    mat.set_base_color_factor(Color::new(1.0, 0.8, 0.2)); // Bright yellow-orange
    mat.set_metallic_factor(0.1);
    mat.set_roughness_factor(0.3);
    let material_handle = engine.add_resource::<PBRMaterial>(mat)?;

    // Camera: positioned to see the motion clearly
    let camera = engine.create_entity(scene)?;
    let cam_xform = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -20.0))
        .rotation(Vector3f::new(0.0, 0.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, cam_xform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Multiple moving pills for motion blur testing
    let num_pills = 12;
    let positions: Vec<Vector3f> = (0..num_pills)
        .map(|i| {
            let angle = i as f32 * (std::f32::consts::TAU / num_pills as f32);
            let radius = 6.0;
            Vector3f::new(angle.cos() * radius, 0.0, angle.sin() * radius)
        })
        .collect();

    for (i, pos) in positions.iter().enumerate() {
        let angle = i as f32 * (std::f32::consts::TAU / num_pills as f32);
        let pill = engine.create_entity(scene)?;
        let transform_component = TransformComponent::builder()
            .scale(Vector3f::new(1.5, 2.0, 1.5))
            .rotation(Vector3f::new(0.0, 0.0, 0.0))
            .position(*pos)
            .build();
        engine.add_component_to_entity(scene, pill, transform_component)?;

        let mesh_rendering_component: MeshRenderingComponent = MeshRenderingComponent::builder()
            .mesh(&pill_mesh_handle)
            .material(&material_handle)
            .build();
        engine.add_component_to_entity(scene, pill, mesh_rendering_component)?;

        // Spread targets around a different circular offset
        let angle_offset = angle + std::f32::consts::PI / 2.0;
        let target_radius = 5.0;
        let target = Vector3f::new(
            angle_offset.cos() * target_radius,
            0.0,
            angle_offset.sin() * target_radius,
        );

        let moving_pill_component = MovingPillComponent {
            target_position: target,
            velocity: 6.0 + i as f32 * 0.3, // vary speed a bit for visualization
            rotation_speed: 0.0,            // Could add spin if you wish
        };
        engine.add_component_to_entity(scene, pill, moving_pill_component)?;
    }

    Ok(scene)
}

fn create_scene_benchmark(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Add systems
    engine.add_system("PillRotation", pill_rotation_system)?;

    // Add meshes
    let pill_mesh_handle = get_or_add_mesh(engine, "Pill", "models/pill.obj")?;
    // Add textures
    let pill_color_texture = Texture::new(
        "PillColor",
        TextureType::Gamma,
        ResourceLoadType::Path("textures/pill_color.png".into()),
    );
    let pill_color_texture_handle = engine.add_resource::<Texture>(pill_color_texture)?;
    let pill_normal_texture = Texture::new(
        "PillNormal",
        TextureType::Linear,
        ResourceLoadType::Path("textures/pill_normal.png".into()),
    );
    let pill_normal_texture_handle = engine.add_resource::<Texture>(pill_normal_texture)?;

    // Material properties showcase: create a small set of materials to reuse
    let mut rng = rand::thread_rng();
    let mut materials = Vec::new();
    for j in 0..10 {
        let mut mat: PBRMaterial = PBRMaterial::new(&format!("PillMat{}", j));
        mat.set_albedo_texture(pill_color_texture_handle.clone());
        mat.set_normal_texture(pill_normal_texture_handle.clone());
        let tint = Color::new(
            rng.gen_range(0.2..=0.8),
            rng.gen_range(0.2..=0.8),
            rng.gen_range(0.2..=0.8),
        );
        let metallic = rng.gen_range(0.0..=1.0);
        let roughness = rng.gen_range(0.0..=1.0);
        mat.set_base_color_factor(tint);
        mat.set_metallic_factor(metallic);
        mat.set_roughness_factor(roughness);
        let handle = engine.add_resource::<PBRMaterial>(mat)?;
        materials.push(handle);
    }

    // Create camera entity
    let camera = engine.create_entity(scene)?;
    let transform_component = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -16.0))
        .rotation(Vector3f::new(0.0, 0.0, -20.0))
        .build();
    engine.add_component_to_entity(scene, camera, transform_component)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Create pill entity
    for i in 0..60000 {
        let pill = engine.create_entity(scene)?;
        let posx = rng.gen_range(-10.0..=20.0);
        let posy = rng.gen_range(-10.0..=10.0);
        let posz = rng.gen_range(-1.0..=1.0);
        let rotx = rng.gen_range(-180.0..=180.0);
        let transform_component = TransformComponent::builder()
            .rotation(Vector3f::new(rotx, 0.0, 0.0))
            .position(Vector3f::new(posx, posy, posz))
            .build();
        engine.add_component_to_entity(scene, pill, transform_component)?;
        let mesh_rendering_component = MeshRenderingComponent::builder()
            .mesh(&pill_mesh_handle)
            .material(&materials[i as usize % materials.len()])
            .build();
        engine.add_component_to_entity(scene, pill, mesh_rendering_component)?;
        engine.add_component_to_entity(scene, pill, PillComponent {})?;
    }

    Ok(scene)
}

fn create_scene_pills(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Add systems
    engine.add_system("PillRotation", pill_rotation_system)?;

    // Add meshes
    let pill_mesh_handle = get_or_add_mesh(engine, "Pill", "models/pill.obj")?;
    // Add textures
    let pill_color_texture = Texture::new(
        "PillColor",
        TextureType::Gamma,
        ResourceLoadType::Path("textures/pill_color.png".into()),
    );
    let pill_color_texture_handle = engine.add_resource::<Texture>(pill_color_texture)?;
    let pill_normal_texture = Texture::new(
        "PillNormal",
        TextureType::Linear,
        ResourceLoadType::Path("textures/pill_normal.png".into()),
    );
    let pill_normal_texture_handle = engine.add_resource::<Texture>(pill_normal_texture)?;

    // Material properties showcase: create a small set of materials to reuse
    let mut rng = rand::thread_rng();
    let mut materials = Vec::new();
    for j in 0..10 {
        let mut mat: PBRMaterial = PBRMaterial::new(&format!("PillMat{}", j));
        mat.set_albedo_texture(pill_color_texture_handle.clone());
        mat.set_normal_texture(pill_normal_texture_handle.clone());
        let tint = Color::new(
            rng.gen_range(0.2..=0.8),
            rng.gen_range(0.2..=0.8),
            rng.gen_range(0.2..=0.8),
        );
        let metallic = rng.gen_range(0.0..=1.0);
        let roughness = rng.gen_range(0.0..=1.0);
        mat.set_base_color_factor(tint);
        mat.set_metallic_factor(metallic);
        mat.set_roughness_factor(roughness);
        let handle = engine.add_resource::<PBRMaterial>(mat)?;
        materials.push(handle);
    }

    // Create camera entity
    let camera = engine.create_entity(scene)?;
    let transform_component = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -16.0))
        .rotation(Vector3f::new(0.0, 0.0, -20.0))
        .build();
    engine.add_component_to_entity(scene, camera, transform_component)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Create pill entity
    for i in 0..3000 {
        let pill = engine.create_entity(scene)?;
        let posx = rng.gen_range(-100.0..=100.0);
        let posy = rng.gen_range(-100.0..=100.0);
        let posz = rng.gen_range(-10.0..=100.0);
        let rotx = rng.gen_range(-180.0..=180.0);
        let roty = rng.gen_range(-180.0..=180.0);
        let rotz = rng.gen_range(-180.0..=180.0);
        let transform_component = TransformComponent::builder()
            .scale(Vector3f::new(0.75, 1.2, 0.75))
            .rotation(Vector3f::new(rotx, roty, rotz))
            .position(Vector3f::new(posx, posy, posz))
            .build();
        engine.add_component_to_entity(scene, pill, transform_component)?;
        let mesh_rendering_component = MeshRenderingComponent::builder()
            .mesh(&pill_mesh_handle)
            .material(&materials[i as usize % materials.len()])
            .build();
        engine.add_component_to_entity(scene, pill, mesh_rendering_component)?;
        engine.add_component_to_entity(scene, pill, PillComponent {})?;
    }

    Ok(scene)
}

fn create_scene_pbr_grid(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Camera
    let camera = engine.create_entity(scene)?;
    let camera_transform = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -16.0))
        .rotation(Vector3f::new(0.0, 0.0, -20.0))
        .build();
    engine.add_component_to_entity(scene, camera, camera_transform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Mesh (use pill as sphere substitute)
    let pill_mesh_handle = get_or_add_mesh(engine, "Pill", "models/pill.obj")?;

    // Create 5x5 grid where:
    // - X axis: roughness [0.0 .. 1.0]
    // - Y axis: metallic  [0.0 .. 1.0]
    // - Base color: (0.35, 0.35, 0.35)
    let grid_size = 5;
    let spacing = 2.5f32;
    for y in 0..grid_size {
        for x in 0..grid_size {
            let roughness = (x as f32) / ((grid_size - 1) as f32);
            let metallic = (y as f32) / ((grid_size - 1) as f32);

            let mut mat: PBRMaterial = PBRMaterial::new(&format!("PBR_{}_{}", x, y));
            mat.set_base_color_factor(Color::new(0.35, 0.35, 0.35));
            mat.set_metallic_factor(metallic);
            mat.set_roughness_factor(roughness);
            let mat_handle = engine.add_resource::<PBRMaterial>(mat)?;

            let entity = engine.create_entity(scene)?;
            let pos_x = (x as f32 - ((grid_size as f32 - 1.0) * 0.5)) * spacing;
            let pos_y = (y as f32 - ((grid_size as f32 - 1.0) * 0.5)) * spacing;
            let transform = TransformComponent::builder()
                .position(Vector3f::new(pos_x, pos_y, 0.0))
                .rotation(Vector3f::new(45.0, 45.0, 45.0))
                .build();
            engine.add_component_to_entity(scene, entity, transform)?;

            let mesh_render = MeshRenderingComponent::builder()
                .mesh(&pill_mesh_handle)
                .material(&mat_handle)
                .build();
            engine.add_component_to_entity(scene, entity, mesh_render)?;
        }
    }

    Ok(scene)
}

fn create_scene_gltf_glb(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    engine.add_system("PillRotation", pill_rotation_system)?;

    // Camera
    let camera = engine.create_entity(scene)?;
    let camera_transform = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -2.0))
        .rotation(Vector3f::new(0.0, 0.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, camera_transform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Load Model from GLTF and assemble
    // https://github.com/KhronosGroup/glTF-Sample-Models/tree/d7a3cc8e51d7c573771ae77a57f16b0662a905c6/2.0/DamagedHelmet
    let model_handle = get_or_add_model(engine, "DamagedHelmet", "models/DamagedHelmet.glb")?;
    let entities = assemble_model_into_scene(engine, scene, &model_handle)?;
    for entity in entities {
        engine.add_component_to_entity(scene, entity, PillComponent {})?;
    }

    Ok(scene)
}

fn create_scene_gltf_pbr(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Camera
    let camera = engine.create_entity(scene)?;
    let camera_transform = TransformComponent::builder()
        .position(Vector3f::new(0.0, -30.0, 0.0))
        .rotation(Vector3f::new(-90.0, 0.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, camera_transform)?;
    let camera_component = CameraComponent::builder().enabled(true).fov(45.0).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Load Model from GLTF and assemble
    // https://github.com/KhronosGroup/glTF-Sample-Assets/blob/ef86ca2c5996146cf1b14f7478842db4eee920b1/Models/MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb
    let model_handle =
        get_or_add_model(engine, "MetalRoughSpheres", "models/MetalRoughSpheres.glb")?;
    let _entities = assemble_model_into_scene(engine, scene, &model_handle)?;

    Ok(scene)
}

fn create_scene_gltf_gltf(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    engine.add_system("PillRotation", pill_rotation_system)?;

    // Camera
    let camera = engine.create_entity(scene)?;
    let camera_transform = TransformComponent::builder()
        .position(Vector3f::new(0.0, 0.0, -3.0))
        .rotation(Vector3f::new(0.0, 0.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, camera_transform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Load Model from GLTF and assemble
    // https://github.com/KhronosGroup/glTF-Sample-Assets/blob/ef86ca2c5996146cf1b14f7478842db4eee920b1/Models/SciFiHelmet/README.md
    let model_handle =
        get_or_add_model(engine, "SciFiHelmet", "models/SciFiHelmet/SciFiHelmet.gltf")?;
    let entities = assemble_model_into_scene(engine, scene, &model_handle)?;
    for entity in entities {
        engine.add_component_to_entity(scene, entity, PillComponent {})?;
    }

    Ok(scene)
}

fn create_scene_gltf_sponza(engine: &mut Engine, name: &str) -> Result<SceneHandle> {
    let scene = engine.create_scene(name)?;

    // Register components
    engine.register_component::<TransformComponent>(scene)?;
    engine.register_component::<MeshRenderingComponent>(scene)?;
    engine.register_component::<AudioListenerComponent>(scene)?;
    engine.register_component::<AudioSourceComponent>(scene)?;
    engine.register_component::<CameraComponent>(scene)?;
    engine.register_component::<PillComponent>(scene)?;

    // Camera
    let camera = engine.create_entity(scene)?;
    let camera_transform = TransformComponent::builder()
        .position(Vector3f::new(5.0, 1.0, 0.0))
        .rotation(Vector3f::new(0.0, -90.0, 0.0))
        .build();
    engine.add_component_to_entity(scene, camera, camera_transform)?;
    let camera_component = CameraComponent::builder().enabled(true).build();
    engine.add_component_to_entity(scene, camera, camera_component)?;

    // Load Model from GLTF and assemble
    // https://github.com/KhronosGroup/glTF-Sample-Assets/blob/ef86ca2c5996146cf1b14f7478842db4eee920b1/Models/SciFiHelmet/README.md
    let model_handle = get_or_add_model(engine, "Sponza", "models/sponza/Sponza.gltf")?;
    let entities = assemble_model_into_scene(engine, scene, &model_handle)?;
    // for entity in entities {
    //     engine.add_component_to_entity(scene, entity, PillComponent {})?;
    // }

    Ok(scene)
}

fn get_or_add_model(engine: &mut Engine, name: &str, path: &str) -> Result<ModelHandle> {
    match engine.get_resource_handle::<Model>(name) {
        Ok(handle) => Ok(handle),
        Err(_) => {
            let model = Model::new(name, path.into());
            engine.add_resource(model)
        }
    }
}

fn assemble_model_into_scene(
    engine: &mut Engine,
    scene: SceneHandle,
    model_handle: &ModelHandle,
) -> Result<Vec<EntityHandle>> {
    // Snapshot slots to avoid holding an immutable borrow of engine while mutating it
    let (material_handles, slots): (
        Vec<PBRMaterialHandle>,
        Vec<(MeshHandle, PBRMaterialHandle, Vector3f, Vector3f, Vector3f)>,
    ) = {
        let model = engine.get_resource::<Model>(model_handle)?;
        let material_handles = model.materials.clone();
        let slots = model
            .material_slots
            .iter()
            .map(|s| {
                (
                    s.mesh.clone(),
                    s.material.clone(),
                    s.translation,
                    s.rotation_euler_deg,
                    s.scale,
                )
            })
            .collect();
        (material_handles, slots)
    };

    // Update emissive for each material handle
    for mh in material_handles {
        if let Ok(mat) = engine.get_resource_mut::<PBRMaterial>(&mh) {
            let cur = mat.emissive;
            mat.set_emissive_factor(cur * 100.0);
        }
    }
    let mut entities: Vec<EntityHandle> = Vec::new();
    for (mesh_h, material_h, translation, rotation_deg, scale) in slots {
        let entity = engine.create_entity(scene)?;
        let transform = TransformComponent::builder()
            .position(translation)
            .rotation(rotation_deg)
            .scale(scale)
            .build();
        engine.add_component_to_entity(scene, entity, transform)?;
        let mesh_render = MeshRenderingComponent::builder()
            .mesh(&mesh_h)
            .material(&material_h)
            .build();
        engine.add_component_to_entity(scene, entity, mesh_render)?;
        entities.push(entity);
    }
    Ok(entities)
}

fn get_or_add_mesh(engine: &mut Engine, name: &str, path: &str) -> Result<MeshHandle> {
    match engine.get_resource_handle::<Mesh>(name) {
        Ok(handle) => Ok(handle),
        Err(_) => {
            let mesh = Mesh::new(name, path.into());
            engine.add_resource(mesh)
        }
    }
}

fn pill_rotation_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    let input_component = engine.get_global_component_mut::<InputComponent>()?;

    // Exit on Escape
    if input_component.get_key_pressed(KeyboardKey::Escape) {
        // TODO: engine.request_shutdown();
        // TODO: standalone handle request_shutdown
        std::process::exit(0);
    }

    // Rotate pill if spacebar is not pressed
    if !input_component.get_key_pressed(KeyboardKey::Space) {
        for (_, transform_component, _) in
            engine.iterate_two_components_mut::<TransformComponent, PillComponent>()?
        {
            transform_component.rotate_around_axis(90.0 * delta_time, Vector3f::new(0.0, 1.0, 0.0));
        }
    }

    Ok(())
}

fn pick_random_target_opposite_hemisphere(
    rng: &mut impl Rng,
    cur_pos: Vector3f,
    prev_dir: Vector3f,
) -> Vector3f {
    let min_distance = 4.0;
    let radius = 13.0;
    let mut new_target = cur_pos;

    for _ in 0..20 {
        let u: f32 = rng.gen();
        let v: f32 = rng.gen();
        let theta = 2.0 * std::f32::consts::PI * u;
        let phi = v.acos();

        let mut rand_dir =
            Vector3f::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
        if rand_dir.dot(prev_dir) > 0.0 {
            rand_dir = -rand_dir;
        }

        let new_distance = rng.gen_range(min_distance..=radius);
        new_target = cur_pos + rand_dir * new_distance;

        new_target.x = new_target.x.max(-12.0).min(12.0);
        new_target.y = new_target.y.max(-8.0).min(8.0);
        new_target.z = new_target.z.max(-3.0).min(20.0);

        if (new_target - cur_pos).magnitude() >= min_distance {
            break;
        }
    }

    new_target
}

fn motion_blur_movement_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    let mut rng = rand::thread_rng();

    for (_, transform_component, moving_pill_component) in
        engine.iterate_two_components_mut::<TransformComponent, MovingPillComponent>()?
    {
        let cur_pos = transform_component.position;
        let last_dir_raw = cur_pos - moving_pill_component.target_position;
        let last_dir = if last_dir_raw.magnitude2() > 0.0001 {
            last_dir_raw.normalize()
        } else {
            Vector3f::new(0.0, 0.0, 1.0)
        };

        let direction_to_target = moving_pill_component.target_position - cur_pos;
        let distance_to_target = direction_to_target.magnitude();
        let step = moving_pill_component.velocity * delta_time;

        if distance_to_target <= step {
            transform_component.set_position(moving_pill_component.target_position);
            let reached_pos = moving_pill_component.target_position;

            moving_pill_component.target_position =
                pick_random_target_opposite_hemisphere(&mut rng, reached_pos, last_dir);
            moving_pill_component.velocity = rng.gen_range(8.0..=25.0);
            continue;
        }

        if distance_to_target > 0.0 {
            let normalized_direction = direction_to_target / distance_to_target;
            let movement = normalized_direction * step;
            let new_position = cur_pos + movement;
            transform_component.set_position(new_position);
        }

        // Update rotation
        // transform_component.rotate_around_axis(
        //     moving_pill_component.rotation_speed * delta_time,
        //     Vector3f::new(0.0, 1.0, 0.0),
        // );
    }

    Ok(())
}
