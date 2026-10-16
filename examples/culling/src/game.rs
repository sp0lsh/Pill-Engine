//! Culling test scene: OBJECT_COUNT kittens (res/models/kitten.obj, 28 944 triangles, from
//! github.com/zeux/niagara, MIT) at seeded-random positions in a 10 m cube around the game camera,
//! with a random uniform scale of 0.2, 1 or 5 and a random rotation of up to 90°. The game camera
//! sits at the origin looking down −Z. The editor (`editor.rs`) shows the scene and the game
//! camera's frustum from an overview camera; both cameras fly with the same controls.

use crate::editor::{self, EditorState, FlyCamera};
use crate::pass_editor::EditorPass;
use glam::{EulerRot, Quat};
use pill_engine::game::*;

const OBJECT_COUNT: usize = 100;
/// Positions are uniform in [−5, 5]³.
const SCENE_RADIUS: f32 = 5.0;
const RNG_SEED: u64 = 0x42;
/// Each kitten gets one of these uniform scales at random.
const KITTEN_SCALES: [f32; 3] = [0.2, 1.0, 5.0];
/// PCG32 (XSH RR) with PCG32_INITIALIZER's increment.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            inc: 0xda3e_39cb_94b9_5bdb,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (1u64 << 32) as f64) as f32
    }
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("culling")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<FlyCamera>(scene)?;

        // Sun from normalize(1, 1, 1)
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .lights[0]
            .direction = (-Vector3f::ONE.normalize()).to_array();

        let mesh = engine.add_resource(Mesh::from_cooked_mesh_bytes(
            "kitten",
            include_bytes!("../res/models/kitten.cooked_mesh"),
        )?)?;
        let material = engine.add_resource(
            Material::new("kitten")
                .albedo(Color::ONE)
                .roughness(0.5)
                .metallic(0.0),
        )?;

        let mut rng = Pcg32::new(RNG_SEED);
        for _ in 0..OBJECT_COUNT {
            // Mesh index draw kept so the sequence is unchanged (one mesh here)
            let _mesh_index = rng.next_u32();
            let position = Vector3f::new(
                rng.next_f32() * SCENE_RADIUS * 2.0 - SCENE_RADIUS,
                rng.next_f32() * SCENE_RADIUS * 2.0 - SCENE_RADIUS,
                rng.next_f32() * SCENE_RADIUS * 2.0 - SCENE_RADIUS,
            );
            let scale = KITTEN_SCALES[rng.next_u32() as usize % KITTEN_SCALES.len()];
            let axis = Vector3f::new(
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            )
            .normalize();
            let angle = (rng.next_f32() * 90.0).to_radians();
            // Pill's rotation is Rx·Ry·Rz in degrees, glam's intrinsic XYZ
            let (x, y, z) = Quat::from_axis_angle(axis, angle).to_euler(EulerRot::XYZ);
            engine
                .build_entity(scene)
                .with_component(
                    TransformComponent::builder()
                        .position(position)
                        .rotation(Vector3f::new(x, y, z) * 180.0 / std::f32::consts::PI)
                        .scale(Vector3f::splat(scale))
                        .build(),
                )
                .with_component(
                    MeshRenderingComponent::builder()
                        .mesh(&mesh)
                        .material(&material)
                        .build(),
                )
                .build();
        }

        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().build())
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(70.0)
                    .range(0.1..100.0)
                    .build(),
            )
            .with_component(FlyCamera::looking_at(Vector3f::ZERO, -Vector3f::Z, false))
            .build();

        // Editor overview camera: never enabled (the engine renders only the game camera), outside
        // the cube looking at its centre
        let overview_position = Vector3f::new(12.0, 8.0, 12.0);
        let mut overview_transform = TransformComponent::builder()
            .position(overview_position)
            .build();
        overview_transform.look_at(Vector3f::ZERO);
        engine
            .build_entity(scene)
            .with_component(overview_transform)
            .with_component(
                CameraComponent::builder()
                    .enabled(false)
                    .fov(60.0)
                    .range(0.1..1000.0)
                    .build(),
            )
            .with_component(FlyCamera::looking_at(
                overview_position,
                Vector3f::ZERO,
                true,
            ))
            .build();

        engine.add_global_component(EditorState { visible: true })?;
        editor::set_editor_visible(engine, true)?;
        engine.add_render_pass(Box::new(EditorPass::new()))?;
        engine.add_system("editor_toggle", editor::editor_toggle_system)?;
        engine.add_system("game_camera_aspect", editor::game_camera_aspect_system)?;
        engine.add_system("fly_camera", editor::fly_camera_system)?;
        Ok(())
    }
}
