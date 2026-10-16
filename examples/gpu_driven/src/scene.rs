//! The scene of bgfx's example 37-gpudrivenrendering (`gpudrivenrendering.cpp`): one unit cube
//! mesh instanced as 4 props — a ground plane, 25 occluder walls, 200 small cubes and 300 tall
//! boxes — at seeded random places in a 100 m square.

use glam::{Mat4, Quat, Vec3};

/// Unit cube centred on the origin: 8 corners, 12 triangles (counter-clockwise from outside).
pub const CUBE_VERTICES: [[f32; 3]; 8] = [
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
];
pub const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 2, 3, 0, // +z
    1, 5, 6, 6, 2, 1, // +x
    5, 4, 7, 7, 6, 5, // −z
    4, 0, 3, 3, 7, 4, // −x
    3, 2, 6, 6, 7, 3, // +y
    4, 5, 1, 1, 0, 4, // −y
];

/// Instance data on the GPU: world matrix (columns) and the prop it belongs to.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub world: [[f32; 4]; 4],
    pub prop: u32,
    pub _padding: [u32; 3],
}

/// World-space axis-aligned bounds of an instance.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Bounds {
    pub min: [f32; 4],
    pub max: [f32; 4],
}

pub struct Prop {
    /// Linear RGB
    pub color: [f32; 3],
    /// Drawn into the occluder depth buffer
    pub occluder: bool,
    /// Range of its instances in `Scene::instances`
    pub first: u32,
    pub count: u32,
}

pub struct Scene {
    pub props: Vec<Prop>,
    /// Ordered by prop (compaction keeps each prop's visible instances together)
    pub instances: Vec<Instance>,
    pub bounds: Vec<Bounds>,
}

/// Seeded random numbers (PCG32).
struct Random(u64);

impl Random {
    fn next(&mut self) -> f32 {
        let old = self.0;
        self.0 = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let value = xorshifted.rotate_right((old >> 59) as u32);
        value as f32 / u32::MAX as f32
    }

    /// Uniform in [low, high]
    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// World bounds of the unit cube under `world`, from all 8 corners.
pub fn cube_bounds(world: &Mat4) -> Bounds {
    let corners = CUBE_VERTICES.map(|corner| world.transform_point3(Vec3::from_array(corner)));
    let min = corners.iter().copied().reduce(Vec3::min).unwrap_or(Vec3::ZERO);
    let max = corners.iter().copied().reduce(Vec3::max).unwrap_or(Vec3::ZERO);
    Bounds {
        min: min.extend(0.0).to_array(),
        max: max.extend(0.0).to_array(),
    }
}

impl Scene {
    /// bgfx's layout, per prop (ground, walls, cubes, boxes): color, occluder, count, scale, y,
    /// random yaw.
    pub fn new(seed: u64) -> Self {
        let layout: [([f32; 3], bool, u32, Vec3, f32, bool); 4] = [
            ([0.0, 0.6, 0.0], false, 1, Vec3::new(100.0, 0.1, 100.0), 0.0, false),
            ([0.0, 0.0, 1.0], true, 25, Vec3::new(40.0, 10.0, 0.1), 5.0, true),
            ([1.0, 1.0, 0.0], false, 200, Vec3::splat(2.0), 1.0, false),
            ([1.0, 0.0, 0.0], false, 300, Vec3::new(2.0, 4.0, 2.0), 2.0, false),
        ];
        let mut random = Random(seed);
        let mut scene = Scene {
            props: Vec::new(),
            instances: Vec::new(),
            bounds: Vec::new(),
        };
        for (prop, (color, occluder, count, scale, y, yaw)) in layout.into_iter().enumerate() {
            scene.props.push(Prop {
                color,
                occluder,
                first: scene.instances.len() as u32,
                count,
            });
            for _ in 0..count {
                let position = if count == 1 {
                    Vec3::new(0.0, y, 0.0)
                } else {
                    Vec3::new(random.range(-50.0, 50.0), y, random.range(-50.0, 50.0))
                };
                let rotation = if yaw {
                    Quat::from_rotation_y(random.range(-1.0, 1.0) * std::f32::consts::FRAC_PI_3)
                } else {
                    Quat::IDENTITY
                };
                let world = Mat4::from_scale_rotation_translation(scale, rotation, position);
                scene.instances.push(Instance {
                    world: world.to_cols_array_2d(),
                    prop: prop as u32,
                    _padding: [0; 3],
                });
                scene.bounds.push(cube_bounds(&world));
            }
        }
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_four_props_526_instances_ordered_by_prop() {
        let scene = Scene::new(1);
        assert_eq!(scene.props.len(), 4);
        assert_eq!(scene.instances.len(), 526);
        assert_eq!(scene.bounds.len(), 526);
        for (index, prop) in scene.props.iter().enumerate() {
            let range = prop.first as usize..(prop.first + prop.count) as usize;
            assert!(scene.instances[range].iter().all(|instance| instance.prop == index as u32));
        }
    }

    #[test]
    fn new_same_seed_same_layout() {
        let (a, b) = (Scene::new(7), Scene::new(7));
        assert!(a
            .instances
            .iter()
            .zip(&b.instances)
            .all(|(x, y)| x.world == y.world));
    }

    #[test]
    fn cube_bounds_rotated_wall_contains_all_corners() {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(40.0, 10.0, 0.1),
            Quat::from_rotation_y(0.9),
            Vec3::new(3.0, 5.0, -7.0),
        );
        let bounds = cube_bounds(&world);
        let (min, max) = (Vec3::from_slice(&bounds.min), Vec3::from_slice(&bounds.max));
        for vertex in CUBE_VERTICES {
            let point = world.transform_point3(Vec3::from_array(vertex));
            assert!(point.cmpge(min - 1e-4).all() && point.cmple(max + 1e-4).all(), "{point}");
        }
        // Rotated, the 40 m wall reaches well past ±0.05 m in z
        assert!(max.z - min.z > 20.0);
    }
}
