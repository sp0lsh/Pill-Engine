//! aras_p's ToyPathTracer workload (Cpp/Source/Test.cpp, Maths.cpp), optimized. Same scene,
//! camera, materials, light sampling, per-row RNG seeding, sample count, depth and ray counting
//! as his code; changed:
//! - Scalar `V3` instead of NEON `float3` (dot products need no lane extraction; LLVM keeps
//!   x/y/z in registers).
//! - Sphere test: 12 groups of 4, branchless, fused multiply-adds, `vsqrtq_f32` instead of
//!   reciprocal-sqrt + 3 Newton steps, and two independent accumulators for instruction-level
//!   parallelism. Padding lanes have r² = -1e30 so they can never be hit (his r² = 0 padding
//!   can, and then indexes past `s_SphereMats`).
//! - Shadow rays are any-hit occlusion tests up to the light's own distance, with an early exit,
//!   instead of a full closest-hit search compared against the light id.
//! - Recursion → loop; hit position and normal only for the closest hit; no `float3pack`
//!   round trips.
//! - Polynomial sincos instead of `cosf`/`sinf`; the metal fuzz sample is skipped for roughness
//!   0 (it is multiplied by 0).

use crate::scene::{
    Material, MaterialType, DO_ANIMATE_SMOOTHING, DO_SAMPLES_PER_PIXEL, K_FLAG_ANIMATE,
    K_FLAG_PROGRESSIVE, K_MAX_DEPTH, K_MAX_T, K_MIN_T, S_SPHERES, S_SPHERE_MATS,
};
use core::arch::aarch64::*;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::sync::{
    atomic::{AtomicI32, AtomicUsize, Ordering},
    Mutex,
};

const SPHERE_COUNT: usize = S_SPHERES.len();
const NO_HIT: usize = usize::MAX;
/// Rows per work item: 1 instead of his 4, so threads finish a frame closer together.
const RANGE_ROWS: usize = 1;

#[derive(Clone, Copy, Default, Debug)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl V3 {
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn dot(self, b: V3) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }

    #[inline(always)]
    pub fn cross(self, b: V3) -> V3 {
        V3::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }

    #[inline(always)]
    pub fn sq_length(self) -> f32 {
        self.dot(self)
    }

    #[inline(always)]
    pub fn normalize(self) -> V3 {
        self * (1.0 / self.sq_length().sqrt())
    }
}

impl Add for V3 {
    type Output = V3;
    #[inline(always)]
    fn add(self, b: V3) -> V3 {
        V3::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}

impl Sub for V3 {
    type Output = V3;
    #[inline(always)]
    fn sub(self, b: V3) -> V3 {
        V3::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}

impl Mul for V3 {
    type Output = V3;
    #[inline(always)]
    fn mul(self, b: V3) -> V3 {
        V3::new(self.x * b.x, self.y * b.y, self.z * b.z)
    }
}

impl Mul<f32> for V3 {
    type Output = V3;
    #[inline(always)]
    fn mul(self, b: f32) -> V3 {
        V3::new(self.x * b, self.y * b, self.z * b)
    }
}

impl Neg for V3 {
    type Output = V3;
    #[inline(always)]
    fn neg(self) -> V3 {
        V3::new(-self.x, -self.y, -self.z)
    }
}

impl AddAssign for V3 {
    #[inline(always)]
    fn add_assign(&mut self, b: V3) {
        *self = *self + b;
    }
}

impl MulAssign for V3 {
    #[inline(always)]
    fn mul_assign(&mut self, b: V3) {
        *self = *self * b;
    }
}

#[inline(always)]
fn xor_shift32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    *state = x;
    x
}

#[inline(always)]
fn random_float01(state: &mut u32) -> f32 {
    (xor_shift32(state) & 0xFF_FFFF) as f32 * (1.0 / 16_777_216.0)
}

/// sin and cos of 2π·t for t in [0, 1): folded to [-π/2, π/2], then Taylor to degree 11 / 10.
#[inline(always)]
fn sin_cos_turns(t: f32) -> (f32, f32) {
    // 2π(x + ½) = 2πx + π: sin and cos flip sign
    let x = t - 0.5;
    // Beyond a quarter turn: sin(π - a) = sin a, cos(π - a) = -cos a
    let fold = x.abs() > 0.25;
    let x = if fold { 0.5f32.copysign(x) - x } else { x };
    let b = x * std::f32::consts::TAU;
    let b2 = b * b;
    let sin = b
        * (1.0
            + b2 * (-1.0 / 6.0
                + b2 * (1.0 / 120.0
                    + b2 * (-1.0 / 5040.0 + b2 * (1.0 / 362_880.0 + b2 * (-1.0 / 39_916_800.0))))));
    let cos = 1.0
        + b2 * (-0.5
            + b2 * (1.0 / 24.0
                + b2 * (-1.0 / 720.0 + b2 * (1.0 / 40_320.0 + b2 * (-1.0 / 3_628_800.0)))));
    (-sin, if fold { cos } else { -cos })
}

#[inline(always)]
fn random_in_unit_disk(state: &mut u32) -> (f32, f32) {
    loop {
        let x = 2.0 * random_float01(state) - 1.0;
        let y = 2.0 * random_float01(state) - 1.0;
        if x * x + y * y < 1.0 {
            return (x, y);
        }
    }
}

#[inline(always)]
fn random_in_unit_sphere(state: &mut u32) -> V3 {
    loop {
        let p = V3::new(
            2.0 * random_float01(state) - 1.0,
            2.0 * random_float01(state) - 1.0,
            2.0 * random_float01(state) - 1.0,
        );
        if p.sq_length() < 1.0 {
            return p;
        }
    }
}

#[inline(always)]
fn random_unit_vector(state: &mut u32) -> V3 {
    let z = random_float01(state) * 2.0 - 1.0;
    let (sin, cos) = sin_cos_turns(random_float01(state));
    let r = (1.0 - z * z).sqrt();
    V3::new(r * cos, r * sin, z)
}

#[derive(Clone, Copy, Default)]
struct Camera {
    origin: V3,
    lower_left_corner: V3,
    horizontal: V3,
    vertical: V3,
    uu: V3,
    vv: V3,
    lens_radius: f32,
}

/// Up to 4 spatially close spheres, tested at once; padding lanes have r² = -1e30, so
/// c = |co|² + 1e30 and discr = nb² - c < 0 for every ray (never hit).
#[derive(Clone, Copy)]
struct Group {
    center_x: [f32; 4],
    center_y: [f32; 4],
    center_z: [f32; 4],
    sq_radius: [f32; 4],
    ids: [u32; 4],
}

const EMPTY_GROUP: Group = Group {
    center_x: [0.0; 4],
    center_y: [0.0; 4],
    center_z: [0.0; 4],
    sq_radius: [-1.0e30; 4],
    ids: [u32::MAX; 4],
};

/// Bounds of 4 groups as [min_x, max_x, min_y, max_y, min_z, max_z] lanes; unused slots hold an
/// inverted box (min +∞, max -∞) that every slab test rejects.
#[derive(Clone, Copy)]
struct Quad {
    bounds: [[f32; 4]; 6],
}

const EMPTY_QUAD: Quad = Quad {
    bounds: [
        [f32::INFINITY; 4],
        [f32::NEG_INFINITY; 4],
        [f32::INFINITY; 4],
        [f32::NEG_INFINITY; 4],
        [f32::INFINITY; 4],
        [f32::NEG_INFINITY; 4],
    ],
};

const MAX_GROUPS: usize = 16;
const MAX_QUADS: usize = MAX_GROUPS / 4;
/// Spheres with a radius above this multiple of the median radius (the ground) are tested on
/// their own for every ray instead of being grouped.
const LARGE_SPHERE_FACTOR: f32 = 10.0;

/// Test.cpp's file statics, laid out for the optimized tracer.
pub struct Test {
    groups: [Group; MAX_GROUPS],
    quads: [Quad; MAX_QUADS],
    quad_count: usize,
    large: Vec<usize>,
    centers: [V3; SPHERE_COUNT],
    radii: [f32; SPHERE_COUNT],
    inv_radii: [f32; SPHERE_COUNT],
    materials: [Material; SPHERE_COUNT],
    lights: Vec<usize>,
    cam: Camera,
}

/// A ray splatted for the 4-wide tests: origin, direction, 1/direction, -origin/direction (one
/// FMA per slab) and, per axis, which bound row is the near one.
struct RayLanes {
    ox: float32x4_t,
    oy: float32x4_t,
    oz: float32x4_t,
    dx: float32x4_t,
    dy: float32x4_t,
    dz: float32x4_t,
    inv: [float32x4_t; 3],
    neg_o_inv: [float32x4_t; 3],
    near: [usize; 3],
}

impl RayLanes {
    #[inline(always)]
    fn new(o: V3, d: V3) -> Self {
        unsafe {
            let o4 = vld1q_f32([o.x, o.y, o.z, 0.0].as_ptr());
            let d4 = vld1q_f32([d.x, d.y, d.z, 1.0].as_ptr());
            // |c| < 1e-20 → ±1e-20, so 0 · ∞ = NaN can't appear in the slab test; one vector
            // divide for all three axes
            let tiny = vdupq_n_f32(1e-20);
            let sign = vandq_u32(vreinterpretq_u32_f32(d4), vdupq_n_u32(0x8000_0000));
            let tiny_signed = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(tiny), sign));
            let safe = vbslq_f32(vcltq_f32(vabsq_f32(d4), tiny), tiny_signed, d4);
            let inv4 = vdivq_f32(vdupq_n_f32(1.0), safe);
            let neg_o_inv4 = vnegq_f32(vmulq_f32(o4, inv4));
            let negative = vcltq_f32(inv4, vdupq_n_f32(0.0));
            let near = [
                vgetq_lane_u32::<0>(negative) as usize & 1,
                2 + (vgetq_lane_u32::<1>(negative) as usize & 1),
                4 + (vgetq_lane_u32::<2>(negative) as usize & 1),
            ];
            Self {
                ox: vdupq_laneq_f32::<0>(o4),
                oy: vdupq_laneq_f32::<1>(o4),
                oz: vdupq_laneq_f32::<2>(o4),
                dx: vdupq_laneq_f32::<0>(d4),
                dy: vdupq_laneq_f32::<1>(d4),
                dz: vdupq_laneq_f32::<2>(d4),
                inv: [
                    vdupq_laneq_f32::<0>(inv4),
                    vdupq_laneq_f32::<1>(inv4),
                    vdupq_laneq_f32::<2>(inv4),
                ],
                neg_o_inv: [
                    vdupq_laneq_f32::<0>(neg_o_inv4),
                    vdupq_laneq_f32::<1>(neg_o_inv4),
                    vdupq_laneq_f32::<2>(neg_o_inv4),
                ],
                near,
            }
        }
    }
}

impl Test {
    pub fn new() -> Self {
        let materials = S_SPHERE_MATS;
        let radii = S_SPHERES.map(|s| s.radius);
        let mut sorted = radii;
        sorted.sort_by(f32::total_cmp);
        let median = sorted[SPHERE_COUNT / 2];
        let mut test = Self {
            groups: [EMPTY_GROUP; MAX_GROUPS],
            quads: [EMPTY_QUAD; MAX_QUADS],
            quad_count: 0,
            large: (0..SPHERE_COUNT)
                .filter(|&i| radii[i] > LARGE_SPHERE_FACTOR * median)
                .collect(),
            centers: S_SPHERES.map(|s| s.center),
            radii,
            inv_radii: radii.map(|r| 1.0 / r),
            materials,
            lights: (0..SPHERE_COUNT)
                .filter(|&i| {
                    let e = materials[i].emissive;
                    e.x > 0.0 || e.y > 0.0 || e.z > 0.0
                })
                .collect(),
            cam: Camera::default(),
        };
        test.build_groups();
        test
    }

    /// Clusters the non-large spheres into groups of up to 4 (median splits along the widest
    /// axis, left side a multiple of 4) and their bounds into quads. Run after spheres move.
    fn build_groups(&mut self) {
        fn split(ids: &mut [usize], centers: &[V3; SPHERE_COUNT], out: &mut Vec<Vec<usize>>) {
            if ids.len() <= 4 {
                out.push(ids.to_vec());
                return;
            }
            let (mut lo, mut hi) = (
                V3::new(f32::MAX, f32::MAX, f32::MAX),
                V3::new(f32::MIN, f32::MIN, f32::MIN),
            );
            for &i in ids.iter() {
                let c = centers[i];
                lo = V3::new(lo.x.min(c.x), lo.y.min(c.y), lo.z.min(c.z));
                hi = V3::new(hi.x.max(c.x), hi.y.max(c.y), hi.z.max(c.z));
            }
            let extent = hi - lo;
            let key = |i: usize| {
                let c = centers[i];
                if extent.x >= extent.y && extent.x >= extent.z {
                    c.x
                } else if extent.y >= extent.z {
                    c.y
                } else {
                    c.z
                }
            };
            ids.sort_by(|&a, &b| key(a).total_cmp(&key(b)));
            let left = 4 * ids.len().div_ceil(8);
            let (a, b) = ids.split_at_mut(left);
            split(a, centers, out);
            split(b, centers, out);
        }

        let mut ids: Vec<usize> = (0..SPHERE_COUNT)
            .filter(|i| !self.large.contains(i))
            .collect();
        let mut clusters = Vec::new();
        split(&mut ids, &self.centers, &mut clusters);
        assert!(clusters.len() <= MAX_GROUPS, "{} groups", clusters.len());

        self.groups = [EMPTY_GROUP; MAX_GROUPS];
        self.quads = [EMPTY_QUAD; MAX_QUADS];
        for (g, cluster) in clusters.iter().enumerate() {
            let group = &mut self.groups[g];
            let quad = &mut self.quads[g / 4];
            let slot = g % 4;
            for (lane, &i) in cluster.iter().enumerate() {
                let c = self.centers[i];
                let r = self.radii[i];
                group.center_x[lane] = c.x;
                group.center_y[lane] = c.y;
                group.center_z[lane] = c.z;
                group.sq_radius[lane] = r * r;
                group.ids[lane] = i as u32;
                for (axis, value) in [c.x, c.y, c.z].into_iter().enumerate() {
                    quad.bounds[2 * axis][slot] = quad.bounds[2 * axis][slot].min(value - r);
                    quad.bounds[2 * axis + 1][slot] =
                        quad.bounds[2 * axis + 1][slot].max(value + r);
                }
            }
        }
        self.quad_count = clusters.len().div_ceil(4);
    }

    fn set_center(&mut self, i: usize, c: V3) {
        self.centers[i] = c;
    }

    /// One group's hit distances: t in (K_MIN_T, ∞) where hit, else +∞.
    #[inline(always)]
    fn group_t(&self, g: usize, ray: &RayLanes) -> float32x4_t {
        unsafe {
            let group = self.groups.get_unchecked(g);
            let cox = vsubq_f32(vld1q_f32(group.center_x.as_ptr()), ray.ox);
            let coy = vsubq_f32(vld1q_f32(group.center_y.as_ptr()), ray.oy);
            let coz = vsubq_f32(vld1q_f32(group.center_z.as_ptr()), ray.oz);
            let nb = vfmaq_f32(vfmaq_f32(vmulq_f32(cox, ray.dx), coy, ray.dy), coz, ray.dz);
            let co2 = vfmaq_f32(vfmaq_f32(vmulq_f32(cox, cox), coy, coy), coz, coz);
            let c = vsubq_f32(co2, vld1q_f32(group.sq_radius.as_ptr()));
            let discr = vnegq_f32(vfmsq_f32(c, nb, nb)); // nb² - c
            let sq = vsqrtq_f32(vmaxq_f32(discr, vdupq_n_f32(0.0)));
            let t_min = vdupq_n_f32(K_MIN_T);
            let t0 = vsubq_f32(nb, sq);
            let t1 = vaddq_f32(nb, sq);
            let t = vbslq_f32(vcgtq_f32(t0, t_min), t0, t1);
            let valid = vandq_u32(vcgtq_f32(discr, vdupq_n_f32(0.0)), vcgtq_f32(t, t_min));
            vbslq_f32(valid, t, vdupq_n_f32(f32::INFINITY))
        }
    }

    /// Bit i set when the ray enters group box i of `quad` within (K_MIN_T, t_max).
    #[inline(always)]
    fn quad_mask(&self, q: usize, ray: &RayLanes, t_max: f32) -> u32 {
        unsafe {
            let quad = self.quads.get_unchecked(q);
            let slab = |row: usize, axis: usize| {
                vfmaq_f32(
                    ray.neg_o_inv[axis],
                    vld1q_f32(quad.bounds[row].as_ptr()),
                    ray.inv[axis],
                )
            };
            let [nx, ny, nz] = ray.near;
            let t_near = vmaxq_f32(
                vmaxq_f32(slab(nx, 0), slab(ny, 1)),
                vmaxq_f32(slab(nz, 2), vdupq_n_f32(K_MIN_T)),
            );
            let t_far = vminq_f32(
                vminq_f32(slab(nx ^ 1, 0), slab(ny ^ 1, 1)),
                vminq_f32(slab(nz ^ 1, 2), vdupq_n_f32(t_max)),
            );
            const WEIGHTS: [u32; 4] = [1, 2, 4, 8];
            vaddvq_u32(vandq_u32(
                vcleq_f32(t_near, t_far),
                vld1q_u32(WEIGHTS.as_ptr()),
            ))
        }
    }

    /// Closest hit with t < t_max: (t, sphere index) or (t_max, NO_HIT).
    #[inline(always)]
    fn closest_hit(&self, o: V3, d: V3, t_max: f32) -> (f32, usize) {
        let mut best_t = t_max;
        let mut best_id = NO_HIT;
        for &i in &self.large {
            if let Some(t) = self.sphere_t(i, o, d) {
                if t < best_t {
                    best_t = t;
                    best_id = i;
                }
            }
        }
        let ray = RayLanes::new(o, d);
        unsafe {
            for q in 0..self.quad_count {
                let mut mask = self.quad_mask(q, &ray, best_t);
                while mask != 0 {
                    let g = q * 4 + mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let t = self.group_t(g, &ray);
                    let min_t = vminvq_f32(t);
                    if min_t < best_t {
                        best_t = min_t;
                        let lane = vaddvq_u32(vandq_u32(
                            vceqq_f32(t, vdupq_n_f32(min_t)),
                            vld1q_u32([1u32, 2, 4, 8].as_ptr()),
                        ))
                        .trailing_zeros();
                        best_id = self.groups.get_unchecked(g).ids[lane as usize] as usize;
                    }
                }
            }
        }
        (best_t, best_id)
    }

    /// True if a sphere other than `light` is hit before `t_light` (the light's own distance).
    #[inline(always)]
    fn occluded(&self, o: V3, d: V3, light: usize, t_light: f32) -> bool {
        for &i in &self.large {
            if i != light && self.sphere_t(i, o, d).is_some_and(|t| t < t_light) {
                return true;
            }
        }
        let ray = RayLanes::new(o, d);
        unsafe {
            let t_limit = vdupq_n_f32(t_light);
            let light = vdupq_n_u32(light as u32);
            for q in 0..self.quad_count {
                let mut mask = self.quad_mask(q, &ray, t_light);
                while mask != 0 {
                    let g = q * 4 + mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let t = self.group_t(g, &ray);
                    let ids = vld1q_u32(self.groups.get_unchecked(g).ids.as_ptr());
                    let blocker =
                        vandq_u32(vcltq_f32(t, t_limit), vmvnq_u32(vceqq_u32(ids, light)));
                    if vmaxvq_u32(blocker) != 0 {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// One sphere's ray/sphere test (ToyPathTracer's scalar `HitSpheres` path).
    #[inline(always)]
    fn sphere_t(&self, i: usize, o: V3, d: V3) -> Option<f32> {
        let co = self.centers[i] - o;
        let nb = co.dot(d);
        let discr = nb * nb - (co.sq_length() - self.radii[i] * self.radii[i]);
        if discr <= 0.0 {
            return None;
        }
        let sq = discr.sqrt();
        let mut t = nb - sq;
        if t <= K_MIN_T {
            t = nb + sq;
        }
        (t > K_MIN_T && t < K_MAX_T).then_some(t)
    }

    /// Test.cpp `Trace` + `Scatter` as a loop.
    #[inline(always)]
    fn trace(&self, mut o: V3, mut d: V3, ray_count: &mut i32, state: &mut u32) -> V3 {
        let mut color = V3::default();
        let mut throughput = V3::new(1.0, 1.0, 1.0);
        let mut do_material_e = true;
        let mut depth = 0;
        loop {
            *ray_count += 1;
            let (t, id) = self.closest_hit(o, d, K_MAX_T);
            if id == NO_HIT {
                // sky
                let t = 0.5 * (d.y + 1.0);
                let sky = (V3::new(1.0, 1.0, 1.0) * (1.0 - t) + V3::new(0.5, 0.7, 1.0) * t) * 0.3;
                return color + throughput * sky;
            }
            let pos = o + d * t;
            let normal = (pos - self.centers[id]) * self.inv_radii[id];
            let mat = &self.materials[id];
            let mut mat_e = mat.emissive;
            if depth >= K_MAX_DEPTH {
                return color + throughput * mat_e;
            }

            let (attenuation, scattered, light_e) = match mat.kind {
                MaterialType::Lambert => {
                    let scattered = (normal + random_unit_vector(state)).normalize();
                    let mut light_e = V3::default();
                    for &i in &self.lights {
                        if i == id {
                            continue; // skip self
                        }
                        let sc = self.centers[i];
                        let sw = (sc - pos).normalize();
                        let axis = if sw.x.abs() > 0.01 {
                            V3::new(0.0, 1.0, 0.0)
                        } else {
                            V3::new(1.0, 0.0, 0.0)
                        };
                        let su = axis.cross(sw).normalize();
                        let sv = sw.cross(su);
                        // sample sphere by solid angle
                        let r = self.radii[i];
                        let cos_a_max = (1.0 - r * r / (pos - sc).sq_length()).sqrt();
                        let eps1 = random_float01(state);
                        let eps2 = random_float01(state);
                        let cos_a = 1.0 - eps1 + eps1 * cos_a_max;
                        let sin_a = (1.0 - cos_a * cos_a).sqrt();
                        let (sin_phi, cos_phi) = sin_cos_turns(eps2);
                        let l = su * (cos_phi * sin_a) + sv * (sin_phi * sin_a) + sw * cos_a;

                        // shoot shadow ray
                        *ray_count += 1;
                        let Some(t_light) = self.sphere_t(i, pos, l) else {
                            continue;
                        };
                        if self.occluded(pos, l, i, t_light) {
                            continue;
                        }
                        let omega = 2.0 * std::f32::consts::PI * (1.0 - cos_a_max);
                        let nl = if normal.dot(d) < 0.0 { normal } else { -normal };
                        light_e += (mat.albedo * self.materials[i].emissive)
                            * (l.dot(nl).max(0.0) * omega / std::f32::consts::PI);
                    }
                    (mat.albedo, scattered, light_e)
                }
                MaterialType::Metal => {
                    let refl = d - normal * (2.0 * d.dot(normal));
                    let scattered = if mat.roughness == 0.0 {
                        refl.normalize()
                    } else {
                        (refl + random_in_unit_sphere(state) * mat.roughness).normalize()
                    };
                    if scattered.dot(normal) <= 0.0 {
                        return color + throughput * mat_e;
                    }
                    (mat.albedo, scattered, V3::default())
                }
                MaterialType::Dielectric => {
                    let refl = d - normal * (2.0 * d.dot(normal));
                    let dn = d.dot(normal);
                    let (outward_n, nint, cosine) = if dn > 0.0 {
                        (-normal, mat.ri, mat.ri * dn)
                    } else {
                        (normal, 1.0 / mat.ri, -dn)
                    };
                    let dt = d.dot(outward_n);
                    let discr = 1.0 - nint * nint * (1.0 - dt * dt);
                    let refl_prob = if discr > 0.0 {
                        let r0 = (1.0 - mat.ri) / (1.0 + mat.ri);
                        let r0 = r0 * r0;
                        let x = 1.0 - cosine;
                        let x2 = x * x;
                        r0 + (1.0 - r0) * (x2 * x2 * x)
                    } else {
                        1.0
                    };
                    let scattered = if random_float01(state) < refl_prob {
                        refl.normalize()
                    } else {
                        ((d - outward_n * dt) * nint - outward_n * discr.sqrt()).normalize()
                    };
                    (V3::new(1.0, 1.0, 1.0), scattered, V3::default())
                }
            };

            if !do_material_e {
                mat_e = V3::default(); // don't add material emission if told so
            }
            do_material_e = mat.kind != MaterialType::Lambert;
            color += throughput * (mat_e + light_e);
            throughput *= attenuation;
            o = pos;
            d = scattered;
            depth += 1;
        }
    }

    #[inline(always)]
    fn get_ray(&self, s: f32, t: f32, state: &mut u32) -> (V3, V3) {
        let cam = &self.cam;
        let (rx, ry) = random_in_unit_disk(state);
        let offset = cam.uu * (rx * cam.lens_radius) + cam.vv * (ry * cam.lens_radius);
        let o = cam.origin + offset;
        let d = (cam.lower_left_corner + cam.horizontal * s + cam.vertical * t - o).normalize();
        (o, d)
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_row_job(
        &self,
        start: usize,
        end: usize,
        frame_count: i32,
        screen_width: usize,
        screen_height: usize,
        backbuffer: &mut [f32],
        test_flags: u32,
    ) -> i32 {
        let inv_width = 1.0 / screen_width as f32;
        let inv_height = 1.0 / screen_height as f32;
        let mut lerp_fac = frame_count as f32 / (frame_count + 1) as f32;
        if test_flags & K_FLAG_ANIMATE != 0 {
            lerp_fac *= DO_ANIMATE_SMOOTHING;
        }
        if test_flags & K_FLAG_PROGRESSIVE == 0 {
            lerp_fac = 0.0;
        }
        let mut ray_count = 0;
        for (row, y) in (start..end).enumerate() {
            let mut state = (y as u32)
                .wrapping_mul(9781)
                .wrapping_add((frame_count as u32).wrapping_mul(6271))
                | 1;
            let row_pixels = &mut backbuffer[row * screen_width * 4..][..screen_width * 4];
            for (x, pixel) in row_pixels.chunks_exact_mut(4).enumerate() {
                let mut col = V3::default();
                for _ in 0..DO_SAMPLES_PER_PIXEL {
                    let u = (x as f32 + random_float01(&mut state)) * inv_width;
                    let v = (y as f32 + random_float01(&mut state)) * inv_height;
                    let (o, d) = self.get_ray(u, v, &mut state);
                    col += self.trace(o, d, &mut ray_count, &mut state);
                }
                col = col * (1.0 / DO_SAMPLES_PER_PIXEL as f32);
                let prev = V3::new(pixel[0], pixel[1], pixel[2]);
                col = prev * lerp_fac + col * (1.0 - lerp_fac);
                pixel[0] = col.x;
                pixel[1] = col.y;
                pixel[2] = col.z;
            }
        }
        ray_count
    }

    /// Test.cpp `UpdateTest`.
    pub fn update_test(
        &mut self,
        time: f32,
        _frame_count: i32,
        screen_width: usize,
        screen_height: usize,
        test_flags: u32,
    ) {
        if test_flags & K_FLAG_ANIMATE != 0 {
            let c1 = self.centers[1];
            self.set_center(1, V3::new(c1.x, time.cos() + 1.0, c1.z));
            let c8 = self.centers[8];
            self.set_center(8, V3::new(c8.x, c8.y, time.sin() * 0.3));
            self.build_groups();
        }
        let look_from = V3::new(0.0, 2.0, 3.0);
        let look_at = V3::new(0.0, 0.0, 0.0);
        let focus_dist = 3.0;
        let aperture = 0.1 * 0.2; // DO_BIG_SCENE
        let aspect = screen_width as f32 / screen_height as f32;
        let theta = 60.0f32 * 3.141_592_6 / 180.0;
        let half_height = (theta / 2.0).tan();
        let half_width = aspect * half_height;
        let w = (look_from - look_at).normalize();
        let u = V3::new(0.0, 1.0, 0.0).cross(w).normalize();
        let v = w.cross(u);
        self.cam = Camera {
            origin: look_from,
            lower_left_corner: look_from
                - u * (half_width * focus_dist)
                - v * (half_height * focus_dist)
                - w * focus_dist,
            horizontal: u * (2.0 * half_width * focus_dist),
            vertical: v * (2.0 * half_height * focus_dist),
            uu: u,
            vv: v,
            lens_radius: aperture / 2.0,
        };
    }

    /// Test.cpp `DrawTest`: one frame into `backbuffer` (RGBA f32, row 0 at the bottom) on
    /// `thread_count` threads taking RANGE_ROWS-row ranges; returns the ray count.
    pub fn draw_test(
        &self,
        frame_count: i32,
        screen_width: usize,
        screen_height: usize,
        backbuffer: &mut [f32],
        test_flags: u32,
        thread_count: usize,
    ) -> i32 {
        let ray_count = AtomicI32::new(0);
        let ranges: Vec<Mutex<(usize, &mut [f32])>> = backbuffer
            .chunks_mut(RANGE_ROWS * screen_width * 4)
            .enumerate()
            .map(|(i, rows)| Mutex::new((i * RANGE_ROWS, rows)))
            .collect();
        let next_range = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..thread_count {
                scope.spawn(|| loop {
                    let index = next_range.fetch_add(1, Ordering::Relaxed);
                    let Some(range) = ranges.get(index) else {
                        break;
                    };
                    let mut range = range.lock().unwrap();
                    let start = range.0;
                    let end = (start + RANGE_ROWS).min(screen_height);
                    let rays = self.trace_row_job(
                        start,
                        end,
                        frame_count,
                        screen_width,
                        screen_height,
                        range.1,
                        test_flags,
                    );
                    ray_count.fetch_add(rays, Ordering::Relaxed);
                });
            }
        });
        ray_count.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_cos_turns_matches_std_within_1e6() {
        for i in 0..10_000 {
            let t = i as f32 / 10_000.0;
            let (sin, cos) = sin_cos_turns(t);
            let (expected_sin, expected_cos) = (t * std::f32::consts::TAU).sin_cos();
            assert!(
                (sin - expected_sin).abs() < 1e-6 && (cos - expected_cos).abs() < 1e-6,
                "t={t}"
            );
        }
    }

    #[test]
    fn scene_46_spheres_lights_8_and_45_ground_separate() {
        let test = Test::new();
        assert_eq!(test.lights, vec![8, 45]);
        assert_eq!(test.large, vec![0]);
        assert_eq!(
            test.groups.iter().filter(|g| g.ids[0] != u32::MAX).count(),
            12
        );
    }
}
