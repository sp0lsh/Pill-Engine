//! aras_p's ToyPathTracer scene and settings, from Cpp/Source/Test.cpp (`DO_BIG_SCENE`:
//! `s_Spheres`, `s_SphereMats`, 46 spheres, emissive #8 and #45) and Config.h.

use crate::fast::V3;

pub const K_FLAG_ANIMATE: u32 = 1 << 0;
pub const K_FLAG_PROGRESSIVE: u32 = 1 << 1;

/// Config.h
pub(crate) const DO_SAMPLES_PER_PIXEL: usize = 4;
pub(crate) const DO_ANIMATE_SMOOTHING: f32 = 0.9;

pub(crate) const K_MIN_T: f32 = 0.001;
pub(crate) const K_MAX_T: f32 = 1.0e7;
pub(crate) const K_MAX_DEPTH: i32 = 10;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum MaterialType {
    Lambert,
    Metal,
    Dielectric,
}

#[derive(Clone, Copy)]
pub(crate) struct Material {
    pub(crate) kind: MaterialType,
    pub(crate) albedo: V3,
    pub(crate) emissive: V3,
    pub(crate) roughness: f32,
    pub(crate) ri: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct Sphere {
    pub(crate) center: V3,
    pub(crate) radius: f32,
}

const fn s(x: f32, y: f32, z: f32, radius: f32) -> Sphere {
    Sphere {
        center: V3::new(x, y, z),
        radius,
    }
}

const fn m(
    kind: MaterialType,
    albedo: (f32, f32, f32),
    emissive: (f32, f32, f32),
    roughness: f32,
    ri: f32,
) -> Material {
    Material {
        kind,
        albedo: V3::new(albedo.0, albedo.1, albedo.2),
        emissive: V3::new(emissive.0, emissive.1, emissive.2),
        roughness,
        ri,
    }
}

use MaterialType::{Dielectric, Lambert, Metal};
const O: (f32, f32, f32) = (0.0, 0.0, 0.0);

// prettier-ignore
pub(crate) const S_SPHERES: [Sphere; 46] = [
    s(0.0, -100.5, -1.0, 100.0),
    s(2.0, 0.0, -1.0, 0.5),
    s(0.0, 0.0, -1.0, 0.5),
    s(-2.0, 0.0, -1.0, 0.5),
    s(2.0, 0.0, 1.0, 0.5),
    s(0.0, 0.0, 1.0, 0.5),
    s(-2.0, 0.0, 1.0, 0.5),
    s(0.5, 1.0, 0.5, 0.5),
    s(-1.5, 1.5, 0.0, 0.3),
    s(4.0, 0.0, -3.0, 0.5),
    s(3.0, 0.0, -3.0, 0.5),
    s(2.0, 0.0, -3.0, 0.5),
    s(1.0, 0.0, -3.0, 0.5),
    s(0.0, 0.0, -3.0, 0.5),
    s(-1.0, 0.0, -3.0, 0.5),
    s(-2.0, 0.0, -3.0, 0.5),
    s(-3.0, 0.0, -3.0, 0.5),
    s(-4.0, 0.0, -3.0, 0.5),
    s(4.0, 0.0, -4.0, 0.5),
    s(3.0, 0.0, -4.0, 0.5),
    s(2.0, 0.0, -4.0, 0.5),
    s(1.0, 0.0, -4.0, 0.5),
    s(0.0, 0.0, -4.0, 0.5),
    s(-1.0, 0.0, -4.0, 0.5),
    s(-2.0, 0.0, -4.0, 0.5),
    s(-3.0, 0.0, -4.0, 0.5),
    s(-4.0, 0.0, -4.0, 0.5),
    s(4.0, 0.0, -5.0, 0.5),
    s(3.0, 0.0, -5.0, 0.5),
    s(2.0, 0.0, -5.0, 0.5),
    s(1.0, 0.0, -5.0, 0.5),
    s(0.0, 0.0, -5.0, 0.5),
    s(-1.0, 0.0, -5.0, 0.5),
    s(-2.0, 0.0, -5.0, 0.5),
    s(-3.0, 0.0, -5.0, 0.5),
    s(-4.0, 0.0, -5.0, 0.5),
    s(4.0, 0.0, -6.0, 0.5),
    s(3.0, 0.0, -6.0, 0.5),
    s(2.0, 0.0, -6.0, 0.5),
    s(1.0, 0.0, -6.0, 0.5),
    s(0.0, 0.0, -6.0, 0.5),
    s(-1.0, 0.0, -6.0, 0.5),
    s(-2.0, 0.0, -6.0, 0.5),
    s(-3.0, 0.0, -6.0, 0.5),
    s(-4.0, 0.0, -6.0, 0.5),
    s(1.5, 1.5, -2.0, 0.3),
];

// prettier-ignore
pub(crate) const S_SPHERE_MATS: [Material; 46] = [
    m(Lambert, (0.8, 0.8, 0.8), O, 0.0, 0.0),
    m(Lambert, (0.8, 0.4, 0.4), O, 0.0, 0.0),
    m(Lambert, (0.4, 0.8, 0.4), O, 0.0, 0.0),
    m(Metal, (0.4, 0.4, 0.8), O, 0.0, 0.0),
    m(Metal, (0.4, 0.8, 0.4), O, 0.0, 0.0),
    m(Metal, (0.4, 0.8, 0.4), O, 0.2, 0.0),
    m(Metal, (0.4, 0.8, 0.4), O, 0.6, 0.0),
    m(Dielectric, (0.4, 0.4, 0.4), O, 0.0, 1.5),
    m(Lambert, (0.8, 0.6, 0.2), (30.0, 25.0, 15.0), 0.0, 0.0),
    m(Lambert, (0.1, 0.1, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.2, 0.2, 0.2), O, 0.0, 0.0),
    m(Lambert, (0.3, 0.3, 0.3), O, 0.0, 0.0),
    m(Lambert, (0.4, 0.4, 0.4), O, 0.0, 0.0),
    m(Lambert, (0.5, 0.5, 0.5), O, 0.0, 0.0),
    m(Lambert, (0.6, 0.6, 0.6), O, 0.0, 0.0),
    m(Lambert, (0.7, 0.7, 0.7), O, 0.0, 0.0),
    m(Lambert, (0.8, 0.8, 0.8), O, 0.0, 0.0),
    m(Lambert, (0.9, 0.9, 0.9), O, 0.0, 0.0),
    m(Metal, (0.1, 0.1, 0.1), O, 0.0, 0.0),
    m(Metal, (0.2, 0.2, 0.2), O, 0.0, 0.0),
    m(Metal, (0.3, 0.3, 0.3), O, 0.0, 0.0),
    m(Metal, (0.4, 0.4, 0.4), O, 0.0, 0.0),
    m(Metal, (0.5, 0.5, 0.5), O, 0.0, 0.0),
    m(Metal, (0.6, 0.6, 0.6), O, 0.0, 0.0),
    m(Metal, (0.7, 0.7, 0.7), O, 0.0, 0.0),
    m(Metal, (0.8, 0.8, 0.8), O, 0.0, 0.0),
    m(Metal, (0.9, 0.9, 0.9), O, 0.0, 0.0),
    m(Metal, (0.8, 0.1, 0.1), O, 0.0, 0.0),
    m(Metal, (0.8, 0.5, 0.1), O, 0.0, 0.0),
    m(Metal, (0.8, 0.8, 0.1), O, 0.0, 0.0),
    m(Metal, (0.4, 0.8, 0.1), O, 0.0, 0.0),
    m(Metal, (0.1, 0.8, 0.1), O, 0.0, 0.0),
    m(Metal, (0.1, 0.8, 0.5), O, 0.0, 0.0),
    m(Metal, (0.1, 0.8, 0.8), O, 0.0, 0.0),
    m(Metal, (0.1, 0.1, 0.8), O, 0.0, 0.0),
    m(Metal, (0.5, 0.1, 0.8), O, 0.0, 0.0),
    m(Lambert, (0.8, 0.1, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.8, 0.5, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.8, 0.8, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.4, 0.8, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.1, 0.8, 0.1), O, 0.0, 0.0),
    m(Lambert, (0.1, 0.8, 0.5), O, 0.0, 0.0),
    m(Lambert, (0.1, 0.8, 0.8), O, 0.0, 0.0),
    m(Lambert, (0.1, 0.1, 0.8), O, 0.0, 0.0),
    m(Metal, (0.5, 0.1, 0.8), O, 0.0, 0.0),
    m(Lambert, (0.1, 0.2, 0.5), (3.0, 10.0, 20.0), 0.0, 0.0),
];
