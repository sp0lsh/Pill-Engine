//! The book's `hittable.h`.

use crate::{
    interval::Interval,
    material::Material,
    ray::Ray,
    vec3::{dot, Point3, Vec3},
};

pub struct HitRecord<'a> {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: &'a dyn Material,
    pub t: f64,
    pub front_face: bool,
}

impl HitRecord<'_> {
    /// Sets the hit record normal vector.
    /// NOTE: the parameter `outward_normal` is assumed to have unit length.
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3) {
        self.front_face = dot(r.direction(), outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>>;
}
