//! The book's `sphere.h`.

use crate::{
    hittable::{HitRecord, Hittable},
    interval::Interval,
    material::Material,
    ray::Ray,
    vec3::{dot, Point3},
};
use std::sync::Arc;

pub struct Sphere {
    center: Point3,
    radius: f64,
    mat: Arc<dyn Material>,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, mat: Arc<dyn Material>) -> Self {
        Self {
            center,
            radius: radius.max(0.0),
            mat,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let oc = self.center - r.origin();
        let a = r.direction().length_squared();
        let h = dot(r.direction(), oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let p = r.at(root);
        let mut rec = HitRecord {
            p,
            normal: Default::default(),
            mat: self.mat.as_ref(),
            t: root,
            front_face: false,
        };
        let outward_normal = (p - self.center) / self.radius;
        rec.set_face_normal(r, outward_normal);
        Some(rec)
    }
}
