//! The book's `hittable_list.h`.

use crate::{
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
};

/// Owns its objects; `Send + Sync` is required here, where the list is shared with the render
/// threads, not on the `Hittable` trait.
#[derive(Default)]
pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable + Send + Sync>>,
}

impl HittableList {
    pub fn add(&mut self, object: impl Hittable + Send + Sync + 'static) {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval) -> Option<HitRecord<'_>> {
        let mut closest: Option<HitRecord<'_>> = None;
        let mut closest_so_far = ray_t.max;

        for object in &self.objects {
            if let Some(rec) = object.hit(r, Interval::new(ray_t.min, closest_so_far)) {
                closest_so_far = rec.t;
                closest = Some(rec);
            }
        }

        closest
    }
}
