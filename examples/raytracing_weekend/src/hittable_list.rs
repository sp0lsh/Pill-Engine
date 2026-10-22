//! The book's `hittable_list.h`.

use crate::{
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
};
use std::sync::Arc;

#[derive(Default)]
pub struct HittableList {
    pub objects: Vec<Arc<dyn Hittable>>,
}

impl HittableList {
    pub fn add(&mut self, object: Arc<dyn Hittable>) {
        self.objects.push(object);
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
