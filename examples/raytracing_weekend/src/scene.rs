//! The book's `main.cc` final scene (section 14.1) and its camera settings.

use crate::{
    camera::Camera,
    color::Color,
    hittable_list::HittableList,
    material::{Dielectric, Lambertian, Metal},
    rtweekend::{random_double, random_double_range},
    sphere::Sphere,
    vec3::{Point3, Vec3},
};
use rand::Rng;

pub fn final_scene(rng: &mut impl Rng) -> HittableList {
    let mut world = HittableList::default();

    let ground_material = Lambertian::new(Color::new(0.5, 0.5, 0.5));
    world.add(Sphere::new(
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        ground_material,
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_double(rng);
            let center = Point3::new(
                a as f64 + 0.9 * random_double(rng),
                0.2,
                b as f64 + 0.9 * random_double(rng),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    let albedo = Color::random(rng) * Color::random(rng);
                    world.add(Sphere::new(center, 0.2, Lambertian::new(albedo)));
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = Color::random_range(rng, 0.5, 1.0);
                    let fuzz = random_double_range(rng, 0.0, 0.5);
                    world.add(Sphere::new(center, 0.2, Metal::new(albedo, fuzz)));
                } else {
                    // glass
                    world.add(Sphere::new(center, 0.2, Dielectric::new(1.5)));
                }
            }
        }
    }

    let material1 = Dielectric::new(1.5);
    world.add(Sphere::new(Point3::new(0.0, 1.0, 0.0), 1.0, material1));

    let material2 = Lambertian::new(Color::new(0.4, 0.2, 0.1));
    world.add(Sphere::new(Point3::new(-4.0, 1.0, 0.0), 1.0, material2));

    let material3 = Metal::new(Color::new(0.7, 0.6, 0.5), 0.0);
    world.add(Sphere::new(Point3::new(4.0, 1.0, 0.0), 1.0, material3));

    world
}

/// The final scene's camera; the book renders it at 1200 wide.
pub fn final_camera(image_width: usize, samples_per_pixel: u32) -> Camera {
    let mut cam = Camera::default();

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth = 50;

    cam.vfov = 20.0;
    cam.lookfrom = Point3::new(13.0, 2.0, 3.0);
    cam.lookat = Point3::new(0.0, 0.0, 0.0);
    cam.vup = Vec3::new(0.0, 1.0, 0.0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    cam.initialize();
    cam
}
