//! The book's `camera.h`. `render` is split: the game's workers loop over pixels and call
//! `pixel_sample`, which is one `get_ray` + `ray_color` of the book's sample loop. The only
//! addition is `rays`, the count of traced ray segments for the Mrays/s readout.

use crate::{
    color::Color,
    hittable::Hittable,
    interval::Interval,
    ray::Ray,
    rtweekend::{degrees_to_radians, random_double},
    vec3::{cross, random_in_unit_disk, unit_vector, Point3, Vec3},
};

pub struct Camera {
    pub aspect_ratio: f64,      // Ratio of image width over height
    pub image_width: usize,     // Rendered image width in pixel count
    pub samples_per_pixel: u32, // Count of random samples for each pixel
    pub max_depth: u32,         // Maximum number of ray bounces into scene

    pub vfov: f64,        // Vertical view angle (field of view)
    pub lookfrom: Point3, // Point camera is looking from
    pub lookat: Point3,   // Point camera is looking at
    pub vup: Vec3,        // Camera-relative "up" direction

    pub defocus_angle: f64, // Variation angle of rays through each pixel
    pub focus_dist: f64,    // Distance from camera lookfrom point to plane of perfect focus

    pub image_height: usize, // Rendered image height
    center: Point3,          // Camera center
    pixel00_loc: Point3,     // Location of pixel 0, 0
    pixel_delta_u: Vec3,     // Offset to pixel to the right
    pixel_delta_v: Vec3,     // Offset to pixel below
    defocus_disk_u: Vec3,    // Defocus disk horizontal radius
    defocus_disk_v: Vec3,    // Defocus disk vertical radius
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            aspect_ratio: 1.0,
            image_width: 100,
            samples_per_pixel: 10,
            max_depth: 10,
            vfov: 90.0,
            lookfrom: Point3::new(0.0, 0.0, 0.0),
            lookat: Point3::new(0.0, 0.0, -1.0),
            vup: Vec3::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_dist: 10.0,
            image_height: 0,
            center: Default::default(),
            pixel00_loc: Default::default(),
            pixel_delta_u: Default::default(),
            pixel_delta_v: Default::default(),
            defocus_disk_u: Default::default(),
            defocus_disk_v: Default::default(),
        }
    }
}

impl Camera {
    pub fn initialize(&mut self) {
        self.image_height = ((self.image_width as f64 / self.aspect_ratio) as usize).max(1);

        self.center = self.lookfrom;

        // Determine viewport dimensions.
        let theta = degrees_to_radians(self.vfov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        let w = unit_vector(self.lookfrom - self.lookat);
        let u = unit_vector(cross(self.vup, w));
        let v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = viewport_width * u; // Vector across viewport horizontal edge
        let viewport_v = viewport_height * -v; // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left =
            self.center - (self.focus_dist * w) - viewport_u / 2.0 - viewport_v / 2.0;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        let defocus_radius = self.focus_dist * degrees_to_radians(self.defocus_angle / 2.0).tan();
        self.defocus_disk_u = u * defocus_radius;
        self.defocus_disk_v = v * defocus_radius;
    }

    /// One sample of pixel (i, j): the body of the book's per-sample loop.
    pub fn pixel_sample(&self, i: usize, j: usize, world: &dyn Hittable, rays: &mut u64) -> Color {
        let r = self.get_ray(i, j);
        Self::ray_color(&r, self.max_depth, world, rays)
    }

    /// Construct a camera ray originating from the defocus disk and directed at a randomly
    /// sampled point around the pixel location i, j.
    fn get_ray(&self, i: usize, j: usize) -> Ray {
        let offset = Self::sample_square();
        let pixel_sample = self.pixel00_loc
            + ((i as f64 + offset.x()) * self.pixel_delta_u)
            + ((j as f64 + offset.y()) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.defocus_disk_sample()
        };
        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    /// Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    fn sample_square() -> Vec3 {
        Vec3::new(random_double() - 0.5, random_double() - 0.5, 0.0)
    }

    /// Returns a random point in the camera defocus disk.
    fn defocus_disk_sample(&self) -> Point3 {
        let p = random_in_unit_disk();
        self.center + (p.x() * self.defocus_disk_u) + (p.y() * self.defocus_disk_v)
    }

    fn ray_color(r: &Ray, depth: u32, world: &dyn Hittable, rays: &mut u64) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth == 0 {
            return Color::new(0.0, 0.0, 0.0);
        }
        *rays += 1;

        if let Some(rec) = world.hit(r, Interval::new(0.001, f64::INFINITY)) {
            if let Some((attenuation, scattered)) = rec.mat.scatter(r, &rec) {
                return attenuation * Self::ray_color(&scattered, depth - 1, world, rays);
            }
            return Color::new(0.0, 0.0, 0.0);
        }

        let unit_direction = unit_vector(r.direction());
        let a = 0.5 * (unit_direction.y() + 1.0);
        (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
    }
}
