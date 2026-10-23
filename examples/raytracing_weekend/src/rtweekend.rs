//! The book's `rtweekend.h`: constants and `random_double`. The book's global `std::rand()` is
//! the `rand` crate, passed in explicitly as `&mut dyn RngCore`.

use rand::{Rng, RngCore};

pub const PI: f64 = std::f64::consts::PI;

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// A random real in [0, 1).
pub fn random_double(rng: &mut dyn RngCore) -> f64 {
    rng.random()
}

/// A random real in [min, max).
pub fn random_double_range(rng: &mut dyn RngCore, min: f64, max: f64) -> f64 {
    min + (max - min) * random_double(rng)
}
