//! The book's `color.h`; `write_color` returns the sRGB8 bytes instead of printing PPM.

use crate::{interval::Interval, vec3::Vec3};

pub type Color = Vec3;

pub fn linear_to_gamma(linear_component: f64) -> f64 {
    if linear_component > 0.0 {
        linear_component.sqrt()
    } else {
        0.0
    }
}

pub fn write_color(pixel_color: Color) -> [u8; 3] {
    let r = linear_to_gamma(pixel_color.x());
    let g = linear_to_gamma(pixel_color.y());
    let b = linear_to_gamma(pixel_color.z());

    // Translate the [0,1] component values to the byte range [0,255]
    let intensity = Interval::new(0.000, 0.999);
    [
        (256.0 * intensity.clamp(r)) as u8,
        (256.0 * intensity.clamp(g)) as u8,
        (256.0 * intensity.clamp(b)) as u8,
    ]
}
