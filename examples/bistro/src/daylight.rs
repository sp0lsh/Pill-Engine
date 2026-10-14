//! Day-light cycle from one float, the hour: sun direction, color and intensity and the sky's
//! intensity, interpolated between keyframes.
//!
//! Sun positions: Rome (41.90° N, 12.50° E) on the spring equinox (20 March, UTC+1), NOAA solar
//! position approximation; the scene's −Z is taken as north. Sun colors: blackbody in linear sRGB
//! (2500 K low sun, 5500 K midday).

use pill_engine::game::Vector3f;

/// Sun intensity of a 45° sun (the panel's former default daylight).
const SUN_INTENSITY: f32 = 4.0;

const LOW_SUN: [f32; 3] = [1.0, 0.38, 0.07]; // 2500 K
const MIDDAY_SUN: [f32; 3] = [1.0, 0.85, 0.76]; // 5500 K

struct Keyframe {
    hour: f32,
    /// Degrees clockwise from north (−Z), and above the horizon.
    azimuth: f32,
    elevation: f32,
    color: [f32; 3],
    /// Image-based light scale (the HDR is a daytime sky; at night only its brightness drops).
    sky: f32,
}

#[rustfmt::skip]
const KEYFRAMES: [Keyframe; 7] = [
    Keyframe { hour: 0.0,  azimuth: 353.5, elevation: -47.8, color: LOW_SUN,    sky: 0.02 },
    Keyframe { hour: 3.0,  azimuth: 52.2,  elevation: -34.6, color: LOW_SUN,    sky: 0.02 },
    // Civil twilight: sky only
    Keyframe { hour: 6.0,  azimuth: 87.2,  elevation: -3.4,  color: LOW_SUN,    sky: 0.15 },
    Keyframe { hour: 12.0, azimuth: 173.5, elevation: 47.9,  color: MIDDAY_SUN, sky: 1.0 },
    Keyframe { hour: 18.0, azimuth: 267.1, elevation: 3.3,   color: LOW_SUN,    sky: 0.4 },
    Keyframe { hour: 20.0, azimuth: 287.8, elevation: -18.7, color: LOW_SUN,    sky: 0.02 },
    Keyframe { hour: 24.0, azimuth: 353.5, elevation: -47.8, color: LOW_SUN,    sky: 0.02 },
];

pub struct Daylight {
    /// Unit vector toward the sun.
    pub to_sun: Vector3f,
    pub elevation: f32,
    pub color: [f32; 3],
    /// Direct sunlight; 0 with the sun below the horizon.
    pub intensity: f32,
    pub sky: f32,
}

/// Light at `hour` (0–24, wrapping).
pub fn at(hour: f32) -> Daylight {
    let hour = hour.rem_euclid(24.0);
    let next = KEYFRAMES
        .iter()
        .position(|key| key.hour > hour)
        .unwrap_or(KEYFRAMES.len() - 1);
    let (a, b) = (&KEYFRAMES[next - 1], &KEYFRAMES[next]);
    let t = (hour - a.hour) / (b.hour - a.hour);
    let lerp = |x: f32, y: f32| x + (y - x) * t;
    // Shortest way around for the azimuth
    let azimuth_step = (b.azimuth - a.azimuth + 540.0).rem_euclid(360.0) - 180.0;
    let azimuth = (a.azimuth + azimuth_step * t).to_radians();
    let elevation = lerp(a.elevation, b.elevation);
    let elevation_radians = elevation.to_radians();
    Daylight {
        to_sun: Vector3f::new(
            azimuth.sin() * elevation_radians.cos(),
            elevation_radians.sin(),
            -azimuth.cos() * elevation_radians.cos(),
        ),
        elevation,
        color: [0, 1, 2].map(|channel| lerp(a.color[channel], b.color[channel])),
        intensity: SUN_INTENSITY * direct_sunlight(elevation) / direct_sunlight(45.0),
        sky: lerp(a.sky, b.sky),
    }
}

/// Direct normal irradiance (kW/m²) of the sun at `elevation` degrees: Kasten–Young air mass,
/// Meinel's attenuation; 0 below the horizon.
fn direct_sunlight(elevation: f32) -> f32 {
    if elevation <= 0.0 {
        return 0.0;
    }
    let zenith = 90.0 - elevation;
    let air_mass = 1.0 / (zenith.to_radians().cos() + 0.50572 * (96.07995 - zenith).powf(-1.6364));
    1.353 * 0.7f32.powf(air_mass.powf(0.678))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hour_12_high_sun_at_about_full_intensity() {
        let noon = at(12.0);
        assert!((noon.elevation - 47.9).abs() < 1e-3);
        assert!(
            (noon.intensity - SUN_INTENSITY).abs() < 0.1,
            "{}",
            noon.intensity
        );
        assert_eq!(noon.color, MIDDAY_SUN);
    }

    #[test]
    fn hour_0_and_hour_24_are_identical() {
        let (midnight, end) = (at(0.0), at(24.0));
        assert!((midnight.to_sun - end.to_sun).length() < 1e-5);
        assert_eq!((midnight.intensity, midnight.sky), (end.intensity, end.sky));
    }

    #[test]
    fn hour_3_sun_below_horizon_gives_no_sunlight() {
        assert_eq!(at(3.0).intensity, 0.0);
    }

    #[test]
    fn hour_18_low_warm_sun() {
        let evening = at(18.0);
        assert!((evening.elevation - 3.3).abs() < 1e-3);
        assert!(evening.intensity > 0.0 && evening.intensity < 1.0);
        assert_eq!(evening.color, LOW_SUN);
    }

    #[test]
    fn azimuth_takes_the_short_way_from_hour_20_to_24() {
        // 287.8° → 353.5°: halfway is about 320.7°, not across south
        let late = at(22.0);
        let azimuth = late
            .to_sun
            .x
            .atan2(-late.to_sun.z)
            .to_degrees()
            .rem_euclid(360.0);
        assert!((azimuth - 320.65).abs() < 0.1, "{azimuth}");
    }
}
