//! Directional shadow map fit and caster culling (CPU side).
//!
//! The light box is a cube around the bounding sphere of the camera frustum slice the shadow map
//! covers (stable under camera rotation), snapped to whole texels so it doesn't shimmer when the
//! camera moves (Filament stable mode, PolyEngine `StablizeShadowProjection`). Casters are culled
//! by their light-space AABB against that box; the box's near plane is then pulled toward the
//! light to include every caster that can shadow it (PolyEngine `CullShadowCasters`).

use crate::graphics::PassTextureHandle;

use pill_core::{Matrix4f, Vector3f};

/// Shadow of the current frame, written by `PassShadow`, read by the PBR passes. The default
/// (`map: None`) means no shadow this frame.
#[derive(Clone, Copy, Default)]
pub(crate) struct ShadowFrame {
    /// Depth map the casters were rendered into.
    pub(crate) map: Option<PassTextureHandle>,
    /// Width and height of `map` in texels.
    pub(crate) map_size: u32,
    /// World → light clip space (depth 0..1).
    pub(crate) view_projection: Matrix4f,
    /// World-space size of one shadow-map texel.
    pub(crate) texel_world_size: f32,
    /// View distance the shadow reaches (`Shadow::distance`, clamped to the camera range).
    pub(crate) distance: f32,
}

/// Light view and the receiver box in light space, before casters extend its near plane.
pub(crate) struct LightFit {
    pub(crate) view: Matrix4f,
    /// Light-space box of the receivers; light space looks down −Z, so `max.z` is nearest the light.
    pub(crate) min: Vector3f,
    pub(crate) max: Vector3f,
    pub(crate) texel_world_size: f32,
}

/// Camera inputs of the fit.
pub(crate) struct CameraSlice {
    pub(crate) position: Vector3f,
    pub(crate) forward: Vector3f,
    /// Vertical field of view in radians.
    pub(crate) fov_y: f32,
    pub(crate) aspect: f32,
    pub(crate) near: f32,
    pub(crate) far: f32,
}

/// Fits the light box to the camera slice `near..far` for a light shining along `direction`.
pub(crate) fn fit_light(camera: &CameraSlice, direction: Vector3f, map_size: u32) -> LightFit {
    let direction = direction.normalize_or(Vector3f::NEG_Y);
    // Bounding sphere of the slice: centred on the view axis, through the far corners. Depends only
    // on the projection, so it doesn't change size when the camera turns.
    let tan_y = (camera.fov_y * 0.5).tan();
    let tan_x = tan_y * camera.aspect;
    let diagonal_squared = tan_x * tan_x + tan_y * tan_y;
    let (near, far) = (camera.near, camera.far.max(camera.near + 1e-3));
    // Centre distance d minimizing the radius over the near and far corners
    let center_distance = ((near + far) * 0.5 * (1.0 + diagonal_squared)).min(far);
    let radius_at = |distance: f32| {
        let along = distance - center_distance;
        (along * along + distance * distance * diagonal_squared).sqrt()
    };
    // Rounded up to 1/16 so float noise in the radius doesn't change the texel size
    let radius = (radius_at(near).max(radius_at(far)) * 16.0).ceil() / 16.0;
    let center = camera.position + camera.forward.normalize() * center_distance;

    // Light view at the world origin: the texel grid stays fixed in world space
    let view = Matrix4f::look_to_rh(Vector3f::ZERO, direction, light_up(direction));
    let texel_world_size = 2.0 * radius / map_size.max(1) as f32;
    let center_light = view.transform_point3(center);
    let snapped = Vector3f::new(
        (center_light.x / texel_world_size).floor() * texel_world_size,
        (center_light.y / texel_world_size).floor() * texel_world_size,
        center_light.z,
    );
    LightFit {
        view,
        min: snapped - Vector3f::splat(radius),
        max: snapped + Vector3f::splat(radius),
        texel_world_size,
    }
}

/// Up vector of the light view: −Z carried along by the shortest rotation from straight down to
/// `direction`. Continuous for every light shining downward, so a moving sun never flips the
/// texel grid; the only singular direction is straight up, where any perpendicular works.
pub(crate) fn light_up(direction: Vector3f) -> Vector3f {
    let down = Vector3f::NEG_Y;
    let cosine = down.dot(direction);
    if cosine < -1.0 + 1e-6 {
        return Vector3f::Z;
    }
    // Rotation taking `down` to `direction` (axis scaled by the sine), applied to −Z
    let axis = down.cross(direction);
    let up = Vector3f::NEG_Z;
    up * cosine + axis.cross(up) + axis * (axis.dot(up) / (1.0 + cosine))
}

/// Light-space AABB of a world-transformed local AABB (8 corners).
pub(crate) fn light_space_bounds(
    light_view: &Matrix4f,
    model: &Matrix4f,
    (local_min, local_max): (Vector3f, Vector3f),
) -> (Vector3f, Vector3f) {
    let to_light = *light_view * *model;
    let mut min = Vector3f::splat(f32::INFINITY);
    let mut max = Vector3f::splat(f32::NEG_INFINITY);
    for corner in 0..8 {
        let local = Vector3f::new(
            if corner & 1 == 0 {
                local_min.x
            } else {
                local_max.x
            },
            if corner & 2 == 0 {
                local_min.y
            } else {
                local_max.y
            },
            if corner & 4 == 0 {
                local_min.z
            } else {
                local_max.z
            },
        );
        let point = to_light.transform_point3(local);
        min = min.min(point);
        max = max.max(point);
    }
    (min, max)
}

/// Whether a caster can shadow the receiver box: overlaps it in xy and isn't entirely beyond its
/// far side (light space looks down −Z, far side = `fit.min.z`).
pub(crate) fn casts_into(fit: &LightFit, (min, max): (Vector3f, Vector3f)) -> bool {
    min.x <= fit.max.x
        && max.x >= fit.min.x
        && min.y <= fit.max.y
        && max.y >= fit.min.y
        && max.z >= fit.min.z
}

/// Light view-projection of the fit, with the near plane pulled toward the light up to
/// `nearest_caster_z` (the largest light-space z of the kept casters).
pub(crate) fn light_view_projection(fit: &LightFit, nearest_caster_z: f32) -> Matrix4f {
    let near_z = fit.max.z.max(nearest_caster_z);
    // orthographic_rh takes distances along −Z and maps them to depth 0..1
    let projection = Matrix4f::orthographic_rh(
        fit.min.x, fit.max.x, fit.min.y, fit.max.y, -near_z, -fit.min.z,
    );
    projection * fit.view
}

/// Clip-space position of a world point (for tests and debugging).
#[cfg(test)]
fn project(matrix: &Matrix4f, point: Vector3f) -> Vector3f {
    let clip = *matrix * pill_core::Vector4f::new(point.x, point.y, point.z, 1.0);
    clip.truncate() / clip.w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraSlice {
        CameraSlice {
            position: Vector3f::new(0.0, 2.0, 5.0),
            forward: Vector3f::NEG_Z,
            fov_y: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 20.0,
        }
    }

    #[test]
    fn fit_contains_the_camera_slice_in_depth_0_to_1() {
        let fit = fit_light(&camera(), Vector3f::new(0.3, -0.9, 0.3), 2048);
        let matrix = light_view_projection(&fit, f32::NEG_INFINITY);
        // Points on the view axis at the slice's near and far ends
        for distance in [0.1_f32, 10.0, 20.0] {
            let point = camera().position + Vector3f::NEG_Z * distance;
            let ndc = project(&matrix, point);
            assert!(ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0, "{ndc:?}");
            assert!((0.0..=1.0).contains(&ndc.z), "{ndc:?}");
        }
    }

    #[test]
    fn fit_size_is_stable_under_camera_rotation() {
        let mut turned = camera();
        turned.forward = Vector3f::new(1.0, -0.3, 0.2).normalize();
        let direction = Vector3f::new(0.3, -0.9, 0.3);
        let a = fit_light(&camera(), direction, 2048);
        let b = fit_light(&turned, direction, 2048);
        assert_eq!(a.texel_world_size, b.texel_world_size);
        assert_eq!(a.max.x - a.min.x, b.max.x - b.min.x);
    }

    #[test]
    fn fit_origin_is_snapped_to_texels() {
        let fit = fit_light(&camera(), Vector3f::new(0.3, -0.9, 0.3), 1024);
        let texels = (fit.min.x + fit.max.x) * 0.5 / fit.texel_world_size;
        assert!((texels - texels.round()).abs() < 1e-3);
    }

    #[test]
    fn caster_beyond_the_far_side_is_culled_and_one_toward_the_light_extends_near() {
        let fit = fit_light(&camera(), Vector3f::NEG_Y, 1024);
        let center = (fit.min + fit.max) * 0.5;
        let beyond = (
            Vector3f::new(center.x, center.y, fit.min.z - 10.0),
            Vector3f::new(center.x + 1.0, center.y + 1.0, fit.min.z - 5.0),
        );
        let above = (
            Vector3f::new(center.x, center.y, fit.max.z + 5.0),
            Vector3f::new(center.x + 1.0, center.y + 1.0, fit.max.z + 10.0),
        );
        let aside = (
            Vector3f::new(fit.max.x + 1.0, center.y, center.z),
            Vector3f::new(fit.max.x + 2.0, center.y + 1.0, center.z + 1.0),
        );
        assert!(!casts_into(&fit, beyond));
        assert!(casts_into(&fit, above));
        assert!(!casts_into(&fit, aside));
        // The caster above the box lands inside depth 0..1 once the near plane is extended
        let matrix = light_view_projection(&fit, above.1.z) * fit.view.inverse();
        let ndc = project(&matrix, above.1);
        assert!((0.0..=1.0).contains(&ndc.z), "{ndc:?}");
    }

    #[test]
    fn light_view_is_continuous_near_the_zenith() {
        // Either side of the old |y| = 0.99 up-vector switch, and through straight down
        let near_zenith = |angle: f32| Vector3f::new(angle.sin(), -angle.cos(), 0.0);
        let boundary = 0.99_f32.acos();
        for (a, b) in [(boundary - 1e-3, boundary + 1e-3), (-1e-3, 1e-3)] {
            let view_a =
                Matrix4f::look_to_rh(Vector3f::ZERO, near_zenith(a), light_up(near_zenith(a)));
            let view_b =
                Matrix4f::look_to_rh(Vector3f::ZERO, near_zenith(b), light_up(near_zenith(b)));
            assert!(view_a.abs_diff_eq(view_b, 1e-2), "{view_a:?} vs {view_b:?}");
        }
    }

    #[test]
    fn light_up_is_perpendicular_to_the_light() {
        for direction in [
            Vector3f::NEG_Y,
            Vector3f::new(0.45, -1.0, 0.08).normalize(),
            Vector3f::X,
            Vector3f::new(0.3, 0.2, -0.9).normalize(),
        ] {
            let up = light_up(direction);
            assert!(up.dot(direction).abs() < 1e-5 && (up.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn light_space_bounds_of_a_unit_cube_at_identity() {
        let view = Matrix4f::IDENTITY;
        let model = Matrix4f::from_translation(Vector3f::new(1.0, 2.0, 3.0))
            * Matrix4f::from_scale(Vector3f::splat(2.0));
        let (min, max) =
            light_space_bounds(&view, &model, (Vector3f::splat(-0.5), Vector3f::splat(0.5)));
        assert_eq!(min, Vector3f::new(0.0, 1.0, 2.0));
        assert_eq!(max, Vector3f::new(2.0, 3.0, 4.0));
    }
}
