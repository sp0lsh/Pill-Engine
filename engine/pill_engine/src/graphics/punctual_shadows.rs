//! Shadow maps of point and spot lights (Filament's model): a per-frame budget of maps, given to
//! the shadow-casting lights nearest the camera; a spot light takes one perspective map, a point
//! light six 90° cube faces. All maps are layers of one depth texture array; a light's record
//! carries its first layer, a point light's face follows from the light-to-surface direction.

use crate::graphics::shadow::light_up;

use pill_core::{Matrix4f, Vector3f};

/// Upper bound of `PunctualLights::max_shadow_maps` (texture array layers).
pub(crate) const MAX_PUNCTUAL_SHADOW_MAPS: usize = 64;
/// Near plane of the light projections (m).
pub(crate) const SHADOW_NEAR: f32 = 0.05;

/// Cube faces in layer order: direction the face looks along, and its up vector.
pub(crate) const CUBE_FACES: [(Vector3f, Vector3f); 6] = [
    (Vector3f::X, Vector3f::Y),
    (Vector3f::NEG_X, Vector3f::Y),
    (Vector3f::Y, Vector3f::Z),
    (Vector3f::NEG_Y, Vector3f::Z),
    (Vector3f::Z, Vector3f::Y),
    (Vector3f::NEG_Z, Vector3f::Y),
];

/// Cube face of a light-to-surface direction: its dominant axis. The reference for `cube_face` in
/// pbr_opaque_fragment.hlsl, checked against the face matrices in the tests.
#[cfg(test)]
fn cube_face(direction: Vector3f) -> usize {
    let magnitude = direction.abs();
    if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
        if direction.x > 0.0 {
            0
        } else {
            1
        }
    } else if magnitude.y >= magnitude.z {
        if direction.y > 0.0 {
            2
        } else {
            3
        }
    } else if direction.z > 0.0 {
        4
    } else {
        5
    }
}

/// Texels around each cube face beyond its 90°: the PCF taps of a receiver on the face's edge
/// (and of one pushed across it by the normal offset) still read the face's own map.
const CUBE_FACE_BORDER_TEXELS: f32 = 4.0;

/// Field of view of a cube face of `size` texels: 90° plus `CUBE_FACE_BORDER_TEXELS` per side.
pub(crate) fn cube_face_fov(size: u32) -> f32 {
    let half_extent = 1.0 + 2.0 * CUBE_FACE_BORDER_TEXELS / size.max(1) as f32;
    2.0 * half_extent.atan()
}

/// World → clip (depth 0..1) of one cube face of a point light (`cube_face_fov`).
pub(crate) fn point_face_view_projection(
    position: Vector3f,
    face: usize,
    range: f32,
    size: u32,
) -> Matrix4f {
    let (forward, up) = CUBE_FACES[face];
    Matrix4f::perspective_rh(
        cube_face_fov(size),
        1.0,
        SHADOW_NEAR,
        range.max(SHADOW_NEAR * 2.0),
    ) * Matrix4f::look_to_rh(position, forward, up)
}

/// Field of view of a spot light's shadow map: its cone (half-angle `outer_angle`, radians),
/// clamped to what a perspective projection covers.
pub(crate) fn spot_fov(outer_angle: f32) -> f32 {
    (outer_angle * 2.0).clamp(1f32.to_radians(), 179f32.to_radians())
}

/// World → clip (depth 0..1) of a spot light's cone (`spot_fov`). The up vector follows the axis
/// continuously (`light_up`), so a moving spot's texel grid doesn't flip.
pub(crate) fn spot_view_projection(
    position: Vector3f,
    axis: Vector3f,
    outer_angle: f32,
    range: f32,
) -> Matrix4f {
    let up = light_up(axis);
    Matrix4f::perspective_rh(
        spot_fov(outer_angle),
        1.0,
        SHADOW_NEAR,
        range.max(SHADOW_NEAR * 2.0),
    ) * Matrix4f::look_to_rh(position, axis, up)
}

/// World-space size of one texel per meter of distance from the light, for a map of `size` texels
/// covering a field of view of `fov` radians.
pub(crate) fn texel_scale(fov: f32, size: u32) -> f32 {
    2.0 * (fov * 0.5).tan() / size.max(1) as f32
}

/// A shadow-casting light competing for the budget.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ShadowCandidate {
    pub(crate) light: usize,
    pub(crate) camera_distance: f32,
    pub(crate) is_point: bool,
}

/// Fills `budget` maps with the candidates nearest the camera (spot: 1 map, point: 6); a light
/// that doesn't fit is skipped and the next ones are still tried. Returns (light, first layer).
pub(crate) fn assign_shadow_layers(
    candidates: &mut [ShadowCandidate],
    budget: usize,
    out: &mut Vec<(usize, usize)>,
) {
    out.clear();
    candidates.sort_by(|a, b| a.camera_distance.total_cmp(&b.camera_distance));
    let mut used = 0;
    for candidate in candidates.iter() {
        let needed = if candidate.is_point { 6 } else { 1 };
        if used + needed <= budget {
            out.push((candidate.light, used));
            used += needed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pill_core::Vector4f;

    fn project(matrix: &Matrix4f, point: Vector3f) -> Vector3f {
        let clip = *matrix * Vector4f::new(point.x, point.y, point.z, 1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn cube_face_of_a_direction_projects_inside_that_face() {
        let light = Vector3f::new(1.0, 2.0, 3.0);
        for index in 0..500 {
            let t = index as f32;
            let direction =
                Vector3f::new((t * 1.7).sin(), (t * 2.3).cos(), (t * 0.9).sin() * 1.3).normalize();
            let point = light + direction * 2.0;
            let ndc = project(
                &point_face_view_projection(light, cube_face(direction), 10.0, 1024),
                point,
            );
            assert!(
                ndc.x.abs() <= 1.0001 && ndc.y.abs() <= 1.0001,
                "{direction} → {ndc}"
            );
            assert!((0.0..=1.0).contains(&ndc.z), "{direction} → {ndc}");
        }
    }

    #[test]
    fn cube_face_border_keeps_edge_taps_inside_the_face() {
        // A point on the boundary of faces +X and +Z: 4 texels from the edge of either map
        let light = Vector3f::ZERO;
        let point = Vector3f::new(1.0, 0.0, 1.0);
        for face in [0, 4] {
            let ndc = project(&point_face_view_projection(light, face, 10.0, 1024), point);
            let texels_from_edge = (1.0 - ndc.x.abs().max(ndc.y.abs())) * 0.5 * 1024.0;
            assert!(
                (texels_from_edge - 4.0).abs() < 0.05,
                "face {face}: {texels_from_edge}"
            );
        }
    }

    #[test]
    fn spot_normal_offset_fov_matches_the_projection_at_the_clamp() {
        // 89.9° half-angle: the map covers spot_fov, not 2 × 89.9°
        let outer = 89.9f32.to_radians();
        let matrix = spot_view_projection(Vector3f::ZERO, Vector3f::NEG_Y, outer, 10.0);
        let half = spot_fov(outer) * 0.5;
        let edge = project(&matrix, Vector3f::new(half.tan() * 2.0, -2.0, 0.0));
        assert!(
            (edge.x.abs().max(edge.y.abs()) - 1.0).abs() < 1e-3,
            "{edge}"
        );
    }

    #[test]
    fn spot_basis_is_continuous_across_near_vertical_axes() {
        // Either side of |axis.y| = 0.99, where the old up vector switched from Y to Z
        let axis = |y: f32| Vector3f::new((1.0 - y * y).sqrt(), -y, 0.0);
        let a = spot_view_projection(Vector3f::ZERO, axis(0.9899), 20f32.to_radians(), 10.0);
        let b = spot_view_projection(Vector3f::ZERO, axis(0.9901), 20f32.to_radians(), 10.0);
        assert!(a.abs_diff_eq(b, 1e-2), "{a:?} vs {b:?}");
    }

    #[test]
    fn spot_matrix_maps_its_cone_to_uv_and_depth_0_to_1() {
        let position = Vector3f::new(0.0, 10.0, 0.0);
        let matrix = spot_view_projection(position, Vector3f::NEG_Y, 20f32.to_radians(), 14.0);
        let on_axis = project(&matrix, position + Vector3f::NEG_Y * 5.0);
        assert!(on_axis.x.abs() < 1e-4 && on_axis.y.abs() < 1e-4);
        assert!((0.0..=1.0).contains(&on_axis.z));
        // A point on the cone's edge lands on the edge of the map
        let edge = position + Vector3f::new(20f32.to_radians().tan() * 5.0, -5.0, 0.0);
        let ndc = project(&matrix, edge);
        assert!((ndc.x.abs().max(ndc.y.abs()) - 1.0).abs() < 1e-3, "{ndc}");
    }

    #[test]
    fn budget_goes_to_the_nearest_lights_and_skips_what_does_not_fit() {
        let candidate = |light, camera_distance, is_point| ShadowCandidate {
            light,
            camera_distance,
            is_point,
        };
        let mut candidates = vec![
            candidate(0, 30.0, false),
            candidate(1, 5.0, true),
            candidate(2, 10.0, true),
            candidate(3, 20.0, false),
        ];
        let mut layers = Vec::new();
        assign_shadow_layers(&mut candidates, 8, &mut layers);
        // Point 1 (6 maps), point 2 doesn't fit, spot 3, spot 0
        assert_eq!(layers, vec![(1, 0), (3, 6), (0, 7)]);
    }
}
