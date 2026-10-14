//! Dynamic diffuse global illumination (DDGI, Majercik et al. 2019), ported from WickedEngine's
//! compute path (`ddgi_raytraceCS`, `ddgi_updateCS`, `ShaderInterop_DDGI.h`): a grid of probes
//! traces rays against a triangle BVH every frame; each probe keeps L1 spherical-harmonics
//! radiance (from 8×8 octahedral texels blended with Wicked's multiscale mean estimator) and
//! 16×16 octahedral depth moments for visibility. The PBR passes read 8 probes per pixel.

pub(crate) mod bvh;
mod pass_ddgi;

pub(crate) use pass_ddgi::{DdgiFrame, PassDdgi};

use pill_core::Vector3f;

/// Octahedral radiance texels per probe side (blended, then projected to SH).
pub(crate) const COLOR_RESOLUTION: u32 = 8;
/// Octahedral depth-moment texels per probe side.
pub(crate) const DEPTH_RESOLUTION: u32 = 16;
/// Depth texels per probe side including the 1-texel border for bilinear filtering.
pub(crate) const DEPTH_TEXELS: u32 = DEPTH_RESOLUTION + 2;
/// Upper bound of `Gi::rays_per_probe`.
pub(crate) const MAX_RAYS_PER_PROBE: u32 = 256;
/// Upper bound of probes per axis.
const MAX_PROBES_PER_AXIS: [u32; 3] = [32, 16, 32];
/// Widest depth atlas (WebGPU default `maxTextureDimension2D`).
const MAX_ATLAS_SIZE: u32 = 8192;

/// Axis-aligned probe grid: probe (x, y, z) sits at `min + (x, y, z) · cell`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ProbeGrid {
    pub(crate) min: Vector3f,
    pub(crate) cell: Vector3f,
    pub(crate) dims: [u32; 3],
}

impl ProbeGrid {
    /// Grid spanning `bounds` with about `spacing` between probes, at least 2 and at most
    /// `MAX_PROBES_PER_AXIS` per axis, and a depth atlas no wider than `MAX_ATLAS_SIZE`.
    pub(crate) fn fit((min, max): (Vector3f, Vector3f), spacing: f32) -> Self {
        let extent = (max - min).max(Vector3f::splat(1e-3));
        let spacing = spacing.max(0.05);
        let mut dims = [0, 1, 2].map(|axis| {
            ((extent[axis] / spacing).ceil() as u32 + 1).clamp(2, MAX_PROBES_PER_AXIS[axis])
        });
        // The atlas lays probes out x-then-y along its width (Wicked's layout)
        while dims[0] * dims[1] * DEPTH_TEXELS > MAX_ATLAS_SIZE {
            let wider = if dims[0] >= dims[1] { 0 } else { 1 };
            dims[wider] -= 1;
        }
        let cell = Vector3f::new(
            extent.x / (dims[0] - 1) as f32,
            extent.y / (dims[1] - 1) as f32,
            extent.z / (dims[2] - 1) as f32,
        );
        Self { min, cell, dims }
    }

    pub(crate) fn probe_count(&self) -> u32 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Probe coordinate of a flat index (x fastest), as `ddgi_probe_coord` in the shaders.
    pub(crate) fn coord(&self, index: u32) -> [u32; 3] {
        let [x, y, _] = self.dims;
        [index % x, (index / x) % y, index / (x * y)]
    }

    pub(crate) fn position(&self, [x, y, z]: [u32; 3]) -> Vector3f {
        self.min + Vector3f::new(x as f32, y as f32, z as f32) * self.cell
    }

    /// Depth atlas size: probes x-then-y along the width, z along the height.
    pub(crate) fn depth_atlas_size(&self) -> (u32, u32) {
        let [x, y, z] = self.dims;
        (x * y * DEPTH_TEXELS, z * DEPTH_TEXELS)
    }

    /// Top-left inner (non-border) texel of a probe's depth tile, as `ddgi_probe_depth_pixel`.
    pub(crate) fn depth_pixel(&self, [x, y, z]: [u32; 3]) -> (u32, u32) {
        (
            x * DEPTH_TEXELS + y * self.dims[0] * DEPTH_TEXELS + 1,
            z * DEPTH_TEXELS + 1,
        )
    }

    /// Ray length: the depth range of a probe (Wicked: a cell diagonal and a half).
    pub(crate) fn max_distance(&self) -> f32 {
        self.cell.length() * 1.5
    }
}

/// Probe-grid uniform shared by the DDGI compute passes and the PBR passes — mirrors
/// `DdgiParams` in include/ddgi_common.hlsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuDdgiParams {
    pub(crate) grid_min: [f32; 3],
    pub(crate) max_distance: f32,
    pub(crate) cell_size: [f32; 3],
    pub(crate) intensity: f32,
    pub(crate) grid_dims: [u32; 3],
    pub(crate) probe_count: u32,
    /// 1 / depth atlas size.
    pub(crate) depth_texel_size: [f32; 2],
    /// 1 when the probes hold valid data for the PBR passes.
    pub(crate) enabled: u32,
    pub(crate) frame: u32,
    /// First probe updated this frame (round robin) and how many.
    pub(crate) probe_offset: u32,
    pub(crate) probes_this_frame: u32,
    pub(crate) rays_per_probe: u32,
    pub(crate) light_count: u32,
    /// Direction from the key light toward the scene.
    pub(crate) sun_direction: [f32; 3],
    /// Multiscale mean estimator short-window blend (Wicked `blendSpeed`).
    pub(crate) blend_speed: f32,
    /// Key light color × intensity.
    pub(crate) sun_radiance: [f32; 3],
    /// IBL scale for rays that miss (sky).
    pub(crate) sky_intensity: f32,
    /// Random rotation of the ray pattern this frame (columns).
    pub(crate) rotation: [[f32; 4]; 3],
}

/// Uniformly random rotation (columns) from a frame number (Shoemake's quaternion method).
pub(crate) fn random_rotation(frame: u32) -> [[f32; 4]; 3] {
    let mut state = frame.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let (u1, u2, u3) = (next(), next(), next());
    let tau = std::f32::consts::TAU;
    let (x, y, z, w) = (
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        u1.sqrt() * (tau * u3).sin(),
        u1.sqrt() * (tau * u3).cos(),
    );
    // Unit quaternion → rotation matrix columns
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
            0.0,
        ],
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
            0.0,
        ],
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
            0.0,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_spans_the_bounds_within_caps_and_atlas_limit() {
        let bounds = (
            Vector3f::new(-15.0, -1.0, -9.5),
            Vector3f::new(14.5, 11.5, 8.8),
        );
        let grid = ProbeGrid::fit(bounds, 1.5);
        assert_eq!(grid.position([0, 0, 0]), bounds.0);
        let last = grid.position([grid.dims[0] - 1, grid.dims[1] - 1, grid.dims[2] - 1]);
        assert!(last.abs_diff_eq(bounds.1, 1e-4), "{last}");
        assert!(grid
            .dims
            .iter()
            .zip(MAX_PROBES_PER_AXIS)
            .all(|(&n, cap)| (2..=cap).contains(&n)));
        assert!(grid.depth_atlas_size().0 <= MAX_ATLAS_SIZE);
    }

    #[test]
    fn huge_scene_is_capped_and_atlas_fits() {
        let grid = ProbeGrid::fit((Vector3f::splat(-500.0), Vector3f::splat(500.0)), 1.0);
        assert!(grid.depth_atlas_size().0 <= MAX_ATLAS_SIZE);
        assert!(grid.dims[2] <= 32);
    }

    #[test]
    fn probe_index_and_coordinate_round_trip() {
        let grid = ProbeGrid::fit((Vector3f::ZERO, Vector3f::new(10.0, 4.0, 6.0)), 1.0);
        for index in 0..grid.probe_count() {
            let [x, y, z] = grid.coord(index);
            assert_eq!(
                x + y * grid.dims[0] + z * grid.dims[0] * grid.dims[1],
                index
            );
        }
    }

    #[test]
    fn depth_tiles_do_not_overlap() {
        let grid = ProbeGrid::fit((Vector3f::ZERO, Vector3f::new(10.0, 4.0, 6.0)), 1.0);
        let (width, height) = grid.depth_atlas_size();
        let mut used = vec![false; (width * height) as usize];
        for index in 0..grid.probe_count() {
            let (x0, y0) = grid.depth_pixel(grid.coord(index));
            for y in y0 - 1..y0 + DEPTH_RESOLUTION + 1 {
                for x in x0 - 1..x0 + DEPTH_RESOLUTION + 1 {
                    let texel = (y * width + x) as usize;
                    assert!(!used[texel]);
                    used[texel] = true;
                }
            }
        }
    }

    #[test]
    fn random_rotation_is_orthonormal() {
        for frame in 0..20 {
            let [x, y, z] =
                random_rotation(frame).map(|column| Vector3f::new(column[0], column[1], column[2]));
            assert!((x.length() - 1.0).abs() < 1e-4 && (y.length() - 1.0).abs() < 1e-4);
            assert!(x.dot(y).abs() < 1e-4 && x.cross(y).abs_diff_eq(z, 1e-4));
        }
    }
}
