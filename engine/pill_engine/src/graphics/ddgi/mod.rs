//! Dynamic diffuse global illumination (DDGI, Majercik et al. 2019; method: WickedEngine's compute
//! path `ddgi_raytraceCS`, `ddgi_updateCS`, `ShaderInterop_DDGI.h`; cascades: Flax Engine's DDGI):
//! grids of probes trace rays against a triangle BVH every frame; each probe keeps L1
//! spherical-harmonics radiance (from 8×8 octahedral texels averaged over frames) and 16×16
//! octahedral depth moments for visibility. The PBR passes read 8 probes per pixel per cascade.

pub(crate) mod bvh;
mod pass_ddgi;
mod pass_ddgi_probes;

pub(crate) use pass_ddgi::{DdgiFrame, PassDdgi};
pub(crate) use pass_ddgi_probes::PassDdgiProbes;

use crate::graphics::shader_bindings::structs::DdgiCascade;

use pill_core::Vector3f;

/// Octahedral radiance texels per probe side (blended, then projected to SH).
pub(crate) const COLOR_RESOLUTION: u32 = 8;
/// Octahedral depth-moment texels per probe side.
pub(crate) const DEPTH_RESOLUTION: u32 = 16;
/// Depth texels per probe side including the 1-texel border for bilinear filtering.
pub(crate) const DEPTH_TEXELS: u32 = DEPTH_RESOLUTION + 2;
/// Upper bound of `Gi::rays_per_probe`.
pub(crate) const MAX_RAYS_PER_PROBE: u32 = 256;
/// `ProbeGrid::exposed` source of a probe with no neighbour to start from.
pub(crate) const NO_SOURCE: u32 = u32::MAX;
/// Upper bound of probes per axis.
const MAX_PROBES_PER_AXIS: [u32; 3] = [32, 16, 32];
/// Widest depth atlas (WebGPU default `maxTextureDimension2D`).
const MAX_ATLAS_SIZE: u32 = 8192;

/// Axis-aligned probe grid: the probe at logical coordinate (x, y, z) sits at
/// `min + (x, y, z) · cell`. A camera-following grid ([`ProbeGrid::around`]) moves in whole cells:
/// `origin` is the world cell of logical probe (0, 0, 0), and probe data is stored toroidally at
/// `(logical + origin) mod dims`, so probes the grid keeps after a move keep their storage (and
/// history). A fixed grid has origin 0: storage = logical.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ProbeGrid {
    pub(crate) min: Vector3f,
    pub(crate) cell: Vector3f,
    pub(crate) dims: [u32; 3],
    pub(crate) origin: [i32; 3],
}

/// `value` modulo `size`, non-negative.
fn wrap(value: i32, size: u32) -> u32 {
    value.rem_euclid(size as i32) as u32
}

impl ProbeGrid {
    /// Grid over `bounds` with about `spacing` between probes, at least 2 and at most
    /// `MAX_PROBES_PER_AXIS` per axis, and a depth atlas no wider than `MAX_ATLAS_SIZE`. Probes
    /// sit at cell centres, half a cell inside the bounds: not on the floor and the outer walls,
    /// where rays would start coplanar with the geometry.
    pub(crate) fn fit((min, max): (Vector3f, Vector3f), spacing: f32) -> Self {
        let extent = (max - min).max(Vector3f::splat(1e-3));
        let spacing = spacing.max(0.05);
        let mut dims = [0, 1, 2].map(|axis| {
            ((extent[axis] / spacing).ceil() as u32).clamp(2, MAX_PROBES_PER_AXIS[axis])
        });
        // The atlas lays probes out x-then-y along its width
        while dims[0] * dims[1] * DEPTH_TEXELS > MAX_ATLAS_SIZE {
            let wider = if dims[0] >= dims[1] { 0 } else { 1 };
            dims[wider] -= 1;
        }
        let cell = Vector3f::new(
            extent.x / dims[0] as f32,
            extent.y / dims[1] as f32,
            extent.z / dims[2] as f32,
        );
        Self {
            min: min + cell * 0.5,
            cell,
            dims,
            origin: [0; 3],
        }
    }

    /// Grid of the largest size the caps and the atlas allow, `spacing` apart, centred on
    /// `center` and snapped to whole cells of the world.
    pub(crate) fn around(center: Vector3f, spacing: f32) -> Self {
        let spacing = spacing.max(0.05);
        let mut dims = MAX_PROBES_PER_AXIS;
        while dims[0] * dims[1] * DEPTH_TEXELS > MAX_ATLAS_SIZE {
            let wider = if dims[0] >= dims[1] { 0 } else { 1 };
            dims[wider] -= 1;
        }
        let origin =
            [0, 1, 2].map(|axis| (center[axis] / spacing).floor() as i32 - (dims[axis] / 2) as i32);
        let cell = Vector3f::splat(spacing);
        Self {
            min: Vector3f::new(origin[0] as f32, origin[1] as f32, origin[2] as f32) * cell,
            cell,
            dims,
            origin,
        }
    }

    /// Same probe layout and storage (only the origin may differ): same dims, spacing and anchor
    /// (world position of cell 0 — constant for a camera-following grid, the fitted minimum
    /// otherwise, so a fitted grid that moves gets new storage).
    pub(crate) fn same_storage(&self, other: &Self) -> bool {
        let anchor = |grid: &Self| {
            grid.min
                - Vector3f::new(
                    grid.origin[0] as f32,
                    grid.origin[1] as f32,
                    grid.origin[2] as f32,
                ) * grid.cell
        };
        self.dims == other.dims
            && self.cell == other.cell
            && anchor(self).abs_diff_eq(anchor(other), 1e-3)
    }

    /// Storage coordinate of a logical one, as `ddgi_storage_coord` in the shaders.
    pub(crate) fn storage_coord(&self, logical: [u32; 3]) -> [u32; 3] {
        [0, 1, 2].map(|axis| wrap(logical[axis] as i32 + self.origin[axis], self.dims[axis]))
    }

    /// Logical coordinate of a storage one, as `ddgi_logical_coord` in the shaders.
    pub(crate) fn logical_coord(&self, storage: [u32; 3]) -> [u32; 3] {
        [0, 1, 2].map(|axis| wrap(storage[axis] as i32 - self.origin[axis], self.dims[axis]))
    }

    /// Flat index of a coordinate (x fastest), as `ddgi_probe_index`.
    pub(crate) fn index(&self, [x, y, z]: [u32; 3]) -> u32 {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    /// The probes this grid has and a grid at `previous_origin` (same storage) did not — the slabs
    /// a move exposes, every probe after a jump of a grid size or more — as `[storage index,
    /// source]`. The source is the storage index of the nearest probe both grids have, which the
    /// exposed probe starts from; `NO_SOURCE` when there is none.
    pub(crate) fn exposed(&self, previous_origin: [i32; 3]) -> Vec<[u32; 2]> {
        (0..self.probe_count())
            .filter_map(|index| {
                let logical = self.logical_coord(self.coord(index));
                // World cell of the exposed probe, and of the previous grid's probe nearest to it
                let cell = [0, 1, 2].map(|axis| logical[axis] as i32 + self.origin[axis]);
                let nearest = [0, 1, 2].map(|axis| {
                    cell[axis].clamp(
                        previous_origin[axis],
                        previous_origin[axis] + self.dims[axis] as i32 - 1,
                    )
                });
                if nearest == cell {
                    return None;
                }
                let in_both = (0..3).all(|axis| {
                    (self.origin[axis]..self.origin[axis] + self.dims[axis] as i32)
                        .contains(&nearest[axis])
                });
                let source = if in_both {
                    self.index([0, 1, 2].map(|axis| wrap(nearest[axis], self.dims[axis])))
                } else {
                    NO_SOURCE
                };
                Some([index, source])
            })
            .collect()
    }

    pub(crate) fn probe_count(&self) -> u32 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Probe coordinate of a flat index (x fastest), as `ddgi_probe_coord` in the shaders.
    pub(crate) fn coord(&self, index: u32) -> [u32; 3] {
        let [x, y, _] = self.dims;
        [index % x, (index / x) % y, index / (x * y)]
    }

    /// World position of the probe at a logical coordinate.
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

    /// Ray length: the depth range of a probe, a cell diagonal and a half.
    pub(crate) fn max_distance(&self) -> f32 {
        self.cell.length() * 1.5
    }
}

/// Most cascades (`Gi::cascades`), as `DDGI_MAX_CASCADES` in include/ddgi_common.hlsl.
pub(crate) const MAX_CASCADES: usize = 4;
/// Default `Gi::cascade_spacings`: 1.5 m × 1, 3, 6, 10.
pub(crate) const CASCADE_SPACINGS: [f32; MAX_CASCADES] = [1.5, 4.5, 9.0, 15.0];
/// Cascade c updates every `CASCADE_UPDATE_EVERY[c]` frames.
pub(crate) const CASCADE_UPDATE_EVERY: [u32; MAX_CASCADES] = [1, 3, 5, 7];

impl DdgiCascade {
    pub(crate) fn new(grid: &ProbeGrid) -> Self {
        Self {
            min: grid.min.extend(0.0).to_array(),
            cell: grid.cell.extend(0.0).to_array(),
            origin: [grid.origin[0], grid.origin[1], grid.origin[2], 0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_probes_sit_half_a_cell_inside_the_bounds_within_caps_and_atlas_limit() {
        let bounds = (
            Vector3f::new(-15.0, -1.0, -9.5),
            Vector3f::new(14.5, 11.5, 8.8),
        );
        let grid = ProbeGrid::fit(bounds, 1.5);
        let first = grid.position([0, 0, 0]);
        assert!(
            first.abs_diff_eq(bounds.0 + grid.cell * 0.5, 1e-4),
            "{first}"
        );
        let last = grid.position([grid.dims[0] - 1, grid.dims[1] - 1, grid.dims[2] - 1]);
        assert!(last.abs_diff_eq(bounds.1 - grid.cell * 0.5, 1e-4), "{last}");
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
    fn storage_and_logical_coordinates_round_trip_for_negative_origins() {
        let mut grid = ProbeGrid::around(Vector3f::new(-37.0, -3.0, 12.0), 2.0);
        grid.origin = [-41, -9, 7];
        for index in 0..grid.probe_count() {
            let logical = grid.coord(index);
            assert_eq!(grid.logical_coord(grid.storage_coord(logical)), logical);
        }
    }

    #[test]
    fn one_cell_step_exposes_exactly_one_slab_per_axis() {
        let grid = ProbeGrid::around(Vector3f::ZERO, 2.0);
        let [x, y, z] = grid.dims;
        for (axis, slab) in [(0, y * z), (1, x * z), (2, x * y)] {
            for step in [1, -1] {
                let mut moved = grid;
                moved.origin[axis] += step;
                assert_eq!(
                    moved.exposed(grid.origin).len() as u32,
                    slab,
                    "axis {axis} {step}"
                );
            }
        }
    }

    #[test]
    fn same_storage_scrolled_camera_grid_yes_moved_fitted_grid_no() {
        let grid = ProbeGrid::around(Vector3f::new(1.0, 2.0, 3.0), 2.0);
        let scrolled = ProbeGrid::around(Vector3f::new(7.0, 2.0, 3.0), 2.0);
        assert!(grid.same_storage(&scrolled));
        let bounds = (Vector3f::ZERO, Vector3f::splat(10.0));
        let fitted = ProbeGrid::fit(bounds, 1.0);
        let moved = ProbeGrid::fit((bounds.0 + Vector3f::X, bounds.1 + Vector3f::X), 1.0);
        assert!(!fitted.same_storage(&moved));
        assert!(fitted.same_storage(&ProbeGrid::fit(bounds, 1.0)));
    }

    #[test]
    fn jump_of_a_grid_size_exposes_every_probe_without_sources() {
        let grid = ProbeGrid::around(Vector3f::ZERO, 2.0);
        let mut moved = grid;
        moved.origin[2] += grid.dims[2] as i32;
        let exposed = moved.exposed(grid.origin);
        assert_eq!(exposed.len() as u32, grid.probe_count());
        assert!(exposed.iter().all(|&[_, source]| source == NO_SOURCE));
        assert!(grid.exposed(grid.origin).is_empty());
    }

    #[test]
    fn one_cell_step_sources_are_the_adjacent_surviving_probes() {
        let grid = ProbeGrid::around(Vector3f::ZERO, 2.0);
        for axis in 0..3 {
            for step in [1, -1] {
                let mut moved = grid;
                moved.origin[axis] += step;
                let exposed = moved.exposed(grid.origin);
                let targets: Vec<u32> = exposed.iter().map(|&[target, _]| target).collect();
                for &[target, source] in &exposed {
                    assert!(
                        !targets.contains(&source),
                        "axis {axis} {step}: source exposed"
                    );
                    let target = moved.logical_coord(moved.coord(target));
                    let source = moved.logical_coord(moved.coord(source));
                    for other in 0..3 {
                        let expected = if other == axis {
                            target[axis] as i32 - step
                        } else {
                            target[other] as i32
                        };
                        assert_eq!(source[other] as i32, expected, "axis {axis} {step}");
                    }
                }
            }
        }
    }

    #[test]
    fn cascades_share_dims_take_their_spacing_and_contain_the_camera() {
        let camera = Vector3f::new(-20.3, 2.5, 7.9);
        let grids: Vec<ProbeGrid> = CASCADE_SPACINGS
            .iter()
            .map(|&spacing| ProbeGrid::around(camera, spacing))
            .collect();
        for (grid, spacing) in grids.iter().zip(CASCADE_SPACINGS) {
            assert_eq!(grid.dims, grids[0].dims);
            assert_eq!(grid.cell, Vector3f::splat(spacing));
            let last = grid.position([grid.dims[0] - 1, grid.dims[1] - 1, grid.dims[2] - 1]);
            assert!(grid.min.cmple(camera).all() && last.cmpge(camera).all());
        }
    }

    #[test]
    fn around_contains_the_center_and_fits_the_atlas() {
        let center = Vector3f::new(13.7, 1.6, -88.2);
        let grid = ProbeGrid::around(center, 2.0);
        let last = grid.position([grid.dims[0] - 1, grid.dims[1] - 1, grid.dims[2] - 1]);
        assert!(grid.min.cmple(center).all() && last.cmpge(center).all());
        assert!(grid.depth_atlas_size().0 <= MAX_ATLAS_SIZE);
    }
}
