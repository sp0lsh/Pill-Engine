//! CPU light assignment for clustered forward shading, after Filament's `Froxelizer`.
//!
//! The view frustum is split into square screen tiles × exponential depth slices ("froxels"). Each
//! point/spot light is tested against the froxels inside its projected bounding box (sphere vs the
//! froxel's side planes and depth range, plus a cone test for spot lights) and recorded in a
//! per-froxel bitset. The bitsets are then compressed into the GPU layout: one (offset, count)
//! entry per froxel into a shared list of light indices, where a froxel with the same lights as
//! its left or upper neighbour reuses that neighbour's list.
//!
//! Differences from Filament, which packs everything into 16 KiB uniform buffers: indices are u16
//! and entries two u32 (WebGPU storage buffers), so up to `MAX_LIGHTS` lights instead of 256; the
//! depth slice of a fragment comes from its linear view depth instead of the depth buffer value.

use pill_core::{Vector3f, Vector4f};

/// Depth slices of the froxel grid.
pub(crate) const FROXEL_SLICE_COUNT: u32 = 16;
/// Upper bound of tiles × slices; sets the tile size for a given render size.
pub(crate) const MAX_FROXELS: usize = 8192;
/// Lights shaded per frame; the rest are dropped.
pub(crate) const MAX_LIGHTS: usize = 1024;
/// Capacity of the shared light-index list (2 MiB of u16). Filament's is 64 Ki, bound by its u16
/// offsets; entries here are u32, so the limit is only memory. Dense scenes need it: 1024 lights of
/// 3.5 m range in Sponza fill ~65 Ki indices at 960×540.
pub(crate) const MAX_RECORDS: usize = 1 << 20;

/// Inputs the froxel grid depends on; the grid is rebuilt when any of them changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FroxelLayoutParams {
    /// Render target size in pixels.
    pub(crate) render_size: (u32, u32),
    /// Vertical field of view in radians.
    pub(crate) fov_y: f32,
    pub(crate) aspect: f32,
    /// Camera near plane.
    pub(crate) near: f32,
    /// View distance the grid covers.
    pub(crate) light_far: f32,
    /// Far end of the first depth slice.
    pub(crate) first_slice_depth: f32,
}

/// Froxel grid of a symmetric perspective camera, in view space (looking down −Z).
pub(crate) struct FroxelLayout {
    pub(crate) params: FroxelLayoutParams,
    pub(crate) count_x: u32,
    pub(crate) count_y: u32,
    pub(crate) count_z: u32,
    /// Square tile size in pixels (multiple of 8).
    pub(crate) tile_size: u32,
    pub(crate) light_far: f32,
    /// Slice of depth `d`: `floor(log2(d / light_far) · inv_linearizer + count_z)`.
    pub(crate) inv_linearizer: f32,
    near: f32,
    tan_x: f32,
    tan_y: f32,
    /// Normals of the tile column edges, left to right; `dot(n, p) > 0` right of the edge.
    planes_x: Vec<Vector3f>,
    /// Normals of the tile row edges, top to bottom; `dot(n, p) > 0` above the edge.
    planes_y: Vec<Vector3f>,
    /// View depths of the slice boundaries; slice `z` spans `distances_z[z]..distances_z[z + 1]`.
    distances_z: Vec<f32>,
    /// Bounding sphere (center xyz, radius w) of every froxel, for the spot cone test.
    bounding_spheres: Vec<Vector4f>,
}

impl FroxelLayout {
    pub(crate) fn new(params: FroxelLayoutParams) -> Self {
        let width = params.render_size.0.max(16);
        let height = params.render_size.1.max(16);
        let count_z = FROXEL_SLICE_COUNT;

        // Square tiles: as many as fit in MAX_FROXELS / slices at the screen's aspect ratio,
        // rounded up to a multiple of 8 pixels (Filament computeFroxelLayout)
        let plane_count = (MAX_FROXELS as u32 / count_z) as f32;
        let columns = ((plane_count * width as f32 / height as f32).sqrt() as u32).max(1);
        let rows = ((plane_count * height as f32 / width as f32).sqrt() as u32).max(1);
        let tile_size = width
            .div_ceil(columns)
            .max(height.div_ceil(rows))
            .next_multiple_of(8);
        let count_x = width.div_ceil(tile_size);
        let count_y = height.div_ceil(tile_size);

        let near = params.near.max(1e-3);
        let light_far = params.light_far.max(near * 2.0);
        let first_slice_depth = params
            .first_slice_depth
            .clamp(near * 1.001, light_far * 0.999);
        let linearizer = (light_far / first_slice_depth).log2() / (count_z - 1) as f32;
        let distances_z: Vec<f32> = (0..=count_z)
            .map(|slice| match slice {
                0 => near,
                _ => light_far * ((slice as f32 - count_z as f32) * linearizer).exp2(),
            })
            .collect();

        let tan_y = (params.fov_y * 0.5).tan();
        let tan_x = tan_y * params.aspect;
        // NDC of the tile edges; the last column/row may extend past the screen edge
        let edge_x = |column: u32| -1.0 + (column * tile_size) as f32 * 2.0 / width as f32;
        let edge_y = |row: u32| 1.0 - (row * tile_size) as f32 * 2.0 / height as f32;
        // The plane through the eye and the line NDC x = e: normal (1, 0, e·tan_x), so that
        // dot(n, p) = tan_x · depth · (ndc_x − e)
        let planes_x = (0..=count_x)
            .map(|column| Vector3f::new(1.0, 0.0, edge_x(column) * tan_x).normalize())
            .collect();
        let planes_y = (0..=count_y)
            .map(|row| Vector3f::new(0.0, 1.0, edge_y(row) * tan_y).normalize())
            .collect();

        let mut bounding_spheres = Vec::with_capacity((count_x * count_y * count_z) as usize);
        for z in 0..count_z {
            for y in 0..count_y {
                for x in 0..count_x {
                    let mut corners = [Vector3f::ZERO; 8];
                    for (corner, point) in corners.iter_mut().enumerate() {
                        let depth = distances_z[z as usize + (corner >> 2)];
                        let ndc_x = edge_x(x + (corner & 1) as u32);
                        let ndc_y = edge_y(y + ((corner >> 1) & 1) as u32);
                        *point =
                            Vector3f::new(ndc_x * tan_x * depth, ndc_y * tan_y * depth, -depth);
                    }
                    let center = corners.iter().copied().sum::<Vector3f>() / 8.0;
                    let radius = corners
                        .iter()
                        .map(|corner| corner.distance(center))
                        .fold(0.0, f32::max);
                    bounding_spheres.push(center.extend(radius));
                }
            }
        }

        Self {
            params,
            count_x,
            count_y,
            count_z,
            tile_size,
            light_far,
            inv_linearizer: 1.0 / linearizer,
            near,
            tan_x,
            tan_y,
            planes_x,
            planes_y,
            distances_z,
            bounding_spheres,
        }
    }

    pub(crate) fn froxel_count(&self) -> usize {
        (self.count_x * self.count_y * self.count_z) as usize
    }

    /// Depth slice of a view depth; same formula as the PBR fragment shader.
    pub(crate) fn slice(&self, depth: f32) -> u32 {
        let slice =
            (depth.max(1e-6) / self.light_far).log2() * self.inv_linearizer + self.count_z as f32;
        slice.floor().clamp(0.0, (self.count_z - 1) as f32) as u32
    }

    fn froxel_index(&self, x: u32, y: u32, z: u32) -> usize {
        (x + y * self.count_x + z * self.count_x * self.count_y) as usize
    }

    /// Tile column of an NDC x, clamped to the grid.
    fn column(&self, ndc_x: f32) -> u32 {
        let pixel = (ndc_x + 1.0) * 0.5 * self.params.render_size.0.max(16) as f32;
        ((pixel / self.tile_size as f32).floor().max(0.0) as u32).min(self.count_x - 1)
    }

    /// Tile row (from the top) of an NDC y, clamped to the grid.
    fn row(&self, ndc_y: f32) -> u32 {
        let pixel = (1.0 - ndc_y) * 0.5 * self.params.render_size.1.max(16) as f32;
        ((pixel / self.tile_size as f32).floor().max(0.0) as u32).min(self.count_y - 1)
    }
}

/// Spot cone of a light, for the froxel test (Filament `sphereConeIntersectionFast`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FroxelCone {
    /// Unit direction the light shines in, view space.
    pub(crate) axis: Vector3f,
    /// 1 / sin(outer angle).
    pub(crate) inv_sin: f32,
    /// cos²(outer angle).
    pub(crate) cos_squared: f32,
}

impl FroxelCone {
    /// Cone of half-angle `outer_angle` (radians), clamped to at least 0.5° like Filament (the
    /// test is imprecise for thinner cones).
    pub(crate) fn new(axis: Vector3f, outer_angle: f32) -> Self {
        let angle = outer_angle.clamp(0.5f32.to_radians(), 89.9f32.to_radians());
        Self {
            axis,
            inv_sin: 1.0 / angle.sin(),
            cos_squared: angle.cos() * angle.cos(),
        }
    }

    fn intersects(&self, apex: Vector3f, sphere: Vector4f) -> bool {
        let shifted_apex = apex - self.axis * (sphere.w * self.inv_sin);
        let offset = sphere.truncate() - shifted_apex;
        let along = self.axis.dot(offset);
        along > 0.0 && along * along >= offset.length_squared() * self.cos_squared
    }
}

/// A light as the froxelizer sees it: bounding sphere in view space, optional cone.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FroxelLight {
    pub(crate) position: Vector3f,
    pub(crate) radius: f32,
    pub(crate) cone: Option<FroxelCone>,
}

/// Per-frame light lists in the GPU layout, with scratch reused across frames.
pub(crate) struct Froxelizer {
    /// Light indices `records` may hold (`MAX_RECORDS`, the GPU buffer size).
    record_capacity: usize,
    /// Light bitset of every froxel, `words` u64 each.
    bits: Vec<u64>,
    words: usize,
    columns: Vec<bool>,
    rows: Vec<bool>,
    /// (offset, count) into `records` per froxel.
    pub(crate) entries: Vec<[u32; 2]>,
    /// Light indices; the first ones are all lights of the frame (the overflow fallback). Padded
    /// to an even length, the GPU reads them as u32 pairs.
    pub(crate) records: Vec<u16>,
}

impl Default for Froxelizer {
    fn default() -> Self {
        Self {
            record_capacity: MAX_RECORDS,
            bits: Vec::new(),
            words: 0,
            columns: Vec::new(),
            rows: Vec::new(),
            entries: Vec::new(),
            records: Vec::new(),
        }
    }
}

impl Froxelizer {
    /// Assigns `lights` (at most `MAX_LIGHTS`) to the froxels of `layout`. Returns false when the
    /// index list overflowed its capacity; froxels past that point then list every light.
    pub(crate) fn froxelize(&mut self, layout: &FroxelLayout, lights: &[FroxelLight]) -> bool {
        let lights = &lights[..lights.len().min(MAX_LIGHTS)];
        let froxel_count = layout.froxel_count();
        self.words = lights.len().div_ceil(64).max(1);
        self.bits.clear();
        self.bits.resize(froxel_count * self.words, 0);
        for (index, light) in lights.iter().enumerate() {
            self.add_light(layout, index, light);
        }
        let fits = self.compress(layout.count_x as usize);
        if self.records.len() % 2 == 1 {
            self.records.push(0);
        }
        fits
    }

    fn add_light(&mut self, layout: &FroxelLayout, index: usize, light: &FroxelLight) {
        let center = light.position;
        let radius = light.radius;
        let depth = -center.z;
        let near_depth = (depth - radius).max(layout.near);
        let far_depth = (depth + radius).min(layout.light_far);
        if far_depth < near_depth {
            return; // entirely in front of the near plane or beyond light_far
        }

        // Froxel range of the light's view-space box, projected at its near and far depth
        let mut min_ndc = (f32::INFINITY, f32::INFINITY);
        let mut max_ndc = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for box_depth in [near_depth, far_depth] {
            for (side_x, side_y) in [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                let ndc_x = (center.x + side_x * radius) / (box_depth * layout.tan_x);
                let ndc_y = (center.y + side_y * radius) / (box_depth * layout.tan_y);
                min_ndc = (min_ndc.0.min(ndc_x), min_ndc.1.min(ndc_y));
                max_ndc = (max_ndc.0.max(ndc_x), max_ndc.1.max(ndc_y));
            }
        }
        let (x0, x1) = (layout.column(min_ndc.0), layout.column(max_ndc.0));
        let (y0, y1) = (layout.row(max_ndc.1), layout.row(min_ndc.1));
        let (z0, z1) = (layout.slice(near_depth), layout.slice(far_depth));

        // Sphere vs the tile column/row planes, once per column and row
        self.columns.clear();
        self.columns.extend((x0..=x1).map(|x| {
            layout.planes_x[x as usize].dot(center) >= -radius
                && layout.planes_x[x as usize + 1].dot(center) <= radius
        }));
        self.rows.clear();
        self.rows.extend((y0..=y1).map(|y| {
            layout.planes_y[y as usize].dot(center) <= radius
                && layout.planes_y[y as usize + 1].dot(center) >= -radius
        }));

        let (word, bit) = (index / 64, 1u64 << (index % 64));
        for z in z0..=z1 {
            for (y, _) in (y0..=y1).zip(&self.rows).filter(|(_, hit)| **hit) {
                for (x, _) in (x0..=x1).zip(&self.columns).filter(|(_, hit)| **hit) {
                    let froxel = layout.froxel_index(x, y, z);
                    if let Some(cone) = &light.cone {
                        if !cone.intersects(center, layout.bounding_spheres[froxel]) {
                            continue;
                        }
                    }
                    self.bits[froxel * self.words + word] |= bit;
                }
            }
        }
    }

    /// Bitsets → entries + records (Filament `froxelizeAssignRecordsCompress`).
    fn compress(&mut self, count_x: usize) -> bool {
        let words = self.words;
        let froxel_count = self.bits.len() / words;
        let bits = &self.bits;
        let froxel_bits = |froxel: usize| &bits[froxel * words..(froxel + 1) * words];
        let push_lights = |records: &mut Vec<u16>, set: &[u64]| {
            for (word_index, &word) in set.iter().enumerate() {
                let mut remaining = word;
                while remaining != 0 {
                    records.push((word_index * 64 + remaining.trailing_zeros() as usize) as u16);
                    remaining &= remaining - 1;
                }
            }
        };

        // First record: every light of the frame, used for froxels past an overflow
        let mut all_lights = vec![0u64; words];
        for froxel in 0..froxel_count {
            for (all, word) in all_lights.iter_mut().zip(froxel_bits(froxel)) {
                *all |= word;
            }
        }
        self.records.clear();
        push_lights(&mut self.records, &all_lights);
        let all_count = self.records.len() as u32;

        self.entries.clear();
        self.entries.resize(froxel_count, [0, 0]);
        let mut froxel = 0;
        while froxel < froxel_count {
            let set = froxel_bits(froxel);
            if set.iter().all(|&word| word == 0) {
                froxel += 1;
                continue;
            }
            let count: u32 = set.iter().map(|word| word.count_ones()).sum();
            let offset = self.records.len();
            if offset + count as usize > self.record_capacity {
                for rest in froxel..froxel_count {
                    let lit = froxel_bits(rest).iter().any(|&word| word != 0);
                    self.entries[rest] = if lit { [0, all_count] } else { [0, 0] };
                }
                return false;
            }
            push_lights(&mut self.records, set);
            let mut entry = [offset as u32, count];
            // Following froxels with the same lights share the record; when the left neighbour
            // differs, the one above (already assigned) is tried
            let mut reference = froxel;
            loop {
                self.entries[froxel] = entry;
                froxel += 1;
                if froxel >= froxel_count {
                    break;
                }
                if froxel_bits(froxel) != froxel_bits(reference) && froxel >= count_x {
                    reference = froxel - count_x;
                    entry = self.entries[reference];
                }
                if froxel_bits(froxel) != froxel_bits(reference) {
                    break;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> FroxelLayout {
        FroxelLayout::new(FroxelLayoutParams {
            render_size: (1920, 1080),
            fov_y: 60f32.to_radians(),
            aspect: 16.0 / 9.0,
            near: 0.1,
            light_far: 100.0,
            first_slice_depth: 5.0,
        })
    }

    fn point(position: Vector3f, radius: f32) -> FroxelLight {
        FroxelLight {
            position,
            radius,
            cone: None,
        }
    }

    /// Light indices of a froxel, decoded from entries + records.
    fn lights_of(froxelizer: &Froxelizer, froxel: usize) -> Vec<u16> {
        let [offset, count] = froxelizer.entries[froxel];
        froxelizer.records[offset as usize..(offset + count) as usize].to_vec()
    }

    /// Froxel containing a view-space point.
    fn froxel_at(layout: &FroxelLayout, point: Vector3f) -> usize {
        let depth = -point.z;
        let x = layout.column(point.x / (depth * layout.tan_x));
        let y = layout.row(point.y / (depth * layout.tan_y));
        layout.froxel_index(x, y, layout.slice(depth))
    }

    #[test]
    fn layout_1080p_tiles_are_multiples_of_8_within_max_froxels() {
        let layout = layout();
        assert_eq!(layout.tile_size % 8, 0);
        assert!(layout.froxel_count() <= MAX_FROXELS);
        assert!(layout.count_x * layout.tile_size >= 1920);
        assert!(layout.count_y * layout.tile_size >= 1080);
    }

    #[test]
    fn slices_start_at_the_first_slice_depth_and_end_at_light_far() {
        let layout = layout();
        assert_eq!(layout.slice(1.0), 0);
        assert_eq!(layout.slice(5.01), 1);
        assert_eq!(layout.slice(99.0), FROXEL_SLICE_COUNT - 1);
        assert_eq!(layout.slice(1000.0), FROXEL_SLICE_COUNT - 1);
        // Every slice boundary maps to the slice starting there
        for slice in 1..FROXEL_SLICE_COUNT {
            let boundary = layout.distances_z[slice as usize];
            assert_eq!(layout.slice(boundary * 1.0001), slice);
            assert_eq!(layout.slice(boundary * 0.9999), slice - 1);
        }
    }

    #[test]
    fn point_light_marks_its_froxel_and_neighbours_but_not_far_ones() {
        let layout = layout();
        let mut froxelizer = Froxelizer::default();
        let center = Vector3f::new(0.3, -0.2, -8.0);
        assert!(froxelizer.froxelize(&layout, &[point(center, 1.0)]));
        assert_eq!(lights_of(&froxelizer, froxel_at(&layout, center)), vec![0]);
        let near_edge = center + Vector3f::new(0.9, 0.0, 0.0);
        assert_eq!(
            lights_of(&froxelizer, froxel_at(&layout, near_edge)),
            vec![0]
        );
        for offset in [
            Vector3f::new(3.0, 0.0, 0.0),
            Vector3f::new(-3.0, 0.0, 0.0),
            Vector3f::new(0.0, 3.0, 0.0),
            Vector3f::new(0.0, -3.0, 0.0),
        ] {
            let outside = center + offset;
            assert!(
                lights_of(&froxelizer, froxel_at(&layout, outside)).is_empty(),
                "{offset}"
            );
        }
        // Lit froxels: the sphere's 2 m extent at 8 m spans only a few tiles and slices
        let lit = froxelizer
            .entries
            .iter()
            .filter(|&&[_, count]| count > 0)
            .count();
        assert!(lit < 60, "{lit} froxels lit");
        let behind = center + Vector3f::new(0.0, 0.0, -20.0);
        assert!(lights_of(&froxelizer, froxel_at(&layout, behind)).is_empty());
    }

    #[test]
    fn light_beyond_light_far_or_behind_the_camera_marks_nothing() {
        let layout = layout();
        let mut froxelizer = Froxelizer::default();
        froxelizer.froxelize(
            &layout,
            &[
                point(Vector3f::new(0.0, 0.0, -150.0), 10.0),
                point(Vector3f::new(0.0, 0.0, 5.0), 1.0),
            ],
        );
        assert!(froxelizer.entries.iter().all(|&[_, count]| count == 0));
    }

    #[test]
    fn spot_light_marks_froxels_in_its_cone_only() {
        let layout = layout();
        let mut froxelizer = Froxelizer::default();
        let apex = Vector3f::new(0.0, 0.0, -10.0);
        // Narrow cone pointing right
        let spot = FroxelLight {
            position: apex,
            radius: 6.0,
            cone: Some(FroxelCone::new(Vector3f::X, 10f32.to_radians())),
        };
        froxelizer.froxelize(&layout, &[spot]);
        let ahead = apex + Vector3f::new(4.0, 0.0, 0.0);
        assert_eq!(lights_of(&froxelizer, froxel_at(&layout, ahead)), vec![0]);
        // Within range but well outside the cone. (The test is conservative: Filament's fast
        // sphere-cone test moves the apex back by the froxel radius / sin(angle).)
        let beside = apex + Vector3f::new(0.0, 5.0, 0.0);
        assert!(lights_of(&froxelizer, froxel_at(&layout, beside)).is_empty());
    }

    #[test]
    fn compressed_lists_match_the_bitsets() {
        let layout = layout();
        let mut froxelizer = Froxelizer::default();
        let lights: Vec<FroxelLight> = (0..300)
            .map(|index| {
                let t = index as f32;
                point(
                    Vector3f::new(
                        (t * 0.37).sin() * 20.0,
                        (t * 0.71).cos() * 6.0,
                        -3.0 - (t * 0.53) % 60.0,
                    ),
                    2.0 + (t * 0.13) % 3.0,
                )
            })
            .collect();
        assert!(froxelizer.froxelize(&layout, &lights));
        let words = froxelizer.words;
        for froxel in 0..layout.froxel_count() {
            let mut expected = Vec::new();
            for (word_index, &word) in froxelizer.bits[froxel * words..(froxel + 1) * words]
                .iter()
                .enumerate()
            {
                for bit in 0..64 {
                    if word & (1 << bit) != 0 {
                        expected.push((word_index * 64 + bit) as u16);
                    }
                }
            }
            assert_eq!(lights_of(&froxelizer, froxel), expected, "froxel {froxel}");
        }
        // Neighbour reuse keeps the list shorter than one list per lit froxel
        let lit = froxelizer
            .entries
            .iter()
            .filter(|&&[_, count]| count > 0)
            .count();
        let listed: usize = froxelizer
            .entries
            .iter()
            .map(|&[_, count]| count as usize)
            .sum();
        assert!(
            froxelizer.records.len() < listed,
            "{} records for {lit} lit froxels",
            froxelizer.records.len()
        );
    }

    #[test]
    fn record_overflow_falls_back_to_all_lights() {
        let layout = layout();
        let mut froxelizer = Froxelizer {
            record_capacity: 65536,
            ..Froxelizer::default()
        };
        // 1024 large lights scattered across the view: neighbouring froxels see different
        // subsets, far more indices than MAX_RECORDS
        let lights: Vec<FroxelLight> = (0..MAX_LIGHTS)
            .map(|index| {
                let t = index as f32;
                let position = Vector3f::new(
                    (t * 37.1).sin() * 30.0,
                    (t * 11.3).cos() * 10.0,
                    -5.0 - (t * 0.61) % 50.0,
                );
                point(position, 10.0)
            })
            .collect();
        assert!(!froxelizer.froxelize(&layout, &lights));
        assert!(froxelizer.records.len() <= 65536);
        // The last lit froxel is past the overflow: it lists every light of the frame, which is
        // the first record
        let &[offset, count] = froxelizer
            .entries
            .iter()
            .rev()
            .find(|entry| entry[1] > 0)
            .unwrap();
        assert_eq!(offset, 0);
        assert!(count > 0 && (count as usize) < froxelizer.records.len());
        let first: Vec<u16> = froxelizer.records[..count as usize].to_vec();
        let mut sorted = first.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(first, sorted);
    }

    #[test]
    fn lights_past_max_lights_are_dropped() {
        let layout = layout();
        let mut froxelizer = Froxelizer::default();
        let lights: Vec<FroxelLight> = (0..MAX_LIGHTS + 1)
            .map(|index| point(Vector3f::new(index as f32 * 0.01, 0.0, -10.0), 0.1))
            .collect();
        froxelizer.froxelize(&layout, &lights);
        assert!(froxelizer
            .records
            .iter()
            .all(|&light| (light as usize) < MAX_LIGHTS));
    }
}
