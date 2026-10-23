//! Tile allocator of the point/spot shadow atlas: square power-of-two tiles carved out of one
//! square power-of-two texture by repeated quartering (a quadtree buddy allocator). Freeing a tile
//! merges it back with its three siblings once they are all free, so the atlas doesn't fragment
//! into small tiles for good.

/// A square region of the atlas, in texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Tile {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) size: u32,
}

impl Tile {
    /// Atlas uv scale and offset of the tile: `uv_atlas = uv_tile · xy + zw`.
    pub(crate) fn uv_rect(&self, atlas_size: u32) -> [f32; 4] {
        let atlas = atlas_size as f32;
        [
            self.size as f32 / atlas,
            self.size as f32 / atlas,
            self.x as f32 / atlas,
            self.y as f32 / atlas,
        ]
    }
}

pub(crate) struct ShadowAtlas {
    size: u32,
    min_tile: u32,
    /// Free tiles per level: level 0 is the whole atlas, each next level a quarter of the size.
    free: Vec<Vec<(u32, u32)>>,
}

impl ShadowAtlas {
    /// `size` and `min_tile` are powers of two, `min_tile` ≤ `size`.
    pub(crate) fn new(size: u32, min_tile: u32) -> Self {
        debug_assert!(size.is_power_of_two() && min_tile.is_power_of_two() && min_tile <= size);
        let levels = (size / min_tile).ilog2() as usize + 1;
        let mut free = vec![Vec::new(); levels];
        free[0].push((0, 0));
        Self {
            size,
            min_tile,
            free,
        }
    }

    pub(crate) fn size(&self) -> u32 {
        self.size
    }

    /// Level of a tile size (clamped to the atlas' sizes).
    fn level(&self, size: u32) -> usize {
        let size = size.clamp(self.min_tile, self.size).next_power_of_two();
        (self.size / size).ilog2() as usize
    }

    fn tile_size(&self, level: usize) -> u32 {
        self.size >> level
    }

    /// A free tile of `size` (rounded up to a power of two, clamped to the atlas), splitting a
    /// larger free tile if needed; `None` when nothing that large is free.
    pub(crate) fn allocate(&mut self, size: u32) -> Option<Tile> {
        let level = self.level(size);
        let source = (0..=level)
            .rev()
            .find(|&level| !self.free[level].is_empty())?;
        // Lowest (y, x) first, so allocation is deterministic and packs from the top left
        let pick = self.free[source]
            .iter()
            .enumerate()
            .min_by_key(|(_, &(x, y))| (y, x))
            .map(|(index, _)| index)?;
        let (x, y) = self.free[source].swap_remove(pick);
        for split in source..level {
            let half = self.tile_size(split + 1);
            // Keep the top-left quarter, free the other three
            self.free[split + 1].extend([(x + half, y), (x, y + half), (x + half, y + half)]);
        }
        Some(Tile {
            x,
            y,
            size: self.tile_size(level),
        })
    }

    /// Returns a tile from `allocate`, merging it with its siblings into their parent while all
    /// four are free.
    pub(crate) fn free(&mut self, tile: Tile) {
        let mut level = self.level(tile.size);
        let (mut x, mut y) = (tile.x, tile.y);
        while level > 0 {
            let size = self.tile_size(level);
            let (parent_x, parent_y) = (x - x % (size * 2), y - y % (size * 2));
            let siblings = [
                (parent_x, parent_y),
                (parent_x + size, parent_y),
                (parent_x, parent_y + size),
                (parent_x + size, parent_y + size),
            ];
            let all_free = siblings
                .iter()
                .all(|&sibling| sibling == (x, y) || self.free[level].contains(&sibling));
            if !all_free {
                break;
            }
            self.free[level].retain(|position| !siblings.contains(position));
            (x, y) = (parent_x, parent_y);
            level -= 1;
        }
        self.free[level].push((x, y));
    }

    /// Texels of the atlas still free.
    #[cfg(test)]
    fn free_area(&self) -> u64 {
        self.free
            .iter()
            .enumerate()
            .map(|(level, tiles)| tiles.len() as u64 * (self.tile_size(level) as u64).pow(2))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn overlaps(a: &Tile, b: &Tile) -> bool {
        a.x < b.x + b.size && b.x < a.x + a.size && a.y < b.y + b.size && b.y < a.y + a.size
    }

    #[test]
    fn atlas_4096_fits_sixteen_1024_tiles_then_full() {
        let mut atlas = ShadowAtlas::new(4096, 128);
        let tiles: Vec<Tile> = (0..16).map(|_| atlas.allocate(1024).unwrap()).collect();
        assert!(atlas.allocate(1024).is_none());
        assert!(atlas.allocate(128).is_none());
        for (index, a) in tiles.iter().enumerate() {
            assert_eq!(a.size, 1024);
            assert!(a.x + a.size <= 4096 && a.y + a.size <= 4096);
            for b in &tiles[index + 1..] {
                assert!(!overlaps(a, b), "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn mixed_sizes_never_overlap_and_account_for_all_area() {
        let mut atlas = ShadowAtlas::new(2048, 128);
        let sizes = [512, 128, 1024, 256, 128, 256, 512, 128];
        let tiles: Vec<Tile> = sizes
            .iter()
            .filter_map(|&size| atlas.allocate(size))
            .collect();
        assert_eq!(tiles.len(), sizes.len());
        for (index, a) in tiles.iter().enumerate() {
            for b in &tiles[index + 1..] {
                assert!(!overlaps(a, b), "{a:?} overlaps {b:?}");
            }
        }
        let used: u64 = tiles.iter().map(|tile| (tile.size as u64).pow(2)).sum();
        assert_eq!(atlas.free_area() + used, 2048 * 2048);
    }

    #[test]
    fn freeing_everything_merges_back_to_one_tile() {
        let mut atlas = ShadowAtlas::new(1024, 128);
        let tiles: Vec<Tile> = (0..64).map(|_| atlas.allocate(128).unwrap()).collect();
        assert!(atlas.allocate(128).is_none());
        for tile in tiles {
            atlas.free(tile);
        }
        assert_eq!(
            atlas.allocate(1024),
            Some(Tile {
                x: 0,
                y: 0,
                size: 1024
            })
        );
    }

    #[test]
    fn freed_tile_reused_by_next_allocation_of_its_size() {
        let mut atlas = ShadowAtlas::new(1024, 256);
        let first = atlas.allocate(256).unwrap();
        let _second = atlas.allocate(256).unwrap();
        atlas.free(first);
        assert_eq!(atlas.allocate(256), Some(first));
    }

    #[test]
    fn size_rounded_up_and_clamped_to_atlas_range() {
        let mut atlas = ShadowAtlas::new(1024, 128);
        assert_eq!(atlas.allocate(300).unwrap().size, 512);
        assert_eq!(atlas.allocate(16).unwrap().size, 128);
    }

    #[test]
    fn uv_rect_maps_tile_into_atlas() {
        let tile = Tile {
            x: 1024,
            y: 512,
            size: 512,
        };
        assert_eq!(tile.uv_rect(2048), [0.25, 0.25, 0.5, 0.25]);
    }
}
