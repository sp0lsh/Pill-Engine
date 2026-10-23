//! Frame-to-frame state of the point/spot shadows (CPU only): which light owns which atlas tiles,
//! at what resolution, how strong its shadow is, and which casters are static.
//!
//! - Resolution and update rate by importance (id Tech 6): a light's importance is the on-screen
//!   radius of its range; it picks the tile size, and how often its dynamic tiles redraw.
//!   Resizing waits until the new size is wanted for a while (hysteresis), as a resize throws
//!   the cached tiles away.
//! - Fade instead of on/off (Source 2): a shadow's strength eases toward a target set by the
//!   light's screen size; a light gaining a slot fades in once its maps are drawn, a light losing
//!   one fades out before its tiles are freed.
//! - Cached static maps with a dynamic layer on top (id Tech 6, Doom Eternal): a light's static
//!   tiles hold the casters that don't move and redraw only when those or the light change;
//!   lights with moving casters nearby also get dynamic tiles, redrawn every frame or every few,
//!   holding only the moving casters. The shader combines both.

use crate::graphics::shadow_atlas::{ShadowAtlas, Tile};

use pill_core::{Matrix4f, Vector3f};
use std::collections::HashMap;

/// Tile size range (texels).
pub(crate) const MIN_TILE: u32 = 128;
pub(crate) const MAX_TILE: u32 = 1024;
/// Frames a different tile size must be wanted before a slot is resized.
const RESIZE_FRAMES: u32 = 30;
/// Frames a caster must keep its model matrix before it counts as static again.
const STATIC_AFTER_FRAMES: u64 = 30;
/// Frames dynamic tiles are kept after their light last had moving casters nearby.
const DYNAMIC_KEEP_FRAMES: u32 = 60;
/// A light without tiles evicts the least important slot only when this much more important, so
/// lights of similar importance don't keep evicting each other.
const EVICT_MARGIN: f32 = 1.5;

/// `PunctualLights` settings the cache uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CacheSettings {
    pub(crate) atlas_size: u32,
    /// Tile texels per pixel of on-screen light radius.
    pub(crate) resolution_scale: f32,
    /// Screen height ÷ 1080: fade thresholds and update rates are in pixels at 1080p, so they
    /// don't depend on the resolution (tile sizes do).
    pub(crate) height_scale: f32,
    /// On-screen light radius (px at 1080p) where the shadow starts fading, and where it is gone.
    pub(crate) fade_start: f32,
    pub(crate) fade_end: f32,
    /// Strength change per second.
    pub(crate) fade_speed: f32,
    /// Dynamic tiles of the least important lights redraw every this many frames.
    pub(crate) dynamic_update_far_frames: u32,
}

/// Stable identity of a light across frames (its entity).
pub(crate) type LightId = u64;

/// A shadow-casting light this frame.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LightRequest {
    pub(crate) id: LightId,
    pub(crate) is_point: bool,
    /// On-screen radius of the light's range, px (0 when not visible).
    pub(crate) importance: f32,
    /// Hash of what its maps depend on besides the casters: position, direction, range, cone.
    pub(crate) light_key: u64,
    /// Moving casters are within its range.
    pub(crate) near_dynamic: bool,
}

/// A light's tiles and shadow state.
#[derive(Debug)]
pub(crate) struct Slot {
    pub(crate) is_point: bool,
    pub(crate) size: u32,
    /// One per map: 6 cube faces (point) or 1 (spot).
    pub(crate) static_tiles: Vec<Tile>,
    /// Same count as `static_tiles` when the light has moving casters nearby, else empty.
    pub(crate) dynamic_tiles: Vec<Tile>,
    pub(crate) strength: f32,
    pub(crate) importance: f32,
    /// Fading out, then freed.
    pub(crate) evicting: bool,
    /// The static tiles have been drawn at least once (until then the strength stays 0).
    pub(crate) static_ready: bool,
    /// What the static tiles were last drawn with: light key, static-caster hash per map (empty
    /// while the current tiles hold no draw), and the scene's static signature then.
    pub(crate) drawn_light_key: u64,
    pub(crate) drawn_caster_hashes: Vec<u64>,
    pub(crate) drawn_signature: u64,
    /// The last static draw ran out of caster draws: redraw when there's room.
    pub(crate) static_dirty: bool,
    /// The dynamic tiles have been drawn since they were allocated.
    pub(crate) dynamic_drawn: bool,
    /// Frames since its dynamic tiles last had moving casters to draw.
    pub(crate) dynamic_idle: u32,
    /// Its dynamic tiles redraw on frames where `frame % period == phase`.
    pub(crate) phase: u32,
    pending_size: u32,
    pending_frames: u32,
}

impl Slot {
    pub(crate) fn map_count(&self) -> usize {
        if self.is_point {
            6
        } else {
            1
        }
    }

    /// Whether its dynamic tiles redraw this frame: every frame when important, else every 2nd or
    /// `far_frames`-th frame, staggered by light.
    pub(crate) fn dynamic_due(&self, frame: u64, settings: &CacheSettings) -> bool {
        let importance = self.importance / settings.height_scale.max(1e-3);
        let period = if importance >= settings.fade_start * 4.0 {
            1
        } else if importance >= settings.fade_start * 2.0 {
            2
        } else {
            settings.dynamic_update_far_frames.max(1)
        };
        frame % period as u64 == (self.phase % period) as u64
    }
}

/// Tile size for an importance: the next power of two of `importance · scale`, clamped.
pub(crate) fn desired_tile_size(importance: f32, settings: &CacheSettings) -> u32 {
    let texels = (importance * settings.resolution_scale).clamp(MIN_TILE as f32, MAX_TILE as f32);
    (texels.ceil() as u32)
        .next_power_of_two()
        .clamp(MIN_TILE, MAX_TILE.min(settings.atlas_size))
}

/// Shadow strength a light's screen size calls for: 1 above `fade_start`, 0 below `fade_end`
/// (both at 1080p).
pub(crate) fn fade_target(importance: f32, settings: &CacheSettings) -> f32 {
    let importance = importance / settings.height_scale.max(1e-3);
    let span = (settings.fade_start - settings.fade_end).max(1e-3);
    ((importance - settings.fade_end) / span).clamp(0.0, 1.0)
}

/// On-screen radius (px) of a light's range sphere seen from `distance`; a camera inside the
/// range counts as if at the range's edge.
pub(crate) fn light_importance(range: f32, distance: f32, pixels_per_unit_at_1m: f32) -> f32 {
    range / distance.max(range) * pixels_per_unit_at_1m
}

#[derive(Default)]
pub(crate) struct ShadowCache {
    atlas: Option<ShadowAtlas>,
    pub(crate) slots: HashMap<LightId, Slot>,
    pub(crate) frame: u64,
}

impl ShadowCache {
    pub(crate) fn atlas_size(&self) -> u32 {
        self.atlas.as_ref().map_or(0, ShadowAtlas::size)
    }

    fn allocate_tiles(atlas: &mut ShadowAtlas, size: u32, count: usize) -> Option<Vec<Tile>> {
        let mut tiles = Vec::with_capacity(count);
        for _ in 0..count {
            match atlas.allocate(size) {
                Some(tile) => tiles.push(tile),
                None => {
                    for tile in tiles {
                        atlas.free(tile);
                    }
                    return None;
                }
            }
        }
        Some(tiles)
    }

    /// Advances one frame of `delta_time` seconds with this frame's shadow-casting lights.
    pub(crate) fn update(
        &mut self,
        requests: &[LightRequest],
        settings: &CacheSettings,
        delta_time: f32,
    ) {
        self.frame += 1;
        let atlas_size = settings.atlas_size.next_power_of_two().max(MIN_TILE);
        if self.atlas.as_ref().map(ShadowAtlas::size) != Some(atlas_size) {
            self.atlas = Some(ShadowAtlas::new(atlas_size, MIN_TILE));
            self.slots.clear();
        }
        let atlas = self.atlas.as_mut().expect("atlas created above");
        let requested: HashMap<LightId, &LightRequest> = requests
            .iter()
            .map(|request| (request.id, request))
            .collect();

        // --- Existing slots: gone lights fade out, resizes after hysteresis, dynamic tiles ---
        for (id, slot) in self.slots.iter_mut() {
            let Some(request) = requested.get(id) else {
                slot.evicting = true;
                slot.importance = 0.0;
                continue;
            };
            slot.importance = request.importance;
            if slot.is_point != request.is_point {
                slot.evicting = true;
                continue;
            }
            let desired = desired_tile_size(request.importance, settings);
            if desired == slot.size {
                slot.pending_frames = 0;
            } else {
                if desired != slot.pending_size {
                    slot.pending_size = desired;
                    slot.pending_frames = 0;
                }
                slot.pending_frames += 1;
                if slot.pending_frames >= RESIZE_FRAMES {
                    let count = slot.map_count();
                    for tile in slot
                        .static_tiles
                        .drain(..)
                        .chain(slot.dynamic_tiles.drain(..))
                    {
                        atlas.free(tile);
                    }
                    let size = match Self::allocate_tiles(atlas, desired, count) {
                        Some(tiles) => {
                            slot.static_tiles = tiles;
                            desired
                        }
                        // No room at the new size: back to the old one (its tiles just freed)
                        None => {
                            slot.static_tiles =
                                Self::allocate_tiles(atlas, slot.size, count).unwrap_or_default();
                            slot.size
                        }
                    };
                    slot.size = size;
                    slot.pending_frames = 0;
                    slot.drawn_caster_hashes.clear();
                    slot.dynamic_drawn = false;
                    if slot.static_tiles.is_empty() {
                        slot.evicting = true;
                        slot.strength = 0.0;
                    }
                }
            }
            if request.near_dynamic {
                slot.dynamic_idle = 0;
                if slot.dynamic_tiles.is_empty() && !slot.evicting {
                    slot.dynamic_tiles = Self::allocate_tiles(atlas, slot.size, slot.map_count())
                        .unwrap_or_default();
                    slot.dynamic_drawn = false;
                }
            } else if !slot.dynamic_tiles.is_empty() {
                slot.dynamic_idle += 1;
                if slot.dynamic_idle >= DYNAMIC_KEEP_FRAMES {
                    for tile in slot.dynamic_tiles.drain(..) {
                        atlas.free(tile);
                    }
                }
            }
        }

        // --- New lights, most important first; a full atlas evicts a less important slot ---
        let mut newcomers: Vec<&LightRequest> = requests
            .iter()
            .filter(|request| {
                !self.slots.contains_key(&request.id)
                    && fade_target(request.importance, settings) > 0.0
            })
            .collect();
        newcomers.sort_by(|a, b| b.importance.total_cmp(&a.importance));
        let mut evicted_this_frame = false;
        for request in newcomers {
            let count = if request.is_point { 6 } else { 1 };
            let mut size = desired_tile_size(request.importance, settings);
            let tiles = loop {
                if let Some(tiles) = Self::allocate_tiles(atlas, size, count) {
                    break Some(tiles);
                }
                if size <= MIN_TILE {
                    break None;
                }
                size /= 2;
            };
            match tiles {
                Some(tiles) => {
                    self.slots.insert(
                        request.id,
                        Slot {
                            is_point: request.is_point,
                            size,
                            static_tiles: tiles,
                            dynamic_tiles: Vec::new(),
                            strength: 0.0,
                            importance: request.importance,
                            evicting: false,
                            static_ready: false,
                            drawn_light_key: request.light_key,
                            drawn_caster_hashes: Vec::new(),
                            drawn_signature: 0,
                            static_dirty: false,
                            dynamic_drawn: false,
                            dynamic_idle: 0,
                            phase: (request.id as u32).wrapping_mul(2_654_435_761) >> 28,
                            pending_size: size,
                            pending_frames: 0,
                        },
                    );
                }
                // One eviction per frame, of the least important slot if this light matters
                // `EVICT_MARGIN` times more:
                // it fades out, and its tiles go to whoever is first in line once it's gone
                None if !evicted_this_frame => {
                    let weakest = self
                        .slots
                        .iter_mut()
                        .filter(|(_, slot)| !slot.evicting)
                        .min_by(|a, b| a.1.importance.total_cmp(&b.1.importance));
                    if let Some((_, slot)) = weakest {
                        if slot.importance * EVICT_MARGIN < request.importance {
                            slot.evicting = true;
                            evicted_this_frame = true;
                        }
                    }
                }
                None => {}
            }
        }

        // --- Strength: ease toward the target; fully faded evictions free their tiles ---
        let step = settings.fade_speed.max(0.0) * delta_time.max(0.0);
        let mut freed = Vec::new();
        for (id, slot) in self.slots.iter_mut() {
            let target = if slot.evicting || !slot.static_ready {
                0.0
            } else {
                fade_target(slot.importance, settings)
            };
            slot.strength += (target - slot.strength).clamp(-step, step);
            if slot.evicting && slot.strength <= 0.0 {
                freed.push(*id);
            }
        }
        for id in freed {
            if let Some(slot) = self.slots.remove(&id) {
                for tile in slot.static_tiles.into_iter().chain(slot.dynamic_tiles) {
                    atlas.free(tile);
                }
            }
        }
    }
}

/// splitmix64 finalizer: scrambles a 64-bit value.
pub(crate) fn mix(hash: u64) -> u64 {
    let mut value = hash.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// Hash of floats, bit-exact.
pub(crate) fn hash_floats(values: &[f32]) -> u64 {
    hash_floats_from(0xCBF2_9CE4_8422_2325, values)
}

/// Hash of floats, bit-exact, continuing from `seed`.
pub(crate) fn hash_floats_from(seed: u64, values: &[f32]) -> u64 {
    values
        .iter()
        .fold(seed, |hash, value| mix(hash ^ value.to_bits() as u64))
}

/// Whether each caster moved recently, by entity slot: a caster is dynamic while its model matrix
/// changed within the last `STATIC_AFTER_FRAMES` frames (or when forced: skinned, `gi_dynamic`).
#[derive(Default)]
pub(crate) struct CasterTracker {
    /// Per entity slot: model matrix hash and the frame it last changed.
    states: Vec<Option<(u64, u64)>>,
    frame: u64,
}

impl CasterTracker {
    pub(crate) fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Records the caster at entity `slot` this frame; returns whether it is dynamic.
    pub(crate) fn update(&mut self, slot: usize, model: &Matrix4f, forced: bool) -> bool {
        if self.states.len() <= slot {
            self.states.resize(slot + 1, None);
        }
        let hash = hash_floats(&model.to_cols_array());
        let state = &mut self.states[slot];
        match state {
            // First sight: static (a level loads standing still)
            None => *state = Some((hash, 0)),
            Some((previous, changed)) => {
                if *previous != hash {
                    *previous = hash;
                    *changed = self.frame;
                }
            }
        }
        let changed = state.map_or(0, |(_, changed)| changed);
        forced || (changed > 0 && self.frame - changed < STATIC_AFTER_FRAMES)
    }
}

/// Hash of a light's shadow inputs: what changes its maps besides the casters.
pub(crate) fn light_key(
    position: Vector3f,
    direction: Vector3f,
    range: f32,
    spot_outer: Option<f32>,
) -> u64 {
    hash_floats(&[
        position.x,
        position.y,
        position.z,
        direction.x,
        direction.y,
        direction.z,
        range,
        spot_outer.unwrap_or(-1.0),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> CacheSettings {
        CacheSettings {
            atlas_size: 2048,
            resolution_scale: 1.0,
            height_scale: 1.0,
            fade_start: 96.0,
            fade_end: 48.0,
            fade_speed: 4.0,
            dynamic_update_far_frames: 4,
        }
    }

    fn point(id: LightId, importance: f32) -> LightRequest {
        LightRequest {
            id,
            is_point: true,
            importance,
            light_key: 1,
            near_dynamic: false,
        }
    }

    /// Runs `frames` frames of 1/60 s; marks every slot's static tiles drawn after each.
    fn run(cache: &mut ShadowCache, requests: &[LightRequest], frames: usize) {
        for _ in 0..frames {
            cache.update(requests, &settings(), 1.0 / 60.0);
            for slot in cache.slots.values_mut() {
                slot.static_ready = true;
            }
        }
    }

    #[test]
    fn desired_size_follows_screen_radius_clamped() {
        let settings = settings();
        assert_eq!(desired_tile_size(10.0, &settings), MIN_TILE);
        assert_eq!(desired_tile_size(300.0, &settings), 512);
        assert_eq!(desired_tile_size(5000.0, &settings), MAX_TILE);
    }

    #[test]
    fn fade_target_ramps_between_end_and_start() {
        let settings = settings();
        assert_eq!(fade_target(40.0, &settings), 0.0);
        assert_eq!(fade_target(72.0, &settings), 0.5);
        assert_eq!(fade_target(200.0, &settings), 1.0);
    }

    #[test]
    fn new_light_fades_in_only_after_its_maps_are_drawn() {
        let mut cache = ShadowCache::default();
        cache.update(&[point(1, 300.0)], &settings(), 0.1);
        assert_eq!(cache.slots[&1].strength, 0.0, "not drawn yet");
        assert_eq!(cache.slots[&1].static_tiles.len(), 6);
        run(&mut cache, &[point(1, 300.0)], 30);
        assert_eq!(cache.slots[&1].strength, 1.0);
    }

    #[test]
    fn removed_light_fades_out_over_quarter_second_then_frees_tiles() {
        let mut cache = ShadowCache::default();
        run(&mut cache, &[point(1, 300.0)], 30);
        // 4/s fade: 15 frames at 60 fps
        run(&mut cache, &[], 14);
        assert!(cache.slots[&1].strength > 0.0 && cache.slots[&1].evicting);
        run(&mut cache, &[], 2);
        assert!(cache.slots.is_empty());
        // All of the atlas is free again: a full-size tile fits
        assert!(cache.atlas.as_mut().unwrap().allocate(2048).is_some());
    }

    #[test]
    fn size_change_shorter_than_hysteresis_keeps_size() {
        let mut cache = ShadowCache::default();
        let spot = |importance| LightRequest {
            is_point: false,
            ..point(1, importance)
        };
        run(&mut cache, &[spot(300.0)], 5);
        assert_eq!(cache.slots[&1].size, 512);
        run(&mut cache, &[spot(900.0)], RESIZE_FRAMES as usize - 1);
        assert_eq!(cache.slots[&1].size, 512);
        run(&mut cache, &[spot(300.0)], 5);
        run(&mut cache, &[spot(900.0)], RESIZE_FRAMES as usize);
        assert_eq!(cache.slots[&1].size, 1024);
        assert!(
            cache.slots[&1].drawn_caster_hashes.is_empty(),
            "resize redraws"
        );
    }

    #[test]
    fn full_atlas_evicts_least_important_slot_then_admits_newcomer() {
        let mut cache = ShadowCache::default();
        let spot = |id, importance| LightRequest {
            is_point: false,
            ..point(id, importance)
        };
        // 2048² holds four 1024 tiles: four spots fill it
        let lights = [
            spot(1, 2000.0),
            spot(2, 2000.0),
            spot(3, 2000.0),
            spot(4, 1100.0),
        ];
        run(&mut cache, &lights, 30);
        assert_eq!(cache.slots.len(), 4);
        let mut more = lights.to_vec();
        more.push(spot(5, 3000.0));
        run(&mut cache, &more, 1);
        assert!(cache.slots[&4].evicting, "least important fades out");
        assert!(!cache.slots.contains_key(&5));
        run(&mut cache, &more, 40);
        assert!(!cache.slots.contains_key(&4));
        assert_eq!(cache.slots[&5].size, 1024);
        assert_eq!(cache.slots[&5].strength, 1.0);
    }

    #[test]
    fn full_atlas_newcomer_within_evict_margin_evicts_nothing() {
        let mut cache = ShadowCache::default();
        let spot = |id, importance| LightRequest {
            is_point: false,
            ..point(id, importance)
        };
        let lights = [
            spot(1, 2000.0),
            spot(2, 2000.0),
            spot(3, 2000.0),
            spot(4, 1100.0),
        ];
        run(&mut cache, &lights, 30);
        let mut more = lights.to_vec();
        more.push(spot(5, 1500.0));
        run(&mut cache, &more, 60);
        assert!(cache.slots.contains_key(&4) && !cache.slots[&4].evicting);
        assert!(!cache.slots.contains_key(&5));
    }

    #[test]
    fn dynamic_tiles_allocated_with_moving_casters_and_freed_when_idle() {
        let mut cache = ShadowCache::default();
        let mut light = point(1, 300.0);
        light.near_dynamic = true;
        // Admitted on the first frame, dynamic tiles from the next
        run(&mut cache, &[light], 2);
        assert_eq!(cache.slots[&1].dynamic_tiles.len(), 6);
        light.near_dynamic = false;
        run(&mut cache, &[light], DYNAMIC_KEEP_FRAMES as usize - 1);
        assert_eq!(cache.slots[&1].dynamic_tiles.len(), 6);
        run(&mut cache, &[light], 1);
        assert!(cache.slots[&1].dynamic_tiles.is_empty());
    }

    #[test]
    fn dynamic_updates_staggered_for_far_lights() {
        let settings = settings();
        let mut cache = ShadowCache::default();
        run(&mut cache, &[point(1, 70.0)], 1);
        let slot = &cache.slots[&1];
        let due = (0..8)
            .filter(|&frame| slot.dynamic_due(frame, &settings))
            .count();
        assert_eq!(due, 2, "every 4th frame");
    }

    #[test]
    fn caster_static_until_moved_then_dynamic_for_a_while() {
        let mut tracker = CasterTracker::default();
        let still = Matrix4f::IDENTITY;
        let moved = Matrix4f::from_translation(Vector3f::X);
        tracker.begin_frame();
        assert!(!tracker.update(3, &still, false), "first sight is static");
        tracker.begin_frame();
        assert!(!tracker.update(3, &still, false));
        tracker.begin_frame();
        assert!(tracker.update(3, &moved, false), "moved");
        for _ in 0..STATIC_AFTER_FRAMES - 1 {
            tracker.begin_frame();
            assert!(tracker.update(3, &moved, false));
        }
        tracker.begin_frame();
        assert!(!tracker.update(3, &moved, false), "still long enough");
        tracker.begin_frame();
        assert!(tracker.update(3, &moved, true), "forced (skinned)");
    }

    #[test]
    fn light_key_changes_with_light_not_camera() {
        let key = light_key(Vector3f::ONE, Vector3f::NEG_Z, 5.0, None);
        assert_eq!(key, light_key(Vector3f::ONE, Vector3f::NEG_Z, 5.0, None));
        assert_ne!(
            key,
            light_key(Vector3f::ONE * 1.01, Vector3f::NEG_Z, 5.0, None)
        );
        assert_ne!(
            key,
            light_key(Vector3f::ONE, Vector3f::NEG_Z, 5.0, Some(0.5))
        );
    }
}
