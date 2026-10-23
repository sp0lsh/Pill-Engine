use crate::ecs::{LightComponent, LightType, RenderStateComponent, TransformComponent};
use crate::graphics::froxelizer::{
    FroxelCone, FroxelLayout, FroxelLayoutParams, FroxelLight, Froxelizer, MAX_FROXELS, MAX_LIGHTS,
    MAX_RECORDS,
};
use crate::graphics::punctual_shadow_cache::{
    hash_floats_from, light_importance, light_key, mix, CacheSettings, CasterTracker, LightId,
    LightRequest, ShadowCache,
};
use crate::graphics::punctual_shadows::{
    cube_face_fov, point_face_view_projection, spot_fov, spot_view_projection, texel_scale,
    MAX_SHADOW_RECORDS, MAX_TILE_DRAWS,
};
use crate::graphics::shader_bindings::structs::{PunctualLight, PunctualShadow};
use crate::graphics::shadow_atlas::Tile;
use crate::graphics::shadow_casters::{
    frustum_planes, gather_casters, keep_nearest, outside_frustum, CasterCandidate,
};
use crate::graphics::{camera_forward, BufferDesc, FrameTarget, Pass, PillRenderer, WorldQuery};

use pill_core::{debug, LogContext, Matrix4f, PillSlotMapKey, Result, Vector3f};
use std::cell::RefCell;
use std::rc::Rc;

/// An atlas tile to draw this frame: a static tile (the casters that don't move) or a dynamic one
/// (only the moving casters).
#[derive(Clone, Copy)]
pub(crate) struct ShadowJob {
    /// World → light clip space; the far plane is the light's range.
    pub(crate) view_projection: Matrix4f,
    /// The light: its casters are those within `light_range` of `light_position`, nearest kept
    /// first when caster draws run out.
    pub(crate) light: LightId,
    pub(crate) light_position: Vector3f,
    pub(crate) light_range: f32,
    pub(crate) tile: Tile,
    pub(crate) dynamic: bool,
}

/// A light of the scene before the `MAX_LIGHTS` cut (nearest to the camera are kept).
struct GatheredLight {
    id: LightId,
    camera_distance_squared: f32,
    gpu: PunctualLight,
    froxel: FroxelLight,
    shadow: Option<ShadowSource>,
}

/// What a light needs for its shadow maps, gathered with the light.
#[derive(Clone, Copy)]
struct ShadowSource {
    position: Vector3f,
    direction: Vector3f,
    range: f32,
    /// Spot cone half-angle (radians); `None` for point lights.
    spot_outer: Option<f32>,
}

/// Storage buffers read by the PBR fragment shader (group 0, bindings 10–12).
#[derive(Clone)]
pub(crate) struct LightBuffers {
    pub(crate) lights: wgpu::Buffer,
    /// (offset, count) per froxel.
    pub(crate) entries: wgpu::Buffer,
    /// u16 light indices, two per u32.
    pub(crate) records: wgpu::Buffer,
    /// Point/spot shadow maps: light matrix and params.
    pub(crate) shadows: wgpu::Buffer,
}

impl LightBuffers {
    pub(crate) fn new(
        renderer: &mut dyn PillRenderer,
        light_capacity: usize,
        entry_capacity: usize,
        record_capacity: usize,
        shadow_capacity: usize,
    ) -> Result<Self> {
        let mut storage = |label: &str, byte_size: usize| {
            renderer.create_buffer(BufferDesc {
                label: Some(label),
                byte_size: byte_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            lights: storage(
                "punctual_lights",
                light_capacity.max(1) * std::mem::size_of::<PunctualLight>(),
            )?,
            entries: storage("froxel_entries", entry_capacity.max(1) * 8)?,
            records: storage(
                "froxel_records",
                record_capacity.max(2).next_multiple_of(2) * 2,
            )?,
            shadows: storage(
                "punctual_shadows",
                shadow_capacity.max(1) * std::mem::size_of::<PunctualShadow>(),
            )?,
        })
    }
}

/// Froxel grid and light count of the current frame, for the scene uniform.
#[derive(Clone, Copy, Default)]
pub(crate) struct LightsParams {
    /// Columns, rows, tile size (px), light count.
    pub(crate) grid: [f32; 4],
    /// 1 / slice linearizer, light far, slice count, 1 = heatmap.
    pub(crate) depth: [f32; 4],
    pub(crate) camera_forward: [f32; 4],
}

/// Light buffers and this frame's grid and shadow work, shared by `PassLights`,
/// `PassPunctualShadows` and the PBR passes.
pub(crate) struct LightsFrame {
    pub(crate) buffers: LightBuffers,
    pub(crate) params: LightsParams,
    /// Atlas tiles to draw this frame, most important light first.
    pub(crate) shadow_jobs: Vec<ShadowJob>,
    /// Width and height of the shadow atlas (0 = none yet).
    pub(crate) atlas_size: u32,
    /// This frame's shadow casters, and per caster whether it is dynamic (moving).
    pub(crate) casters: Vec<CasterCandidate>,
    pub(crate) caster_dynamic: Vec<bool>,
    /// Lights whose static tiles ran out of caster draws (`PassPunctualShadows`), redrawn later.
    pub(crate) truncated_lights: Vec<LightId>,
    /// Shadow atlas view, set by `PassPunctualShadows`; `generation` changes when it is
    /// recreated.
    pub(crate) shadow_view: Option<wgpu::TextureView>,
    pub(crate) shadow_generation: u64,
}

impl LightsFrame {
    pub(crate) fn new(buffers: LightBuffers) -> Self {
        Self {
            buffers,
            params: LightsParams::default(),
            shadow_jobs: Vec::new(),
            atlas_size: 0,
            casters: Vec::new(),
            caster_dynamic: Vec::new(),
            truncated_lights: Vec::new(),
            shadow_view: None,
            shadow_generation: 0,
        }
    }
}

/// Clustered forward lighting, CPU side: gathers the `LightComponent`s, assigns them to the
/// froxels of the active camera (`froxelizer.rs`) and uploads lights and froxel lists for the PBR
/// passes. No GPU pass of its own. Settings: `RenderStateComponent::punctual_lights`.
pub(crate) struct PassLights {
    frame: Rc<RefCell<LightsFrame>>,
    layout: Option<FroxelLayout>,
    froxelizer: Froxelizer,
    // Per-frame scratch, capacity reused across frames
    gpu_lights: Vec<PunctualLight>,
    froxel_lights: Vec<FroxelLight>,
    gathered: Vec<GatheredLight>,
    // Per light: its id, shadow inputs and whether it casts
    light_ids: Vec<LightId>,
    shadow_sources: Vec<Option<ShadowSource>>,
    // Shadow state kept across frames
    shadow_cache: ShadowCache,
    caster_tracker: CasterTracker,
    // Per caster this frame: its hash when static (0 when moving)
    caster_hashes: Vec<u64>,
    skinned: Vec<crate::graphics::RendererMeshHandle>,
    // Per light: its `light_key` this frame
    light_keys: Vec<u64>,
    // Per-frame scratch
    shadow_requests: Vec<LightRequest>,
    shadow_order: Vec<usize>,
    gpu_shadow_maps: Vec<PunctualShadow>,
    overflow_logged: bool,
    /// `PILL_SHADOW_STATS=<file>`: one CSV line of shadow-cache stats per frame (native debug aid;
    /// the game's log doesn't reach stdout). `None` until first checked.
    #[cfg(not(target_arch = "wasm32"))]
    shadow_stats: Option<Option<std::io::BufWriter<std::fs::File>>>,
}

impl PassLights {
    pub(crate) fn new(frame: Rc<RefCell<LightsFrame>>) -> Self {
        Self {
            frame,
            layout: None,
            froxelizer: Froxelizer::default(),
            gpu_lights: Vec::new(),
            froxel_lights: Vec::new(),
            gathered: Vec::new(),
            light_ids: Vec::new(),
            shadow_sources: Vec::new(),
            shadow_cache: ShadowCache::default(),
            caster_tracker: CasterTracker::default(),
            caster_hashes: Vec::new(),
            skinned: Vec::new(),
            light_keys: Vec::new(),
            shadow_requests: Vec::new(),
            shadow_order: Vec::new(),
            gpu_shadow_maps: Vec::new(),
            overflow_logged: false,
            #[cfg(not(target_arch = "wasm32"))]
            shadow_stats: None,
        }
    }

    /// Buffers sized for `MAX_LIGHTS` lights and the largest froxel grid.
    pub(crate) fn create_buffers(renderer: &mut dyn PillRenderer) -> Result<LightBuffers> {
        LightBuffers::new(
            renderer,
            MAX_LIGHTS,
            MAX_FROXELS,
            MAX_RECORDS,
            MAX_SHADOW_RECORDS,
        )
    }

    /// Collects the lights of the scene into `gpu_lights`, `froxel_lights` (view space) and
    /// `shadow_sources`; past `MAX_LIGHTS`, the ones nearest to the camera.
    fn gather_lights(
        &mut self,
        world: &WorldQuery<'_>,
        view: &Matrix4f,
        camera_position: Vector3f,
    ) {
        self.gpu_lights.clear();
        self.froxel_lights.clear();
        self.light_ids.clear();
        self.shadow_sources.clear();
        self.gathered.clear();
        // Scenes that don't register LightComponent have no lights
        let (Ok(lights), Ok(transforms)) = (
            world.query::<LightComponent>(),
            world.query::<TransformComponent>(),
        ) else {
            return;
        };
        for (slot, (light, transform)) in lights.data.iter().zip(&transforms.data).enumerate() {
            let (Some(light), Some(transform)) = (light, transform) else {
                continue;
            };
            // Dark lights add nothing (and would dilute random light picks of GI rays)
            if light.intensity <= 0.0 || light.color.iter().all(|&channel| channel <= 0.0) {
                continue;
            }
            let range = light.range.max(1e-3);
            let direction = transform
                .get_forward_direction()
                .normalize_or(Vector3f::NEG_Z);
            let (spot_scale_offset, cone) = match light.light_type {
                LightType::Point => ([0.0, 1.0, -1.0, 1.0], None),
                LightType::Spot {
                    inner_angle,
                    outer_angle,
                } => {
                    let outer = outer_angle.to_radians().clamp(0.0, 89.9f32.to_radians());
                    let inner = inner_angle.to_radians().clamp(0.0, outer);
                    // Filament: full intensity inside the inner cone, zero at the outer
                    let scale = 1.0 / (inner.cos() - outer.cos()).max(1e-4);
                    let offset = -outer.cos() * scale;
                    let axis = view.transform_vector3(direction);
                    (
                        [scale, offset, -1.0, 0.0],
                        Some(FroxelCone::new(axis, outer)),
                    )
                }
            };
            let position = transform.position;
            let gpu = PunctualLight {
                position_falloff: position.extend(1.0 / (range * range)).to_array(),
                direction: direction.extend(0.0).to_array(),
                color_intensity: [
                    light.color[0],
                    light.color[1],
                    light.color[2],
                    light.intensity,
                ],
                spot_scale_offset,
            };
            let froxel = FroxelLight {
                position: view.transform_point3(position),
                radius: range,
                cone,
            };
            let shadow = light.cast_shadows.then(|| ShadowSource {
                position,
                direction,
                range,
                spot_outer: match light.light_type {
                    LightType::Point => None,
                    LightType::Spot { outer_angle, .. } => Some(
                        outer_angle
                            .to_radians()
                            .clamp(0.5f32.to_radians(), 89.9f32.to_radians()),
                    ),
                },
            });
            self.gathered.push(GatheredLight {
                id: slot as LightId,
                camera_distance_squared: position.distance_squared(camera_position),
                gpu,
                froxel,
                shadow,
            });
        }
        keep_nearest(&mut self.gathered, MAX_LIGHTS, |light| {
            light.camera_distance_squared
        });
        for light in &self.gathered {
            self.gpu_lights.push(light.gpu);
            self.froxel_lights.push(light.froxel);
            self.light_ids.push(light.id);
            self.shadow_sources.push(light.shadow);
        }
    }
}

impl PassLights {
    /// This frame's casters, which of them move (`CasterTracker`) and each static one's hash;
    /// returns the hash of the static set (changes when any static caster appears, moves or goes).
    fn classify_casters(&mut self, world: &WorldQuery<'_>, frame: &mut LightsFrame) -> u64 {
        gather_casters(world, &mut frame.casters);
        self.caster_tracker.begin_frame();
        // Skinned meshes change shape without their transform changing
        self.skinned.clear();
        #[cfg(not(feature = "headless"))]
        if let Ok(skinning) = world.get_global::<crate::ecs::GpuSkinningComponent>() {
            self.skinned.extend(
                skinning
                    .instances
                    .iter()
                    .map(|instance| instance.renderer_mesh),
            );
            self.skinned.sort_unstable();
        }
        frame.caster_dynamic.clear();
        self.caster_hashes.clear();
        let mut signature = 0u64;
        for candidate in &frame.casters {
            let forced =
                candidate.gi_dynamic || self.skinned.binary_search(&candidate.caster.mesh).is_ok();
            let dynamic = self
                .caster_tracker
                .update(candidate.entity, &candidate.model, forced);
            let hash = if dynamic { 0 } else { caster_hash(candidate) };
            frame.caster_dynamic.push(dynamic);
            self.caster_hashes.push(hash);
            signature = signature.wrapping_add(hash);
        }
        signature
    }

    /// Order-independent hash of the static casters inside a map's frustum (`light_position`,
    /// `range`: its light, for a cheap sphere test first).
    fn static_casters_hash(
        &self,
        view_projection: &Matrix4f,
        light_position: Vector3f,
        range: f32,
        casters: &[CasterCandidate],
    ) -> u64 {
        let planes = frustum_planes(view_projection);
        casters
            .iter()
            .zip(&self.caster_hashes)
            .filter(|(candidate, &hash)| {
                hash != 0
                    && candidate.within(light_position, range)
                    && !outside_frustum(&planes, &candidate.world_corners)
            })
            .fold(0u64, |sum, (_, &hash)| sum.wrapping_add(hash))
    }

    /// Shadow work of the frame: updates the cache (tiles, sizes, fades) with this frame's
    /// shadow-casting lights, writes their shadow records and each light's first record, and
    /// lists the atlas tiles to draw (static tiles whose light or casters changed, dynamic tiles
    /// that are due), most important light first, up to `MAX_TILE_DRAWS`.
    #[allow(clippy::too_many_arguments)]
    fn assign_shadows(
        &mut self,
        world: &WorldQuery<'_>,
        camera_position: Vector3f,
        pixels_per_meter: f32,
        settings: &CacheSettings,
        normal_bias: f32,
        delta_time: f32,
        frame: &mut LightsFrame,
    ) {
        frame.shadow_jobs.clear();
        self.gpu_shadow_maps.clear();
        for id in frame.truncated_lights.drain(..) {
            if let Some(slot) = self.shadow_cache.slots.get_mut(&id) {
                slot.static_dirty = true;
            }
        }
        let any_shadow = self.shadow_sources.iter().any(Option::is_some);
        let static_signature = if any_shadow {
            self.classify_casters(world, frame)
        } else {
            frame.casters.clear();
            frame.caster_dynamic.clear();
            self.caster_hashes.clear();
            0
        };

        // Requests: importance = on-screen radius of the range; moving casters within range
        self.shadow_requests.clear();
        self.light_keys.clear();
        self.light_keys.resize(self.shadow_sources.len(), 0);
        for (light, source) in self.shadow_sources.iter().enumerate() {
            let Some(source) = source else {
                continue;
            };
            let key = light_key(
                source.position,
                source.direction,
                source.range,
                source.spot_outer,
            );
            self.light_keys[light] = key;
            let near_dynamic =
                frame
                    .casters
                    .iter()
                    .zip(&frame.caster_dynamic)
                    .any(|(candidate, &dynamic)| {
                        dynamic && candidate.within(source.position, source.range)
                    });
            self.shadow_requests.push(LightRequest {
                id: self.light_ids[light],
                is_point: source.spot_outer.is_none(),
                importance: light_importance(
                    source.range,
                    source.position.distance(camera_position),
                    pixels_per_meter,
                ),
                light_key: key,
                near_dynamic,
            });
        }
        self.shadow_cache
            .update(&self.shadow_requests, settings, delta_time);
        frame.atlas_size = self.shadow_cache.atlas_size();

        // Most important first: they get the tile draws of this frame
        self.shadow_order.clear();
        self.shadow_order
            .extend((0..self.shadow_sources.len()).filter(|&light| {
                self.shadow_sources[light].is_some()
                    && self.shadow_cache.slots.contains_key(&self.light_ids[light])
            }));
        let slots = &self.shadow_cache.slots;
        let ids = &self.light_ids;
        self.shadow_order.sort_by(|&a, &b| {
            slots[&ids[b]]
                .importance
                .total_cmp(&slots[&ids[a]].importance)
        });

        let atlas_size = frame.atlas_size;
        let cache_frame = self.shadow_cache.frame;
        for order in 0..self.shadow_order.len() {
            let light = self.shadow_order[order];
            let Some(source) = self.shadow_sources[light] else {
                continue;
            };
            let id = self.light_ids[light];
            let Some(slot) = self.shadow_cache.slots.get(&id) else {
                continue;
            };
            // A light that changed type keeps its old tiles while they fade out: no shadow
            if slot.is_point != source.spot_outer.is_none() {
                continue;
            }
            let size = slot.size;
            let view_projections: Vec<Matrix4f> = match source.spot_outer {
                Some(outer) => vec![spot_view_projection(
                    source.position,
                    source.direction,
                    outer,
                    source.range,
                )],
                None => (0..6)
                    .map(|face| {
                        point_face_view_projection(source.position, face, source.range, size)
                    })
                    .collect(),
            };
            let key = self.light_keys[light];

            // Static tiles: redraw when new, resized, the light changed, its static casters did,
            // or the last draw ran out of casters. The hashes are only recomputed when some
            // static caster in the scene changed since this light's last draw.
            let tiles_drawn = slot.drawn_caster_hashes.len() == view_projections.len();
            let face_hashes = |pass: &Self| -> Vec<u64> {
                view_projections
                    .iter()
                    .map(|view_projection| {
                        pass.static_casters_hash(
                            view_projection,
                            source.position,
                            source.range,
                            &frame.casters,
                        )
                    })
                    .collect()
            };
            let mut hashes = (tiles_drawn && slot.drawn_signature != static_signature)
                .then(|| face_hashes(self));
            let slot = self
                .shadow_cache
                .slots
                .get_mut(&id)
                .expect("slot looked up above");
            let needs_static = !tiles_drawn
                || slot.static_dirty
                || slot.drawn_light_key != key
                || hashes
                    .as_ref()
                    .is_some_and(|hashes| *hashes != slot.drawn_caster_hashes);
            if !needs_static {
                slot.drawn_signature = static_signature;
            }
            let room = MAX_TILE_DRAWS.saturating_sub(frame.shadow_jobs.len());
            let mut drew_static = false;
            if needs_static && room >= view_projections.len() {
                push_jobs(
                    &mut frame.shadow_jobs,
                    &view_projections,
                    &slot.static_tiles,
                    id,
                    &source,
                    false,
                );
                drew_static = true;
            } else if needs_static && !tiles_drawn {
                // The tiles hold no draw yet: no shadow until their turn
                continue;
            }

            // Dynamic tiles: only the moving casters, every frame or every few
            let has_dynamic = slot.dynamic_tiles.len() == view_projections.len();
            if has_dynamic
                && slot.dynamic_due(cache_frame, settings)
                && MAX_TILE_DRAWS.saturating_sub(frame.shadow_jobs.len()) >= view_projections.len()
            {
                push_jobs(
                    &mut frame.shadow_jobs,
                    &view_projections,
                    &slot.dynamic_tiles,
                    id,
                    &source,
                    true,
                );
                slot.dynamic_drawn = true;
            }
            let dynamic_valid = has_dynamic && slot.dynamic_drawn;
            let strength = slot.strength;
            let static_tiles = slot.static_tiles.clone();
            let dynamic_tiles = slot.dynamic_tiles.clone();

            if drew_static {
                let hashes = hashes.take().unwrap_or_else(|| face_hashes(self));
                let slot = self
                    .shadow_cache
                    .slots
                    .get_mut(&id)
                    .expect("slot looked up above");
                slot.static_ready = true;
                slot.static_dirty = false;
                slot.drawn_light_key = key;
                slot.drawn_caster_hashes = hashes;
                slot.drawn_signature = static_signature;
            }

            // Records: none while fully faded out
            if strength <= 0.0
                || self.gpu_shadow_maps.len() + view_projections.len() > MAX_SHADOW_RECORDS
            {
                continue;
            }
            let fov = match source.spot_outer {
                Some(outer) => spot_fov(outer),
                None => cube_face_fov(size),
            };
            let normal_offset = normal_bias * texel_scale(fov, size);
            self.gpu_lights[light].spot_scale_offset[2] = self.gpu_shadow_maps.len() as f32;
            for (face, view_projection) in view_projections.iter().enumerate() {
                self.gpu_shadow_maps.push(PunctualShadow {
                    view_projection: view_projection.to_cols_array_2d(),
                    params: [normal_offset, size as f32, 1.0 / size as f32, strength],
                    static_rect: static_tiles[face].uv_rect(atlas_size),
                    dynamic_rect: if dynamic_valid {
                        dynamic_tiles[face].uv_rect(atlas_size)
                    } else {
                        [0.0; 4]
                    },
                });
            }
        }
    }
}

/// One job per map of a light: `tiles[face]` drawn with `view_projections[face]`.
fn push_jobs(
    jobs: &mut Vec<ShadowJob>,
    view_projections: &[Matrix4f],
    tiles: &[Tile],
    light: LightId,
    source: &ShadowSource,
    dynamic: bool,
) {
    for (view_projection, tile) in view_projections.iter().zip(tiles) {
        jobs.push(ShadowJob {
            view_projection: *view_projection,
            light,
            light_position: source.position,
            light_range: source.range,
            tile: *tile,
            dynamic,
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PassLights {
    /// Appends this frame's line to the `PILL_SHADOW_STATS` file, if set: frame, slots, slots
    /// fading out, mean strength, static and dynamic tiles drawn, atlas texels in use, records.
    fn write_shadow_stats(&mut self, frame: &LightsFrame) {
        use std::io::Write;
        let file = self.shadow_stats.get_or_insert_with(|| {
            let path = std::env::var("PILL_SHADOW_STATS").ok()?;
            let mut file = std::io::BufWriter::new(std::fs::File::create(path).ok()?);
            let _ = writeln!(
                file,
                "frame,slots,evicting,mean_strength,static_drawn,dynamic_drawn,atlas_used,records"
            );
            Some(file)
        });
        let Some(file) = file else {
            return;
        };
        let slots = &self.shadow_cache.slots;
        let evicting = slots.values().filter(|slot| slot.evicting).count();
        let mean_strength =
            slots.values().map(|slot| slot.strength).sum::<f32>() / slots.len().max(1) as f32;
        let atlas_used: u64 = slots
            .values()
            .flat_map(|slot| slot.static_tiles.iter().chain(&slot.dynamic_tiles))
            .map(|tile| (tile.size as u64).pow(2))
            .sum();
        let dynamic_drawn = frame.shadow_jobs.iter().filter(|job| job.dynamic).count();
        let _ = writeln!(
            file,
            "{},{},{},{:.3},{},{},{},{}",
            self.shadow_cache.frame,
            slots.len(),
            evicting,
            mean_strength,
            frame.shadow_jobs.len() - dynamic_drawn,
            dynamic_drawn,
            atlas_used,
            self.gpu_shadow_maps.len(),
        );
    }
}

/// What a static caster contributes to a map: entity, mesh and model matrix.
fn caster_hash(candidate: &CasterCandidate) -> u64 {
    let seed = mix(candidate.entity as u64 ^ ((candidate.caster.mesh.data().index as u64) << 32));
    hash_floats_from(seed, &candidate.model.to_cols_array())
}

impl Pass for PassLights {
    fn get_label(&self) -> &str {
        "pass_lights"
    }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> {
        Ok(())
    }

    fn draw(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let settings = world.get_global::<RenderStateComponent>()?.punctual_lights;
        let (camera, camera_transform) = world.active_camera()?;
        let forward = camera_forward(camera_transform);
        // Same view as camera_view_projection
        let view = Matrix4f::look_to_rh(camera_transform.position, forward, Vector3f::Y);
        self.gather_lights(world, &view, camera_transform.position);

        let params = FroxelLayoutParams {
            render_size: renderer.get_render_size(),
            fov_y: camera.fov.to_radians(),
            aspect: camera.aspect.get_value(),
            near: camera.range.start,
            light_far: settings.light_far.min(camera.range.end),
            first_slice_depth: settings.first_slice_depth,
        };
        if self.layout.as_ref().map(|layout| layout.params) != Some(params) {
            self.layout = Some(FroxelLayout::new(params));
        }
        let layout = self
            .layout
            .as_ref()
            .expect("PassLights: layout built above");
        let fits = self.froxelizer.froxelize(layout, &self.froxel_lights);
        if !fits && !self.overflow_logged {
            debug!(LogContext::Rendering =>
                "PassLights: froxel light lists exceed {MAX_RECORDS} entries; the overflow lists all lights"
            );
            self.overflow_logged = true;
        }

        let params = LightsParams {
            grid: [
                layout.count_x as f32,
                layout.count_y as f32,
                layout.tile_size as f32,
                self.gpu_lights.len() as f32,
            ],
            depth: [
                layout.inv_linearizer,
                layout.light_far,
                layout.count_z as f32,
                if settings.debug_heatmap { 1.0 } else { 0.0 },
            ],
            camera_forward: forward.extend(0.0).to_array(),
        };

        let frame_cell = self.frame.clone();
        let mut frame = frame_cell.borrow_mut();
        frame.params = params;
        if settings.shadows_enabled {
            let render_height = renderer.get_render_size().1.max(1) as f32;
            let max_size = renderer.get_device().limits().max_texture_dimension_2d;
            let cache_settings = CacheSettings {
                atlas_size: settings.atlas_size.min(max_size),
                resolution_scale: settings.resolution_scale,
                height_scale: render_height / 1080.0,
                fade_start: settings.fade_start_px,
                fade_end: settings.fade_end_px,
                fade_speed: settings.fade_speed,
                dynamic_update_far_frames: settings.dynamic_update_far_frames,
            };
            let pixels_per_meter = render_height / (2.0 * (camera.fov.to_radians() * 0.5).tan());
            self.assign_shadows(
                world,
                camera_transform.position,
                pixels_per_meter,
                &cache_settings,
                settings.shadow_normal_bias,
                // WorldQuery::delta_time is in milliseconds
                world.delta_time / 1000.0,
                &mut frame,
            );
            #[cfg(not(target_arch = "wasm32"))]
            self.write_shadow_stats(&frame);
        } else {
            frame.shadow_jobs.clear();
            self.gpu_shadow_maps.clear();
        }
        if self.gpu_lights.is_empty() {
            return Ok(());
        }
        let queue = renderer.get_queue();
        if !self.gpu_shadow_maps.is_empty() {
            queue.write_buffer(
                &frame.buffers.shadows,
                0,
                bytemuck::cast_slice(&self.gpu_shadow_maps),
            );
        }
        queue.write_buffer(
            &frame.buffers.lights,
            0,
            bytemuck::cast_slice(&self.gpu_lights),
        );
        queue.write_buffer(
            &frame.buffers.entries,
            0,
            bytemuck::cast_slice(&self.froxelizer.entries),
        );
        // u16 indices read as u32 words (little-endian: even index in the low half)
        if !self.froxelizer.records.is_empty() {
            queue.write_buffer(
                &frame.buffers.records,
                0,
                bytemuck::cast_slice(&self.froxelizer.records),
            );
        }
        Ok(())
    }
}
