use crate::ecs::{LightComponent, LightType, RenderStateComponent, TransformComponent};
use crate::graphics::froxelizer::{
    FroxelCone, FroxelLayout, FroxelLayoutParams, FroxelLight, Froxelizer, MAX_FROXELS, MAX_LIGHTS,
    MAX_RECORDS,
};
use crate::graphics::punctual_shadows::{
    assign_shadow_layers, cube_face_fov, point_face_view_projection, spot_fov,
    spot_view_projection, texel_scale, ShadowCandidate, MAX_PUNCTUAL_SHADOW_MAPS,
};
use crate::graphics::shadow_casters::keep_nearest;
use crate::graphics::{camera_forward, BufferDesc, FrameTarget, Pass, PillRenderer, WorldQuery};

use pill_core::{debug, LogContext, Matrix4f, Result, Vector3f};
use std::cell::RefCell;
use std::rc::Rc;

/// One light in the lights storage buffer — mirrors `PunctualLight` in pbr_opaque_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position_falloff: [f32; 4], // xyz = world position, w = 1 / range²
    direction: [f32; 4],        // xyz = direction the spot shines in
    color_intensity: [f32; 4],  // rgb, intensity
    // x, y: angle term saturate(cos · x + y), point light (0, 1); z: first shadow layer (−1 = no
    // shadow); w: 1 = point light (6 cube-face layers)
    spot_scale_offset: [f32; 4],
}

/// One point/spot shadow map in the shadow storage buffer — mirrors `PunctualShadow` in
/// pbr_opaque_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuShadowMap {
    view_projection: [[f32; 4]; 4], // world → light clip space (depth 0..1)
    // x = normal offset per meter of light distance (world), y = map size, z = 1 / map size
    params: [f32; 4],
}

/// A point/spot shadow map to render this frame.
#[derive(Clone, Copy)]
pub(crate) struct PunctualShadowMap {
    /// World → light clip space; the far plane is the light's range.
    pub(crate) view_projection: Matrix4f,
    /// Casters nearest to it are kept first when caster draws run out.
    pub(crate) light_position: Vector3f,
}

/// A light of the scene before the `MAX_LIGHTS` cut (nearest to the camera are kept).
struct GatheredLight {
    camera_distance_squared: f32,
    gpu: GpuLight,
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
                light_capacity.max(1) * std::mem::size_of::<GpuLight>(),
            )?,
            entries: storage("froxel_entries", entry_capacity.max(1) * 8)?,
            records: storage(
                "froxel_records",
                record_capacity.max(2).next_multiple_of(2) * 2,
            )?,
            shadows: storage(
                "punctual_shadows",
                shadow_capacity.max(1) * std::mem::size_of::<GpuShadowMap>(),
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

/// Light buffers and this frame's grid and shadow maps, shared by `PassLights`,
/// `PassPunctualShadows` and the PBR passes.
pub(crate) struct LightsFrame {
    pub(crate) buffers: LightBuffers,
    pub(crate) params: LightsParams,
    /// Shadow maps to render this frame, in layer order.
    pub(crate) shadow_maps: Vec<PunctualShadowMap>,
    pub(crate) shadow_map_size: u32,
    /// Shadow texture array view, set by `PassPunctualShadows`; `generation` changes when it is
    /// recreated.
    pub(crate) shadow_view: Option<wgpu::TextureView>,
    pub(crate) shadow_generation: u64,
}

impl LightsFrame {
    pub(crate) fn new(buffers: LightBuffers) -> Self {
        Self {
            buffers,
            params: LightsParams::default(),
            shadow_maps: Vec::new(),
            shadow_map_size: 1,
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
    gpu_lights: Vec<GpuLight>,
    froxel_lights: Vec<FroxelLight>,
    gathered: Vec<GatheredLight>,
    // Per light: its shadow inputs and whether it casts
    shadow_sources: Vec<Option<ShadowSource>>,
    shadow_candidates: Vec<ShadowCandidate>,
    shadow_layers: Vec<(usize, usize)>,
    gpu_shadow_maps: Vec<GpuShadowMap>,
    overflow_logged: bool,
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
            shadow_sources: Vec::new(),
            shadow_candidates: Vec::new(),
            shadow_layers: Vec::new(),
            gpu_shadow_maps: Vec::new(),
            overflow_logged: false,
        }
    }

    /// Buffers sized for `MAX_LIGHTS` lights and the largest froxel grid.
    pub(crate) fn create_buffers(renderer: &mut dyn PillRenderer) -> Result<LightBuffers> {
        LightBuffers::new(
            renderer,
            MAX_LIGHTS,
            MAX_FROXELS,
            MAX_RECORDS,
            MAX_PUNCTUAL_SHADOW_MAPS,
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
        self.shadow_sources.clear();
        self.gathered.clear();
        // Scenes that don't register LightComponent have no lights
        let (Ok(lights), Ok(transforms)) = (
            world.query::<LightComponent>(),
            world.query::<TransformComponent>(),
        ) else {
            return;
        };
        for (light, transform) in lights.data.iter().zip(&transforms.data) {
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
            let gpu = GpuLight {
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
            self.shadow_sources.push(light.shadow);
        }
    }
}

impl PassLights {
    /// Gives the shadow-map budget to the visible shadow-casting lights nearest the camera,
    /// writes their first layer into their light records and lists the maps in `frame`.
    fn assign_shadows(
        &mut self,
        camera_position: Vector3f,
        budget: usize,
        map_size: u32,
        normal_bias: f32,
        frame: &mut LightsFrame,
    ) {
        self.shadow_candidates.clear();
        for (light, source) in self.shadow_sources.iter().enumerate() {
            let Some(source) = source else {
                continue;
            };
            if !self.froxelizer.visible.get(light).copied().unwrap_or(false) {
                continue;
            }
            self.shadow_candidates.push(ShadowCandidate {
                light,
                camera_distance: source.position.distance(camera_position),
                is_point: source.spot_outer.is_none(),
            });
        }
        assign_shadow_layers(&mut self.shadow_candidates, budget, &mut self.shadow_layers);

        frame.shadow_maps.clear();
        frame.shadow_map_size = map_size;
        self.gpu_shadow_maps.clear();
        for &(light, first_layer) in &self.shadow_layers {
            let Some(source) = self.shadow_sources[light] else {
                continue;
            };
            self.gpu_lights[light].spot_scale_offset[2] = first_layer as f32;
            let (fov, face_count) = match source.spot_outer {
                Some(outer) => (spot_fov(outer), 1),
                None => (cube_face_fov(map_size), 6),
            };
            let normal_offset = normal_bias * texel_scale(fov, map_size);
            for face in 0..face_count {
                let view_projection = match source.spot_outer {
                    Some(outer) => {
                        spot_view_projection(source.position, source.direction, outer, source.range)
                    }
                    None => {
                        point_face_view_projection(source.position, face, source.range, map_size)
                    }
                };
                frame.shadow_maps.push(PunctualShadowMap {
                    view_projection,
                    light_position: source.position,
                });
                self.gpu_shadow_maps.push(GpuShadowMap {
                    view_projection: view_projection.to_cols_array_2d(),
                    params: [normal_offset, map_size as f32, 1.0 / map_size as f32, 0.0],
                });
            }
        }
    }
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
        let budget = if settings.shadows_enabled {
            (settings.max_shadow_maps as usize).min(MAX_PUNCTUAL_SHADOW_MAPS)
        } else {
            0
        };
        // One clamp for the texture (PassPunctualShadows) and the shader's map params
        let max_size = renderer.get_device().limits().max_texture_dimension_2d;
        self.assign_shadows(
            camera_transform.position,
            budget,
            settings.shadow_map_size.clamp(1, max_size),
            settings.shadow_normal_bias,
            &mut frame,
        );
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
