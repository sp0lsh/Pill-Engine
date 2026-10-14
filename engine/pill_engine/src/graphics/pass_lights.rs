use crate::ecs::{LightComponent, LightType, RenderStateComponent, TransformComponent};
use crate::graphics::froxelizer::{
    FroxelCone, FroxelLayout, FroxelLayoutParams, FroxelLight, Froxelizer, MAX_FROXELS, MAX_LIGHTS,
    MAX_RECORDS,
};
use crate::graphics::{camera_forward, BufferDesc, FrameTarget, Pass, PillRenderer, WorldQuery};

use pill_core::{Matrix4f, Result, Vector3f};
use std::cell::RefCell;
use std::rc::Rc;

/// One light in the lights storage buffer — mirrors `PunctualLight` in pbr_opaque_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position_falloff: [f32; 4],  // xyz = world position, w = 1 / range²
    direction: [f32; 4],         // xyz = direction the spot shines in
    color_intensity: [f32; 4],   // rgb, intensity
    spot_scale_offset: [f32; 4], // angle term saturate(cos · x + y); point light: (0, 1)
}

/// Storage buffers read by the PBR fragment shader (group 0, bindings 10–12).
#[derive(Clone)]
pub(crate) struct LightBuffers {
    pub(crate) lights: wgpu::Buffer,
    /// (offset, count) per froxel.
    pub(crate) entries: wgpu::Buffer,
    /// u16 light indices, two per u32.
    pub(crate) records: wgpu::Buffer,
}

impl LightBuffers {
    pub(crate) fn new(
        renderer: &mut dyn PillRenderer,
        light_capacity: usize,
        entry_capacity: usize,
        record_capacity: usize,
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

/// Light buffers and this frame's grid, shared by `PassLights` and the PBR passes.
pub(crate) struct LightsFrame {
    pub(crate) buffers: LightBuffers,
    pub(crate) params: LightsParams,
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
    capacity_warned: bool,
    overflow_warned: bool,
}

impl PassLights {
    pub(crate) fn new(frame: Rc<RefCell<LightsFrame>>) -> Self {
        Self {
            frame,
            layout: None,
            froxelizer: Froxelizer::default(),
            gpu_lights: Vec::new(),
            froxel_lights: Vec::new(),
            capacity_warned: false,
            overflow_warned: false,
        }
    }

    /// Buffers sized for `MAX_LIGHTS` lights and the largest froxel grid.
    pub(crate) fn create_buffers(renderer: &mut dyn PillRenderer) -> Result<LightBuffers> {
        LightBuffers::new(renderer, MAX_LIGHTS, MAX_FROXELS, MAX_RECORDS)
    }

    /// Collects the lights of the scene into `gpu_lights` and `froxel_lights` (view space).
    fn gather_lights(&mut self, world: &WorldQuery<'_>, view: &Matrix4f) {
        self.gpu_lights.clear();
        self.froxel_lights.clear();
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
            if self.gpu_lights.len() == MAX_LIGHTS {
                if !self.capacity_warned {
                    log::error!(
                        "PassLights: {MAX_LIGHTS} lights reached; extra lights are not shaded"
                    );
                    self.capacity_warned = true;
                }
                break;
            }
            let range = light.range.max(1e-3);
            let direction = transform
                .get_forward_direction()
                .normalize_or(Vector3f::NEG_Z);
            let (spot_scale_offset, cone) = match light.light_type {
                LightType::Point => ([0.0, 1.0, 0.0, 0.0], None),
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
                        [scale, offset, 0.0, 0.0],
                        Some(FroxelCone::new(axis, outer)),
                    )
                }
            };
            let position = transform.position;
            self.gpu_lights.push(GpuLight {
                position_falloff: position.extend(1.0 / (range * range)).to_array(),
                direction: direction.extend(0.0).to_array(),
                color_intensity: [
                    light.color[0],
                    light.color[1],
                    light.color[2],
                    light.intensity,
                ],
                spot_scale_offset,
            });
            self.froxel_lights.push(FroxelLight {
                position: view.transform_point3(position),
                radius: range,
                cone,
            });
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
        self.gather_lights(world, &view);

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
        if !fits && !self.overflow_warned {
            log::warn!(
                "PassLights: froxel light lists exceed {MAX_RECORDS} entries; the overflow lists all lights"
            );
            self.overflow_warned = true;
        }

        let mut frame = self.frame.borrow_mut();
        frame.params = LightsParams {
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
        if self.gpu_lights.is_empty() {
            return Ok(());
        }
        let queue = renderer.get_queue();
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
