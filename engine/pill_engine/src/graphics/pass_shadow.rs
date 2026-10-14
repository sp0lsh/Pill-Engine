use crate::config::{
    DEPTH_TARGET_FORMAT, SHADOW_CASTER_FRAGMENT_SHADER, SHADOW_CASTER_VERTEX_SHADER,
};
use crate::ecs::{MeshRenderingComponent, RenderStateComponent, TransformComponent};
use crate::graphics::fullscreen::uniform_entry_for;
use crate::graphics::pass_pbr_opaque::{
    is_translucent, per_draw_storage_entry, DrawTransform, PER_DRAW_STRIDE_BYTES,
};
use crate::graphics::shadow::{
    casts_into, fit_light, light_space_bounds, light_view_projection, CameraSlice, ShadowFrame,
};
use crate::graphics::{
    camera_forward, decompose_render_queue_key, BufferDesc, FrameTarget, Pass, PassTextureHandle,
    PillRenderer, Pipeline, PipelineDesc, RendererMeshHandle, RendererTargetDesc, ShaderDesc,
    TargetSize, WorldQuery,
};
use crate::renderer::resources::{RendererMesh, Vertex};

use pill_core::{debug, LogContext, PillSlotMapKey, Result, Vector3f};
use std::cell::RefCell;
use std::num::NonZeroU32;
use std::rc::Rc;

struct ShadowState {
    pipeline: Pipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
}

/// One caster that passed culling.
struct Caster {
    mesh: RendererMeshHandle,
    transform: DrawTransform,
}

/// Depth-only pass rendering the key light's (`lights[0]`) shadow casters with an orthographic
/// projection fitted to the camera (see `shadow.rs`). Owns the shadow map: created when shadows
/// are first enabled, recreated when `map_size` changes. Publishes map and light matrix in `frame`
/// for the PBR passes. Settings: `RenderStateComponent::shadow`.
pub(crate) struct PassShadow {
    frame: Rc<RefCell<ShadowFrame>>,
    max_instances: usize,
    // Hardware depth bias, fixed in the pipeline
    constant_bias: i32,
    slope_bias: f32,
    // Current shadow map and its size; kept while shadows are off
    shadow_map: Option<(PassTextureHandle, u32)>,
    // Per-frame scratch, capacity reused across frames
    casters: Vec<Caster>,
    staging_buffer: Vec<u8>,
    state: Option<ShadowState>,
    // Caster total of the last culling log line; logged again when it changes
    logged_total: usize,
    capacity_warned: bool,
    size_clamp_warned: bool,
}

impl PassShadow {
    pub(crate) fn new(
        frame: Rc<RefCell<ShadowFrame>>,
        max_instances: usize,
        constant_bias: i32,
        slope_bias: f32,
    ) -> Self {
        Self {
            frame,
            max_instances: max_instances.max(1),
            constant_bias,
            slope_bias,
            shadow_map: None,
            casters: Vec::new(),
            staging_buffer: Vec::new(),
            state: None,
            logged_total: 0,
            capacity_warned: false,
            size_clamp_warned: false,
        }
    }

    /// Shadow map of `requested` size (clamped to the device limit), recreated when it changes.
    fn shadow_map(
        &mut self,
        renderer: &mut dyn PillRenderer,
        requested: u32,
    ) -> Result<(PassTextureHandle, u32)> {
        let max_size = renderer.get_device().limits().max_texture_dimension_2d;
        let size = requested.clamp(1, max_size);
        if size != requested && !self.size_clamp_warned {
            log::warn!("PassShadow: map_size {requested} clamped to {size} (device limit)");
            self.size_clamp_warned = true;
        }
        if let Some((map, current)) = self.shadow_map {
            if current == size {
                return Ok((map, size));
            }
            renderer.remove_render_target(map);
        }
        let map = renderer.create_render_target(RendererTargetDesc {
            name: "shadow_map".to_string(),
            format: DEPTH_TARGET_FORMAT,
            size: TargetSize::Fixed(size, size),
        })?;
        self.shadow_map = Some((map, size));
        Ok((map, size))
    }
}

impl Pass for PassShadow {
    fn get_label(&self) -> &str {
        "pass_shadow"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some("pass_shadow"),
            vs: ShaderDesc {
                source: SHADOW_CASTER_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: SHADOW_CASTER_FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            vertex_buffers: &[RendererMesh::data_layout_descriptor()],
            bind_groups: vec![
                vec![uniform_entry_for(0, wgpu::ShaderStages::VERTEX)],
                vec![per_draw_storage_entry(0)],
            ],
            targets: &[],
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TARGET_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                // Polygon offset against self-shadowing (Filament's default approach)
                bias: wgpu::DepthBiasState {
                    constant: self.constant_bias,
                    slope_scale: self.slope_bias,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
        })?;

        let camera_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_shadow_camera"),
            byte_size: std::mem::size_of::<[[f32; 4]; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;
        let per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_shadow_per_draw"),
            byte_size: (PER_DRAW_STRIDE_BYTES * self.max_instances) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })?;
        let device = renderer.get_device();
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass_shadow_camera"),
            layout: &pipeline.bind_group_layouts[0],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let per_draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass_shadow_per_draw"),
            layout: &pipeline.bind_group_layouts[1],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: per_draw_buffer.as_entire_binding(),
            }],
        });
        self.state = Some(ShadowState {
            pipeline,
            camera_buffer,
            camera_bind_group,
            per_draw_buffer,
            per_draw_bind_group,
        });
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let render_state = world.get_global::<RenderStateComponent>()?;
        let settings = render_state.shadow;
        if !settings.enabled {
            *self.frame.borrow_mut() = ShadowFrame::default();
            return Ok(());
        }
        let (shadow_map, map_size) = self.shadow_map(renderer, settings.map_size)?;
        let (camera, camera_transform) = world.active_camera()?;
        let distance = settings.distance.min(camera.range.end);
        let fit = fit_light(
            &CameraSlice {
                position: camera_transform.position,
                forward: camera_forward(camera_transform),
                fov_y: camera.fov.to_radians(),
                aspect: camera.aspect.get_value(),
                near: camera.range.start,
                far: distance,
            },
            Vector3f::from_array(render_state.lights[0].direction),
            map_size,
        );

        // Caster queue: casters whose light-space AABB can reach the receiver box
        self.casters.clear();
        let mut total = 0;
        let mut nearest_caster_z = f32::NEG_INFINITY;
        let transforms = &world.query::<TransformComponent>()?.data;
        let mesh_renderings = &world.query::<MeshRenderingComponent>()?.data;
        for (transform, mesh_rendering) in transforms.iter().zip(mesh_renderings) {
            let (Some(transform), Some(mesh_rendering)) = (transform, mesh_rendering) else {
                continue;
            };
            let Some(key) = mesh_rendering.render_queue_key else {
                continue;
            };
            // Translucent materials don't cast (no partial coverage in a depth map)
            if !mesh_rendering.cast_shadows || is_translucent(world, key) {
                continue;
            }
            total += 1;
            let fields = decompose_render_queue_key(key);
            let Some(version) = NonZeroU32::new(fields.mesh_version.into()) else {
                continue;
            };
            let mesh_handle = RendererMeshHandle::new(fields.mesh_index.into(), version);
            let Ok(mesh) = world.resources.get_resource::<RendererMesh>(&mesh_handle) else {
                continue;
            };
            let draw_transform = DrawTransform::from(transform);
            let bounds =
                light_space_bounds(&fit.view, &draw_transform.model_matrix(), mesh.local_bounds);
            if !casts_into(&fit, bounds) {
                continue;
            }
            if self.casters.len() == self.max_instances {
                if !self.capacity_warned {
                    log::error!(
                        "PassShadow: per-draw capacity {} reached; extra casters cast no shadow",
                        self.max_instances
                    );
                    self.capacity_warned = true;
                }
                continue;
            }
            nearest_caster_z = nearest_caster_z.max(bounds.1.z);
            self.casters.push(Caster {
                mesh: mesh_handle,
                transform: draw_transform,
            });
        }
        if total != self.logged_total {
            debug!(LogContext::Rendering =>
                "PassShadow: {} of {} shadow casters kept after culling",
                self.casters.len(),
                total
            );
            self.logged_total = total;
        }
        self.casters
            .sort_unstable_by_key(|caster| caster.mesh.data().index);

        let view_projection = light_view_projection(&fit, nearest_caster_z);
        *self.frame.borrow_mut() = ShadowFrame {
            map: Some(shadow_map),
            map_size,
            view_projection,
            texel_world_size: fit.texel_world_size,
            distance,
        };

        let state = self
            .state
            .as_ref()
            .expect("PassShadow: state not initialized — call init() before draw()");
        self.staging_buffer.clear();
        for caster in &self.casters {
            self.staging_buffer
                .extend_from_slice(bytemuck::bytes_of(&caster.transform));
        }
        let queue = renderer.get_queue();
        queue.write_buffer(
            &state.camera_buffer,
            0,
            bytemuck::bytes_of(&view_projection.to_cols_array_2d()),
        );
        queue.write_buffer(&state.per_draw_buffer, 0, &self.staging_buffer);

        let depth_view = renderer
            .get_render_target_view(shadow_map)
            .ok_or("PassShadow: shadow map missing")?;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&state.pipeline.pipeline);
        render_pass.set_bind_group(0, &state.camera_bind_group, &[]);
        render_pass.set_bind_group(1, &state.per_draw_bind_group, &[]);
        // One instanced draw per run of casters sharing a mesh
        let mut first = 0;
        while first < self.casters.len() {
            let mesh_handle = self.casters[first].mesh;
            let mut end = first + 1;
            while end < self.casters.len() && self.casters[end].mesh == mesh_handle {
                end += 1;
            }
            if let Ok(mesh) = world.resources.get_resource::<RendererMesh>(&mesh_handle) {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, first as u32..end as u32);
            }
            first = end;
        }
        Ok(())
    }
}
