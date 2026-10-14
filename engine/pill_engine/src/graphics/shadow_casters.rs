//! Depth-only caster rendering shared by the shadow passes (key light, point/spot lights): one
//! pipeline, one per-draw transform buffer holding the casters of every shadow map this frame and
//! one light matrix per map (dynamic uniform offsets).

use crate::config::{
    DEPTH_TARGET_FORMAT, SHADOW_CASTER_FRAGMENT_SHADER, SHADOW_CASTER_VERTEX_SHADER,
};
use crate::ecs::{MeshRenderingComponent, TransformComponent};
use crate::graphics::pass_pbr_opaque::{
    is_translucent, per_draw_storage_entry, DrawTransform, PER_DRAW_STRIDE_BYTES,
};
use crate::graphics::{
    decompose_render_queue_key, BufferDesc, PillRenderer, Pipeline, PipelineDesc,
    RendererMeshHandle, ShaderDesc, WorldQuery,
};
use crate::renderer::resources::{RendererMesh, Vertex};

use pill_core::{Matrix4f, PillSlotMapKey, Result, Vector3f, Vector4f};
use std::num::NonZeroU32;
use std::ops::Range;

/// A mesh drawn into a shadow map.
#[derive(Clone, Copy)]
pub(crate) struct Caster {
    pub(crate) mesh: RendererMeshHandle,
    pub(crate) transform: DrawTransform,
}

/// A shadow-casting mesh entity: model matrix, mesh-local bounds and the world-space corners and
/// center of those bounds (computed once, tested against every shadow map).
pub(crate) struct CasterCandidate {
    pub(crate) caster: Caster,
    pub(crate) model: Matrix4f,
    pub(crate) local_bounds: (Vector3f, Vector3f),
    pub(crate) world_corners: [Vector3f; 8],
    pub(crate) world_center: Vector3f,
}

/// Mesh entities with `cast_shadows` whose material is opaque; none when the scene has no mesh
/// rendering.
pub(crate) fn gather_casters(world: &WorldQuery<'_>, out: &mut Vec<CasterCandidate>) {
    out.clear();
    let (Ok(transforms), Ok(mesh_renderings)) = (
        world.query::<TransformComponent>(),
        world.query::<MeshRenderingComponent>(),
    ) else {
        return;
    };
    for (transform, mesh_rendering) in transforms.data.iter().zip(&mesh_renderings.data) {
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
        let fields = decompose_render_queue_key(key);
        let Some(version) = NonZeroU32::new(fields.mesh_version.into()) else {
            continue;
        };
        let mesh = RendererMeshHandle::new(fields.mesh_index.into(), version);
        let Ok(renderer_mesh) = world.resources.get_resource::<RendererMesh>(&mesh) else {
            continue;
        };
        let transform = DrawTransform::from(transform);
        let caster = Caster { mesh, transform };
        let model = transform.model_matrix();
        let world_corners =
            box_corners(renderer_mesh.local_bounds).map(|corner| model.transform_point3(corner));
        out.push(CasterCandidate {
            caster,
            model,
            local_bounds: renderer_mesh.local_bounds,
            world_corners,
            world_center: world_corners.iter().copied().sum::<Vector3f>() / 8.0,
        });
    }
}

/// The 8 corners of an axis-aligned box.
fn box_corners((min, max): (Vector3f, Vector3f)) -> [Vector3f; 8] {
    std::array::from_fn(|corner| {
        Vector3f::new(
            if corner & 1 == 0 { min.x } else { max.x },
            if corner & 2 == 0 { min.y } else { max.y },
            if corner & 4 == 0 { min.z } else { max.z },
        )
    })
}

/// The six planes (xyz normal pointing inside, w offset) of a view-projection with 0..1 depth.
pub(crate) fn frustum_planes(view_projection: &Matrix4f) -> [Vector4f; 6] {
    let rows = view_projection.transpose();
    let (x, y, z, w) = (rows.x_axis, rows.y_axis, rows.z_axis, rows.w_axis);
    [w + x, w - x, w + y, w - y, z, w - z]
}

/// Whether a world-space box (its 8 corners) is entirely outside one frustum plane.
pub(crate) fn outside_frustum(planes: &[Vector4f; 6], corners: &[Vector3f; 8]) -> bool {
    planes.iter().any(|plane| {
        corners
            .iter()
            .all(|corner| plane.truncate().dot(*corner) + plane.w < 0.0)
    })
}

/// Best effort over a capacity: keeps the `capacity` items with the smallest `distance`, in any
/// order (lights nearest the camera, casters nearest their light or the camera).
pub(crate) fn keep_nearest<T>(items: &mut Vec<T>, capacity: usize, distance: impl Fn(&T) -> f32) {
    if items.len() > capacity {
        items.select_nth_unstable_by(capacity, |a, b| distance(a).total_cmp(&distance(b)));
        items.truncate(capacity);
    }
}

/// Pipeline and buffers of the caster rendering; `map_capacity` light matrices,
/// `max_instances` caster transforms across all maps of a frame.
pub(crate) struct ShadowCasters {
    pipeline: Pipeline,
    camera_buffer: wgpu::Buffer,
    camera_stride: u64,
    camera_bind_group: wgpu::BindGroup,
    per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
    map_capacity: usize,
    max_instances: usize,
    label: String,
    staging: Vec<u8>,
}

impl ShadowCasters {
    pub(crate) fn new(
        renderer: &mut dyn PillRenderer,
        label: &str,
        map_capacity: usize,
        max_instances: usize,
        constant_bias: i32,
        slope_bias: f32,
    ) -> Result<Self> {
        let map_capacity = map_capacity.max(1);
        let max_instances = max_instances.max(1);
        let pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some(label),
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
                vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
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
                    constant: constant_bias,
                    slope_scale: slope_bias,
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

        let matrix_size = std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        let alignment = renderer
            .get_device()
            .limits()
            .min_uniform_buffer_offset_alignment as u64;
        let camera_stride = matrix_size.next_multiple_of(alignment);
        let camera_buffer = renderer.create_buffer(BufferDesc {
            label: Some(label),
            byte_size: camera_stride * map_capacity as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;
        let per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some(label),
            byte_size: (PER_DRAW_STRIDE_BYTES * max_instances) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })?;
        let device = renderer.get_device();
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.bind_group_layouts[0],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(matrix_size),
                }),
            }],
        });
        let per_draw_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.bind_group_layouts[1],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: per_draw_buffer.as_entire_binding(),
            }],
        });
        Ok(Self {
            pipeline,
            camera_buffer,
            camera_stride,
            camera_bind_group,
            per_draw_buffer,
            per_draw_bind_group,
            map_capacity,
            max_instances,
            label: label.to_string(),
            staging: Vec::new(),
        })
    }

    /// Uploads the light matrices of this frame's maps and all their casters; `casters` holds
    /// each map's casters as a contiguous range, sorted by mesh within it.
    pub(crate) fn upload(
        &mut self,
        queue: &wgpu::Queue,
        view_projections: &[Matrix4f],
        casters: &[Caster],
    ) {
        // Light matrices at their dynamic-offset stride, one write
        self.staging.clear();
        for view_projection in view_projections.iter().take(self.map_capacity) {
            let start = self.staging.len();
            self.staging
                .extend_from_slice(bytemuck::bytes_of(&view_projection.to_cols_array_2d()));
            self.staging.resize(start + self.camera_stride as usize, 0);
        }
        if !self.staging.is_empty() {
            queue.write_buffer(&self.camera_buffer, 0, &self.staging);
        }
        self.staging.clear();
        for caster in casters.iter().take(self.max_instances) {
            self.staging
                .extend_from_slice(bytemuck::bytes_of(&caster.transform));
        }
        if !self.staging.is_empty() {
            queue.write_buffer(&self.per_draw_buffer, 0, &self.staging);
        }
    }

    /// Clears `depth_view` and draws `casters[range]` with the light matrix of `map`: one
    /// instanced draw per run of casters sharing a mesh.
    pub(crate) fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        map: usize,
        casters: &[Caster],
        range: Range<usize>,
        world: &WorldQuery<'_>,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&self.label),
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
        if map >= self.map_capacity {
            return;
        }
        render_pass.set_pipeline(&self.pipeline.pipeline);
        render_pass.set_bind_group(
            0,
            &self.camera_bind_group,
            &[(map as u64 * self.camera_stride) as u32],
        );
        render_pass.set_bind_group(1, &self.per_draw_bind_group, &[]);
        let end = range.end.min(casters.len()).min(self.max_instances);
        let mut first = range.start;
        while first < end {
            let mesh_handle = casters[first].mesh;
            let mut run_end = first + 1;
            while run_end < end && casters[run_end].mesh == mesh_handle {
                run_end += 1;
            }
            if let Ok(mesh) = world.resources.get_resource::<RendererMesh>(&mesh_handle) {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, first as u32..run_end as u32);
            }
            first = run_end;
        }
    }
}

/// Sorts `casters[range]` by mesh so equal meshes form instanced runs.
pub(crate) fn sort_by_mesh(casters: &mut [Caster]) {
    casters.sort_unstable_by_key(|caster| caster.mesh.data().index);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit box centred at (x, y, z).
    fn unit_box_at(x: f32, y: f32, z: f32) -> [Vector3f; 8] {
        box_corners((Vector3f::splat(-0.5), Vector3f::splat(0.5)))
            .map(|corner| corner + Vector3f::new(x, y, z))
    }

    #[test]
    fn box_outside_one_frustum_plane_is_culled() {
        let projection = Matrix4f::perspective_rh(90f32.to_radians(), 1.0, 0.1, 10.0);
        let planes = frustum_planes(&projection);
        assert!(!outside_frustum(&planes, &unit_box_at(0.0, 0.0, -5.0)));
        assert!(outside_frustum(&planes, &unit_box_at(0.0, 0.0, 5.0))); // behind
        assert!(outside_frustum(&planes, &unit_box_at(0.0, 0.0, -20.0))); // beyond far
        assert!(outside_frustum(&planes, &unit_box_at(20.0, 0.0, -5.0))); // right
        assert!(!outside_frustum(&planes, &unit_box_at(5.0, 0.0, -5.0))); // straddles the edge
    }

    #[test]
    fn over_capacity_keeps_the_nearest_casters() {
        let candidates: Vec<CasterCandidate> = [5.0, 1.0, 9.0, 3.0]
            .into_iter()
            .map(|x| CasterCandidate {
                caster: Caster {
                    mesh: RendererMeshHandle::default(),
                    transform: DrawTransform::from(&crate::ecs::TransformComponent::default()),
                },
                model: Matrix4f::IDENTITY,
                local_bounds: (Vector3f::ZERO, Vector3f::ZERO),
                world_corners: [Vector3f::new(x, 0.0, 0.0); 8],
                world_center: Vector3f::new(x, 0.0, 0.0),
            })
            .collect();
        let mut selected: Vec<usize> = (0..candidates.len()).collect();
        keep_nearest(&mut selected, 2, |&index| {
            candidates[index].world_center.length()
        });
        selected.sort_unstable();
        assert_eq!(selected, vec![1, 3]);
    }
}
