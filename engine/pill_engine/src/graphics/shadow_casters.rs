//! Depth-only caster rendering shared by the shadow passes (key light, point/spot lights): one
//! pipeline, one per-draw transform buffer holding the casters of every shadow map this frame and
//! one light matrix per map (dynamic uniform offsets).

use crate::config::{
    SHADOW_CASTER_CUTOUT_FRAGMENT_SHADER, SHADOW_CASTER_CUTOUT_VERTEX_SHADER,
    SHADOW_CASTER_FRAGMENT_SHADER, SHADOW_CASTER_VERTEX_SHADER, SHADOW_MAP_FORMAT,
};
use crate::ecs::{MeshRenderingComponent, TransformComponent};
use crate::graphics::pass_pbr_opaque::{DrawTransform, PER_DRAW_STRIDE_BYTES};
use crate::graphics::shader_bindings::{
    merge_layouts, shadow_caster_cutout_fragment, shadow_caster_cutout_vertex,
    shadow_caster_fragment, shadow_caster_vertex, structs::ShadowCamera,
};
use crate::graphics::{
    decompose_render_queue_key, BufferDesc, PillRenderer, Pipeline, PipelineDesc,
    RendererMaterialHandle, RendererMeshHandle, ShaderDesc, WorldQuery,
};
use crate::renderer::resources::{
    material_textures_layout_entries, RendererMaterial, RendererMesh, Vertex,
};

use pill_core::{Matrix4f, PillSlotMapKey, Result, Vector3f, Vector4f};
use std::num::NonZeroU32;
use std::ops::Range;

/// A mesh drawn into a shadow map; `cutout` = its alpha-tested material, drawn with the alpha
/// test and without culling; `double_sided` opaque casters are drawn without culling.
#[derive(Clone, Copy)]
pub(crate) struct Caster {
    pub(crate) mesh: RendererMeshHandle,
    pub(crate) cutout: Option<RendererMaterialHandle>,
    pub(crate) double_sided: bool,
    pub(crate) transform: DrawTransform,
}

impl Caster {
    /// Draw group, in draw order: culled opaque, double-sided opaque, then cutout by material.
    fn group(&self) -> u64 {
        match self.cutout {
            Some(material) => material.data().index as u64 + 2,
            None => self.double_sided as u64,
        }
    }
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
        if !mesh_rendering.cast_shadows {
            continue;
        }
        let fields = decompose_render_queue_key(key);
        let Some(material_version) = NonZeroU32::new(fields.material_version.into()) else {
            continue;
        };
        let material = RendererMaterialHandle::new(fields.material_index.into(), material_version);
        let Ok(renderer_material) = world.resources.get_resource::<RendererMaterial>(&material)
        else {
            continue;
        };
        // Translucent materials don't cast (no partial coverage in a depth map)
        if renderer_material.translucent {
            continue;
        }
        // Only the alpha test needs the material in the caster shader; double-sided opaque casters
        // just skip culling
        let cutout = renderer_material.alpha_tested.then_some(material);
        let double_sided = renderer_material.double_sided;
        let Some(version) = NonZeroU32::new(fields.mesh_version.into()) else {
            continue;
        };
        let mesh = RendererMeshHandle::new(fields.mesh_index.into(), version);
        let Ok(renderer_mesh) = world.resources.get_resource::<RendererMesh>(&mesh) else {
            continue;
        };
        let transform = DrawTransform::from(transform).with_material_index(fields.material_index);
        let caster = Caster {
            mesh,
            cutout,
            double_sided,
            transform,
        };
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
pub(crate) fn box_corners((min, max): (Vector3f, Vector3f)) -> [Vector3f; 8] {
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
    // Opaque, no culling: double-sided casters without an alpha test
    two_sided_pipeline: Pipeline,
    // Alpha test, no culling: casters with a `cutout` material
    cutout_pipeline: Pipeline,
    camera_buffer: wgpu::Buffer,
    camera_stride: u64,
    camera_bind_group: wgpu::BindGroup,
    per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
    material_array_bind_group: wgpu::BindGroup,
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
        let mut opaque_pipeline = |cull_mode: Option<wgpu::Face>| {
            renderer.create_pipeline(PipelineDesc {
                label: Some(label),
                vs: ShaderDesc {
                    source: SHADOW_CASTER_VERTEX_SHADER,
                    entry_func: "vs_main",
                },
                ps: ShaderDesc {
                    source: SHADOW_CASTER_FRAGMENT_SHADER,
                    entry_func: "fs_main",
                },
                fragment_constants: &[],
                vertex_buffers: &[RendererMesh::position_layout_descriptor()],
                bind_groups: merge_layouts(&[
                    shadow_caster_vertex::layout(),
                    shadow_caster_fragment::layout(),
                ]),
                targets: &[],
                depth_stencil: Some(caster_depth_state(constant_bias, slope_bias)),
                multisample: wgpu::MultisampleState::default(),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    ..Default::default()
                },
            })
        };
        let pipeline = opaque_pipeline(Some(wgpu::Face::Back))?;
        let two_sided_pipeline = opaque_pipeline(None)?;

        let cutout_pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some(label),
            vs: ShaderDesc {
                source: SHADOW_CASTER_CUTOUT_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: SHADOW_CASTER_CUTOUT_FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            fragment_constants: &[],
            vertex_buffers: &RendererMesh::data_layout_descriptors(),
            // Group 2 is the material's texture bind group (all its textures, the shader reads
            // base_color only), group 3 the material array
            bind_groups: merge_layouts(&[
                shadow_caster_cutout_vertex::layout(),
                shadow_caster_cutout_fragment::layout(),
                vec![vec![], vec![], material_textures_layout_entries()],
            ]),
            targets: &[],
            depth_stencil: Some(caster_depth_state(constant_bias, slope_bias)),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
        })?;

        let matrix_size = std::mem::size_of::<ShadowCamera>() as u64;
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
        let material_array_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &cutout_pipeline.bind_group_layouts[3],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: renderer.get_material_buffer().as_entire_binding(),
            }],
        });
        Ok(Self {
            pipeline,
            two_sided_pipeline,
            cutout_pipeline,
            camera_buffer,
            camera_stride,
            camera_bind_group,
            per_draw_buffer,
            per_draw_bind_group,
            material_array_bind_group,
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
                .extend_from_slice(bytemuck::bytes_of(&ShadowCamera {
                    view_projection: view_projection.to_cols_array_2d(),
                }));
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
        // Read by the cutout pipeline only; bound once for all of its materials
        render_pass.set_bind_group(3, &self.material_array_bind_group, &[]);
        let end = range.end.min(casters.len()).min(self.max_instances);
        let mut first = range.start;
        // Casters are sorted by group (`Caster::group`): culled opaque, double-sided opaque, then
        // cutout by material
        let mut bound_cutout: Option<RendererMaterialHandle> = None;
        let mut two_sided_bound = false;
        while first < end {
            let (mesh_handle, group) = (casters[first].mesh, casters[first].group());
            let cutout = casters[first].cutout;
            let mut run_end = first + 1;
            while run_end < end
                && casters[run_end].mesh == mesh_handle
                && casters[run_end].group() == group
            {
                run_end += 1;
            }
            if group == 1 && !two_sided_bound {
                render_pass.set_pipeline(&self.two_sided_pipeline.pipeline);
                two_sided_bound = true;
            }
            if let Some(material_handle) = cutout.filter(|handle| bound_cutout != Some(*handle)) {
                let Ok(material) = world
                    .resources
                    .get_resource::<RendererMaterial>(&material_handle)
                else {
                    first = run_end;
                    continue;
                };
                if bound_cutout.is_none() {
                    render_pass.set_pipeline(&self.cutout_pipeline.pipeline);
                }
                render_pass.set_bind_group(2, &material.textures_bind_group, &[]);
                bound_cutout = Some(material_handle);
            }
            if let Ok(mesh) = world.resources.get_resource::<RendererMesh>(&mesh_handle) {
                // Cutouts read uvs from the attribute stream; the other casters only positions
                if cutout.is_some() {
                    mesh.set_vertex_buffers(&mut render_pass);
                } else {
                    mesh.set_position_buffer(&mut render_pass);
                }
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.index_count, 0, first as u32..run_end as u32);
            }
            first = run_end;
        }
    }
}

/// Sorts casters so equal meshes form instanced runs within each draw group (`Caster::group`):
/// one pipeline switch per group and one material bind per cutout material.
pub(crate) fn sort_by_mesh(casters: &mut [Caster]) {
    casters.sort_unstable_by_key(|caster| (caster.group(), caster.mesh.data().index));
}

/// Depth write with polygon offset against self-shadowing (Filament's default approach).
fn caster_depth_state(constant_bias: i32, slope_bias: f32) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: SHADOW_MAP_FORMAT,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: constant_bias,
            slope_scale: slope_bias,
            clamp: 0.0,
        },
    }
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
                    cutout: None,
                    double_sided: false,
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
