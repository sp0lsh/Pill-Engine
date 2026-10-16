//! Editor view (from the teaser's `PassWireframePreview`): every mesh as colored wireframe plus
//! its bounds at `BOUNDS_OPACITY` (meshes outside the game camera's frustum: bounds only), the
//! floor grid and the game camera's frustum, seen from the overview `FlyCamera`, drawn into an HDR
//! render target (`EditorPass` composites it into the bottom half of the frame).
//!
//! Wireframe is `LineList` over an edge index buffer built from each mesh's triangles (a-b, b-c,
//! c-a); `PolygonMode::Line` needs POLYGON_MODE_LINE, which Metal and WebGPU don't have.

use std::collections::HashMap;

use pill_engine::game::*;
use pill_engine::internal::{Mesh, ResourceManager};
use wgpu::util::DeviceExt;

use crate::editor::FlyCamera;

const VERTEX_SHADER: &str = include_str!("../res/shaders/wireframe_vertex.wgsl");
const FRAGMENT_SHADER: &str = include_str!("../res/shaders/wireframe_fragment.wgsl");

/// Column-major view-projection (the shader's `column_major float4x4`).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
}

/// Per-instance data indexed by `instance_index`: the PBR pass's raw position / rotation (radians)
/// / scale, so the lines align with the shaded geometry, plus a flat color.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PerDraw {
    position: [f32; 4],
    rotation: [f32; 4],
    scale: [f32; 4],
    color: [f32; 4],
}

/// Edge index buffer of one mesh (positions come from its `RendererMesh::position_buffer`) and the
/// 8 corners of its local bounds, in `FRUSTUM_NDC` order.
struct MeshLines {
    index_buffer: wgpu::Buffer,
    index_count: u32,
    bounds_vertex_buffer: wgpu::Buffer,
}

const BOUNDS_OPACITY: f32 = 0.5;

/// The 8 corners of the box `(min, max)`, in `FRUSTUM_NDC` order (NDC −1 → min, 1 → max; z 0 → min).
fn box_corners((min, max): (Vector3f, Vector3f)) -> [Vector3f; 8] {
    FRUSTUM_NDC.map(|[x, y, z]| {
        Vector3f::new(
            if x < 0.0 { min.x } else { max.x },
            if y < 0.0 { min.y } else { max.y },
            if z <= 0.0 { min.z } else { max.z },
        )
    })
}

/// Whether the world-space box `corners` is entirely outside one plane of the `view_projection`
/// frustum (wgpu clip space: −w ≤ x, y ≤ w, 0 ≤ z ≤ w).
fn outside_frustum(view_projection: &glam::Mat4, corners: &[Vector3f; 8]) -> bool {
    let clip = corners.map(|corner| *view_projection * corner.extend(1.0));
    let all = |outside: fn(&glam::Vec4) -> bool| clip.iter().all(outside);
    all(|c| c.x < -c.w)
        || all(|c| c.x > c.w)
        || all(|c| c.y < -c.w)
        || all(|c| c.y > c.w)
        || all(|c| c.z < 0.0)
        || all(|c| c.z > c.w)
}

/// World-space gizmo vertex, the `position_buffer` layout.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WireVertex {
    position: [f32; 3],
}

/// NDC cube corners (wgpu: x, y ∈ [−1, 1], z ∈ [0, 1]); near plane 0–3, far plane 4–7.
const FRUSTUM_NDC: [[f32; 3]; 8] = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// 12 frustum edges: near loop, far loop, then the 4 connectors.
const FRUSTUM_LINE_INDICES: [u32; 24] = [
    0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7,
];
const MAX_FRUSTUMS: usize = 8;

/// Floor grid: N×N cells of `GRID_GAP` m on the XZ plane, centred at the origin.
const GRID_CELLS: i32 = 10;
const GRID_GAP: f32 = 1.0;

fn grid_lines() -> (Vec<WireVertex>, Vec<u32>) {
    let half = GRID_CELLS as f32 * GRID_GAP * 0.5;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=GRID_CELLS {
        let offset = -half + i as f32 * GRID_GAP;
        for (a, b) in [
            ([-half, 0.0, offset], [half, 0.0, offset]),
            ([offset, 0.0, -half], [offset, 0.0, half]),
        ] {
            let base = vertices.len() as u32;
            vertices.push(WireVertex { position: a });
            vertices.push(WireVertex { position: b });
            indices.extend_from_slice(&[base, base + 1]);
        }
    }
    (vertices, indices)
}

/// sRGB display value → the HDR value the engine's default tonemap (Reinhard c / (1 + c)) and
/// sRGB encode turn back into it.
fn display_to_hdr(c: f32) -> f32 {
    linear_to_hdr(c.powf(2.2))
}

fn linear_to_hdr(linear: f32) -> f32 {
    let linear = linear.min(0.99);
    linear / (1.0 - linear)
}

/// `color` (HDR, from `display_to_hdr`) at `opacity` over the black background, as it shows after
/// tonemapping; blending in HDR would come out brighter.
fn with_opacity(color: [f32; 4], opacity: f32) -> [f32; 4] {
    let scale = |hdr: f32| linear_to_hdr(hdr / (1.0 + hdr) * opacity);
    [scale(color[0]), scale(color[1]), scale(color[2]), 1.0]
}

/// Identity transform with a flat sRGB color, for the world-space gizmos.
fn solid_per_draw(color_srgb: [f32; 3]) -> PerDraw {
    PerDraw {
        position: [0.0; 4],
        rotation: [0.0; 4],
        scale: [1.0, 1.0, 1.0, 0.0],
        color: [
            display_to_hdr(color_srgb[0]),
            display_to_hdr(color_srgb[1]),
            display_to_hdr(color_srgb[2]),
            1.0,
        ],
    }
}

fn storage_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct WireframeState {
    pipeline: Pipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Grown to the instance count; rebuilt with its bind group
    per_draw: Option<(usize, wgpu::Buffer, wgpu::BindGroup)>,
    frustum_vertex_buffer: wgpu::Buffer,
    frustum_index_buffer: wgpu::Buffer,
    grid_vertex_buffer: wgpu::Buffer,
    grid_index_buffer: wgpu::Buffer,
    grid_index_count: u32,
}

pub struct PassWireframePreview {
    state: Option<WireframeState>,
    mesh_lines: HashMap<MeshHandle, MeshLines>,
}

impl PassWireframePreview {
    pub fn new() -> Self {
        Self {
            state: None,
            mesh_lines: HashMap::new(),
        }
    }

    /// Builds the edge index buffer and bounds corners of `handle` once.
    fn ensure_mesh(
        &mut self,
        renderer: &dyn PillRenderer,
        resources: &ResourceManager,
        handle: MeshHandle,
        local_bounds: (Vector3f, Vector3f),
    ) {
        if self.mesh_lines.contains_key(&handle) {
            return;
        }
        let Some(data) = resources
            .get_resource::<Mesh>(&handle)
            .ok()
            .and_then(Mesh::data)
        else {
            return;
        };
        let mut edges: Vec<u32> = Vec::with_capacity(data.indices.len() * 2);
        for triangle in data.indices.chunks_exact(3) {
            edges.extend_from_slice(&[
                triangle[0],
                triangle[1],
                triangle[1],
                triangle[2],
                triangle[2],
                triangle[0],
            ]);
        }
        let index_buffer =
            renderer
                .get_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wireframe_line_indices"),
                    contents: bytemuck::cast_slice(&edges),
                    usage: wgpu::BufferUsages::INDEX,
                });
        let corners = box_corners(local_bounds).map(|corner| WireVertex {
            position: corner.to_array(),
        });
        let bounds_vertex_buffer =
            renderer
                .get_device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("wireframe_bounds_vertices"),
                    contents: bytemuck::cast_slice(&corners),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        self.mesh_lines.insert(
            handle,
            MeshLines {
                index_buffer,
                index_count: edges.len() as u32,
                bounds_vertex_buffer,
            },
        );
    }
}

impl PassWireframePreview {
    pub fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let camera_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some("pass_wireframe_preview"),
            vs: ShaderDesc {
                source: VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            fragment_constants: &[],
            vertex_buffers: &[RendererMesh::position_layout_descriptor()],
            bind_groups: vec![vec![camera_entry], vec![storage_entry()]],
            targets: &[Some(wgpu::ColorTargetState {
                format: renderer.get_hdr_format(),
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            // No depth: an X-ray view
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
        })?;

        let device = renderer.get_device();
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass_wireframe_preview_camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pass_wireframe_preview_camera"),
            layout: &pipeline.bind_group_layouts[0],
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let frustum_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pass_wireframe_preview_frustum_indices"),
            contents: bytemuck::cast_slice(&FRUSTUM_LINE_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let frustum_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass_wireframe_preview_frustum_vertices"),
            size: (std::mem::size_of::<WireVertex>() * 8 * MAX_FRUSTUMS) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (grid_vertices, grid_indices) = grid_lines();
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pass_wireframe_preview_grid_vertices"),
            contents: bytemuck::cast_slice(&grid_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grid_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pass_wireframe_preview_grid_indices"),
            contents: bytemuck::cast_slice(&grid_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.state = Some(WireframeState {
            pipeline,
            camera_buffer,
            camera_bind_group,
            per_draw: None,
            frustum_vertex_buffer,
            frustum_index_buffer,
            grid_vertex_buffer,
            grid_index_buffer,
            grid_index_count: grid_indices.len() as u32,
        });
        Ok(())
    }

    /// Clears `target` (an HDR render target) and draws the wireframe view into all of it.
    pub fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        target: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let cameras = &world.query::<CameraComponent>()?.data;
        let transforms = &world.query::<TransformComponent>()?.data;
        let renderables = &world.query::<MeshRenderingComponent>()?.data;
        let Some(preview_index) = world
            .query::<FlyCamera>()?
            .data
            .iter()
            .position(|fly| fly.as_ref().is_some_and(|fly| fly.overview))
        else {
            return Ok(());
        };
        let (Some(Some(preview_camera)), Some(Some(preview_transform))) =
            (cameras.get(preview_index), transforms.get(preview_index))
        else {
            return Ok(());
        };
        // The engine renders the enabled camera; meshes outside its frustum count as culled
        let game_view_projection =
            cameras
                .iter()
                .zip(transforms.iter())
                .find_map(|pair| match pair {
                    (Some(camera), Some(transform)) if camera.enabled => {
                        Some(camera_view_projection(camera, transform))
                    }
                    _ => None,
                });

        // One palette color per entity, instances grouped per mesh: (mesh, in view, culled)
        let mut groups: Vec<(MeshHandle, Vec<PerDraw>, Vec<PerDraw>)> = Vec::new();
        for (entity_index, (transform, renderable)) in
            transforms.iter().zip(renderables.iter()).enumerate()
        {
            let (Some(transform), Some(renderable)) = (transform, renderable) else {
                continue;
            };
            let Some(mesh_handle) = renderable.mesh_handle else {
                continue;
            };
            if transform.scale == Vector3f::ZERO {
                continue;
            }
            let local_bounds = world.renderer_mesh(&mesh_handle)?.local_bounds;
            self.ensure_mesh(renderer, world.resources, mesh_handle, local_bounds);
            if !self.mesh_lines.contains_key(&mesh_handle) {
                continue;
            }
            let world_corners = box_corners(local_bounds).map(|corner| {
                transform.position + transform.rotation_matrix() * (transform.scale * corner)
            });
            let culled = game_view_projection
                .is_some_and(|view_projection| outside_frustum(&view_projection, &world_corners));
            let palette = DISTINCT_COLOR_PALETTE[entity_index % DISTINCT_COLOR_PALETTE.len()];
            let rotation = transform.rotation * (std::f32::consts::PI / 180.0);
            let per_draw = PerDraw {
                position: transform.position.extend(0.0).to_array(),
                rotation: rotation.extend(0.0).to_array(),
                scale: transform.scale.extend(0.0).to_array(),
                color: [
                    display_to_hdr(palette.0),
                    display_to_hdr(palette.1),
                    display_to_hdr(palette.2),
                    1.0,
                ],
            };
            let group = match groups
                .iter()
                .position(|(handle, ..)| *handle == mesh_handle)
            {
                Some(index) => &mut groups[index],
                None => {
                    groups.push((mesh_handle, Vec::new(), Vec::new()));
                    groups.last_mut().expect("just pushed")
                }
            };
            if culled {
                group.2.push(per_draw);
            } else {
                group.1.push(per_draw);
            }
        }
        // Per mesh: the edges of the meshes in view, then the bounds of all its meshes
        let mut staging: Vec<PerDraw> = Vec::new();
        let mut ranges: Vec<(MeshHandle, u32, u32)> = Vec::new();
        let mut bounds_ranges: Vec<(MeshHandle, u32, u32)> = Vec::new();
        for (mesh_handle, in_view, culled) in &groups {
            ranges.push((*mesh_handle, staging.len() as u32, in_view.len() as u32));
            staging.extend_from_slice(in_view);
            bounds_ranges.push((
                *mesh_handle,
                staging.len() as u32,
                (in_view.len() + culled.len()) as u32,
            ));
            staging.extend(in_view.iter().chain(culled).map(|per_draw| PerDraw {
                color: with_opacity(per_draw.color, BOUNDS_OPACITY),
                ..*per_draw
            }));
        }

        // Every other camera as its frustum's world-space edges: inverse(view-projection) × NDC cube
        let mut frustum_vertices: Vec<WireVertex> = Vec::new();
        let frustum_instance_base = staging.len() as u32;
        for (index, (camera, transform)) in cameras.iter().zip(transforms.iter()).enumerate() {
            if index == preview_index || frustum_vertices.len() / 8 >= MAX_FRUSTUMS {
                continue;
            }
            let (Some(camera), Some(transform)) = (camera, transform) else {
                continue;
            };
            let inverse = camera_view_projection(camera, transform).inverse();
            for ndc in FRUSTUM_NDC {
                frustum_vertices.push(WireVertex {
                    position: inverse.project_point3(Vector3f::from_array(ndc)).to_array(),
                });
            }
            staging.push(solid_per_draw([1.0, 0.85, 0.1]));
        }
        let frustum_count = frustum_vertices.len() / 8;

        let grid_instance = staging.len() as u32;
        staging.push(solid_per_draw([0.28, 0.30, 0.34]));

        let state = self
            .state
            .as_mut()
            .expect("PassWireframePreview: init() must run before draw()");
        if state
            .per_draw
            .as_ref()
            .is_none_or(|(capacity, _, _)| *capacity < staging.len())
        {
            let capacity = staging.len().next_power_of_two();
            let buffer = renderer.create_buffer(BufferDesc {
                label: Some("pass_wireframe_preview_per_draw"),
                byte_size: (std::mem::size_of::<PerDraw>() * capacity) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })?;
            let bind_group = renderer
                .get_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pass_wireframe_preview_per_draw"),
                    layout: &state.pipeline.bind_group_layouts[1],
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
            state.per_draw = Some((capacity, buffer, bind_group));
        }
        let (_, per_draw_buffer, per_draw_bind_group) = state
            .per_draw
            .as_ref()
            .expect("per-draw buffer built above");

        let queue = renderer.get_queue();
        queue.write_buffer(
            &state.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform {
                view_projection: camera_view_projection(preview_camera, preview_transform)
                    .to_cols_array_2d(),
            }),
        );
        queue.write_buffer(per_draw_buffer, 0, bytemuck::cast_slice(&staging));
        if frustum_count > 0 {
            queue.write_buffer(
                &state.frustum_vertex_buffer,
                0,
                bytemuck::cast_slice(&frustum_vertices),
            );
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_wireframe_preview"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_pipeline(&state.pipeline.pipeline);
        render_pass.set_bind_group(0, &state.camera_bind_group, &[]);
        render_pass.set_bind_group(1, per_draw_bind_group, &[]);

        // Grid first, so everything else draws over it
        render_pass.set_vertex_buffer(0, state.grid_vertex_buffer.slice(..));
        render_pass.set_index_buffer(state.grid_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(
            0..state.grid_index_count,
            0,
            grid_instance..grid_instance + 1,
        );

        for (mesh_handle, base, count) in &ranges {
            let lines = &self.mesh_lines[mesh_handle];
            world
                .renderer_mesh(mesh_handle)?
                .set_position_buffer(&mut render_pass);
            render_pass.set_index_buffer(lines.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..lines.index_count, 0, *base..*base + *count);
        }

        render_pass.set_index_buffer(
            state.frustum_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        for (mesh_handle, base, count) in &bounds_ranges {
            render_pass.set_vertex_buffer(
                0,
                self.mesh_lines[mesh_handle].bounds_vertex_buffer.slice(..),
            );
            render_pass.draw_indexed(
                0..FRUSTUM_LINE_INDICES.len() as u32,
                0,
                *base..*base + *count,
            );
        }

        if frustum_count > 0 {
            render_pass.set_vertex_buffer(0, state.frustum_vertex_buffer.slice(..));
            render_pass.set_index_buffer(
                state.frustum_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for i in 0..frustum_count {
                let instance = frustum_instance_base + i as u32;
                render_pass.draw_indexed(
                    0..FRUSTUM_LINE_INDICES.len() as u32,
                    (i * 8) as i32,
                    instance..instance + 1,
                );
            }
        }
        Ok(())
    }
}
