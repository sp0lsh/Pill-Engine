use crate::graphics::RendererMeshHandle;
use crate::resources::{
    vertex_bounds, MeshData, MeshVertex, PackedVertexAttributes, Resource, ResourceStorage,
};

use pill_core::{PillTypeMapKey, Result, Vector3f};
use wgpu::util::DeviceExt;

// --- Vertex ---

/// GPU vertex layout of `RendererMesh`: slot 0 = positions, slot 1 = packed attributes.
pub trait Vertex {
    /// Both streams, for pipelines reading positions and attributes (bind with
    /// `RendererMesh::set_vertex_buffers`).
    fn data_layout_descriptors<'a>() -> [wgpu::VertexBufferLayout<'a>; 2];
    /// Positions only, for depth-only pipelines (bind with `RendererMesh::set_position_buffer`).
    fn position_layout_descriptor<'a>() -> wgpu::VertexBufferLayout<'a>;
}

// --- Mesh ---

/// Mesh on the GPU as two vertex streams: positions (Float32x3, 12 B) and packed attributes
/// (normal, tangent + bitangent sign, uv: 12 B), so depth-only passes fetch positions alone.
pub struct RendererMesh {
    pub name: String,
    pub position_buffer: wgpu::Buffer,
    pub attribute_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    /// Local-space AABB `(min, max)` of the vertex positions (shadow caster culling).
    pub local_bounds: (Vector3f, Vector3f),
}

/// Splits vertices into the position stream (slot 0) and the packed attribute stream (slot 1).
fn vertex_streams(vertices: &[MeshVertex]) -> (Vec<[f32; 3]>, Vec<PackedVertexAttributes>) {
    (
        vertices.iter().map(MeshVertex::position).collect(),
        vertices.iter().map(MeshVertex::packed_attributes).collect(),
    )
}

impl RendererMesh {
    pub fn new(device: &wgpu::Device, name: &str, mesh_data: &MeshData) -> Result<Self> {
        let (positions, attributes) = vertex_streams(&mesh_data.vertices);
        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?}_position_buffer", name)),
            contents: bytemuck::cast_slice(&positions),
            // STORAGE: written by the GPU skin cache (pass_skinning.rs)
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        });

        let attribute_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?}_attribute_buffer", name)),
            contents: bytemuck::cast_slice(&attributes),
            // STORAGE: written by the GPU skin cache (pass_skinning.rs)
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?}_index_buffer", name)),
            contents: bytemuck::cast_slice(&mesh_data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Ok(Self {
            name: name.to_string(),
            position_buffer,
            attribute_buffer,
            index_buffer,
            index_count: mesh_data.indices.len() as u32,
            local_bounds: mesh_data.bounds(),
        })
    }

    /// Overwrites both vertex streams in place (indices unchanged) and refits `local_bounds`;
    /// `vertices` must have the mesh's vertex count.
    pub fn update_vertices(&mut self, queue: &wgpu::Queue, vertices: &[MeshVertex]) -> Result<()> {
        let vertex_count = self.position_buffer.size() as usize / std::mem::size_of::<[f32; 3]>();
        if vertices.len() != vertex_count {
            return Err(format!(
                "Mesh {:?}: {} vertices given, it has {vertex_count}",
                self.name,
                vertices.len()
            )
            .into());
        }
        let (positions, attributes) = vertex_streams(vertices);
        queue.write_buffer(&self.position_buffer, 0, bytemuck::cast_slice(&positions));
        queue.write_buffer(&self.attribute_buffer, 0, bytemuck::cast_slice(&attributes));
        self.local_bounds = vertex_bounds(vertices);
        Ok(())
    }

    /// Binds both streams (slots 0 and 1) for pipelines built with `data_layout_descriptors`.
    pub fn set_vertex_buffers(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_vertex_buffer(0, self.position_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.attribute_buffer.slice(..));
    }

    /// Binds the position stream (slot 0) for pipelines built with `position_layout_descriptor`.
    pub fn set_position_buffer(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_vertex_buffer(0, self.position_buffer.slice(..));
    }
}

impl PillTypeMapKey for RendererMesh {
    type Storage = ResourceStorage<RendererMesh>;
}

impl Resource for RendererMesh {
    type Handle = RendererMeshHandle;

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

// slangc maps TEXCOORD0 → @location(4), NORMAL → 5, TANGENT → 6
const POSITION_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
const PACKED_ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
    5 => Unorm10_10_10_2,
    6 => Unorm10_10_10_2,
    4 => Float16x2,
];

impl Vertex for RendererMesh {
    fn data_layout_descriptors<'a>() -> [wgpu::VertexBufferLayout<'a>; 2] {
        [
            Self::position_layout_descriptor(),
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<PackedVertexAttributes>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &PACKED_ATTRIBUTES,
            },
        ]
    }

    fn position_layout_descriptor<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSITION_ATTRIBUTES,
        }
    }
}
