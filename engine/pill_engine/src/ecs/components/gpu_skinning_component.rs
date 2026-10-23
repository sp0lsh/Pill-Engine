//! Engine-internal state of skeletal animation: the models added with
//! `Engine::add_skeletal_model`, and the CPU side of the GPU skin cache (`graphics::pass_skinning`):
//! skin sources (rest vertices, joints and weights of one glTF primitive, shared by its
//! instances), skinned instances (one per animated entity, each with its own mesh) and the joint
//! matrices of all instances, which `skeletal_animation_system` rewrites every frame.

use crate::ecs::components::{GlobalComponent, GlobalComponentStorage};
use crate::resources::MeshHandle;

use pill_core::PillTypeMapKey;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SkinSourceId(pub(crate) usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SkinInstanceId(pub(crate) usize);

#[cfg(feature = "gltf_loading")]
pub(crate) struct SkeletalModel {
    pub data: pill_assets::gltf_skin::GltfSkinnedData,
    /// One per `data.primitives`
    pub sources: Vec<SkinSourceId>,
}

pub(crate) struct SkinSource {
    pub vertex_count: u32,
    pub joint_count: u32,
    /// One thread per triangle, face normals (see skinning_compute.hlsl)
    pub flat: bool,
    /// The GPU inputs of skinning_compute.hlsl group 0
    pub rest_positions: Vec<f32>,
    pub rest_frames: Vec<[f32; 4]>,
    pub skin_vertices: Vec<[u32; 4]>,
    pub skin_weights: Vec<[f32; 4]>,
    #[cfg(feature = "gltf_loading")]
    pub bounds: pill_assets::skinning::SkinBounds,
}

pub(crate) struct SkinInstance {
    pub mesh: MeshHandle,
    /// Where the system refits the culling bounds
    #[cfg(not(feature = "headless"))]
    pub renderer_mesh: crate::graphics::RendererMeshHandle,
    pub source: SkinSourceId,
    /// First joint of this instance in `GpuSkinningComponent::joint_rows`
    pub joint_offset: u32,
}

#[derive(Default)]
pub(crate) struct GpuSkinningComponent {
    #[cfg(feature = "gltf_loading")]
    pub(crate) models: Vec<SkeletalModel>,
    /// CPU time of the last `skeletal_animation_system` run, ms
    pub(crate) cpu_ms: f32,
    pub(crate) sources: Vec<SkinSource>,
    pub(crate) instances: Vec<SkinInstance>,
    /// Row-major 3x4 joint matrices of every instance, 3 rows per joint
    pub(crate) joint_rows: Vec<[f32; 4]>,
}

impl PillTypeMapKey for GpuSkinningComponent {
    type Storage = GlobalComponentStorage<GpuSkinningComponent>;
}

impl GlobalComponent for GpuSkinningComponent {}
