use crate::graphics::{
    RendererCameraHandle, RendererMaterialHandle, RendererMeshHandle, RendererShaderHandle,
    RendererTextureHandle,
};
use crate::renderer::resources::{
    EngineParameters, RendererCamera, RendererMaterial, RendererMesh, RendererShader,
    RendererTexture,
};

use anyhow::Result;
use pill_core::PillSlotMap;

pub const MAX_SHADERS: usize = 10;
pub const MAX_TEXTURES: usize = 10;
pub const MAX_MATERIALS: usize = 10;
pub const MAX_MESHES: usize = 10;
pub const MAX_CAMERAS: usize = 10;

pub struct RendererResourceStorage {
    pub(crate) shaders: PillSlotMap<RendererShaderHandle, RendererShader>,
    pub(crate) materials: PillSlotMap<RendererMaterialHandle, RendererMaterial>,
    pub(crate) textures: PillSlotMap<RendererTextureHandle, RendererTexture>,
    pub(crate) meshes: PillSlotMap<RendererMeshHandle, RendererMesh>,
    pub(crate) cameras: PillSlotMap<RendererCameraHandle, RendererCamera>,
    pub(crate) engine_parameters: EngineParameters,
}

impl RendererResourceStorage {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        Ok(RendererResourceStorage {
            shaders: PillSlotMap::<RendererShaderHandle, RendererShader>::with_capacity_and_key(
                MAX_SHADERS,
            ),
            textures: PillSlotMap::<RendererTextureHandle, RendererTexture>::with_capacity_and_key(
                MAX_TEXTURES,
            ),
            materials: PillSlotMap::<RendererMaterialHandle, RendererMaterial>::with_capacity_and_key(
                MAX_MATERIALS,
            ),
            meshes: PillSlotMap::<RendererMeshHandle, RendererMesh>::with_capacity_and_key(
                MAX_MESHES,
            ),
            cameras: PillSlotMap::<RendererCameraHandle, RendererCamera>::with_capacity_and_key(
                MAX_CAMERAS,
            ),
            engine_parameters: EngineParameters::new(device)?,
        })
    }
}
