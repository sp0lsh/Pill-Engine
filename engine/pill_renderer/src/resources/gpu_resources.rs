use pill_core::{
    RendererCameraTag, RendererMaterialTag, RendererMeshTag, RendererTextureTag, ResourcePool,
};

use super::{RendererCamera, RendererMaterial, RendererMesh, RendererTexture};

pub struct GpuResources {
    pub textures: ResourcePool<RendererTextureTag, RendererTexture>,
    pub materials: ResourcePool<RendererMaterialTag, RendererMaterial>,
    pub meshes: ResourcePool<RendererMeshTag, RendererMesh>,
    pub cameras: ResourcePool<RendererCameraTag, RendererCamera>,
}

impl GpuResources {
    pub fn new() -> Self {
        Self {
            textures: ResourcePool::new(),
            materials: ResourcePool::new(),
            meshes: ResourcePool::new(),
            cameras: ResourcePool::new(),
        }
    }
}

impl Default for GpuResources {
    fn default() -> Self {
        Self::new()
    }
}
