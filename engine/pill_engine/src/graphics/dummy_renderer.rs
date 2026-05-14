use crate::{
    ecs::{CameraComponent, ComponentStorage, EguiClient, EntityHandle, TransformComponent},
    graphics::{
        BufferDesc, Pass, PillRenderer, PipelineV2, PipelineV2Desc, RendererTargetDesc,
        RenderQueueItem, RendererCameraHandle, RendererMaterialHandle, RendererMeshHandle,
        RendererShaderHandle, RendererTextureHandle, WorldQuery,
    },
    internal::{MaterialParameter, MaterialTexture},
    resources::{MeshData, ShaderParameterSlot, ShaderTextureSlot, TextureType},
};

use anyhow::Result;
use image::DynamicImage;
use indexmap::IndexMap;
use pill_core::Timer;
use std::{collections::HashMap, sync::Arc};
use winit::{dpi::PhysicalSize, event::WindowEvent, window::Window};

pub struct DummyRenderer;

impl PillRenderer for DummyRenderer {
    fn new(_window: Arc<Window>, _config: config::Config) -> Result<Self> {
        Ok(DummyRenderer)
    }

    // --- Create ---

    fn create_shader(
        &mut self,
        _name: &str,
        _vertex_wgsl: &str,
        _fragment_wgsl: &str,
        _texture_slots: &HashMap<String, ShaderTextureSlot>,
        _parameter_slots: &IndexMap<String, ShaderParameterSlot>,
        _pass_engine_parameters: bool,
        _pass_camera_parameters: bool,
    ) -> Result<RendererShaderHandle> {
        Ok(RendererShaderHandle::default())
    }

    fn create_material(
        &mut self,
        _name: &str,
        _renderer_shader_handle: RendererShaderHandle,
        _textures: &IndexMap<String, MaterialTexture>,
        _parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<RendererMaterialHandle> {
        Ok(RendererMaterialHandle::default())
    }

    fn create_texture(
        &mut self,
        _name: &str,
        _image_data: &DynamicImage,
        _texture_type: TextureType,
    ) -> Result<RendererTextureHandle> {
        Ok(RendererTextureHandle::default())
    }

    fn create_mesh(&mut self, _name: &str, _mesh_data: &MeshData) -> Result<RendererMeshHandle> {
        Ok(RendererMeshHandle::default())
    }

    fn create_camera(&mut self) -> Result<RendererCameraHandle> {
        Ok(RendererCameraHandle::default())
    }

    // --- Update ---

    fn update_material_textures(
        &mut self,
        _renderer_material_handle: RendererMaterialHandle,
        _textures: &IndexMap<String, MaterialTexture>,
    ) -> Result<()> {
        Ok(())
    }

    fn update_material_parameters(
        &mut self,
        _renderer_material_handle: RendererMaterialHandle,
        _parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<()> {
        Ok(())
    }

    // --- Destroy ---

    fn destroy_shader(&mut self, _renderer_shader_handle: RendererShaderHandle) -> Result<()> {
        Ok(())
    }

    fn destroy_material(
        &mut self,
        _renderer_material_handle: RendererMaterialHandle,
    ) -> Result<()> {
        Ok(())
    }

    fn destroy_texture(&mut self, _renderer_texture_handle: RendererTextureHandle) -> Result<()> {
        Ok(())
    }

    fn destroy_mesh(&mut self, _renderer_mesh_handle: RendererMeshHandle) -> Result<()> {
        Ok(())
    }

    fn destroy_camera(&mut self, _renderer_camera_handle: RendererCameraHandle) -> Result<()> {
        Ok(())
    }

    // --- Other ---

    fn resize(&mut self, _new_window_size: PhysicalSize<u32>) {}

    fn pass_input_to_egui(&mut self, _event: &WindowEvent) -> Result<()> {
        Ok(())
    }

    fn render(
        &mut self,
        _active_camera_entity_handle: EntityHandle,
        _render_queue: &[RenderQueueItem],
        _camera_component_storage: &ComponentStorage<CameraComponent>,
        _transform_component_storage: &ComponentStorage<TransformComponent>,
        _delta_time: f32,
        _timer: &mut Timer,
    ) -> Result<()> {
        Ok(())
    }

    // --- Pass API ---

    fn set_passes(&mut self, _passes: Vec<Box<dyn Pass>>) -> Result<()> {
        Ok(())
    }

    fn init_default_passes(&mut self, _egui_client: Arc<EguiClient>) -> Result<()> {
        Ok(())
    }

    fn get_device(&self) -> &wgpu::Device {
        unimplemented!("DummyRenderer has no wgpu Device")
    }

    fn get_queue(&self) -> &wgpu::Queue {
        unimplemented!("DummyRenderer has no wgpu Queue")
    }

    fn get_surface_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba8UnormSrgb
    }

    fn create_buffer(&mut self, _desc: BufferDesc) -> Result<wgpu::Buffer> {
        unimplemented!("DummyRenderer has no wgpu Device")
    }

    fn create_pipeline_v2(&mut self, _desc: PipelineV2Desc) -> Result<PipelineV2> {
        unimplemented!("DummyRenderer has no wgpu Device")
    }

    fn create_render_target(
        &mut self,
        _desc: RendererTargetDesc,
    ) -> Result<RendererTextureHandle> {
        Ok(RendererTextureHandle::default())
    }

    fn create_depth_texture(&mut self, _label: &str) -> Result<RendererTextureHandle> {
        Ok(RendererTextureHandle::default())
    }

    fn get_render_target_view(
        &self,
        _handle: RendererTextureHandle,
    ) -> Option<&wgpu::TextureView> {
        None
    }

    fn record_scene_pass(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _view: &wgpu::TextureView,
        _world: &WorldQuery<'_>,
    ) -> Result<()> {
        Ok(())
    }
}
