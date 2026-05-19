#![allow(clippy::too_many_arguments)]
use crate::{
    ecs::{CameraComponent, ComponentStorage, EguiClient, EntityHandle, TransformComponent},
    graphics::RenderQueueItem,
    resources::{
        ResourceManager, ShaderParameterSlot, ShaderTextureSlot, TextureType,
    },
};

use indexmap::IndexMap;
use pill_core::Timer;

use anyhow::Result;
use std::{collections::HashMap, sync::Arc};

// --- Renderer resource handles ---

pill_core::define_new_pill_slotmap_key! {
    pub struct RendererMaterialHandle;
}

pill_core::define_new_pill_slotmap_key! {
    pub struct RendererMeshHandle;
}

pill_core::define_new_pill_slotmap_key! {
    pub struct RendererCameraHandle;
}

pill_core::define_new_pill_slotmap_key! {
    pub struct RendererTextureHandle;
}

pill_core::define_new_pill_slotmap_key! {
    pub struct RendererShaderHandle;
}

// --- Pass API types ---

pub struct WorldQuery<'a> {
    pub active_camera: EntityHandle,
    pub render_queue: &'a [RenderQueueItem],
    pub camera_components: &'a ComponentStorage<CameraComponent>,
    pub transform_components: &'a ComponentStorage<TransformComponent>,
    pub delta_time: f32,
    pub resources: &'a ResourceManager,
}

#[derive(Clone, Copy)]
pub struct BufferDesc<'a> {
    pub label: Option<&'a str>,
    pub byte_size: u64,
    pub usage: wgpu::BufferUsages,
}

#[derive(Clone, Copy, Debug)]
pub struct ShaderDesc<'a> {
    pub source: &'a str,
    pub entry_func: &'a str,
}

#[derive(Clone, Debug)]
pub struct PipelineV2Desc<'a> {
    pub label: Option<&'a str>,
    pub vs: ShaderDesc<'a>,
    pub ps: ShaderDesc<'a>,
    pub vertex_buffers: &'a [wgpu::VertexBufferLayout<'a>],
    pub bind_groups: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
    pub targets: &'a [Option<wgpu::ColorTargetState>],
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
}

pub struct RendererTargetDesc {
    pub name: String,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

pub struct PipelineV2 {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

// --- Pass trait ---

pub trait Pass {
    fn get_label(&self) -> &str;
    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()>;
    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()>;
}

// --- Renderer trait definition ---

pub trait PillRenderer {
    fn new(window: Arc<winit::window::Window>, config: config::Config) -> Result<Self>
    where
        Self: Sized;

    // --- Create ---

    fn create_shader_struct(
        &mut self,
        name: &str,
        vertex_wgsl: &str,
        fragment_wgsl: &str,
        texture_slots: &HashMap<String, ShaderTextureSlot>,
        parameter_slots: &IndexMap<String, ShaderParameterSlot>,
        pass_engine_parameters: bool,
        pass_camera_parameters: bool,
    ) -> Result<crate::renderer::resources::RendererShader>;

    fn create_camera(&mut self) -> Result<RendererCameraHandle>;

    // --- Destroy ---

    fn destroy_camera(&mut self, renderer_camera_handle: RendererCameraHandle) -> Result<()>;

    // --- Other ---

    fn resize(&mut self, new_window_size: winit::dpi::PhysicalSize<u32>);

    fn pass_input_to_egui(&mut self, event: &winit::event::WindowEvent) -> Result<()>;

    fn render(
        &mut self,
        active_camera_entity_handle: EntityHandle,
        render_queue: &[RenderQueueItem],
        camera_component_storage: &ComponentStorage<CameraComponent>,
        transform_component_storage: &ComponentStorage<TransformComponent>,
        delta_time: f32,
        timer: &mut Timer,
        resource_manager: &ResourceManager,
    ) -> Result<()>;

    // --- Pass API ---

    fn set_passes(&mut self, passes: Vec<Box<dyn Pass>>) -> Result<()>;

    fn init_default_passes(&mut self, egui_client: Arc<EguiClient>) -> Result<()>;

    fn get_device(&self) -> &wgpu::Device;

    fn get_queue(&self) -> &wgpu::Queue;

    fn get_surface_format(&self) -> wgpu::TextureFormat;

    fn get_window(&self) -> Arc<winit::window::Window>;

    fn create_buffer(&mut self, desc: BufferDesc) -> Result<wgpu::Buffer>;

    fn create_pipeline_v2(&mut self, desc: PipelineV2Desc) -> Result<PipelineV2>;

    fn create_render_target(&mut self, desc: RendererTargetDesc) -> Result<RendererTextureHandle>;

    fn create_depth_texture(&mut self, label: &str) -> Result<RendererTextureHandle>;

    fn get_render_target_view(
        &self,
        handle: RendererTextureHandle,
    ) -> Option<&wgpu::TextureView>;

    fn record_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()>;
}

pub type Renderer = Box<dyn PillRenderer>;
