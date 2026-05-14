#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

#[cfg(feature = "headless")]
mod dummy_renderer;
mod egui;
mod pass_egui;
mod pass_scene;
mod render_queue;
mod renderer;

// --- Use ---

pub use renderer::{
    BufferDesc, Pass, PillRenderer, PipelineV2, PipelineV2Desc, Renderer, RendererCameraHandle,
    RendererMaterialHandle, RendererMeshHandle, RendererShaderHandle, RendererTargetDesc,
    RendererTextureHandle, ShaderDesc, WorldQuery,
};

#[cfg(feature = "headless")]
pub use self::dummy_renderer::DummyRenderer;
pub use egui::EguiUI;
pub use pass_egui::PassEgui;
pub use pass_scene::PassScene;

pub use render_queue::{
    compose_pbr_render_queue_key, compose_render_queue_key, decompose_render_queue_key,
    RenderQueueItem, RenderQueueKey, RenderQueueKeyFields, RENDER_QUEUE_KEY_ORDER,
};
