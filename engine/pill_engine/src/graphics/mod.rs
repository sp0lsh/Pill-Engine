#![cfg_attr(debug_assertions, allow(dead_code, unused_imports))]

pub(crate) mod ddgi;
#[cfg(feature = "headless")]
mod dummy_renderer;
mod froxelizer;
mod fullscreen;
pub(crate) mod ibl;
#[cfg(feature = "post_fx")]
mod pass_bloom;
#[cfg(feature = "post_fx")]
mod pass_depth_of_field;
#[cfg(feature = "ui")]
mod pass_egui;
#[cfg(feature = "post_fx")]
mod pass_fxaa;
pub(crate) mod pass_ibl_bake;
#[cfg(feature = "post_fx")]
mod pass_lens;
pub(crate) mod pass_lights;
#[cfg(feature = "post_fx")]
mod pass_motion_blur;
mod pass_pbr_opaque;
mod pass_punctual_shadows;
mod pass_shadow;
#[cfg(feature = "post_fx")]
mod pass_ssao;
#[cfg(feature = "post_fx")]
mod pass_ssr;
mod pass_tonemap;
#[cfg(feature = "post_fx")]
mod projection;
mod punctual_shadows;
mod render_queue;
mod renderer;
mod scene_uniform;
pub(crate) mod shadow;
mod shadow_casters;

// --- Use ---

pub(crate) use renderer::mip_copy_layout;
pub use renderer::{
    camera_forward, camera_view_projection, BufferDesc, ComputePipeline, ComputePipelineDesc,
    FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, PipelineDesc, Renderer,
    RendererMaterialHandle, RendererMeshHandle, RendererTargetDesc, RendererTextureHandle,
    ShaderDesc, TargetSize, TextureData, WorldQuery,
};

#[cfg(feature = "headless")]
pub use self::dummy_renderer::DummyRenderer;
#[cfg(feature = "post_fx")]
pub use pass_bloom::PassBloom;
#[cfg(feature = "post_fx")]
pub use pass_depth_of_field::PassDepthOfField;
#[cfg(feature = "ui")]
pub use pass_egui::PassEgui;
#[cfg(feature = "post_fx")]
pub use pass_fxaa::PassFxaa;
#[cfg(feature = "post_fx")]
pub use pass_lens::PassLens;
pub(crate) use pass_lights::PassLights;
#[cfg(feature = "post_fx")]
pub use pass_motion_blur::PassMotionBlur;
pub use pass_pbr_opaque::PassPBROpaque;
#[cfg(feature = "post_fx")]
pub use pass_pbr_opaque::PassPBRTranslucent;
pub(crate) use pass_punctual_shadows::PassPunctualShadows;
pub(crate) use pass_shadow::PassShadow;
#[cfg(feature = "post_fx")]
pub use pass_ssao::PassSsao;
#[cfg(feature = "post_fx")]
pub use pass_ssr::PassSsr;
pub use pass_tonemap::PassTonemap;
pub(crate) use scene_uniform::SceneUniform;

pub use render_queue::{
    compose_render_queue_key, decompose_render_queue_key, RenderQueueKey, RenderQueueKeyFields,
    RENDER_QUEUE_KEY_MATERIAL_INDEX, RENDER_QUEUE_KEY_MESH_INDEX,
};
