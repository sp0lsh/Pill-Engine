//! Shared plumbing of the full-screen post-processing passes: one bind group (group 0), a
//! full-screen triangle and one color target.

use crate::config::FULLSCREEN_VERTEX_SHADER;
use crate::graphics::shader_bindings::BindGroupResources;
use crate::graphics::{PassTextureHandle, PillRenderer, Pipeline, PipelineDesc, ShaderDesc};

use pill_core::Result;

/// Full-screen pipeline: `fragment_shader` (`fs_main`) with its generated `layout` (group 0 only),
/// writing one `format` target with `blend` (None = replace).
pub(crate) fn create_fullscreen_pipeline(
    renderer: &mut dyn PillRenderer,
    label: &str,
    fragment_shader: &str,
    layout: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> Result<Pipeline> {
    renderer.create_pipeline(PipelineDesc {
        label: Some(label),
        vs: ShaderDesc {
            source: FULLSCREEN_VERTEX_SHADER,
            entry_func: "vs_main",
        },
        ps: ShaderDesc {
            source: fragment_shader,
            entry_func: "fs_main",
        },
        fragment_constants: &[],
        vertex_buffers: &[],
        bind_groups: layout,
        targets: &[Some(wgpu::ColorTargetState {
            format,
            blend,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        primitive: wgpu::PrimitiveState::default(),
    })
}

/// Bilinear, clamp-to-edge.
pub(crate) fn linear_clamp_sampler(renderer: &dyn PillRenderer) -> wgpu::Sampler {
    renderer
        .get_device()
        .create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
}

pub(crate) fn uniform_buffer<T>(renderer: &dyn PillRenderer, label: &str) -> wgpu::Buffer {
    renderer
        .get_device()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
}

/// Group 0 of one draw site, kept until a resize replaces the target views it holds. Each call
/// site owns one and always passes the same resources.
#[derive(Default)]
pub(crate) struct CachedBindGroup {
    bind_group: Option<(u64, wgpu::BindGroup)>,
}

impl CachedBindGroup {
    /// The cached group, rebuilt from `resources` after a resize.
    fn get(
        &mut self,
        renderer: &dyn PillRenderer,
        pipeline: &Pipeline,
        resources: &dyn BindGroupResources,
    ) -> &wgpu::BindGroup {
        let generation = renderer.target_generation();
        if !matches!(&self.bind_group, Some((built, _)) if *built == generation) {
            let bind_group =
                resources.create(renderer.get_device(), &pipeline.bind_group_layouts[0]);
            self.bind_group = Some((generation, bind_group));
        }
        &self.bind_group.as_ref().expect("bind group just built").1
    }
}

/// Draws the full-screen triangle into `output`, cleared first, with group 0 from `cache`.
pub(crate) fn draw_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &dyn PillRenderer,
    label: &str,
    pipeline: &Pipeline,
    cache: &mut CachedBindGroup,
    resources: &dyn BindGroupResources,
    output: &wgpu::TextureView,
) -> Result<()> {
    draw_fullscreen_with_load(
        encoder,
        renderer,
        label,
        pipeline,
        cache,
        resources,
        output,
        wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    )
}

/// `draw_fullscreen` with an explicit load op (`Load` to blend onto existing content).
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_fullscreen_with_load(
    encoder: &mut wgpu::CommandEncoder,
    renderer: &dyn PillRenderer,
    label: &str,
    pipeline: &Pipeline,
    cache: &mut CachedBindGroup,
    resources: &dyn BindGroupResources,
    output: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> Result<()> {
    let bind_group = cache.get(renderer, pipeline, resources);

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_pipeline(&pipeline.pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
    Ok(())
}

/// View of a pass target read by a full-screen pass.
pub(crate) fn input_view<'a>(
    renderer: &'a dyn PillRenderer,
    handle: PassTextureHandle,
    label: &str,
) -> Result<&'a wgpu::TextureView> {
    renderer
        .get_render_target_view(handle)
        .ok_or_else(|| format!("{label}: input target missing").into())
}

/// View of a pass output target.
pub(crate) fn target_view<'a>(
    renderer: &'a dyn PillRenderer,
    handle: PassTextureHandle,
    label: &str,
) -> Result<&'a wgpu::TextureView> {
    renderer
        .get_render_target_view(handle)
        .ok_or_else(|| format!("{label}: output target missing").into())
}
