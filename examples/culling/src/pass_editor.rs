//! Editor composite, a game pass before tonemapping: copies the engine's frame (game camera) into
//! the game render target, draws the wireframe view into the editor render target, then draws
//! both into their window rects of the HDR target (`GAME_VIEWPORT` / `EDITOR_VIEWPORT`). The
//! engine tonemap then presents the whole frame. Off while the editor is hidden.

use crate::editor::{EditorState, ViewportRect, EDITOR_VIEWPORT, GAME_VIEWPORT};
use crate::pass_wireframe::PassWireframePreview;
use pill_engine::game::*;

// Generated from res/shaders/*.hlsl by pill_assets
const VERTEX_SHADER: &str = include_str!("../res/shaders/editor_fullscreen_vertex.wgsl");
const COMPOSITE_SHADER: &str = include_str!("../res/shaders/editor_composite_fragment.wgsl");
const COPY_SHADER: &str = include_str!("../res/shaders/editor_composite_copy_fragment.wgsl");

struct EditorTargets {
    composite: Pipeline,
    copy: Pipeline,
    sampler: wgpu::Sampler,
    game: PassTextureHandle,
    wireframe: PassTextureHandle,
    // (target generation, copy group, game group, wireframe group): views change on resize
    bind_groups: Option<(u64, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup)>,
}

pub struct EditorPass {
    wireframe: PassWireframePreview,
    state: Option<EditorTargets>,
}

impl EditorPass {
    pub fn new() -> Self {
        Self {
            wireframe: PassWireframePreview::new(),
            state: None,
        }
    }
}

fn texture_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn fullscreen_pipeline(
    renderer: &mut dyn PillRenderer,
    label: &str,
    fragment_shader: &str,
    entries: Vec<wgpu::BindGroupLayoutEntry>,
) -> Result<Pipeline> {
    let format = renderer.get_hdr_format();
    renderer.create_pipeline(PipelineDesc {
        label: Some(label),
        vs: ShaderDesc {
            source: VERTEX_SHADER,
            entry_func: "vs_main",
        },
        ps: ShaderDesc {
            source: fragment_shader,
            entry_func: "fs_main",
        },
        fragment_constants: &[],
        vertex_buffers: &[],
        bind_groups: vec![entries],
        targets: &[Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        primitive: wgpu::PrimitiveState::default(),
    })
}

fn bind_group(
    renderer: &dyn PillRenderer,
    pipeline: &Pipeline,
    source: &wgpu::TextureView,
    sampler: Option<&wgpu::Sampler>,
) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(source),
    }];
    if let Some(sampler) = sampler {
        entries.push(wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    renderer
        .get_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("editor_pass"),
            layout: &pipeline.bind_group_layouts[0],
            entries: &entries,
        })
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    label: &str,
    output: &'a wgpu::TextureView,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    })
}

/// Draws the full-screen triangle into the `viewport` rect of a `width` × `height` target.
fn draw_in_rect(
    render_pass: &mut wgpu::RenderPass,
    group: &wgpu::BindGroup,
    viewport: ViewportRect,
    (width, height): (u32, u32),
) {
    let (x, y, rect_width, rect_height) = viewport.to_pixels(width, height);
    render_pass.set_viewport(
        x as f32,
        y as f32,
        rect_width as f32,
        rect_height as f32,
        0.0,
        1.0,
    );
    render_pass.set_scissor_rect(x, y, rect_width, rect_height);
    render_pass.set_bind_group(0, group, &[]);
    render_pass.draw(0..3, 0..1);
}

impl Pass for EditorPass {
    fn get_label(&self) -> &str {
        "editor_pass"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.wireframe.init(renderer)?;
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let composite = fullscreen_pipeline(
            renderer,
            "editor_composite",
            COMPOSITE_SHADER,
            vec![texture_entry(), sampler_entry],
        )?;
        let copy =
            fullscreen_pipeline(renderer, "editor_copy", COPY_SHADER, vec![texture_entry()])?;
        let hdr_format = renderer.get_hdr_format();
        let game = renderer.create_render_target(RendererTargetDesc {
            name: "editor_game".to_string(),
            format: hdr_format,
            size: TargetSize::Surface,
        })?;
        let wireframe = renderer.create_render_target(RendererTargetDesc {
            name: "editor_wireframe".to_string(),
            format: hdr_format,
            size: TargetSize::Surface,
        })?;
        let sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("editor_composite"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
        self.state = Some(EditorTargets {
            composite,
            copy,
            sampler,
            game,
            wireframe,
            bind_groups: None,
        });
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        if !world.get_global::<EditorState>()?.visible {
            return Ok(());
        }
        let state = self
            .state
            .as_mut()
            .expect("EditorPass: init() must run before draw()");
        let generation = renderer.target_generation();
        if !matches!(&state.bind_groups, Some((built, ..)) if *built == generation) {
            let view = |handle| {
                renderer
                    .get_render_target_view(handle)
                    .ok_or("EditorPass: render target missing")
            };
            let hdr_target = world.get_global::<RenderStateComponent>()?.hdr_target();
            state.bind_groups = Some((
                generation,
                bind_group(renderer, &state.copy, view(hdr_target)?, None),
                bind_group(
                    renderer,
                    &state.composite,
                    view(state.game)?,
                    Some(&state.sampler),
                ),
                bind_group(
                    renderer,
                    &state.composite,
                    view(state.wireframe)?,
                    Some(&state.sampler),
                ),
            ));
        }

        let wireframe_view = renderer
            .get_render_target_view(state.wireframe)
            .ok_or("EditorPass: wireframe target missing")?
            .clone();
        self.wireframe
            .draw(encoder, renderer, &wireframe_view, world)?;

        let (_, copy_group, game_group, wireframe_group) =
            state.bind_groups.as_ref().expect("bind groups built above");
        let size = renderer.get_render_size();
        let hdr_view = renderer
            .get_render_target_view(world.get_global::<RenderStateComponent>()?.hdr_target())
            .ok_or("EditorPass: HDR target missing")?;
        let game_view = renderer
            .get_render_target_view(state.game)
            .ok_or("EditorPass: game target missing")?;

        // The engine's frame (game camera) into the game target
        {
            let mut render_pass = begin_pass(encoder, "editor_copy", game_view);
            render_pass.set_pipeline(&state.copy.pipeline);
            draw_in_rect(&mut render_pass, copy_group, ViewportRect::FULL, size);
        }
        // Both targets into their rects of the frame
        let mut render_pass = begin_pass(encoder, "editor_composite", hdr_view);
        render_pass.set_pipeline(&state.composite.pipeline);
        draw_in_rect(&mut render_pass, game_group, GAME_VIEWPORT, size);
        draw_in_rect(&mut render_pass, wireframe_group, EDITOR_VIEWPORT, size);
        Ok(())
    }
}
