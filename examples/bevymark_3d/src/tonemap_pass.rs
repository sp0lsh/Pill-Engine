//! Bevy's Camera3d display transform as a game pass: Tony McMapface + deband dither, applied to
//! the engine's HDR target after the scene passes (the engine tonemap runs with Tonemap::None).

use pill_engine::game::*;

// Generated from res/shaders/*.hlsl by pill_assets
const VERTEX_SHADER: &str = include_str!("../res/shaders/fullscreen_vertex.wgsl");
const TONEMAP_SHADER: &str = include_str!("../res/shaders/tony_mc_mapface_fragment.wgsl");
const COPY_SHADER: &str = include_str!("../res/shaders/tony_mc_mapface_copy_fragment.wgsl");
/// Tony McMapface (Tomasz Stachowiak, Apache-2.0, github.com/h3r2tic/tony-mc-mapface): Bevy's
/// tony_mc_mapface.ktx2 level 0, zstd-decompressed; 48³ RGB9E5 texels, x fastest.
const LUT_SIZE: u32 = 48;
const LUT_TEXELS: &[u8] = include_bytes!("../res/tonemap/tony_mc_mapface.rgb9e5");
/// Holds the tonemapped image between the two draws.
const SCRATCH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

struct TonemapState {
    tonemap: Pipeline,
    copy: Pipeline,
    lut_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    scratch: PassTextureHandle,
    // (target generation, tonemap group, copy group): views change when targets are resized
    bind_groups: Option<(u64, wgpu::BindGroup, wgpu::BindGroup)>,
}

pub struct TonyMcMapfacePass {
    state: Option<TonemapState>,
}

impl TonyMcMapfacePass {
    pub fn new() -> Self {
        Self { state: None }
    }
}

fn texture_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
    filterable: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension,
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
    format: wgpu::TextureFormat,
) -> Result<Pipeline> {
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

fn create_lut_view(renderer: &dyn PillRenderer) -> wgpu::TextureView {
    let extent = wgpu::Extent3d {
        width: LUT_SIZE,
        height: LUT_SIZE,
        depth_or_array_layers: LUT_SIZE,
    };
    let texture = renderer
        .get_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("tony_mc_mapface_lut"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgb9e5Ufloat,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    renderer.get_queue().write_texture(
        texture.as_image_copy(),
        LUT_TEXELS,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LUT_SIZE * 4),
            rows_per_image: Some(LUT_SIZE),
        },
        extent,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn draw_fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &Pipeline,
    bind_group: &wgpu::BindGroup,
    output: &wgpu::TextureView,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    });
    render_pass.set_pipeline(&pipeline.pipeline);
    render_pass.set_bind_group(0, bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

impl Pass for TonyMcMapfacePass {
    fn get_label(&self) -> &str {
        "tony_mc_mapface_pass"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let tonemap = fullscreen_pipeline(
            renderer,
            "tony_mc_mapface",
            TONEMAP_SHADER,
            vec![
                // Read with textureLoad
                texture_entry(0, wgpu::TextureViewDimension::D2, false),
                texture_entry(1, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            SCRATCH_FORMAT,
        )?;
        let hdr_format = renderer.get_hdr_format();
        let copy = fullscreen_pipeline(
            renderer,
            "tony_mc_mapface_copy",
            COPY_SHADER,
            vec![texture_entry(0, wgpu::TextureViewDimension::D2, false)],
            hdr_format,
        )?;
        let scratch = renderer.create_render_target(RendererTargetDesc {
            name: "tony_mc_mapface_scratch".to_string(),
            format: SCRATCH_FORMAT,
            size: TargetSize::Surface,
        })?;
        // Bevy's LUT sampler: linear, clamp to edge
        let sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("tony_mc_mapface_lut"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
        self.state = Some(TonemapState {
            tonemap,
            copy,
            lut_view: create_lut_view(renderer),
            sampler,
            scratch,
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
        let state = self
            .state
            .as_mut()
            .expect("TonyMcMapfacePass: init() must run before draw()");
        let hdr_target = world.get_global::<RenderStateComponent>()?.hdr_target();
        let hdr_view = renderer
            .get_render_target_view(hdr_target)
            .ok_or("TonyMcMapfacePass: HDR target missing")?;
        let scratch_view = renderer
            .get_render_target_view(state.scratch)
            .ok_or("TonyMcMapfacePass: scratch target missing")?;

        let generation = renderer.target_generation();
        if !matches!(&state.bind_groups, Some((built, _, _)) if *built == generation) {
            let device = renderer.get_device();
            let tonemap_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tony_mc_mapface"),
                layout: &state.tonemap.bind_group_layouts[0],
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&state.lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&state.sampler),
                    },
                ],
            });
            let copy_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tony_mc_mapface_copy"),
                layout: &state.copy.bind_group_layouts[0],
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(scratch_view),
                }],
            });
            state.bind_groups = Some((generation, tonemap_group, copy_group));
        }
        let (_, tonemap_group, copy_group) =
            state.bind_groups.as_ref().expect("bind groups built above");

        draw_fullscreen(
            encoder,
            "tony_mc_mapface",
            &state.tonemap,
            tonemap_group,
            scratch_view,
        );
        draw_fullscreen(
            encoder,
            "tony_mc_mapface_copy",
            &state.copy,
            copy_group,
            hdr_view,
        );
        Ok(())
    }
}
