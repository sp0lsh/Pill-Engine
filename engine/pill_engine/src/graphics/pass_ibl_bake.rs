use crate::config::{IBL_BAKE_FRAGMENT_SHADER, IBL_DOWNSAMPLE_FRAGMENT_SHADER};
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{create_fullscreen_pipeline, uniform_buffer};
use crate::graphics::ibl::{
    IblFrame, IBL_FORMAT, IBL_IRRADIANCE_SIZE, IBL_SPECULAR_MIPS, IBL_SPECULAR_MIP_ROUGHNESS,
    IBL_SPECULAR_SIZE,
};
use crate::graphics::shader_bindings::{
    ibl_bake_fragment, ibl_downsample_fragment, structs::IblBakeParams,
};
use crate::graphics::{FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, WorldQuery};

use pill_core::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// Mip 0 of the source chain the bake samples (the environment is resampled to it).
const SOURCE_WIDTH: u32 = 512;
const SOURCE_HEIGHT: u32 = 256;
/// 512×256 down to 1×1 (widths; heights clamp at 1).
const SOURCE_MIPS: u32 = 10;
/// GGX samples per specular texel; filtered importance sampling keeps them noise-free.
const SPECULAR_SAMPLES: u32 = 64;
/// Cosine samples per irradiance texel.
const IRRADIANCE_SAMPLES: u32 = 256;

struct IblBakeState {
    downsample: Pipeline,
    bake: Pipeline,
    // Environment resampled to SOURCE_WIDTH × SOURCE_HEIGHT with a box-filtered mip chain
    source_mip_views: Vec<wgpu::TextureView>,
    source_view: wgpu::TextureView,
    // Equirect addressing: u wraps around the horizon, v clamps at the poles
    equirect_sampler: wgpu::Sampler,
    // One per bake draw: the specular mips, then the irradiance map; `values` as last written
    params: Vec<wgpu::Buffer>,
    values: Vec<IblBakeParams>,
}

/// Bakes the octahedral IBL maps of `IblFrame` on the GPU from
/// `RenderStateComponent::environment` (an equirect radiance image) whenever it changes, before
/// the passes that read them: a source mip chain, then one full-screen draw per specular mip
/// (GGX prefilter for its roughness) and one for the diffuse irradiance.
pub(crate) struct PassIblBake {
    frame: Rc<RefCell<IblFrame>>,
    // Baked when the game's environment is unset or not a valid texture
    default_environment: PassTextureHandle,
    // Environment and rotation (turns) of the current maps
    baked: Option<(PassTextureHandle, f32)>,
    state: Option<IblBakeState>,
}

impl PassIblBake {
    pub(crate) fn new(
        frame: Rc<RefCell<IblFrame>>,
        default_environment: PassTextureHandle,
    ) -> Self {
        Self {
            frame,
            default_environment,
            baked: None,
            state: None,
        }
    }
}

/// Full-screen draw of `pipeline` with `bind_group` into `target`.
fn draw_into(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    pipeline: &Pipeline,
    bind_group: &wgpu::BindGroup,
    target: &wgpu::TextureView,
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
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

/// View of mip `level` alone (a render target, or the sampled input of the next level).
fn mip_view(texture: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: level,
        mip_level_count: Some(1),
        ..Default::default()
    })
}

impl Pass for PassIblBake {
    fn get_label(&self) -> &str {
        "pass_ibl_bake"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let downsample = create_fullscreen_pipeline(
            renderer,
            "pass_ibl_bake_downsample",
            IBL_DOWNSAMPLE_FRAGMENT_SHADER,
            ibl_downsample_fragment::layout(),
            IBL_FORMAT,
            None,
        )?;
        let bake = create_fullscreen_pipeline(
            renderer,
            "pass_ibl_bake",
            IBL_BAKE_FRAGMENT_SHADER,
            ibl_bake_fragment::layout(),
            IBL_FORMAT,
            None,
        )?;
        let source = renderer
            .get_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("ibl_bake_source"),
                size: wgpu::Extent3d {
                    width: SOURCE_WIDTH,
                    height: SOURCE_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: SOURCE_MIPS,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: IBL_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let equirect_sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

        let source_params =
            |size: u32, roughness: f32, mode: f32, sample_count: u32| IblBakeParams {
                size: size as f32,
                roughness,
                mode,
                sample_count: sample_count as f32,
                source_size: [SOURCE_WIDTH as f32, SOURCE_HEIGHT as f32],
                source_max_lod: (SOURCE_MIPS - 1) as f32,
                rotation: 0.0,
            };
        let mut params = Vec::with_capacity(IBL_SPECULAR_MIPS as usize + 1);
        let values: Vec<IblBakeParams> = (0..IBL_SPECULAR_MIPS)
            .map(|level| {
                source_params(
                    IBL_SPECULAR_SIZE >> level,
                    IBL_SPECULAR_MIP_ROUGHNESS[level as usize],
                    0.0,
                    SPECULAR_SAMPLES,
                )
            })
            .chain([source_params(
                IBL_IRRADIANCE_SIZE,
                1.0,
                1.0,
                IRRADIANCE_SAMPLES,
            )])
            .collect();
        for value in &values {
            let buffer = uniform_buffer::<IblBakeParams>(renderer, "pass_ibl_bake_params");
            renderer
                .get_queue()
                .write_buffer(&buffer, 0, bytemuck::bytes_of(value));
            params.push(buffer);
        }

        self.state = Some(IblBakeState {
            downsample,
            bake,
            source_mip_views: (0..SOURCE_MIPS)
                .map(|level| mip_view(&source, level))
                .collect(),
            source_view: source.create_view(&Default::default()),
            equirect_sampler,
            params,
            values,
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
        let render_state = world.get_global::<RenderStateComponent>()?;
        let environment = render_state
            .environment
            .filter(|handle| renderer.has_texture(*handle))
            .unwrap_or(self.default_environment);
        let rotation = render_state.environment_rotation_turns();
        let previous = self.baked;
        if previous == Some((environment, rotation)) {
            return Ok(());
        }
        let state = self
            .state
            .as_mut()
            .expect("PassIblBake: state not initialized — call init() before draw()");
        if previous.is_some_and(|(_, baked_rotation)| baked_rotation != rotation) {
            for (buffer, value) in state.params.iter().zip(state.values.iter_mut()) {
                value.rotation = rotation;
                renderer
                    .get_queue()
                    .write_buffer(buffer, 0, bytemuck::bytes_of(value));
            }
        }
        let state = &*state;
        let device = renderer.get_device();
        let environment_view = renderer
            .get_texture_view(environment)
            .ok_or("PassIblBake: environment texture missing")?;

        // Source chain: the environment resampled into mip 0, each further mip from the one above;
        // a rotation alone keeps it (the bake turns its lookups)
        let source_current = previous.is_some_and(|(baked, _)| baked == environment);
        for level in (0..SOURCE_MIPS as usize).filter(|_| !source_current) {
            let input = if level == 0 {
                &environment_view
            } else {
                &state.source_mip_views[level - 1]
            };
            let bind_group = ibl_downsample_fragment::Group0 {
                source: input,
                source_sampler: &state.equirect_sampler,
            }
            .create(device, &state.downsample.bind_group_layouts[0]);
            draw_into(
                encoder,
                "pass_ibl_bake_downsample",
                &state.downsample,
                &bind_group,
                &state.source_mip_views[level],
            );
        }

        let mut frame = self.frame.borrow_mut();
        let targets = (0..IBL_SPECULAR_MIPS)
            .map(|level| mip_view(&frame.specular_texture, level))
            .chain([mip_view(&frame.irradiance_texture, 0)]);
        for (target, params) in targets.zip(&state.params) {
            let bind_group = ibl_bake_fragment::Group0 {
                source: &state.source_view,
                source_sampler: &state.equirect_sampler,
                params,
            }
            .create(device, &state.bake.bind_group_layouts[0]);
            draw_into(encoder, "pass_ibl_bake", &state.bake, &bind_group, &target);
        }
        frame.generation += 1;
        self.baked = Some((environment, rotation));
        Ok(())
    }
}
