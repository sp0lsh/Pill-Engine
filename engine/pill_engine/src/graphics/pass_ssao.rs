use crate::config::{
    SSAO_APPLY_FRAGMENT_SHADER, SSAO_BLUR_FRAGMENT_SHADER, SSAO_DEPTH_DOWNSAMPLE_FRAGMENT_SHADER,
    SSAO_FRAGMENT_SHADER,
};
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{
    create_fullscreen_pipeline, draw_fullscreen, target_view, texture_entry, uniform_buffer,
    uniform_entry, Binding, CachedBindGroup,
};
use crate::graphics::projection::depth_unpack;
use crate::graphics::{
    FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, RendererTargetDesc, TargetSize,
    WorldQuery,
};

use pill_core::Result;
use std::cell::Cell;
use std::rc::Rc;

/// AO visibility at half resolution; read with Load only. The bilateral passes take view depth
/// from the depth chain's first level, which has the same size.
const SSAO_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// View-space depth chain; half floats quantise too coarsely at range for the AO radius.
const DEPTH_CHAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Depth chain levels at ½, ¼, ⅛, 1/16 of the surface (`MAX_MIP` + 1 in ssao_fragment.hlsl).
const DEPTH_CHAIN_DIVISORS: [u32; 4] = [2, 4, 8, 16];
/// Blur spatial sigma in half-res pixels: the 9-tap kernel's edge at 2σ covers the AO pass's 4×4
/// rotation cell.
const BLUR_SIGMA_PX: f32 = 2.0;
/// Blur depth sigma as a fraction of view depth (an absolute threshold suits only one distance).
const BLUR_SIGMA_DEPTH_RELATIVE: f32 = 0.02;
const UPSAMPLE_SIGMA_DEPTH_RELATIVE: f32 = 0.01;

/// Mirrors `SsaoParams` in ssao_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    tan_half_x: f32,
    tan_half_y: f32,
    radius: f32,
    intensity: f32,
    power: f32,
    _padding: [f32; 3],
}

/// Mirrors `SsaoDownsampleParams` in ssao_depth_downsample_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoDownsampleParams {
    depth_mul: f32,
    depth_add: f32,
    /// 1: the source is the scene depth buffer.
    linearize: f32,
    _padding: f32,
}

/// Mirrors `SsaoBlurParams` in ssao_blur_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoBlurParams {
    blur_dir: [f32; 2],
    sigma_px: f32,
    sigma_depth_relative: f32,
}

/// Mirrors `SsaoApplyParams` in ssao_apply_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoApplyParams {
    strength: f32,
    sigma_depth_relative: f32,
    depth_mul: f32,
    depth_add: f32,
}

struct SsaoState {
    downsample: Pipeline,
    /// Downsample parameters: from the scene depth (linearizing), between chain levels.
    downsample_first_params: wgpu::Buffer,
    downsample_params: wgpu::Buffer,
    depth_chain: [PassTextureHandle; 4],
    occlusion: Pipeline,
    blur: Pipeline,
    apply: Pipeline,
    occlusion_params: wgpu::Buffer,
    blur_horizontal_params: wgpu::Buffer,
    blur_vertical_params: wgpu::Buffer,
    apply_params: wgpu::Buffer,
    raw: PassTextureHandle,
    blurred_horizontal: PassTextureHandle,
    blurred: PassTextureHandle,
    bind_groups: [CachedBindGroup; 8], // one per step, in draw order
    // Apply step reading `color_when_inactive`
    inactive_apply_bind_group: CachedBindGroup,
}

/// Horizon-based ambient occlusion (GTAO) at half resolution from a view-space depth chain built
/// from the depth buffer, smoothed by a separable depth-aware blur and upsampled bilaterally. Only the indirect
/// (IBL) light is darkened: `output = color − ambient · (1 − ao)`, with `ambient` from
/// `PassPBROpaque::with_ambient_target`. Parameters: `RenderStateComponent::ssao`.
pub struct PassSsao {
    color: PassTextureHandle,
    ambient: PassTextureHandle,
    depth: PassTextureHandle,
    output: PassTextureHandle,
    // Color to read while the pass writing `color` is off (PassSsr), and its flag
    color_when_inactive: Option<(PassTextureHandle, Rc<Cell<bool>>)>,
    state: Option<SsaoState>,
}

impl PassSsao {
    pub fn new(
        color: PassTextureHandle,
        ambient: PassTextureHandle,
        depth: PassTextureHandle,
        output: PassTextureHandle,
    ) -> Self {
        Self {
            color,
            ambient,
            depth,
            output,
            color_when_inactive: None,
            state: None,
        }
    }

    /// Reads `color` instead while `active` is false (the pass that writes the main color input
    /// skipped this frame).
    pub(crate) fn with_color_when_inactive(
        mut self,
        color: PassTextureHandle,
        active: Rc<Cell<bool>>,
    ) -> Self {
        self.color_when_inactive = Some((color, active));
        self
    }
}

impl Pass for PassSsao {
    fn get_label(&self) -> &str {
        "pass_ssao"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let hdr_format = renderer.get_hdr_format();
        let mut half_target = |name: &str| {
            renderer.create_render_target(RendererTargetDesc {
                name: name.to_string(),
                format: SSAO_TARGET_FORMAT,
                size: TargetSize::SurfaceDivisor(2),
            })
        };
        let raw = half_target("ssao_raw")?;
        let blurred_horizontal = half_target("ssao_blurred_horizontal")?;
        let blurred = half_target("ssao_blurred")?;
        let mut depth_chain = [PassTextureHandle::default(); 4];
        for (level, divisor) in DEPTH_CHAIN_DIVISORS.into_iter().enumerate() {
            depth_chain[level] = renderer.create_render_target(RendererTargetDesc {
                name: format!("ssao_depth_{level}"),
                format: DEPTH_CHAIN_FORMAT,
                size: TargetSize::SurfaceDivisor(divisor),
            })?;
        }

        let downsample = create_fullscreen_pipeline(
            renderer,
            "pass_ssao_depth_downsample",
            SSAO_DEPTH_DOWNSAMPLE_FRAGMENT_SHADER,
            vec![texture_entry(0, false), uniform_entry(1)],
            DEPTH_CHAIN_FORMAT,
            None,
        )?;
        let occlusion = create_fullscreen_pipeline(
            renderer,
            "pass_ssao_occlusion",
            SSAO_FRAGMENT_SHADER,
            vec![
                texture_entry(0, false),
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, false),
                uniform_entry(4),
            ],
            SSAO_TARGET_FORMAT,
            None,
        )?;
        let blur = create_fullscreen_pipeline(
            renderer,
            "pass_ssao_blur",
            SSAO_BLUR_FRAGMENT_SHADER,
            vec![
                texture_entry(0, false),
                texture_entry(1, false),
                uniform_entry(2),
            ],
            SSAO_TARGET_FORMAT,
            None,
        )?;
        let apply = create_fullscreen_pipeline(
            renderer,
            "pass_ssao_apply",
            SSAO_APPLY_FRAGMENT_SHADER,
            vec![
                texture_entry(0, false),
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, false),
                texture_entry(4, false),
                uniform_entry(5),
            ],
            hdr_format,
            None,
        )?;

        let blur_horizontal_params = uniform_buffer::<SsaoBlurParams>(renderer, "pass_ssao_blur_h");
        let blur_vertical_params = uniform_buffer::<SsaoBlurParams>(renderer, "pass_ssao_blur_v");
        for (buffer, blur_dir) in [
            (&blur_horizontal_params, [1.0, 0.0]),
            (&blur_vertical_params, [0.0, 1.0]),
        ] {
            renderer.get_queue().write_buffer(
                buffer,
                0,
                bytemuck::bytes_of(&SsaoBlurParams {
                    blur_dir,
                    sigma_px: BLUR_SIGMA_PX,
                    sigma_depth_relative: BLUR_SIGMA_DEPTH_RELATIVE,
                }),
            );
        }

        self.state = Some(SsaoState {
            downsample,
            downsample_first_params: uniform_buffer::<SsaoDownsampleParams>(
                renderer,
                "pass_ssao_downsample_first",
            ),
            downsample_params: uniform_buffer::<SsaoDownsampleParams>(
                renderer,
                "pass_ssao_downsample",
            ),
            depth_chain,
            occlusion,
            blur,
            apply,
            occlusion_params: uniform_buffer::<SsaoParams>(renderer, "pass_ssao_params"),
            blur_horizontal_params,
            blur_vertical_params,
            apply_params: uniform_buffer::<SsaoApplyParams>(renderer, "pass_ssao_apply"),
            raw,
            blurred_horizontal,
            blurred,
            bind_groups: Default::default(),
            inactive_apply_bind_group: CachedBindGroup::default(),
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
            .expect("PassSsao: state not initialized — call init() before draw()");
        let settings = world.get_global::<RenderStateComponent>()?.ssao;
        let (camera, _) = world.active_camera()?;
        let [depth_mul, depth_add] = depth_unpack(camera.range.start, camera.range.end);
        let tan_half_y = (camera.fov.to_radians() * 0.5).tan();

        let queue = renderer.get_queue();
        queue.write_buffer(
            &state.occlusion_params,
            0,
            bytemuck::bytes_of(&SsaoParams {
                tan_half_x: tan_half_y * camera.aspect.get_value(),
                tan_half_y,
                radius: settings.radius.max(1e-3),
                intensity: settings.intensity,
                power: settings.power,
                _padding: [0.0; 3],
            }),
        );
        for (buffer, linearize) in [
            (&state.downsample_first_params, 1.0),
            (&state.downsample_params, 0.0),
        ] {
            queue.write_buffer(
                buffer,
                0,
                bytemuck::bytes_of(&SsaoDownsampleParams {
                    depth_mul,
                    depth_add,
                    linearize,
                    _padding: 0.0,
                }),
            );
        }
        queue.write_buffer(
            &state.apply_params,
            0,
            bytemuck::bytes_of(&SsaoApplyParams {
                strength: settings.strength.clamp(0.0, 1.0),
                sigma_depth_relative: UPSAMPLE_SIGMA_DEPTH_RELATIVE,
                depth_mul,
                depth_add,
            }),
        );

        let (color, inactive) = match &self.color_when_inactive {
            Some((color, active)) if !active.get() => (*color, true),
            _ => (self.color, false),
        };
        let [depth0, depth1, depth2, depth3] = state.depth_chain;
        let steps: [(&str, &Pipeline, Vec<Binding>, PassTextureHandle); 8] = [
            (
                "pass_ssao_depth_0",
                &state.downsample,
                vec![
                    Binding::Target(self.depth),
                    Binding::Buffer(&state.downsample_first_params),
                ],
                depth0,
            ),
            (
                "pass_ssao_depth_1",
                &state.downsample,
                vec![
                    Binding::Target(depth0),
                    Binding::Buffer(&state.downsample_params),
                ],
                depth1,
            ),
            (
                "pass_ssao_depth_2",
                &state.downsample,
                vec![
                    Binding::Target(depth1),
                    Binding::Buffer(&state.downsample_params),
                ],
                depth2,
            ),
            (
                "pass_ssao_depth_3",
                &state.downsample,
                vec![
                    Binding::Target(depth2),
                    Binding::Buffer(&state.downsample_params),
                ],
                depth3,
            ),
            (
                "pass_ssao_occlusion",
                &state.occlusion,
                vec![
                    Binding::Target(depth0),
                    Binding::Target(depth1),
                    Binding::Target(depth2),
                    Binding::Target(depth3),
                    Binding::Buffer(&state.occlusion_params),
                ],
                state.raw,
            ),
            (
                "pass_ssao_blur_horizontal",
                &state.blur,
                vec![
                    Binding::Target(state.raw),
                    Binding::Target(depth0),
                    Binding::Buffer(&state.blur_horizontal_params),
                ],
                state.blurred_horizontal,
            ),
            (
                "pass_ssao_blur_vertical",
                &state.blur,
                vec![
                    Binding::Target(state.blurred_horizontal),
                    Binding::Target(depth0),
                    Binding::Buffer(&state.blur_vertical_params),
                ],
                state.blurred,
            ),
            (
                "pass_ssao_apply",
                &state.apply,
                vec![
                    Binding::Target(color),
                    Binding::Target(self.ambient),
                    Binding::Target(state.blurred),
                    Binding::Target(depth0),
                    Binding::Target(self.depth),
                    Binding::Buffer(&state.apply_params),
                ],
                self.output,
            ),
        ];
        // Off: only the apply step, which then copies the color through
        let skipped = if settings.strength <= 0.0 { 7 } else { 0 };
        let [depth0_group, depth1_group, depth2_group, depth3_group, occlusion_group, horizontal_group, vertical_group, apply_group] =
            &mut state.bind_groups;
        let apply_group = if inactive {
            &mut state.inactive_apply_bind_group
        } else {
            apply_group
        };
        let bind_groups = [
            depth0_group,
            depth1_group,
            depth2_group,
            depth3_group,
            occlusion_group,
            horizontal_group,
            vertical_group,
            apply_group,
        ];
        for ((label, pipeline, bindings, output), bind_group) in
            steps.into_iter().zip(bind_groups).skip(skipped)
        {
            draw_fullscreen(
                encoder,
                renderer,
                label,
                pipeline,
                bind_group,
                &bindings,
                target_view(renderer, output, label)?,
            )?;
        }
        Ok(())
    }
}
