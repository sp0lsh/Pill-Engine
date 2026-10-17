use crate::config::{LDR_TARGET_FORMAT, SHARPEN_FRAGMENT_SHADER};
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{
    create_fullscreen_pipeline, draw_fullscreen, input_view, linear_clamp_sampler, target_view,
    uniform_buffer, CachedBindGroup,
};
use crate::graphics::shader_bindings::{sharpen_fragment, structs::SharpenParams};
use crate::graphics::{FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, WorldQuery};

use pill_core::Result;

struct SharpenState {
    pipeline: Pipeline,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,
    bind_group: CachedBindGroup,
}

/// Contrast-adaptive sharpening (`RenderStateComponent::sharpness`, 0 = copy) of the tonemapped
/// image, at the render size before the lens pass upscales it.
pub struct PassSharpen {
    input: PassTextureHandle,
    output: PassTextureHandle,
    state: Option<SharpenState>,
}

impl PassSharpen {
    pub fn new(input: PassTextureHandle, output: PassTextureHandle) -> Self {
        Self {
            input,
            output,
            state: None,
        }
    }
}

impl Pass for PassSharpen {
    fn get_label(&self) -> &str {
        "pass_sharpen"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let pipeline = create_fullscreen_pipeline(
            renderer,
            "pass_sharpen",
            SHARPEN_FRAGMENT_SHADER,
            sharpen_fragment::layout(),
            LDR_TARGET_FORMAT,
            None,
        )?;
        self.state = Some(SharpenState {
            pipeline,
            sampler: linear_clamp_sampler(renderer),
            params_buffer: uniform_buffer::<SharpenParams>(renderer, "pass_sharpen_params"),
            bind_group: CachedBindGroup::default(),
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
        let sharpness = world.get_global::<RenderStateComponent>()?.sharpness;
        let state = self
            .state
            .as_mut()
            .expect("PassSharpen: state not initialized — call init() before draw()");
        let (width, height) = renderer.get_render_size();
        let params = SharpenParams {
            rcp_frame: [1.0 / width.max(1) as f32, 1.0 / height.max(1) as f32],
            sharpness,
            ..bytemuck::Zeroable::zeroed()
        };
        renderer
            .get_queue()
            .write_buffer(&state.params_buffer, 0, bytemuck::bytes_of(&params));
        draw_fullscreen(
            encoder,
            renderer,
            "pass_sharpen",
            &state.pipeline,
            &mut state.bind_group,
            &sharpen_fragment::Group0 {
                t_src: input_view(renderer, self.input, "pass_sharpen")?,
                s_src: &state.sampler,
                params: &state.params_buffer,
            },
            target_view(renderer, self.output, "pass_sharpen")?,
        )
    }
}
