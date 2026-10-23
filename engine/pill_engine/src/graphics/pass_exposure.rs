//! Auto exposure (eye adaptation): one compute dispatch per frame meters the image the tonemap
//! pass is about to read and updates the exposure multiplier it applies (`exposure_compute`).
//! The multiplier lives in a GPU buffer shared with `PassTonemap`; with auto exposure off it is
//! `2^RenderStateComponent::exposure`.

use std::rc::Rc;

use crate::config::EXPOSURE_COMPUTE_SHADER;
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{input_view, uniform_buffer};
use crate::graphics::shader_bindings::{exposure_compute, structs::ExposureParams};
use crate::graphics::{
    ComputePipeline, ComputePipelineDesc, FrameTarget, Pass, PassTextureHandle, PillRenderer,
    ShaderDesc, WorldQuery,
};

use pill_core::Result;
use wgpu::util::DeviceExt;

/// The exposure state both passes bind: [adapted mean log2 luminance, exposure multiplier].
pub(crate) fn exposure_state_buffer(renderer: &dyn PillRenderer) -> Rc<wgpu::Buffer> {
    Rc::new(
        renderer
            .get_device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("exposure_state"),
                contents: bytemuck::cast_slice(&[0.0f32, 1.0]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
    )
}

pub struct PassExposure {
    hdr_source: PassTextureHandle,
    exposure_state: Rc<wgpu::Buffer>,
    state: Option<ExposureState>,
    /// `motion_history_epoch` and `auto_exposure.enabled` of the last frame: a camera cut, or
    /// turning auto exposure on (the adapted value went stale while off), snaps the adaptation
    epoch: Option<u32>,
    enabled: bool,
}

struct ExposureState {
    pipeline: ComputePipeline,
    params_buffer: wgpu::Buffer,
}

impl PassExposure {
    pub fn new(hdr_source: PassTextureHandle, exposure_state: Rc<wgpu::Buffer>) -> Self {
        Self {
            hdr_source,
            exposure_state,
            state: None,
            epoch: None,
            enabled: false,
        }
    }
}

impl Pass for PassExposure {
    fn get_label(&self) -> &str {
        "pass_exposure"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let pipeline = renderer.create_compute_pipeline(ComputePipelineDesc {
            label: Some("exposure"),
            cs: ShaderDesc {
                source: EXPOSURE_COMPUTE_SHADER,
                entry_func: "cs_main",
            },
            bind_groups: exposure_compute::layout(),
        })?;
        self.state = Some(ExposureState {
            pipeline,
            params_buffer: uniform_buffer::<ExposureParams>(renderer, "pass_exposure_params"),
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
            .as_ref()
            .expect("PassExposure: state not initialized — call init() before draw()");
        let render_state = world.get_global::<RenderStateComponent>()?;
        let auto = &render_state.auto_exposure;
        let epoch = render_state.motion_history_epoch;
        let snap = self.epoch != Some(epoch) || (auto.enabled && !self.enabled);
        self.epoch = Some(epoch);
        self.enabled = auto.enabled;
        let params = ExposureParams {
            exposure_ev: render_state.exposure,
            compensation_ev: auto.compensation_ev,
            min_ev: auto.min_ev,
            max_ev: auto.max_ev,
            adapt_to_bright_speed: auto.adapt_to_bright_speed,
            adapt_to_dark_speed: auto.adapt_to_dark_speed,
            key: auto.key,
            // WorldQuery::delta_time is in milliseconds
            delta_seconds: world.delta_time / 1000.0,
            enabled: auto.enabled as u32,
            snap: snap as u32,
            ..bytemuck::Zeroable::zeroed()
        };
        renderer
            .get_queue()
            .write_buffer(&state.params_buffer, 0, bytemuck::bytes_of(&params));

        // Rebuilt per frame: the HDR view changes on resize, and one small group is cheap
        let bind_group = exposure_compute::Group0 {
            hdr_texture: input_view(renderer, self.hdr_source, "pass_exposure")?,
            params: &state.params_buffer,
            exposure_state: &self.exposure_state,
        }
        .create(renderer.get_device(), &state.pipeline.bind_group_layouts[0]);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass_exposure"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&state.pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }
}
