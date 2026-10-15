use crate::config::SSR_FRAGMENT_SHADER;
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{
    create_fullscreen_pipeline, draw_fullscreen, input_view, linear_clamp_sampler, target_view,
    uniform_buffer, CachedBindGroup,
};
use crate::graphics::ibl::IblFrame;
use crate::graphics::projection::depth_unpack;
use crate::graphics::renderer::camera_view_projection;
use crate::graphics::shader_bindings::{ssr_fragment, structs::SsrParams};
use crate::graphics::{FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, WorldQuery};

use pill_core::Result;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

struct SsrState {
    pipeline: Pipeline,
    params: wgpu::Buffer,
    prefilter_sampler: wgpu::Sampler,
    bind_group: CachedBindGroup,
}

/// Screen-space reflections (`RenderStateComponent::ssr`): marches the mirror ray of each opaque
/// pixel through the depth buffer and adds `W · lerp(ibl_specular, reflected, confidence)` to the
/// color, with the specular weight W, roughness and normal from
/// `PassPBROpaque::with_ssr_targets` (which leaves the specular IBL term out of the color while
/// SSR is on). Off: draws nothing and clears `active`, so `PassSsao` reads the opaque color
/// instead of `output`.
pub struct PassSsr {
    color: PassTextureHandle,
    depth: PassTextureHandle,
    surface: [PassTextureHandle; 2],
    // Octahedral specular map baked by PassIblBake (a fixed view; the bake rewrites its content)
    ibl: Rc<RefCell<IblFrame>>,
    output: PassTextureHandle,
    active: Rc<Cell<bool>>,
    state: Option<SsrState>,
}

impl PassSsr {
    /// `surface` = [specular weight + roughness, octahedral normal]; `ibl` holds the baked
    /// specular map; `active` is set each frame to whether `output` was written.
    pub(crate) fn new(
        color: PassTextureHandle,
        depth: PassTextureHandle,
        surface: [PassTextureHandle; 2],
        ibl: Rc<RefCell<IblFrame>>,
        output: PassTextureHandle,
        active: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            color,
            depth,
            surface,
            ibl,
            output,
            active,
            state: None,
        }
    }
}

impl Pass for PassSsr {
    fn get_label(&self) -> &str {
        "pass_ssr"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let hdr_format = renderer.get_hdr_format();
        let pipeline = create_fullscreen_pipeline(
            renderer,
            "pass_ssr",
            SSR_FRAGMENT_SHADER,
            ssr_fragment::layout(),
            hdr_format,
            None,
        )?;
        // As the PBR pass's: the oct map has seam borders and mips are picked explicitly
        let prefilter_sampler = linear_clamp_sampler(renderer);
        self.state = Some(SsrState {
            pipeline,
            params: uniform_buffer::<SsrParams>(renderer, "pass_ssr_params"),
            prefilter_sampler,
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
        let state = self
            .state
            .as_mut()
            .expect("PassSsr: state not initialized — call init() before draw()");
        let render_state = world.get_global::<RenderStateComponent>()?;
        let settings = render_state.ssr;
        // Off: draw nothing; SSAO reads the opaque color directly
        self.active.set(settings.enabled);
        if !settings.enabled {
            return Ok(());
        }
        let (camera, transform) = world.active_camera()?;
        let [depth_mul, depth_add] = depth_unpack(camera.range.start, camera.range.end);
        let view_projection = camera_view_projection(camera, transform);
        let mode = if settings.debug_reflections_only {
            2.0
        } else {
            1.0
        };
        let position = transform.position;
        renderer.get_queue().write_buffer(
            &state.params,
            0,
            bytemuck::bytes_of(&SsrParams {
                view_projection: view_projection.to_cols_array_2d(),
                inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
                camera: [
                    position.x,
                    position.y,
                    position.z,
                    settings.thickness.max(1e-3),
                ],
                depth: [depth_mul, depth_add, render_state.ibl_intensity, mode],
                march: [
                    settings.linear_steps.max(2) as f32,
                    settings.bisection_steps as f32,
                    settings.roughness_fade.0,
                    settings
                        .roughness_fade
                        .1
                        .max(settings.roughness_fade.0 + 1e-3),
                ],
                screen: [settings.edge_fade, 0.0, 0.0, 0.0],
            }),
        );

        let ibl = self.ibl.borrow();
        draw_fullscreen(
            encoder,
            renderer,
            "pass_ssr",
            &state.pipeline,
            &mut state.bind_group,
            &ssr_fragment::Group0 {
                t_color: input_view(renderer, self.color, "pass_ssr")?,
                t_depth: input_view(renderer, self.depth, "pass_ssr")?,
                t_surface: input_view(renderer, self.surface[0], "pass_ssr")?,
                t_normal: input_view(renderer, self.surface[1], "pass_ssr")?,
                t_prefilter: &ibl.specular_view,
                s_prefilter: &state.prefilter_sampler,
                params: &state.params,
            },
            target_view(renderer, self.output, "pass_ssr")?,
        )
    }
}
