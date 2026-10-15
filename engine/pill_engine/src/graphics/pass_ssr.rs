use crate::config::SSR_FRAGMENT_SHADER;
use crate::ecs::RenderStateComponent;
use crate::graphics::fullscreen::{
    create_fullscreen_pipeline, draw_fullscreen, sampler_entry, target_view, texture_entry,
    uniform_buffer, uniform_entry, Binding, CachedBindGroup,
};
use crate::graphics::projection::depth_unpack;
use crate::graphics::renderer::camera_view_projection;
use crate::graphics::{FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, WorldQuery};

use pill_core::Result;
use std::cell::Cell;
use std::rc::Rc;

/// Mirrors `SsrParams` in ssr_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SsrParams {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    camera: [f32; 4],
    depth: [f32; 4],
    march: [f32; 4],
    screen: [f32; 4],
}

struct SsrState {
    pipeline: Pipeline,
    params: wgpu::Buffer,
    prefilter_sampler: wgpu::Sampler,
    bind_group: CachedBindGroup,
    // Specular IBL handle the bind group was built with
    prefilter: PassTextureHandle,
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
    fallback_prefilter: PassTextureHandle,
    output: PassTextureHandle,
    active: Rc<Cell<bool>>,
    state: Option<SsrState>,
}

impl PassSsr {
    /// `surface` = [specular weight + roughness, octahedral normal]; `fallback_prefilter` is the
    /// engine's default specular IBL, used when `RenderStateComponent::ibl_specular` is invalid;
    /// `active` is set each frame to whether `output` was written.
    pub fn new(
        color: PassTextureHandle,
        depth: PassTextureHandle,
        surface: [PassTextureHandle; 2],
        fallback_prefilter: PassTextureHandle,
        output: PassTextureHandle,
        active: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            color,
            depth,
            surface,
            fallback_prefilter,
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
            vec![
                texture_entry(0, false),
                texture_entry(1, false),
                texture_entry(2, false),
                texture_entry(3, false),
                texture_entry(4, true),
                sampler_entry(5),
                uniform_entry(6),
            ],
            hdr_format,
            None,
        )?;
        // As the PBR pass's prefilter sampler: equirect u wraps, trilinear across roughness mips
        let prefilter_sampler = renderer
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
        self.state = Some(SsrState {
            pipeline,
            params: uniform_buffer::<SsrParams>(renderer, "pass_ssr_params"),
            prefilter_sampler,
            bind_group: CachedBindGroup::default(),
            prefilter: self.fallback_prefilter,
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

        // The game may swap the specular IBL; rebuild the bind group then
        let prefilter = if renderer.has_texture(render_state.ibl_specular) {
            render_state.ibl_specular
        } else {
            self.fallback_prefilter
        };
        if prefilter != state.prefilter {
            state.prefilter = prefilter;
            state.bind_group = CachedBindGroup::default();
        }
        draw_fullscreen(
            encoder,
            renderer,
            "pass_ssr",
            &state.pipeline,
            &mut state.bind_group,
            &[
                Binding::Target(self.color),
                Binding::Target(self.depth),
                Binding::Target(self.surface[0]),
                Binding::Target(self.surface[1]),
                Binding::Texture(prefilter),
                Binding::Sampler(&state.prefilter_sampler),
                Binding::Buffer(&state.params),
            ],
            target_view(renderer, self.output, "pass_ssr")?,
        )
    }
}
