use crate::ecs::{DebugView, RenderStateComponent};
use crate::graphics::pass_lights::LightsParams;
use crate::graphics::shader_bindings::structs::Scene;
use crate::graphics::shadow::ShadowFrame;

/// Scene globals of the PBR passes (group 0, binding 7): fog, IBL intensity, the three
/// directional lights from `RenderStateComponent`, the key light's shadow and the froxel grid of
/// the point/spot lights.
impl Scene {
    pub(crate) fn from_render_state(
        render_state: &RenderStateComponent,
        shadow: &ShadowFrame,
        punctual: &LightsParams,
        ssr: bool,
    ) -> Self {
        let [r, g, b] = render_state.fog_color;
        // Per light: direction (xyz) row, then color (rgb) + intensity (w) row
        let [light0, light1, light2] = render_state.lights.map(|light| {
            let [x, y, z] = light.direction;
            let [red, green, blue] = light.color;
            ([x, y, z, 0.0], [red, green, blue, light.intensity])
        });
        let size = shadow.map_size.max(1) as f32;
        let (shadow_view_projection, shadow_params) = match shadow.map {
            Some(_) => (
                shadow.view_projection.to_cols_array_2d(),
                [
                    render_state.shadow.normal_bias * shadow.texel_world_size,
                    shadow.distance,
                    1.0 / size,
                    size,
                ],
            ),
            None => ([[0.0; 4]; 4], [0.0; 4]),
        };
        Self {
            fog_color_density: [r, g, b, render_state.fog_density],
            ibl_intensity: [
                render_state.ibl_intensity,
                ssr as u32 as f32,
                render_state.wetness.clamp(0.0, 1.0),
                match render_state.debug_view {
                    DebugView::Lit => 0.0,
                    DebugView::LightingOnly => 1.0,
                    DebugView::DetailLighting => 2.0,
                    DebugView::Unlit => 3.0,
                },
            ],
            light_direction0: light0.0,
            light_color0: light0.1,
            light_direction1: light1.0,
            light_color1: light1.1,
            light_direction2: light2.0,
            light_color2: light2.1,
            shadow_view_projection,
            shadow_params,
            froxel_grid: punctual.grid,
            froxel_depth: punctual.depth,
            camera_forward: punctual.camera_forward,
        }
    }
}
