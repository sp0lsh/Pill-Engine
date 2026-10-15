use crate::ecs::RenderStateComponent;
use crate::graphics::pass_lights::LightsParams;
use crate::graphics::shadow::ShadowFrame;

/// Scene globals of the PBR passes (group 0, binding 7): fog, IBL intensity, the three
/// directional lights from `RenderStateComponent`, the key light's shadow and the froxel grid of
/// the point/spot lights. All float4 rows so the std140 layout is
/// unambiguous; mirrors `Scene` in pbr_opaque_fragment.hlsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SceneUniform {
    fog_color_density: [f32; 4], // xyz = fog color, w = fog density
    ibl_intensity: [f32; 4],     // x, y = 1 when PassSsr adds the specular IBL term, z = wetness
    // Per light: direction (xyz) row, then color (rgb) + intensity (w) row
    lights: [[f32; 4]; 6],
    shadow_view_projection: [[f32; 4]; 4], // world → light clip space (column-major)
    // x = normal offset (world units), y = shadow distance (0 = no shadow), z = 1 / map size,
    // w = map size
    shadow_params: [f32; 4],
    froxel_grid: [f32; 4],    // columns, rows, tile size (px), light count
    froxel_depth: [f32; 4],   // 1 / slice linearizer, light far, slice count, 1 = heatmap
    camera_forward: [f32; 4], // xyz
}

impl SceneUniform {
    pub(crate) fn from_render_state(
        render_state: &RenderStateComponent,
        shadow: &ShadowFrame,
        punctual: &LightsParams,
        ssr: bool,
    ) -> Self {
        let [r, g, b] = render_state.fog_color;
        let mut lights = [[0.0; 4]; 6];
        for (index, light) in render_state.lights.iter().enumerate() {
            let [x, y, z] = light.direction;
            let [red, green, blue] = light.color;
            lights[index * 2] = [x, y, z, 0.0];
            lights[index * 2 + 1] = [red, green, blue, light.intensity];
        }
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
                0.0,
            ],
            lights,
            shadow_view_projection,
            shadow_params,
            froxel_grid: punctual.grid,
            froxel_depth: punctual.depth,
            camera_forward: punctual.camera_forward,
        }
    }
}
