use crate::config::DEPTH_TARGET_FORMAT;
use crate::ecs::RenderStateComponent;
use crate::graphics::shadow::{
    casts_into, fit_light, light_space_bounds, light_view_projection, CameraSlice, ShadowFrame,
};
use crate::graphics::shadow_casters::{
    gather_casters, keep_nearest, sort_by_mesh, Caster, CasterCandidate, ShadowCasters,
};
use crate::graphics::{
    camera_forward, FrameTarget, Pass, PassTextureHandle, PillRenderer, RendererTargetDesc,
    TargetSize, WorldQuery,
};

use pill_core::{debug, LogContext, Result, Vector3f};
use std::cell::RefCell;
use std::rc::Rc;

/// Depth-only pass rendering the key light's (`lights[0]`) shadow casters with an orthographic
/// projection fitted to the camera (see `shadow.rs`). Owns the shadow map: created when shadows
/// are first enabled, recreated when `map_size` changes. Publishes map and light matrix in `frame`
/// for the PBR passes. Settings: `RenderStateComponent::shadow`.
pub(crate) struct PassShadow {
    frame: Rc<RefCell<ShadowFrame>>,
    max_instances: usize,
    // Hardware depth bias, fixed in the pipeline
    constant_bias: i32,
    slope_bias: f32,
    // Current shadow map and its size; kept while shadows are off
    shadow_map: Option<(PassTextureHandle, u32)>,
    // Per-frame scratch, capacity reused across frames
    candidates: Vec<CasterCandidate>,
    selected: Vec<usize>,
    casters: Vec<Caster>,
    shadow_casters: Option<ShadowCasters>,
    // Caster total of the last culling log line; logged again when it changes
    logged_total: usize,
    size_clamp_warned: bool,
}

impl PassShadow {
    pub(crate) fn new(
        frame: Rc<RefCell<ShadowFrame>>,
        max_instances: usize,
        constant_bias: i32,
        slope_bias: f32,
    ) -> Self {
        Self {
            frame,
            max_instances: max_instances.max(1),
            constant_bias,
            slope_bias,
            shadow_map: None,
            candidates: Vec::new(),
            selected: Vec::new(),
            casters: Vec::new(),
            shadow_casters: None,
            logged_total: 0,
            size_clamp_warned: false,
        }
    }

    /// Shadow map of `requested` size (clamped to the device limit), recreated when it changes.
    fn shadow_map(
        &mut self,
        renderer: &mut dyn PillRenderer,
        requested: u32,
    ) -> Result<(PassTextureHandle, u32)> {
        let max_size = renderer.get_device().limits().max_texture_dimension_2d;
        let size = requested.clamp(1, max_size);
        if size != requested && !self.size_clamp_warned {
            log::warn!("PassShadow: map_size {requested} clamped to {size} (device limit)");
            self.size_clamp_warned = true;
        }
        if let Some((map, current)) = self.shadow_map {
            if current == size {
                return Ok((map, size));
            }
            renderer.remove_render_target(map);
        }
        let map = renderer.create_render_target(RendererTargetDesc {
            name: "shadow_map".to_string(),
            format: DEPTH_TARGET_FORMAT,
            size: TargetSize::Fixed(size, size),
        })?;
        self.shadow_map = Some((map, size));
        Ok((map, size))
    }
}

impl Pass for PassShadow {
    fn get_label(&self) -> &str {
        "pass_shadow"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.shadow_casters = Some(ShadowCasters::new(
            renderer,
            "pass_shadow",
            1,
            self.max_instances,
            self.constant_bias,
            self.slope_bias,
        )?);
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
        let settings = render_state.shadow;
        if !settings.enabled {
            *self.frame.borrow_mut() = ShadowFrame::default();
            return Ok(());
        }
        let (shadow_map, map_size) = self.shadow_map(renderer, settings.map_size)?;
        let (camera, camera_transform) = world.active_camera()?;
        let distance = settings.distance.min(camera.range.end);
        let fit = fit_light(
            &CameraSlice {
                position: camera_transform.position,
                forward: camera_forward(camera_transform),
                fov_y: camera.fov.to_radians(),
                aspect: camera.aspect.get_value(),
                near: camera.range.start,
                far: distance,
            },
            Vector3f::from_array(render_state.lights[0].direction),
            map_size,
        );

        // Caster queue: casters whose light-space AABB can reach the receiver box
        // Best effort past the per-draw capacity: the casters nearest to the camera
        gather_casters(world, &mut self.candidates);
        let light_bounds = |candidate: &CasterCandidate| {
            light_space_bounds(&fit.view, &candidate.model, candidate.local_bounds)
        };
        self.selected.clear();
        self.selected.extend(
            (0..self.candidates.len())
                .filter(|&index| casts_into(&fit, light_bounds(&self.candidates[index]))),
        );
        let candidates = &self.candidates;
        keep_nearest(&mut self.selected, self.max_instances, |&index| {
            candidates[index]
                .world_center
                .distance_squared(camera_transform.position)
        });
        self.casters.clear();
        let mut nearest_caster_z = f32::NEG_INFINITY;
        for &index in &self.selected {
            let candidate = &self.candidates[index];
            nearest_caster_z = nearest_caster_z.max(light_bounds(candidate).1.z);
            self.casters.push(candidate.caster);
        }
        let total = self.candidates.len();
        if total != self.logged_total {
            debug!(LogContext::Rendering =>
                "PassShadow: {} of {} shadow casters kept after culling",
                self.casters.len(),
                total
            );
            self.logged_total = total;
        }
        sort_by_mesh(&mut self.casters);

        let view_projection = light_view_projection(&fit, nearest_caster_z);
        *self.frame.borrow_mut() = ShadowFrame {
            map: Some(shadow_map),
            map_size,
            view_projection,
            texel_world_size: fit.texel_world_size,
            distance,
        };

        let shadow_casters = self
            .shadow_casters
            .as_mut()
            .expect("PassShadow: not initialized — call init() before draw()");
        shadow_casters.upload(renderer.get_queue(), &[view_projection], &self.casters);
        let depth_view = renderer
            .get_render_target_view(shadow_map)
            .ok_or("PassShadow: shadow map missing")?;
        shadow_casters.draw(
            encoder,
            depth_view,
            0,
            &self.casters,
            0..self.casters.len(),
            world,
        );
        Ok(())
    }
}
