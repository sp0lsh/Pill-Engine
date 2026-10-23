use crate::config::SHADOW_MAP_FORMAT;
use crate::graphics::pass_lights::{LightsFrame, ShadowJob};
use crate::graphics::punctual_shadows::MAX_TILE_DRAWS;
use crate::graphics::shadow_casters::{
    frustum_planes, keep_nearest, outside_frustum, sort_by_mesh, Caster, ShadowCasters,
};
use crate::graphics::{FrameTarget, Pass, PillRenderer, WorldQuery};

use pill_core::{Matrix4f, Result};
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

/// The shadow atlas texture and its view.
struct ShadowAtlasTexture {
    size: u32,
    view: wgpu::TextureView,
}

/// Depth-only pass drawing the point/spot shadow atlas tiles `PassLights` scheduled this frame
/// (`LightsFrame::shadow_jobs`): each tile is cleared and redrawn in place, the rest of the atlas
/// (cached tiles) is kept. Owns the atlas texture and publishes it in the same frame. Settings:
/// `RenderStateComponent::punctual_lights`.
pub(crate) struct PassPunctualShadows {
    frame: Rc<RefCell<LightsFrame>>,
    max_instances: usize,
    // Hardware depth bias, fixed in the pipeline
    constant_bias: i32,
    slope_bias: f32,
    atlas: Option<ShadowAtlasTexture>,
    // Per-frame scratch, capacity reused across frames
    jobs: Vec<ShadowJob>,
    view_projections: Vec<Matrix4f>,
    // Casters of the current light and kind within its range, then those in the tile's frustum
    in_range: Vec<usize>,
    selected: Vec<usize>,
    casters: Vec<Caster>,
    ranges: Vec<Range<usize>>,
    shadow_casters: Option<ShadowCasters>,
}

impl PassPunctualShadows {
    /// `max_instances`: caster draws across all tiles of a frame; past it, tiles get the casters
    /// nearest their light that still fit.
    pub(crate) fn new(
        frame: Rc<RefCell<LightsFrame>>,
        max_instances: usize,
        constant_bias: i32,
        slope_bias: f32,
    ) -> Self {
        Self {
            frame,
            max_instances,
            constant_bias,
            slope_bias,
            atlas: None,
            jobs: Vec::new(),
            view_projections: Vec::new(),
            in_range: Vec::new(),
            selected: Vec::new(),
            casters: Vec::new(),
            ranges: Vec::new(),
            shadow_casters: None,
        }
    }

    /// (Re)creates the atlas when its size changes and publishes the sampling view; the cache
    /// starts over at a new size (`ShadowCache::update`), so nothing is lost.
    fn ensure_atlas(&mut self, renderer: &dyn PillRenderer, size: u32) {
        if self.atlas.as_ref().is_some_and(|atlas| atlas.size == size) {
            return;
        }
        let texture = renderer
            .get_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("punctual_shadow_atlas"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW_MAP_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut frame = self.frame.borrow_mut();
        frame.shadow_view = Some(view.clone());
        frame.shadow_generation += 1;
        self.atlas = Some(ShadowAtlasTexture { size, view });
    }
}

impl Pass for PassPunctualShadows {
    fn get_label(&self) -> &str {
        "pass_punctual_shadows"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.shadow_casters = Some(ShadowCasters::new(
            renderer,
            "pass_punctual_shadows",
            MAX_TILE_DRAWS,
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
        let frame_cell = self.frame.clone();
        let frame = frame_cell.borrow();
        self.jobs.clear();
        self.jobs.extend_from_slice(&frame.shadow_jobs);
        let atlas_size = frame.atlas_size;
        if self.jobs.is_empty() || atlas_size == 0 {
            return Ok(());
        }
        drop(frame);
        self.ensure_atlas(renderer, atlas_size);
        let mut frame = frame_cell.borrow_mut();

        // Per tile, best effort: the casters of its kind (static or moving) inside its frustum
        // (far plane = the light's range), nearest to the light first while caster draws are
        // left; tiles are in priority order. A light's tiles of one kind are consecutive, so the
        // range test runs once per light and kind.
        self.casters.clear();
        self.ranges.clear();
        self.view_projections.clear();
        let mut current = None;
        for job in &self.jobs {
            if current != Some((job.light, job.dynamic)) {
                current = Some((job.light, job.dynamic));
                self.in_range.clear();
                self.in_range
                    .extend((0..frame.casters.len()).filter(|&index| {
                        frame.caster_dynamic[index] == job.dynamic
                            && frame.casters[index].within(job.light_position, job.light_range)
                    }));
            }
            let planes = frustum_planes(&job.view_projection);
            self.selected.clear();
            self.selected.extend(
                self.in_range.iter().copied().filter(|&index| {
                    !outside_frustum(&planes, &frame.casters[index].world_corners)
                }),
            );
            let room = self.max_instances.saturating_sub(self.casters.len());
            // A static tile drawn without all its casters is redrawn when there's room
            if self.selected.len() > room
                && !job.dynamic
                && !frame.truncated_lights.contains(&job.light)
            {
                frame.truncated_lights.push(job.light);
            }
            let candidates = &frame.casters;
            keep_nearest(&mut self.selected, room, |&index| {
                candidates[index]
                    .world_center
                    .distance_squared(job.light_position)
            });
            let start = self.casters.len();
            self.casters.extend(
                self.selected
                    .iter()
                    .map(|&index| frame.casters[index].caster),
            );
            sort_by_mesh(&mut self.casters[start..]);
            self.ranges.push(start..self.casters.len());
            self.view_projections.push(job.view_projection);
        }
        drop(frame);

        let shadow_casters = self
            .shadow_casters
            .as_mut()
            .expect("PassPunctualShadows: not initialized — call init() before draw()");
        shadow_casters.upload(renderer.get_queue(), &self.view_projections, &self.casters);
        let atlas = self
            .atlas
            .as_ref()
            .expect("PassPunctualShadows: atlas created above");
        let mut render_pass = shadow_casters.begin_pass(encoder, &atlas.view, wgpu::LoadOp::Load);
        for (map, (job, range)) in self.jobs.iter().zip(&self.ranges).enumerate() {
            let tile = job.tile;
            render_pass.set_viewport(
                tile.x as f32,
                tile.y as f32,
                tile.size as f32,
                tile.size as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(tile.x, tile.y, tile.size, tile.size);
            shadow_casters.clear_viewport(&mut render_pass);
            shadow_casters.draw_into(&mut render_pass, map, &self.casters, range.clone(), world);
        }
        Ok(())
    }
}
