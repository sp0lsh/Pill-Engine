use crate::config::SHADOW_MAP_FORMAT;
use crate::graphics::pass_lights::{LightsFrame, PunctualShadowMap};
use crate::graphics::punctual_shadows::MAX_PUNCTUAL_SHADOW_MAPS;
use crate::graphics::shadow_casters::{
    frustum_planes, gather_casters, keep_nearest, outside_frustum, sort_by_mesh, Caster,
    CasterCandidate, ShadowCasters,
};
use crate::graphics::{FrameTarget, Pass, PillRenderer, WorldQuery};

use pill_core::{Matrix4f, Result};
use std::cell::RefCell;
use std::ops::Range;
use std::rc::Rc;

/// Shadow texture array: `layers` maps of `size`², a render view per layer and the array view the
/// PBR passes sample.
struct ShadowArray {
    size: u32,
    layers: u32,
    layer_views: Vec<wgpu::TextureView>,
}

/// Depth-only pass rendering the point/spot shadow maps `PassLights` assigned this frame
/// (`LightsFrame::shadow_maps`) into the layers of a depth texture array, which it owns and
/// publishes in the same frame. Settings: `RenderStateComponent::punctual_lights`.
pub(crate) struct PassPunctualShadows {
    frame: Rc<RefCell<LightsFrame>>,
    max_instances: usize,
    // Hardware depth bias, fixed in the pipeline
    constant_bias: i32,
    slope_bias: f32,
    array: Option<ShadowArray>,
    // Per-frame scratch, capacity reused across frames
    maps: Vec<PunctualShadowMap>,
    view_projections: Vec<Matrix4f>,
    candidates: Vec<CasterCandidate>,
    selected: Vec<usize>,
    casters: Vec<Caster>,
    ranges: Vec<Range<usize>>,
    shadow_casters: Option<ShadowCasters>,
}

impl PassPunctualShadows {
    /// `max_instances`: caster draws across all maps of a frame; past it, maps get the casters
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
            array: None,
            maps: Vec::new(),
            view_projections: Vec::new(),
            candidates: Vec::new(),
            selected: Vec::new(),
            casters: Vec::new(),
            ranges: Vec::new(),
            shadow_casters: None,
        }
    }

    /// (Re)creates the texture array when the map size changes or more layers are needed (grows
    /// only) and publishes the sampling view. `size` is clamped to the device by `PassLights`.
    fn ensure_array(&mut self, renderer: &dyn PillRenderer, size: u32, layers: u32) {
        let device = renderer.get_device();
        if self
            .array
            .as_ref()
            .is_some_and(|array| array.size == size && array.layers >= layers)
        {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("punctual_shadow_maps"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_MAP_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layer_views = (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("punctual_shadow_map_layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("punctual_shadow_maps"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let mut frame = self.frame.borrow_mut();
        frame.shadow_view = Some(array_view);
        frame.shadow_generation += 1;
        self.array = Some(ShadowArray {
            size,
            layers,
            layer_views,
        });
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
            MAX_PUNCTUAL_SHADOW_MAPS,
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
        let map_size = {
            let frame = self.frame.borrow();
            self.maps.clear();
            self.maps.extend_from_slice(&frame.shadow_maps);
            frame.shadow_map_size
        };
        if self.maps.is_empty() {
            return Ok(());
        }
        // Layers in multiples of 4, grown only, so a changing map count doesn't churn the texture
        self.ensure_array(
            renderer,
            map_size,
            self.maps.len().next_multiple_of(4) as u32,
        );

        // Per map, best effort: the casters inside its frustum (far plane = the light's range),
        // nearest to the light first while caster draws are left; maps are in priority order
        gather_casters(world, &mut self.candidates);
        self.casters.clear();
        self.ranges.clear();
        self.view_projections.clear();
        for map in &self.maps {
            let planes = frustum_planes(&map.view_projection);
            self.selected.clear();
            self.selected.extend(
                (0..self.candidates.len()).filter(|&index| {
                    !outside_frustum(&planes, &self.candidates[index].world_corners)
                }),
            );
            let room = self.max_instances - self.casters.len();
            let candidates = &self.candidates;
            keep_nearest(&mut self.selected, room, |&index| {
                candidates[index]
                    .world_center
                    .distance_squared(map.light_position)
            });
            let start = self.casters.len();
            self.casters.extend(
                self.selected
                    .iter()
                    .map(|&index| self.candidates[index].caster),
            );
            sort_by_mesh(&mut self.casters[start..]);
            self.ranges.push(start..self.casters.len());
            self.view_projections.push(map.view_projection);
        }

        let shadow_casters = self
            .shadow_casters
            .as_mut()
            .expect("PassPunctualShadows: not initialized — call init() before draw()");
        shadow_casters.upload(renderer.get_queue(), &self.view_projections, &self.casters);
        let array = self
            .array
            .as_ref()
            .expect("PassPunctualShadows: array created above");
        for (map, range) in self.ranges.iter().enumerate() {
            let Some(layer_view) = array.layer_views.get(map) else {
                break;
            };
            shadow_casters.draw(
                encoder,
                layer_view,
                map,
                &self.casters,
                range.clone(),
                world,
            );
        }
        Ok(())
    }
}
