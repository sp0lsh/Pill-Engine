use crate::ecs::{Component, ComponentStorage};

use pill_core::PillTypeMapKey;

/// Shape of a punctual light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LightType {
    Point,
    /// Cone along the entity's forward direction; full intensity inside `inner_angle`, fading to
    /// zero at `outer_angle` (half-angles, degrees).
    Spot {
        inner_angle: f32,
        outer_angle: f32,
    },
}

// --- Builder ---

pub struct LightComponentBuilder {
    component: LightComponent,
}

impl LightComponentBuilder {
    pub fn light_type(mut self, light_type: LightType) -> Self {
        self.component.light_type = light_type;
        self
    }

    pub fn color(mut self, color: [f32; 3]) -> Self {
        self.component.color = color;
        self
    }

    pub fn intensity(mut self, intensity: f32) -> Self {
        self.component.intensity = intensity;
        self
    }

    pub fn range(mut self, range: f32) -> Self {
        self.component.range = range;
        self
    }

    /// Renders a shadow map for the light (spot: 1, point: 6), within the per-frame budget
    /// `PunctualLights::max_shadow_maps`.
    pub fn cast_shadows(mut self, cast_shadows: bool) -> Self {
        self.component.cast_shadows = cast_shadows;
        self
    }

    pub fn build(self) -> LightComponent {
        self.component
    }
}

// --- Light Component ---

/// Point or spot light at the entity's `TransformComponent` position, shaded by the PBR passes
/// through clustered (froxel) light lists. Register it per scene to use it.
#[derive(Clone, Copy, Debug)]
pub struct LightComponent {
    pub light_type: LightType,
    /// Linear RGB, multiplied by `intensity`.
    pub color: [f32; 3],
    /// Same scale as `DirectionalLight::intensity`, before the inverse-square falloff.
    pub intensity: f32,
    /// Distance (m) at which the light fades to zero.
    pub range: f32,
    /// Shadowed when within the shadow-map budget; nearer lights first.
    pub cast_shadows: bool,
}

impl Default for LightComponent {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: [1.0; 3],
            intensity: 1.0,
            range: 5.0,
            cast_shadows: false,
        }
    }
}

impl LightComponent {
    pub fn builder() -> LightComponentBuilder {
        LightComponentBuilder {
            component: Self::default(),
        }
    }
}

impl PillTypeMapKey for LightComponent {
    type Storage = ComponentStorage<LightComponent>;
}

impl Component for LightComponent {}
