//! Skeletal animation of one entity: which model and clip it plays and at what time. The engine
//! (`skeletal_animation_system`) poses the skeleton and deforms the entity's mesh every frame;
//! games only set these fields.

use crate::ecs::{Component, ComponentStorage, SkinInstanceId};

use pill_core::PillTypeMapKey;

/// A skinned glTF model added with `Engine::add_skeletal_model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkeletalModelHandle(pub(crate) usize);

pub struct SkeletalAnimationComponent {
    /// Index into the model's `animations`
    pub animation: usize,
    /// Seconds into the clip
    pub time: f32,
    /// Wrap `time` into the clip's duration; otherwise it is clamped to its ends
    pub looping: bool,
    /// Fixed at creation: the entity's mesh is made for this model and primitive
    pub(crate) model: SkeletalModelHandle,
    pub(crate) primitive: usize,
    /// Created by the system on the entity's first frame, together with the entity's mesh
    pub(crate) instance: Option<SkinInstanceId>,
}

impl SkeletalAnimationComponent {
    /// Plays `animation` of `model` from time 0, looping, on the model's first skinned primitive.
    /// The entity also needs a `MeshRenderingComponent` with a material; the engine sets its mesh.
    pub fn new(model: SkeletalModelHandle, animation: usize) -> Self {
        Self {
            model,
            animation,
            time: 0.0,
            looping: true,
            primitive: 0,
            instance: None,
        }
    }

    /// Shows skinned primitive `primitive` of the model instead of the first
    pub fn with_primitive(mut self, primitive: usize) -> Self {
        self.primitive = primitive;
        self
    }

    pub fn model(&self) -> SkeletalModelHandle {
        self.model
    }

    pub fn primitive(&self) -> usize {
        self.primitive
    }
}

impl PillTypeMapKey for SkeletalAnimationComponent {
    type Storage = ComponentStorage<SkeletalAnimationComponent>;
}

impl Component for SkeletalAnimationComponent {}
