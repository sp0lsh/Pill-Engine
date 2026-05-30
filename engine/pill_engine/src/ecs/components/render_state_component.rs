use crate::ecs::components::{GlobalComponent, GlobalComponentStorage};

use pill_core::PillTypeMapKey;

pub struct RenderStateComponent {
    pub boot_done: bool,
}

impl RenderStateComponent {
    /// Creates the component in its pre-boot state.
    pub fn new() -> Self {
        Self { boot_done: false }
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
