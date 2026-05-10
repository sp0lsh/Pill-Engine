use std::sync::Arc;

use crate::ecs::egui_client::EguiClient;

use crate::ecs::components::{GlobalComponent, GlobalComponentStorage};

use pill_core::PillTypeMapKey;

pub struct RenderStateComponent {
    pub boot_done: bool,
    pub egui_client: Arc<EguiClient>,
}

impl RenderStateComponent {
    pub fn new(egui_client: Arc<EguiClient>) -> Self {
        Self {
            boot_done: false,
            egui_client,
        }
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
