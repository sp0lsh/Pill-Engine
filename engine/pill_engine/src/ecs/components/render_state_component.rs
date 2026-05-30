use crate::ecs::components::{GlobalComponent, GlobalComponentStorage};

use pill_core::PillTypeMapKey;

pub struct RenderStateComponent {
    pub boot_done: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub egui_client: std::sync::Arc<crate::ecs::egui_client::EguiClient>,
}

impl RenderStateComponent {
    /// Creates the component in its pre-boot state with the provided egui client.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(egui_client: std::sync::Arc<crate::ecs::egui_client::EguiClient>) -> Self {
        Self {
            boot_done: false,
            egui_client,
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Self {
        Self { boot_done: false }
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
