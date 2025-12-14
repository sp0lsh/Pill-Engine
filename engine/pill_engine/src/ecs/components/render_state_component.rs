use crate::ecs::{GlobalComponent, GlobalComponentStorage};
use crate::PillTypeMapKey;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy)]
pub struct PostProcessParams {
    pub time_s: f32,      // seconds since start
    pub focus_point: f32, // meters
    pub focus_scale: f32,
    pub mb_enabled: bool,
    pub mb_debug_velocity: bool,
    pub mb_strength: f32,
    pub mb_max_samples: u32,
    pub mb_min_speed: f32,
    pub mb_depth_softness: f32,
}

// Keeps renderer/bootstrap state; minimal for now
pub struct RenderStateComponent {
    pub boot_done: bool,
    pub egui_client: Option<std::sync::Arc<crate::ecs::EguiClient>>,
    pub post_process: Arc<Mutex<PostProcessParams>>,
}

impl RenderStateComponent {
    pub fn new() -> Self {
        Self {
            boot_done: false,
            egui_client: None,
            post_process: Arc::new(Mutex::new(PostProcessParams {
                time_s: 0.0,
                focus_point: 0.0,
                focus_scale: 0.0,
                mb_enabled: true,
                mb_debug_velocity: false,
                mb_strength: 1.0,
                mb_max_samples: 16,
                mb_min_speed: 0.001,
                mb_depth_softness: 1.0,
            })),
        }
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
