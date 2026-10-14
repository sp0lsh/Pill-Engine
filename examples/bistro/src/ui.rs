//! Render settings panel. egui callbacks get no `&mut Engine`, so the panel edits `SETTINGS` and
//! `apply_settings_system` copies them into `RenderStateComponent` every frame; the keyboard
//! shortcuts edit the same settings.

use pill_engine::game::*;
use std::sync::Mutex;

use crate::daylight;

#[derive(Clone, Copy)]
pub struct Settings {
    /// Hour of the day (0–24): sun and sky from `daylight::at`.
    pub hour: f32,
    pub shadows: bool,
    pub gi: bool,
    pub gi_indirect_only: bool,
    pub ssr: bool,
    pub ssr_reflections_only: bool,
    pub ssao: bool,
    pub bloom: bool,
    pub depth_of_field: bool,
    pub motion_blur: bool,
    /// Rain-darkened, glossy pavement (Exterior): shows the screen-space reflections.
    pub wet_street: bool,
}

pub static SETTINGS: Mutex<Settings> = Mutex::new(Settings {
    // Golden hour
    hour: 17.5,
    shadows: true,
    gi: true,
    gi_indirect_only: false,
    ssr: true,
    ssr_reflections_only: false,
    ssao: true,
    bloom: true,
    depth_of_field: false,
    motion_blur: false,
    wet_street: true,
});

/// What the panel shows besides the settings (written by the game systems).
pub static STATUS: Mutex<String> = Mutex::new(String::new());

pub fn settings() -> Settings {
    *SETTINGS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub fn edit(change: impl FnOnce(&mut Settings)) {
    change(
        &mut SETTINGS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    );
}

pub fn set_status(status: String) {
    log::info!("{status}");
    *STATUS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = status;
}

pub fn register(engine: &mut Engine) -> Result<()> {
    engine
        .get_global_component_mut::<EguiComponent>()?
        .register_ui("bistro.settings", |ctx| {
            egui::Window::new("Bistro")
                .default_open(false)
                .show(ctx, |ui| {
                    ui.label(
                        STATUS
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .as_str(),
                    );
                    ui.separator();
                    edit(|settings| {
                        ui.add(egui::Slider::new(&mut settings.hour, 0.0..=24.0).text("Hour"));
                        ui.label(format!(
                            "Sun elevation {:.1}°",
                            daylight::at(settings.hour).elevation
                        ));
                        ui.separator();
                        ui.checkbox(&mut settings.shadows, "Sun shadows (K)");
                        ui.checkbox(&mut settings.gi, "Diffuse GI (G)");
                        ui.checkbox(&mut settings.gi_indirect_only, "Indirect light only (H)");
                        ui.checkbox(&mut settings.ssr, "Screen-space reflections (R)");
                        ui.checkbox(&mut settings.ssr_reflections_only, "Reflections only (T)");
                        ui.checkbox(&mut settings.ssao, "SSAO (O)");
                        ui.checkbox(&mut settings.bloom, "Bloom (B)");
                        ui.checkbox(&mut settings.depth_of_field, "Depth of field (F)");
                        ui.checkbox(&mut settings.motion_blur, "Motion blur (M)");
                        ui.checkbox(&mut settings.wet_street, "Wet street (P)");
                    });
                    ui.separator();
                    ui.label("1–4: viewpoints");
                    ui.label("Hold LMB to look, WASD + E/Q to move, Shift faster");
                });
        });
    Ok(())
}

/// Keyboard shortcuts for the toggles.
type Shortcut = (KeyboardKey, fn(&mut Settings) -> &mut bool);
const SHORTCUTS: [Shortcut; 10] = [
    (KeyboardKey::KeyK, |settings| &mut settings.shadows),
    (KeyboardKey::KeyG, |settings| &mut settings.gi),
    (KeyboardKey::KeyH, |settings| &mut settings.gi_indirect_only),
    (KeyboardKey::KeyR, |settings| &mut settings.ssr),
    (KeyboardKey::KeyT, |settings| {
        &mut settings.ssr_reflections_only
    }),
    (KeyboardKey::KeyO, |settings| &mut settings.ssao),
    (KeyboardKey::KeyB, |settings| &mut settings.bloom),
    (KeyboardKey::KeyF, |settings| &mut settings.depth_of_field),
    (KeyboardKey::KeyM, |settings| &mut settings.motion_blur),
    (KeyboardKey::KeyP, |settings| &mut settings.wet_street),
];

pub fn shortcuts_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component::<InputComponent>()?;
    edit(|settings| {
        for (key, flag) in SHORTCUTS {
            if input.get_key_pressed(key) {
                let flag = flag(settings);
                *flag = !*flag;
            }
        }
    });
    Ok(())
}

/// Copies the settings into the render state.
pub fn apply_settings_system(engine: &mut Engine) -> Result<()> {
    let settings = settings();
    let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
    let light = daylight::at(settings.hour);
    // Direction from the sun toward the scene
    render_state.lights[0].direction = (-light.to_sun).to_array();
    render_state.lights[0].color = light.color;
    render_state.lights[0].intensity = light.intensity;
    render_state.ibl_intensity = light.sky;
    render_state.shadow.enabled = settings.shadows;
    render_state.gi.enabled = settings.gi;
    render_state.gi.debug_indirect_only = settings.gi_indirect_only;
    render_state.ssr.enabled = settings.ssr;
    render_state.ssr.debug_reflections_only = settings.ssr_reflections_only;
    render_state.ssao.strength = if settings.ssao { 1.0 } else { 0.0 };
    render_state.bloom.strength = if settings.bloom { 0.04 } else { 0.0 };
    render_state.depth_of_field.focus_scale = if settings.depth_of_field { 2.0 } else { 0.0 };
    render_state.motion_blur = if settings.motion_blur { 0.5 } else { 0.0 };
    Ok(())
}
