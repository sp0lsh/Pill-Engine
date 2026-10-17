//! Render settings panel. egui callbacks get no `&mut Engine`, so the panel edits `SETTINGS` and
//! `apply_settings_system` copies them into `RenderStateComponent` every frame; the keyboard
//! shortcuts edit the same settings.

use pill_engine::game::*;
use std::sync::Mutex;

use crate::daylight;

#[derive(Clone, Copy)]
pub struct Settings {
    /// Hour (0–48, two days, so the slider crosses midnight smoothly): sun and sky from
    /// `daylight::at`, which wraps it to the day.
    pub hour: f32,
    /// Degrees the sun path turns about +Y (clockwise seen from above): `daylight` takes the
    /// scene's −Z as north, rotating the sun is easier than rotating the scene.
    pub sun_rotation: f32,
    pub shadows: bool,
    pub gi: bool,
    pub gi_indirect_only: bool,
    /// GI debug: probes as spheres, surfaces coloured by their probe cell.
    pub gi_probes: bool,
    pub gi_probe_cells: bool,
    /// Lighting-only view (every surface white, diffuse only), with or without normal maps.
    pub lighting_only: bool,
    pub lighting_normal_maps: bool,
    /// Unlit view: the materials' base color only (wins over the lighting-only view).
    pub unlit: bool,
    /// Probe spacing (m) of the fine GI cascade around the camera (the coarse one is 6 m).
    pub gi_fine_spacing: f32,
    pub ssr: bool,
    pub ssr_reflections_only: bool,
    pub ssao: bool,
    pub bloom: bool,
    pub depth_of_field: bool,
    pub motion_blur: bool,
    /// Rain: wetness of every surface, 0–1 (`RenderStateComponent::wetness`).
    pub wet: f32,
    /// Scales the sun's intensity from the time of day (0–5).
    pub sun_intensity: f32,
}

pub static SETTINGS: Mutex<Settings> = Mutex::new(Settings {
    // Golden hour
    hour: 17.5,
    sun_rotation: 0.0,
    shadows: true,
    gi: true,
    gi_indirect_only: false,
    gi_probes: false,
    gi_probe_cells: false,
    lighting_only: false,
    lighting_normal_maps: false,
    unlit: false,
    gi_fine_spacing: 2.0,
    ssr: true,
    ssr_reflections_only: false,
    ssao: true,
    bloom: true,
    depth_of_field: false,
    motion_blur: false,
    wet: 0.0,
    sun_intensity: 1.0,
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
                        ui.add(
                            egui::Slider::new(&mut settings.hour, 0.0..=48.0)
                                .custom_formatter(|hour, _| {
                                    let minutes = (hour.rem_euclid(24.0) * 60.0).round() as u32;
                                    format!("{:02}:{:02}", minutes / 60 % 24, minutes % 60)
                                })
                                .text("Hour"),
                        );
                        ui.add(
                            egui::Slider::new(&mut settings.sun_rotation, 0.0..=360.0)
                                .suffix("°")
                                .text("Sun rotation"),
                        );
                        ui.add(
                            egui::Slider::new(&mut settings.sun_intensity, 0.0..=5.0)
                                .text("Sun intensity"),
                        );
                        ui.add(egui::Slider::new(&mut settings.wet, 0.0..=1.0).text("Wet"));
                        ui.label(format!(
                            "Sun elevation {:.1}°",
                            daylight::at(settings.hour).elevation
                        ));
                        ui.separator();
                        ui.checkbox(&mut settings.shadows, "Sun shadows (K)");
                        ui.checkbox(&mut settings.gi, "Diffuse GI (G)");
                        ui.checkbox(&mut settings.gi_indirect_only, "Indirect light only (H)");
                        ui.checkbox(&mut settings.gi_probes, "GI probes");
                        ui.checkbox(&mut settings.gi_probe_cells, "GI probe cells");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut settings.lighting_only, "Lighting only (L)");
                            ui.add_enabled(
                                settings.lighting_only,
                                egui::Checkbox::new(
                                    &mut settings.lighting_normal_maps,
                                    "with normal maps",
                                ),
                            );
                        });
                        ui.checkbox(&mut settings.unlit, "Unlit (U)");
                        ui.add(
                            egui::Slider::new(&mut settings.gi_fine_spacing, 0.5..=3.0)
                                .suffix(" m")
                                .text("GI fine probe spacing"),
                        );
                        ui.checkbox(&mut settings.ssr, "Screen-space reflections (R)");
                        ui.checkbox(&mut settings.ssr_reflections_only, "Reflections only (T)");
                        ui.checkbox(&mut settings.ssao, "SSAO (O)");
                        ui.checkbox(&mut settings.bloom, "Bloom (B)");
                        ui.checkbox(&mut settings.depth_of_field, "Depth of field (F)");
                        ui.checkbox(&mut settings.motion_blur, "Motion blur (M)");
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
const SHORTCUTS: [Shortcut; 11] = [
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
    (KeyboardKey::KeyL, |settings| &mut settings.lighting_only),
    (KeyboardKey::KeyU, |settings| &mut settings.unlit),
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
    // Turn the sun path about +Y by `sun_rotation` (adds to the azimuth)
    let (sin, cos) = settings.sun_rotation.to_radians().sin_cos();
    let to_sun = Vector3f::new(
        light.to_sun.x * cos - light.to_sun.z * sin,
        light.to_sun.y,
        light.to_sun.z * cos + light.to_sun.x * sin,
    );
    // Direction from the sun toward the scene
    render_state.lights[0].direction = (-to_sun).to_array();
    render_state.lights[0].color = light.color;
    render_state.lights[0].intensity = light.intensity * settings.sun_intensity;
    render_state.wetness = settings.wet;
    render_state.debug_view = match (settings.unlit, settings.lighting_only, settings.lighting_normal_maps) {
        (true, _, _) => DebugView::Unlit,
        (false, false, _) => DebugView::Lit,
        (false, true, false) => DebugView::LightingOnly,
        (false, true, true) => DebugView::DetailLighting,
    };
    render_state.ibl_intensity = light.sky;
    // The visible sky dims with its light
    render_state.background_intensity = light.sky;
    render_state.shadow.enabled = settings.shadows;
    render_state.gi.enabled = settings.gi;
    render_state.gi.debug_indirect_only = settings.gi_indirect_only;
    render_state.gi.debug_probes = settings.gi_probes;
    render_state.gi.debug_probe_cells = settings.gi_probe_cells;
    render_state.gi.cascade_spacings[0] = settings.gi_fine_spacing;
    render_state.ssr.enabled = settings.ssr;
    render_state.ssr.debug_reflections_only = settings.ssr_reflections_only;
    render_state.ssao.strength = if settings.ssao { 1.0 } else { 0.0 };
    render_state.bloom.strength = if settings.bloom { 0.04 } else { 0.0 };
    render_state.depth_of_field.focus_scale = if settings.depth_of_field { 2.0 } else { 0.0 };
    render_state.motion_blur = if settings.motion_blur { 0.5 } else { 0.0 };
    Ok(())
}
