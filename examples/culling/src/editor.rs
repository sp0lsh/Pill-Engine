//! Game-side editor (from the teaser's previs layout): the game camera in the top half of the
//! window, the overview camera's wireframe view (`PassWireframePreview`) in the bottom half,
//! composited by `EditorPass`. F toggles the editor (game view full window).
//!
//! Both cameras have the same controls, active while the cursor is over the camera's view: hold
//! RMB or L = mouse-look + WASD fly, E/Q up/down, Shift faster; Alt+LMB = orbit the origin;
//! MMB = pan; scroll = dolly.

use pill_engine::{define_component, define_global_component, game::*};

/// Sub-rectangle of the window, as fractions of its size.
#[derive(Clone, Copy)]
pub struct ViewportRect {
    pub x_frac: f32,
    pub y_frac: f32,
    pub w_frac: f32,
    pub h_frac: f32,
}

impl ViewportRect {
    pub const FULL: ViewportRect = ViewportRect {
        x_frac: 0.0,
        y_frac: 0.0,
        w_frac: 1.0,
        h_frac: 1.0,
    };
    pub const TOP_HALF: ViewportRect = ViewportRect {
        x_frac: 0.0,
        y_frac: 0.0,
        w_frac: 1.0,
        h_frac: 0.5,
    };
    pub const BOTTOM_HALF: ViewportRect = ViewportRect {
        x_frac: 0.0,
        y_frac: 0.5,
        w_frac: 1.0,
        h_frac: 0.5,
    };

    /// Integer pixels `(x, y, w, h)` of the rect on a `width` × `height` surface, at least 1×1.
    pub fn to_pixels(self, width: u32, height: u32) -> (u32, u32, u32, u32) {
        let (width, height) = (width as f32, height as f32);
        (
            (self.x_frac * width) as u32,
            (self.y_frac * height) as u32,
            (self.w_frac * width).max(1.0) as u32,
            (self.h_frac * height).max(1.0) as u32,
        )
    }

    /// Pixel aspect ratio of the rect on a `width` × `height` surface.
    pub fn pixel_aspect(self, width: u32, height: u32) -> f32 {
        (width as f32 * self.w_frac) / (height as f32 * self.h_frac).max(1e-6)
    }
}

/// Where the overview camera's wireframe is drawn; the game frame goes to `GAME_VIEWPORT`.
pub const EDITOR_VIEWPORT: ViewportRect = ViewportRect::BOTTOM_HALF;
pub const GAME_VIEWPORT: ViewportRect = ViewportRect::TOP_HALF;

const LOOK_SENSITIVITY: f32 = 0.12; // degrees per pixel
const PITCH_LIMIT: f32 = 89.0;
const ORBIT_SENSITIVITY: f32 = 0.01; // radians per pixel
const PAN_PER_PIXEL: f32 = 0.0015; // scaled by the distance to the pivot
                                   // Metres per second
const FLY_SPEED: f32 = 5.0;
const FLY_SPEED_SPRINT: f32 = 15.0;
const ZOOM_PER_SCROLL: f32 = 0.6;
const PIVOT: Vector3f = Vector3f::ZERO;
/// Cursor jumps at least this large (window re-entry) are not treated as mouse motion.
const MAX_CURSOR_STEP: f32 = 200.0;

define_global_component!(EditorState { visible: bool });

// Free-fly camera: view angles in degrees (yaw 0 looks down −Z, positive pitch looks up), the last
// cursor position for mouse deltas, and whether it is the editor's overview camera
define_component!(FlyCamera {
    yaw: f32,
    pitch: f32,
    last_cursor: Option<Vector2f>,
    overview: bool,
});

impl FlyCamera {
    /// Looking from `position` at `target`.
    pub fn looking_at(position: Vector3f, target: Vector3f, overview: bool) -> Self {
        let direction = (target - position).normalize();
        Self {
            yaw: (-direction.x).atan2(-direction.z).to_degrees(),
            pitch: direction.y.asin().to_degrees(),
            last_cursor: None,
            overview,
        }
    }

    /// The window rect of this camera's view; `None` while it isn't shown.
    fn viewport(&self, editor_visible: bool) -> Option<ViewportRect> {
        match (self.overview, editor_visible) {
            (true, true) => Some(EDITOR_VIEWPORT),
            (true, false) => None,
            (false, true) => Some(GAME_VIEWPORT),
            (false, false) => Some(ViewportRect::FULL),
        }
    }
}

/// Forward and right of a yaw / pitch view.
fn basis(yaw: f32, pitch: f32) -> (Vector3f, Vector3f) {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    let forward = Vector3f::new(
        -yaw.sin() * pitch.cos(),
        pitch.sin(),
        -yaw.cos() * pitch.cos(),
    );
    (forward, forward.cross(Vector3f::Y).normalize_or_zero())
}

/// Whether `cursor` (physical pixels) is inside `viewport` of a `surface` sized window.
fn cursor_in(cursor: Vector2f, viewport: ViewportRect, (width, height): (u32, u32)) -> bool {
    let (x, y, rect_width, rect_height) = viewport.to_pixels(width, height);
    cursor.x >= x as f32
        && cursor.x < (x + rect_width) as f32
        && cursor.y >= y as f32
        && cursor.y < (y + rect_height) as f32
}

/// Shows or hides the editor: the game frame takes the top half or the whole window.
pub fn set_editor_visible(engine: &mut Engine, visible: bool) -> Result<()> {
    engine.get_global_component_mut::<EditorState>()?.visible = visible;
    engine
        .get_global_component_mut::<EguiComponent>()?
        .show_engine_windows = visible;
    Ok(())
}

pub fn editor_toggle_system(engine: &mut Engine) -> Result<()> {
    if engine
        .get_global_component::<InputComponent>()?
        .get_key_pressed(KeyboardKey::KeyF)
    {
        let visible = engine.get_global_component::<EditorState>()?.visible;
        set_editor_visible(engine, !visible)?;
    }
    Ok(())
}

/// The game camera frames the game rect while the editor is shown; the engine sets `Automatic`
/// to the window aspect.
pub fn game_camera_aspect_system(engine: &mut Engine) -> Result<()> {
    let visible = engine.get_global_component::<EditorState>()?.visible;
    let (surface_width, surface_height) = engine.get_surface_size();
    let aspect = GAME_VIEWPORT.pixel_aspect(surface_width, surface_height);
    for (_, camera) in engine.iterate_one_component_mut::<CameraComponent>()? {
        if camera.enabled {
            camera.aspect = if visible {
                CameraAspectRatio::Manual(aspect)
            } else {
                CameraAspectRatio::Automatic(aspect)
            };
        }
    }
    Ok(())
}

pub fn fly_camera_system(engine: &mut Engine) -> Result<()> {
    let editor_visible = engine.get_global_component::<EditorState>()?.visible;
    let surface = engine.get_surface_size();
    let overview_aspect = EDITOR_VIEWPORT.pixel_aspect(surface.0, surface.1);
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .min(1.0 / 30.0);

    let input = engine.get_global_component::<InputComponent>()?;
    let look = input.get_mouse_button(MouseButton::Right) || input.get_key(KeyboardKey::KeyL);
    let orbit = input.get_mouse_button(MouseButton::Left)
        && (input.get_key(KeyboardKey::AltLeft) || input.get_key(KeyboardKey::AltRight));
    let pan = input.get_mouse_button(MouseButton::Middle);
    let axis = |positive: KeyboardKey, negative: KeyboardKey| {
        input.get_key(positive) as i32 as f32 - input.get_key(negative) as i32 as f32
    };
    // x = right, y = up, z = forward
    let movement = Vector3f::new(
        axis(KeyboardKey::KeyD, KeyboardKey::KeyA),
        axis(KeyboardKey::KeyE, KeyboardKey::KeyQ),
        axis(KeyboardKey::KeyW, KeyboardKey::KeyS),
    );
    let sprint = input.get_key(KeyboardKey::ShiftLeft) || input.get_key(KeyboardKey::ShiftRight);
    let speed = if sprint { FLY_SPEED_SPRINT } else { FLY_SPEED };
    let cursor = input.get_mouse_position();
    let scroll = input.get_mouse_scroll_delta();

    for (_, transform, camera, fly) in
        engine.iterate_three_components_mut::<TransformComponent, CameraComponent, FlyCamera>()?
    {
        if fly.overview {
            // The engine only sets the aspect of the active camera
            camera.aspect = CameraAspectRatio::Manual(overview_aspect);
        }

        // Frame-to-frame cursor delta
        let delta = fly
            .last_cursor
            .map(|last| cursor - last)
            .filter(|delta| delta.abs().max_element() < MAX_CURSOR_STEP)
            .unwrap_or(Vector2f::ZERO);
        fly.last_cursor = Some(cursor);
        if !fly
            .viewport(editor_visible)
            .is_some_and(|viewport| cursor_in(cursor, viewport, surface))
        {
            continue;
        }

        let (forward, right) = basis(fly.yaw, fly.pitch);
        if scroll.y != 0.0 {
            transform.translate_world(forward * scroll.y * ZOOM_PER_SCROLL);
        }
        if look {
            fly.yaw -= delta.x * LOOK_SENSITIVITY;
            fly.pitch = (fly.pitch - delta.y * LOOK_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            let (forward, right) = basis(fly.yaw, fly.pitch);
            let step = (right * movement.x + Vector3f::Y * movement.y + forward * movement.z)
                .normalize_or_zero()
                * speed
                * delta_time;
            transform.translate_world(step);
        } else if orbit {
            // Around the pivot at a constant radius, then aimed back at it
            let offset = transform.position - PIVOT;
            let radius = offset.length().max(0.01);
            let azimuth = offset.x.atan2(offset.z) - delta.x * ORBIT_SENSITIVITY;
            let elevation =
                ((offset.y / radius).asin() + delta.y * ORBIT_SENSITIVITY).clamp(-1.5, 1.5);
            transform.set_position(
                PIVOT
                    + Vector3f::new(
                        radius * elevation.cos() * azimuth.sin(),
                        radius * elevation.sin(),
                        radius * elevation.cos() * azimuth.cos(),
                    ),
            );
            let direction = (PIVOT - transform.position).normalize();
            fly.yaw = (-direction.x).atan2(-direction.z).to_degrees();
            fly.pitch = direction.y.asin().to_degrees();
        } else if pan {
            let up = right.cross(forward);
            let scale = PAN_PER_PIXEL * (transform.position - PIVOT).length().max(1.0);
            transform.translate_world((right * -delta.x + up * delta.y) * scale);
        }
        let (forward, _) = basis(fly.yaw, fly.pitch);
        let target = transform.position + forward;
        transform.look_at(target);
    }
    Ok(())
}
