//! The browser's scene viewport: orbit camera, picking, selection outline with
//! highlighted entities, and the transform gizmo. A UI canvas pixel is a scene pixel.

// Standard library
use std::collections::HashSet;

// External crates
use pill_core::telemetry::telemetry_target;
use pill_core::info;
use pill_editor_core::boundary::{EguiShellFrame, ShellRect};
use pill_editor_core::camera::{
    navigation_binding, CameraButton, CameraNavigation, Modifiers, OrbitCamera,
    PIXELS_PER_WHEEL_LINE, ZOOM_RATIO_PER_LINE,
};
use pill_editor_core::commands::{KeymapPreset, PendingEditorCommands};
use pill_editor_core::editor_state::{EditorCommand, EditorSelection};
use pill_editor_core::gizmo::{
    self, GizmoGeometry, GizmoMode, GizmoState, HandleKind, HandleStates, TransformField,
};
use pill_editor_core::inspector;
use pill_editor_core::preferences::GIZMO_HIT_RADIUS_DEFAULT;
use pill_editor_core::edit_stream::{to_edit_op, EditOutbox};
use pill_engine_core::common_components::TRANSFORM_SHARED_NAME;
use pill_engine_core::input::{ButtonState, KeyCode, MouseButton, ScrollDelta};
use pill_engine_core::{DebugDraw, Engine, Entity, FieldValue, InputEvent, RunMode, World};
use winit::event::{Touch, TouchPhase};
use pill_player::{
    set_surface_camera, CameraBinding, CameraComponent, PickHit, PickRequests, PickResults,
    SelectionOutlines, SurfaceId, TransformComponent,
};

/// Handle length as a fraction of the camera's distance to the gizmo.
const GIZMO_SCREEN_FRACTION: f32 = 0.18;

/// Fly speed while the right button is held, in camera distances per second.
const FLY_SPEED: f32 = 0.8;

/// The drag's handle id when the press grabbed the object itself.
const BODY_HANDLE: u32 = u32::MAX;

/// How far an empty-space press may travel, in pixels, and still deselect.
const CLICK_SLOP: f32 = 4.0;

struct CameraDrag {
    button: CameraButton,
    navigation: CameraNavigation,
    last: (f32, f32),
    deselect: bool,
    travel: f32,
}

struct GizmoDrag {
    handle: u32,
    kind: HandleKind,
    entity: Entity,
    original: TransformComponent,
    basis: [[f32; 3]; 3],
    start_value: f32,
    start_plane: Option<(f32, f32)>,
    reference: f32,
    written: Option<(TransformField, Vec<f32>)>,
    sent: Vec<f32>,
}

/// The scene viewport's state across frames.
pub struct SceneViewport {
    camera: OrbitCamera,
    camera_dirty: bool,
    camera_drag: Option<CameraDrag>,
    gizmo_drag: Option<GizmoDrag>,
    geometry: GizmoGeometry,
    reference: f32,
    hovered: Option<u32>,
    modifiers: Modifiers,
    cursor: Option<(f32, f32)>,
    keys: HashSet<KeyCode>,
    press_pick: Option<(u64, (f32, f32))>,
    left_held: bool,
    last_focus: Option<[f32; 3]>,
    touches: Vec<(u64, (f32, f32))>,
    touch_queue: Vec<Touch>,
    multi_touch: bool,
    outlined: Vec<Entity>,
    scene_rect: Option<ShellRect>,
    overlay_regions: Vec<ShellRect>,
    wants_pointer: bool,
    bound_mode: Option<RunMode>,
}

impl Default for SceneViewport {
    fn default() -> Self {
        Self {
            camera: OrbitCamera::default(),
            camera_dirty: true,
            camera_drag: None,
            gizmo_drag: None,
            geometry: GizmoGeometry::default(),
            reference: 0.0,
            hovered: None,
            modifiers: Modifiers::default(),
            cursor: None,
            keys: HashSet::new(),
            press_pick: None,
            left_held: false,
            last_focus: None,
            touches: Vec::new(),
            touch_queue: Vec::new(),
            multi_touch: false,
            outlined: Vec::new(),
            scene_rect: None,
            overlay_regions: Vec::new(),
            wants_pointer: false,
            bound_mode: None,
        }
    }
}

impl SceneViewport {
    /// Reads where the shell drew this frame, so the next frame's pointer
    /// knows whether it is over the scene.
    pub fn read_shell_frame(&mut self, engine: &Engine) {
        let Some(frame) = engine.world().get_resource::<EguiShellFrame>() else {
            return;
        };
        self.scene_rect = frame.scene_rects.first().copied().flatten();
        self.overlay_regions.clone_from(&frame.overlay_regions);
        self.wants_pointer = frame.wants_pointer;
    }

    /// Runs this frame's input, then publishes the camera, outline and gizmo.
    /// `size` is the scene canvas in physical pixels.
    pub fn frame(
        &mut self,
        engine: &mut Engine,
        events: &[InputEvent],
        size: (f32, f32),
        scale_factor: f32,
        delta_seconds: f32,
    ) {
        let editing = engine.run_mode() == RunMode::Editing;
        self.drain_picks(engine.world_mut(), size);
        if let Some(point) = focus_point(engine.world()) {
            self.last_focus = Some(point);
        }
        if editing {
            for event in events {
                self.input(engine, event, size, scale_factor);
            }
            for touch in std::mem::take(&mut self.touch_queue) {
                self.touch(engine, touch, size, scale_factor);
            }
            self.fly(delta_seconds);
            self.send_live_drag(engine.world_mut());
        } else {
            self.end_gizmo_drag(engine.world_mut());
            self.camera_drag = None;
            self.press_pick = None;
            self.left_held = false;
            self.keys.clear();
            self.touch_queue.clear();
            self.touches.clear();
            self.multi_touch = false;
        }
        self.publish_camera(engine, editing);
        publish_outline(engine.world_mut(), &mut self.outlined);
        if editing {
            self.publish_gizmo(engine.world_mut());
        }
    }

    fn input(&mut self, engine: &mut Engine, event: &InputEvent, size: (f32, f32), scale: f32) {
        match event {
            InputEvent::ModifiersChanged { modifiers } => {
                self.modifiers = Modifiers {
                    shift: modifiers.shift,
                    control: modifiers.control,
                    alt: modifiers.alt,
                };
            }
            InputEvent::Key { key, state } => {
                match state {
                    ButtonState::Pressed => self.keys.insert(*key),
                    ButtonState::Released => self.keys.remove(key),
                };
                let over_scene = self.cursor.is_some_and(|point| self.over_scene(point, scale));
                if *key == KeyCode::KeyF && *state == ButtonState::Pressed && over_scene {
                    if let Some(point) = self.focus() {
                        self.camera.focus(point);
                        self.camera_dirty = true;
                    }
                }
            }
            InputEvent::CursorMoved { position } => {
                self.pointer_moved(engine, (position.x, position.y), size, scale);
            }
            InputEvent::CursorLeft => self.cursor = None,
            InputEvent::MouseWheel { delta } => {
                let Some(point) = self.cursor else {
                    return;
                };
                if !self.over_scene(point, scale) {
                    return;
                }
                let lines = match delta {
                    ScrollDelta::Lines(lines) => lines.y,
                    ScrollDelta::Pixels(pixels) => pixels.y / PIXELS_PER_WHEEL_LINE,
                };
                match self.focus() {
                    Some(point) => self.camera.zoom_toward(point, lines),
                    None => self.camera.zoom(lines),
                }
                self.camera_dirty = true;
            }
            InputEvent::MouseButton { button, state } => {
                self.button(engine.world_mut(), *button, *state, size, scale);
            }
            _ => {}
        }
    }

    fn pointer_moved(&mut self, engine: &mut Engine, point: (f32, f32), size: (f32, f32), scale: f32) {
        self.cursor = Some(point);
        if let Some(drag) = self.camera_drag.as_mut() {
            let delta = (point.0 - drag.last.0, point.1 - drag.last.1);
            drag.last = point;
            drag.travel += delta.0.abs() + delta.1.abs();
            let navigation = drag.navigation;
            let focus = self.focus();
            self.navigate(navigation, delta, size.1, focus);
        } else if self.gizmo_drag.is_some() {
            self.update_gizmo_drag(engine.world_mut(), point, size);
        } else if self.over_scene(point, scale) {
            self.hovered = self.handle_at(point, size);
        }
    }

    /// The selection's position, else the last selection's.
    fn focus(&self) -> Option<[f32; 3]> {
        self.last_focus
    }

    /// Queues a touch for the next frame.
    pub fn push_touch(&mut self, touch: Touch) {
        self.touch_queue.push(touch);
    }

    /// One finger acts as the left button; two pan by their midpoint and zoom
    /// by their spread, and the gesture stays two-finger until every finger lifts.
    fn touch(&mut self, engine: &mut Engine, touch: Touch, size: (f32, f32), scale: f32) {
        let point = (touch.location.x as f32, touch.location.y as f32);
        let before = self.pinch();
        match touch.phase {
            TouchPhase::Started => self.touches.push((touch.id, point)),
            TouchPhase::Moved => {
                if let Some(entry) = self.touches.iter_mut().find(|(id, _)| *id == touch.id) {
                    entry.1 = point;
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => self.touches.retain(|(id, _)| *id != touch.id),
        }
        if self.touches.len() >= 2 && !self.multi_touch {
            self.multi_touch = true;
            self.end_gizmo_drag(engine.world_mut());
            self.camera_drag = None;
            self.press_pick = None;
            self.left_held = false;
            self.cursor = None;
            return;
        }
        if self.multi_touch {
            let pinch = (touch.phase == TouchPhase::Moved).then(|| before.zip(self.pinch())).flatten();
            if let Some(((last_mid, last_spread), (mid, spread))) = pinch {
                let delta = (mid.0 - last_mid.0, mid.1 - last_mid.1);
                self.navigate(CameraNavigation::Pan, delta, size.1, None);
                if last_spread > 1.0 && spread > 1.0 {
                    let lines = (spread / last_spread).ln() / ZOOM_RATIO_PER_LINE.ln();
                    match self.focus() {
                        Some(focus) => self.camera.zoom_toward(focus, lines),
                        None => self.camera.zoom(lines),
                    }
                }
            }
            if self.touches.is_empty() {
                self.multi_touch = false;
            }
            return;
        }
        match touch.phase {
            TouchPhase::Started => {
                self.cursor = Some(point);
                self.button(engine.world_mut(), MouseButton::Left, ButtonState::Pressed, size, scale);
            }
            TouchPhase::Moved => self.pointer_moved(engine, point, size, scale),
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.button(engine.world_mut(), MouseButton::Left, ButtonState::Released, size, scale);
                self.cursor = None;
            }
        }
    }

    /// The first two fingers' midpoint and spread.
    fn pinch(&self) -> Option<((f32, f32), f32)> {
        let [(_, first), (_, second), ..] = self.touches.as_slice() else {
            return None;
        };
        let mid = ((first.0 + second.0) * 0.5, (first.1 + second.1) * 0.5);
        let spread = ((first.0 - second.0).powi(2) + (first.1 - second.1).powi(2)).sqrt();
        Some((mid, spread))
    }

    fn button(
        &mut self,
        world: &mut World,
        button: MouseButton,
        state: ButtonState,
        size: (f32, f32),
        scale: f32,
    ) {
        let camera_button = match button {
            MouseButton::Left => CameraButton::Left,
            MouseButton::Middle => CameraButton::Middle,
            MouseButton::Right => CameraButton::Right,
            _ => return,
        };
        if camera_button == CameraButton::Left {
            self.left_held = state == ButtonState::Pressed;
        }
        if state == ButtonState::Released {
            if self
                .camera_drag
                .as_ref()
                .is_some_and(|drag| drag.button == camera_button)
            {
                let drag = self.camera_drag.take();
                if drag.is_some_and(|drag| drag.deselect && drag.travel < CLICK_SLOP) {
                    clear_selection(world);
                }
            }
            if camera_button == CameraButton::Left {
                self.end_gizmo_drag(world);
            }
            return;
        }
        let Some(point) = self.cursor else {
            return;
        };
        if !self.over_scene(point, scale) || self.gizmo_drag.is_some() {
            return;
        }
        if let Some(navigation) = self.navigation(camera_button) {
            self.camera_drag = Some(CameraDrag {
                button: camera_button,
                navigation,
                last: point,
                deselect: false,
                travel: 0.0,
            });
            return;
        }
        if camera_button != CameraButton::Left {
            return;
        }
        if let Some(handle) = self.handle_at(point, size) {
            self.begin_gizmo_drag(world, handle, point, size);
            return;
        }
        if world.get_resource::<PickRequests>().is_none() {
            world.insert_resource(PickRequests::default());
        }
        if let Some(requests) = world.get_resource_mut::<PickRequests>() {
            let x = point.0.max(0.0).round() as u32;
            let y = point.1.max(0.0).round() as u32;
            self.press_pick = Some((requests.request(SurfaceId::PRIMARY, x, y), point));
        }
    }

    /// The editor's default chords, plus right-drag orbit (Shift pans, Ctrl
    /// dollies) and Alt+left orbit for a trackpad with no middle button.
    fn navigation(&self, button: CameraButton) -> Option<CameraNavigation> {
        if let Some(navigation) = navigation_binding(KeymapPreset::default(), button, self.modifiers) {
            return Some(navigation);
        }
        match (button, self.modifiers.shift, self.modifiers.control, self.modifiers.alt) {
            (CameraButton::Right, false, false, _) => Some(CameraNavigation::Orbit),
            (CameraButton::Right, true, false, _) => Some(CameraNavigation::Pan),
            (CameraButton::Right, false, true, _) => Some(CameraNavigation::Dolly),
            (CameraButton::Left, false, false, true) => Some(CameraNavigation::Orbit),
            _ => None,
        }
    }

    /// Orbit and dolly turn around `focus`, the selection, when there is one.
    fn navigate(
        &mut self,
        navigation: CameraNavigation,
        delta: (f32, f32),
        height: f32,
        focus: Option<[f32; 3]>,
    ) {
        match (navigation, focus) {
            (CameraNavigation::Orbit, Some(point)) => self.camera.orbit_around(point, delta.0, delta.1),
            (CameraNavigation::Orbit, None) => self.camera.orbit(delta.0, delta.1),
            (CameraNavigation::Pan, _) => {
                let (tangent, _) = gizmo::projection_terms(1.0, 1.0);
                let world_per_pixel = 2.0 * self.camera.distance * tangent / height.max(1.0);
                self.camera.pan(delta.0, delta.1, world_per_pixel);
            }
            (CameraNavigation::Dolly, Some(point)) => self.camera.dolly_toward(point, delta.1),
            (CameraNavigation::Dolly, None) => self.camera.dolly(delta.1),
        }
        self.camera_dirty = true;
    }

    /// WASD/QE fly while the right button holds the view, as in a game editor.
    fn fly(&mut self, delta_seconds: f32) {
        let right_held = self
            .camera_drag
            .as_ref()
            .is_some_and(|drag| drag.button == CameraButton::Right);
        if !right_held || delta_seconds <= 0.0 {
            return;
        }
        let axis = |positive: KeyCode, negative: KeyCode| {
            f32::from(u8::from(self.keys.contains(&positive)))
                - f32::from(u8::from(self.keys.contains(&negative)))
        };
        let forward = axis(KeyCode::KeyW, KeyCode::KeyS);
        let right = axis(KeyCode::KeyD, KeyCode::KeyA);
        let up = axis(KeyCode::KeyE, KeyCode::KeyQ);
        if forward == 0.0 && right == 0.0 && up == 0.0 {
            return;
        }
        let boost = if self.modifiers.shift { 3.0 } else { 1.0 };
        let step = FLY_SPEED * boost * self.camera.distance.max(1.0) * delta_seconds;
        let motion = gizmo::add(
            gizmo::add(
                gizmo::scale(self.camera.forward(), forward * step),
                gizmo::scale(self.camera.right(), right * step),
            ),
            gizmo::scale([0.0, 1.0, 0.0], up * step),
        );
        self.camera.pivot = gizmo::add(self.camera.pivot, motion);
        self.camera_dirty = true;
    }

    /// Whether a physical pixel is on the scene view and off the shell's chips;
    /// without a scene tab, anywhere egui does not want the pointer.
    fn over_scene(&self, point: (f32, f32), scale: f32) -> bool {
        let (x, y) = (point.0 / scale, point.1 / scale);
        if self.overlay_regions.iter().any(|region| region.contains(x, y)) {
            return false;
        }
        match self.scene_rect {
            Some(rect) => rect.contains(x, y),
            None => !self.wants_pointer,
        }
    }

    fn publish_camera(&mut self, engine: &mut Engine, editing: bool) {
        let mode = engine.run_mode();
        if !self.camera_dirty && self.bound_mode == Some(mode) {
            return;
        }
        let binding = editing.then(|| CameraBinding::View {
            camera: CameraComponent::default(),
            transform: TransformComponent {
                translation: self.camera.eye(),
                rotation: self.camera.rotation(),
                ..Default::default()
            },
        });
        set_surface_camera(engine.world_mut(), SurfaceId::PRIMARY, binding);
        self.camera_dirty = false;
        self.bound_mode = Some(mode);
    }

    /// Applies the press's pick; a hit with the button still down grabs the
    /// object, so the drag moves it in the view plane.
    fn drain_picks(&mut self, world: &mut World, size: (f32, f32)) {
        let Some(results) = world.get_resource_mut::<PickResults>().map(|results| results.take())
        else {
            return;
        };
        for result in results {
            let Some((id, press_point)) = self.press_pick else {
                continue;
            };
            if result.request.id != id {
                continue;
            }
            self.press_pick = None;
            let Some(selection) = world.get_resource_mut::<EditorSelection>() else {
                return;
            };
            match result.hit {
                PickHit::Entity(entity) => {
                    selection.select_only(entity);
                    selection.clear_elements();
                    info!(
                        target: telemetry_target::EDITOR,
                        entity = %entity,
                        "[editor] scene pick entity"
                    );
                    if self.left_held {
                        self.begin_body_drag(world, entity, press_point, size);
                    }
                    return;
                }
                PickHit::Nothing if self.left_held => {
                    self.camera_drag = Some(CameraDrag {
                        button: CameraButton::Left,
                        navigation: CameraNavigation::Orbit,
                        last: self.cursor.unwrap_or(press_point),
                        deselect: true,
                        travel: 0.0,
                    });
                }
                PickHit::Nothing => clear_selection(world),
                PickHit::Handle(_) => {}
            }
        }
    }

    fn ray(&self, point: (f32, f32), size: (f32, f32)) -> Option<([f32; 3], [f32; 3])> {
        gizmo::camera_ray(&self.camera, point, size)
    }

    fn handle_at(&self, point: (f32, f32), size: (f32, f32)) -> Option<u32> {
        if self.geometry.lines.is_empty() && self.geometry.triangles.is_empty() {
            return None;
        }
        let (origin, direction) = self.ray(point, size)?;
        let radius = (self.reference * GIZMO_HIT_RADIUS_DEFAULT).max(1e-4);
        gizmo::hit_test(&self.geometry, origin, direction, radius)
    }

    fn gizmo_length(&self, origin: [f32; 3]) -> f32 {
        (gizmo::distance(self.camera.eye(), origin) * GIZMO_SCREEN_FRACTION).clamp(0.05, 50.0)
    }

    fn publish_gizmo(&mut self, world: &mut World) {
        let state = world.get_resource::<GizmoState>().copied().unwrap_or_default();
        let selected = world
            .get_resource::<EditorSelection>()
            .and_then(|selection| selection.primary());
        let Some(transform) = selected.and_then(|entity| gizmo::read_transform(world, entity)) else {
            self.geometry = GizmoGeometry::default();
            self.reference = 0.0;
            self.hovered = None;
            return;
        };
        let basis = gizmo::space_basis(state.settings.space, &transform);
        let length = self.gizmo_length(transform.translation);
        let lengths = match self.gizmo_drag.as_ref() {
            Some(drag) => match drag.kind {
                HandleKind::Axis {
                    mode: GizmoMode::Scale,
                    axis,
                } => gizmo::scale_handle_lengths(drag.reference, drag.original.scale, transform.scale, axis),
                _ => [length; 3],
            },
            None => [length; 3],
        };
        let states = HandleStates {
            hovered: self.hovered,
            active: self.gizmo_drag.as_ref().map(|drag| drag.handle),
        };
        let geometry =
            gizmo::build_geometry_styled(state.settings.mode, transform.translation, basis, lengths, states);
        if world.get_resource::<DebugDraw>().is_none() {
            world.insert_resource(DebugDraw::new(true));
        }
        if let Some(debug_draw) = world.get_resource_mut::<DebugDraw>() {
            for (start, end, color, handle) in &geometry.lines {
                debug_draw.gizmo_line(*start, *end, *color, *handle);
            }
            for (corners, color, handle) in &geometry.triangles {
                debug_draw.gizmo_triangle(*corners, *color, *handle);
            }
        }
        self.geometry = geometry;
        self.reference = length;
    }

    fn begin_gizmo_drag(&mut self, world: &mut World, handle: u32, point: (f32, f32), size: (f32, f32)) {
        let Some(kind) = gizmo::handle_kind(handle) else {
            return;
        };
        let state = world.get_resource::<GizmoState>().copied().unwrap_or_default();
        let Some(entity) = world
            .get_resource::<EditorSelection>()
            .and_then(|selection| selection.primary())
        else {
            return;
        };
        let Some(original) = gizmo::read_transform(world, entity) else {
            return;
        };
        let Some((ray_origin, ray_direction)) = self.ray(point, size) else {
            return;
        };
        let basis = gizmo::space_basis(state.settings.space, &original);
        let origin = original.translation;
        let (start_value, start_plane) = match kind {
            HandleKind::Axis {
                mode: GizmoMode::Rotate,
                axis,
            } => {
                let Some(hit) = gizmo::ray_plane_point(origin, basis[axis], ray_origin, ray_direction)
                else {
                    return;
                };
                (gizmo::plane_angle(origin, basis[axis], hit), None)
            }
            HandleKind::Axis { axis, .. } => {
                let Some(value) =
                    gizmo::closest_axis_point(origin, basis[axis], ray_origin, ray_direction)
                else {
                    return;
                };
                (value, None)
            }
            HandleKind::Plane { first, second } => {
                let normal = basis[3 - first - second];
                let Some(hit) = gizmo::ray_plane_point(origin, normal, ray_origin, ray_direction)
                else {
                    return;
                };
                (0.0, Some(gizmo::plane_coordinates(origin, basis[first], basis[second], hit)))
            }
        };
        self.gizmo_drag = Some(GizmoDrag {
            handle,
            kind,
            entity,
            original,
            basis,
            start_value,
            start_plane,
            reference: self.gizmo_length(origin),
            written: None,
            sent: Vec::new(),
        });
        self.hovered = Some(handle);
    }

    fn begin_body_drag(&mut self, world: &mut World, entity: Entity, point: (f32, f32), size: (f32, f32)) {
        let Some(original) = gizmo::read_transform(world, entity) else {
            return;
        };
        let Some((ray_origin, ray_direction)) = self.ray(point, size) else {
            return;
        };
        let basis = [self.camera.right(), self.camera.up(), self.camera.forward()];
        let origin = original.translation;
        let Some(hit) = gizmo::ray_plane_point(origin, basis[2], ray_origin, ray_direction) else {
            return;
        };
        self.gizmo_drag = Some(GizmoDrag {
            handle: BODY_HANDLE,
            kind: HandleKind::Plane {
                first: 0,
                second: 1,
            },
            entity,
            original,
            basis,
            start_value: 0.0,
            start_plane: Some(gizmo::plane_coordinates(origin, basis[0], basis[1], hit)),
            reference: self.gizmo_length(origin),
            written: None,
            sent: Vec::new(),
        });
    }

    /// One move of the drag, measured from the press; the value goes straight
    /// into the world, and the release records the undo step.
    fn update_gizmo_drag(&mut self, world: &mut World, point: (f32, f32), size: (f32, f32)) {
        let Some(drag) = self.gizmo_drag.as_ref() else {
            return;
        };
        let Some((ray_origin, ray_direction)) = self.ray(point, size) else {
            return;
        };
        let settings = world.get_resource::<GizmoState>().copied().unwrap_or_default().settings;
        let origin = drag.original.translation;
        let basis = drag.basis;
        let (field, values) = match drag.kind {
            HandleKind::Axis {
                mode: GizmoMode::Rotate,
                axis,
            } => {
                let Some(hit) = gizmo::ray_plane_point(origin, basis[axis], ray_origin, ray_direction)
                else {
                    return;
                };
                let delta =
                    gizmo::wrap_angle(gizmo::plane_angle(origin, basis[axis], hit) - drag.start_value);
                let mut degrees = delta.to_degrees();
                if settings.snap {
                    degrees = gizmo::snap_value(degrees, settings.rotate_step_degrees);
                }
                let turn = gizmo::axis_angle_quaternion(basis[axis], degrees.to_radians());
                (
                    TransformField::Rotation,
                    gizmo::quaternion_product(turn, drag.original.rotation).to_vec(),
                )
            }
            HandleKind::Axis {
                mode: GizmoMode::Translate,
                axis,
            } => {
                let Some(value) =
                    gizmo::closest_axis_point(origin, basis[axis], ray_origin, ray_direction)
                else {
                    return;
                };
                let mut delta = value - drag.start_value;
                if settings.snap {
                    delta = gizmo::snap_value(delta, settings.translate_step);
                }
                (
                    TransformField::Translation,
                    gizmo::add(origin, gizmo::scale(basis[axis], delta)).to_vec(),
                )
            }
            HandleKind::Axis {
                mode: GizmoMode::Scale,
                axis,
            } => {
                let Some(value) =
                    gizmo::closest_axis_point(origin, basis[axis], ray_origin, ray_direction)
                else {
                    return;
                };
                let factor = 1.0 + (value - drag.start_value) / drag.reference.max(1e-4);
                let mut axis_scale = drag.original.scale[axis] * factor;
                if settings.snap {
                    axis_scale = gizmo::snap_value(axis_scale, settings.scale_step);
                }
                let mut scale = drag.original.scale;
                scale[axis] = axis_scale.max(0.01);
                (TransformField::Scale, scale.to_vec())
            }
            HandleKind::Plane { first, second } => {
                let Some((start_first, start_second)) = drag.start_plane else {
                    return;
                };
                let normal = basis[3 - first - second];
                let Some(hit) = gizmo::ray_plane_point(origin, normal, ray_origin, ray_direction)
                else {
                    return;
                };
                let (along_first, along_second) =
                    gizmo::plane_coordinates(origin, basis[first], basis[second], hit);
                let (mut delta_first, mut delta_second) =
                    (along_first - start_first, along_second - start_second);
                if settings.snap {
                    delta_first = gizmo::snap_value(delta_first, settings.translate_step);
                    delta_second = gizmo::snap_value(delta_second, settings.translate_step);
                }
                let translation = gizmo::add(
                    gizmo::add(origin, gizmo::scale(basis[first], delta_first)),
                    gizmo::scale(basis[second], delta_second),
                );
                (TransformField::Translation, translation.to_vec())
            }
        };
        let entity = drag.entity;
        if gizmo::write_transform_field(world, entity, field.name(), &values) {
            if let Some(drag) = self.gizmo_drag.as_mut() {
                drag.written = Some((field, values));
            }
        }
    }

    /// Streams the drag's newest value once a frame, only the
    /// components that changed since the last send.
    fn send_live_drag(&mut self, world: &mut World) {
        let Some(drag) = self.gizmo_drag.as_mut() else {
            return;
        };
        let Some((field, values)) = drag.written.as_ref() else {
            return;
        };
        let ops: Vec<_> = values
            .iter()
            .enumerate()
            .filter(|(index, value)| drag.sent.get(*index) != Some(*value))
            .filter_map(|(index, value)| {
                to_edit_op(world, &transform_element_edit(drag.entity, *field, index, *value))
            })
            .collect();
        drag.sent.clone_from(values);
        if let Some(outbox) = world.get_resource_mut::<EditOutbox>() {
            outbox.ops.extend(ops);
        }
    }

    /// Ends the drag as one undo step: the press value is written back first,
    /// because the recorded command takes its inverse from the world.
    fn end_gizmo_drag(&mut self, world: &mut World) {
        let Some(drag) = self.gizmo_drag.take() else {
            return;
        };
        self.hovered = None;
        let Some((field, values)) = drag.written else {
            return;
        };
        let original = field.of(&drag.original);
        let Some(edits) = gizmo::release_edits(&original, &values) else {
            return;
        };
        gizmo::write_transform_field(world, drag.entity, field.name(), &original);
        let commands = edits
            .into_iter()
            .map(|(index, value)| transform_element_edit(drag.entity, field, index, value))
            .collect();
        let mode = match drag.kind {
            HandleKind::Axis { mode, .. } => mode,
            HandleKind::Plane { .. } => GizmoMode::Translate,
        };
        if let Some(pending) = world.get_resource_mut::<PendingEditorCommands>() {
            pending.push_recorded(commands, format!("gizmo {}", gizmo::mode_name(mode)), None);
        }
    }
}

/// Outlines the local selection and the highlighted entities.
fn publish_outline(world: &mut World, outlined: &mut Vec<Entity>) {
    let mut entities: Vec<Entity> = world
        .get_resource::<EditorSelection>()
        .map(|selection| selection.entities().to_vec())
        .unwrap_or_default();
    crate::edit_stream::with_highlighted(|ids| {
        for id in ids {
            if let Some(entity) = world.entity_by_id(*id) {
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
        }
    });
    if entities == *outlined {
        return;
    }
    outlined.clone_from(&entities);
    if world.get_resource::<SelectionOutlines>().is_none() {
        world.insert_resource(SelectionOutlines::default());
    }
    if let Some(outlines) = world.get_resource_mut::<SelectionOutlines>() {
        outlines.entities = entities;
    }
}

/// The primary selection's position, which the camera orbits and dollies around.
fn focus_point(world: &World) -> Option<[f32; 3]> {
    let entity = world.get_resource::<EditorSelection>()?.primary()?;
    gizmo::read_transform(world, entity).map(|transform| transform.translation)
}

fn clear_selection(world: &mut World) {
    if let Some(selection) = world.get_resource_mut::<EditorSelection>() {
        selection.clear_entities();
        selection.clear_elements();
    }
}

fn transform_element_edit(entity: Entity, field: TransformField, index: usize, value: f32) -> EditorCommand {
    inspector::element_edit(
        entity,
        TRANSFORM_SHARED_NAME,
        field.name(),
        index,
        FieldValue::F32(value),
    )
    .0
}
