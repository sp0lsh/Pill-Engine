//! The editor without egui: one scene canvas with picking and the gizmo.
//! The host page draws the panels and edits through the edit stream.

// Standard library
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

// External crates
use pill_core::platform::Instant;
use pill_core::telemetry::telemetry_target;
use pill_core::{error, info, warn};
use pill_editor_core::commands::ensure_command_resources;
use pill_editor_core::editor_state::EditorSelection;
use pill_editor_core::gizmo::GizmoState;
use pill_engine_core::input::InputEvent;
use pill_input::winit_events::translate_window_event;
use pill_player::{
    report_failure, FrameDriver, RenderingRuntime, Runtime, StaticProject, SurfaceId,
    WindowedFrameDriver,
};
use pill_web::{canvas_element, canvas_physical_size, focus_without_scrolling, start_project};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::{Window, WindowId};

// Current crate
use crate::scripted::take_scripted_lines;
use crate::viewport::SceneViewport;

struct ViewerApplication {
    canvas_id: String,
    runtime: Option<Runtime>,
    window: Option<Arc<Window>>,
    driver: Rc<RefCell<Option<RenderingRuntime>>>,
    pending_events: Vec<InputEvent>,
    started_at: Instant,
    last_frame_time: Option<Instant>,
    viewport: SceneViewport,
}

impl ApplicationHandler for ViewerApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        let canvas = match canvas_element(&self.canvas_id) {
            Ok(canvas) => canvas,
            Err(message) => {
                error!(
                    target: telemetry_target::EDITOR,
                    error = message.as_str(),
                    "scene canvas not found; the viewer does not start"
                );
                return;
            }
        };
        let laid_out_size = canvas_physical_size(&canvas);
        let focus_target = canvas.clone();
        let attributes = Window::default_attributes()
            .with_canvas(Some(canvas))
            .with_focusable(true);
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => return report_failure("scene canvas window creation", &error),
        };
        if let Err(error) = focus_without_scrolling(&focus_target) {
            warn!(
                target: telemetry_target::EDITOR,
                error = ?error,
                "could not focus the scene canvas; click it to send keys"
            );
        }
        self.window = Some(Arc::clone(&window));

        // WebGPU refuses a zero-sized surface, so a canvas winit has not
        // measured yet is attached at its laid-out size.
        let size = window.inner_size();
        let (width, height) = if size.width > 0 && size.height > 0 {
            (size.width, size.height)
        } else {
            laid_out_size
        };
        let driver = Rc::clone(&self.driver);
        pill_core::platform::futures::spawn(async move {
            match pill_player::attach_renderer(runtime, Arc::clone(&window), width, height).await {
                Ok(mut attached) => {
                    let size = window.inner_size();
                    if size.width > 0
                        && size.height > 0
                        && (size.width, size.height) != (width, height)
                    {
                        attached.resize_surface(SurfaceId::PRIMARY, size.width, size.height);
                    }
                    attached
                        .engine_mut()
                        .set_run_mode(pill_engine_core::RunMode::Editing);
                    *driver.borrow_mut() = Some(attached);
                    info!(target: telemetry_target::EDITOR, "renderer attached to the scene canvas");
                    window.request_redraw();
                }
                Err(error) => report_failure("renderer attachment", &error),
            }
        });
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                if let Some(driver) = self.driver.borrow_mut().as_mut() {
                    driver.resize_surface(SurfaceId::PRIMARY, size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            WindowEvent::Touch(touch) => self.viewport.push_touch(touch),
            other => self.pending_events.extend(translate_window_event(&other)),
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let Some(input) = pill_input::winit_events::translate_device_event(&event) {
            self.pending_events.push(input);
        }
    }
}

impl ViewerApplication {
    /// Input, the page's ops and the queued edits, the viewport, then the
    /// world frame.
    fn frame(&mut self) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let now = Instant::now();
        let delta_seconds = self
            .last_frame_time
            .map_or(0.0, |last| now.duration_since(last).as_secs_f32());
        self.last_frame_time = Some(now);

        // Input and scripted lines wait in their queues until the renderer attaches.
        let mut driver_slot = self.driver.borrow_mut();
        let Some(driver) = driver_slot.as_mut() else {
            return;
        };
        let events: Vec<InputEvent> = std::mem::take(&mut self.pending_events);
        let scripted_lines = take_scripted_lines();
        for event in &events {
            driver.push_input(event.clone());
        }

        // The shell registers these in the egui posture; here nothing else does.
        {
            let world = driver.engine_mut().world_mut();
            ensure_command_resources(world);
            world.register_resource::<GizmoState>();
            if world.get_resource::<GizmoState>().is_none() {
                world.insert_resource(GizmoState::default());
            }
            world.register_resource::<EditorSelection>();
            if world.get_resource::<EditorSelection>().is_none() {
                world.insert_resource(EditorSelection::default());
            }
        }

        crate::page_api::apply_runtime_mounts(driver.engine_mut().world_mut());
        crate::apply_pending_commands(
            driver.engine_mut(),
            now.duration_since(self.started_at),
            &scripted_lines,
        );
        let size = window.inner_size();
        let scale_factor = (window.scale_factor() as f32).max(f32::EPSILON);
        self.viewport.frame(
            driver.engine_mut(),
            &events,
            (size.width as f32, size.height as f32),
            scale_factor,
            delta_seconds,
        );
        crate::edit_stream::emit_outbox(driver.engine_mut());

        if let Err(error) = driver.run_frame() {
            report_failure("frame render", &error);
            return;
        }
        window.request_redraw();
    }
}

/// Run `project` as the shell-less editor on the canvas named `canvas_id`.
pub fn run_viewer(project: StaticProject, canvas_id: &str) {
    let canvas_id = canvas_id.to_owned();
    start_project(project, move |runtime| {
        let event_loop = match EventLoop::new() {
            Ok(event_loop) => event_loop,
            Err(error) => return report_failure("event loop creation", &error),
        };
        info!(
            target: telemetry_target::EDITOR,
            canvas = %canvas_id,
            "starting the shell-less editor in the browser"
        );
        event_loop.spawn_app(ViewerApplication {
            canvas_id: canvas_id.clone(),
            runtime: Some(runtime),
            window: None,
            driver: Rc::new(RefCell::new(None)),
            pending_events: Vec::new(),
            started_at: Instant::now(),
            last_frame_time: None,
            viewport: SceneViewport::default(),
        });
    });
}
