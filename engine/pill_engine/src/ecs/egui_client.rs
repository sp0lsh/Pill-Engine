use std::sync::{Arc, Mutex};

pub struct EguiClient {
    events: Mutex<Vec<winit::event::WindowEvent>>,
    ui: Mutex<Option<Box<dyn Fn(&egui::Context) + Send>>>,
}

impl EguiClient {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            events: Mutex::new(Vec::new()),
            ui: Mutex::new(None),
        })
    }

    pub fn handle_input(&self, event: winit::event::WindowEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub fn take_events(&self) -> Vec<winit::event::WindowEvent> {
        std::mem::take(&mut self.events.lock().unwrap())
    }

    pub fn set_ui(&self, f: impl Fn(&egui::Context) + Send + 'static) {
        *self.ui.lock().unwrap() = Some(Box::new(f));
    }

    pub fn take_ui(&self) -> Option<Box<dyn Fn(&egui::Context) + Send>> {
        self.ui.lock().unwrap().take()
    }
}
