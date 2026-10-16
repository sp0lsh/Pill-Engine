//! Port of bgfx's example 37-gpudrivenrendering: 526 instances of one cube mesh (ground, 25
//! occluder walls, 200 cubes, 300 boxes) culled on the GPU against a Hi-Z pyramid of the walls'
//! depth and drawn with one indirect draw per prop (`gpu_driven`). "Use Draw Indirect" off draws
//! every instance directly instead. The drawn count is read back from the indirect draws.
//! Camera: left-drag orbits, right-drag or scroll dollies.

use crate::gpu_driven::{hiz_size, GpuDriven, View, DRAW_ARGS};
use crate::scene::Scene;
use glam::{Mat4, Vec3};
use pill_engine::{define_global_component, game::*};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const SEED: u64 = 0x37;
const FOV: f32 = 60.0;
const NEAR: f32 = 0.1;
const FAR: f32 = 500.0;
const START_EYE: Vec3 = Vec3::new(55.0, 20.0, 65.0);
const MIN_DISTANCE: f32 = 1.0;
const MAX_DISTANCE: f32 = 100.0;
/// Seconds the camera takes to settle on its target (bgfx: `min(dt / 0.12, 1)`)
const SMOOTHING: f32 = 0.12;
/// Degrees per pixel of mouse drag, metres per pixel of dolly drag and per scroll line.
const ORBIT_SPEED: f32 = 0.3;
const DOLLY_SPEED: f32 = 0.2;
const SCROLL_SPEED: f32 = 4.0;

/// What the settings window and the pass share.
struct Settings {
    use_indirect: bool,
    /// Instances drawn by the last read-back indirect draws
    drawn: Option<u32>,
    total: u32,
}

static SETTINGS: Mutex<Settings> = Mutex::new(Settings {
    use_indirect: true,
    drawn: None,
    total: 0,
});

fn with_settings<T>(f: impl FnOnce(&mut Settings) -> T) -> T {
    f(&mut SETTINGS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()))
}

// Orbit around the origin: yaw / pitch in degrees, distance in metres; targets and smoothed
// values
define_global_component!(Orbit {
    target: Vec3,
    current: Vec3,
});

/// (yaw, pitch, distance) of an eye position around the origin.
fn orbit_of(eye: Vec3) -> Vec3 {
    let distance = eye.length();
    Vec3::new(
        eye.x.atan2(eye.z).to_degrees(),
        (eye.y / distance).asin().to_degrees(),
        distance,
    )
}

fn eye_of(orbit: Vec3) -> Vec3 {
    let (yaw, pitch) = (orbit.x.to_radians(), orbit.y.to_radians());
    Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * orbit.z
}

pub struct Game {}

impl PillGame for Game {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("gpu_driven")?;
        engine.set_active_scene(scene)?;
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        // Queried by the engine's PBR pass; the props are drawn by GpuDrivenPass
        engine.register_component::<MeshRenderingComponent>(scene)?;
        let mut transform = TransformComponent::builder().position(START_EYE).build();
        transform.look_at(Vector3f::ZERO);
        engine
            .build_entity(scene)
            .with_component(transform)
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(FOV)
                    .range(NEAR..FAR)
                    .build(),
            )
            .build();
        engine.add_global_component(Orbit {
            target: orbit_of(START_EYE),
            current: orbit_of(START_EYE),
        })?;

        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            // bgfx's flat material colors
            render_state.tonemap = Tonemap::None;
            render_state.background_color = [0.19, 0.19, 0.19];
        }
        let egui = engine.get_global_component_mut::<EguiComponent>()?;
        egui.show_engine_windows = false;
        egui.register_ui("gpu_driven.settings", settings_ui);

        engine.add_render_pass(Box::new(GpuDrivenPass {
            gpu_driven: None,
            render_size: (0, 0),
            readback: None,
        }))?;
        engine.add_system("orbit_camera", orbit_camera_system)?;
        Ok(())
    }
}

fn orbit_camera_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine.get_global_component::<TimeComponent>()?.delta_time;
    let input = engine.get_global_component::<InputComponent>()?;
    let drag = input.get_mouse_delta();
    let (orbiting, dollying) = (
        input.get_mouse_button(MouseButton::Left),
        input.get_mouse_button(MouseButton::Right),
    );
    let scroll = input.get_mouse_scroll_delta().y;
    let orbit = engine.get_global_component_mut::<Orbit>()?;
    if orbiting {
        orbit.target.x -= drag.x * ORBIT_SPEED;
        orbit.target.y = (orbit.target.y + drag.y * ORBIT_SPEED).clamp(2.0, 88.0);
    }
    if dollying {
        orbit.target.z += drag.y * DOLLY_SPEED;
    }
    orbit.target.z = (orbit.target.z - scroll * SCROLL_SPEED).clamp(MIN_DISTANCE, MAX_DISTANCE);
    let amount = (delta_time / SMOOTHING).min(1.0);
    orbit.current = orbit.current.lerp(orbit.target, amount);
    let eye = eye_of(orbit.current);
    for (_, transform, camera) in
        engine.iterate_two_components_mut::<TransformComponent, CameraComponent>()?
    {
        if camera.enabled {
            transform.set_position(eye);
            transform.look_at(Vector3f::ZERO);
        }
    }
    Ok(())
}

/// Read-back of the indirect draws, one frame behind: copied in one frame, mapped after that
/// frame is submitted, read when mapped.
enum Readback {
    Copied(wgpu::Buffer),
    Mapping(wgpu::Buffer, Arc<AtomicBool>),
}

/// Culls and draws the props into the engine's HDR target, after the engine's opaque pass.
struct GpuDrivenPass {
    gpu_driven: Option<GpuDriven>,
    /// Render size the Hi-Z was built for
    render_size: (u32, u32),
    readback: Option<Readback>,
}

impl GpuDrivenPass {
    /// Advances the read-back; on a culled frame without one in flight, records the copy.
    fn update_readback(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, culled: bool) {
        let Some(gpu_driven) = &self.gpu_driven else {
            return;
        };
        let size = (gpu_driven.prop_count() as usize * DRAW_ARGS * 4) as u64;
        self.readback = match self.readback.take() {
            Some(Readback::Copied(buffer)) => {
                let mapped = Arc::new(AtomicBool::new(false));
                let flag = mapped.clone();
                buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    flag.store(result.is_ok(), Ordering::Release);
                });
                Some(Readback::Mapping(buffer, mapped))
            }
            Some(Readback::Mapping(buffer, mapped)) if mapped.load(Ordering::Acquire) => {
                let drawn: u32 = bytemuck::cast_slice::<u8, u32>(&buffer.slice(..).get_mapped_range())
                    .chunks(DRAW_ARGS)
                    .map(|draw| draw[1])
                    .sum();
                buffer.unmap();
                with_settings(|settings| settings.drawn = Some(drawn));
                None
            }
            pending @ Some(Readback::Mapping(..)) => pending,
            None if culled => {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gdr_readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(gpu_driven.draws(), 0, &buffer, 0, size);
                Some(Readback::Copied(buffer))
            }
            None => None,
        };
    }
}

impl Pass for GpuDrivenPass {
    fn get_label(&self) -> &str {
        "gpu_driven_pass"
    }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> {
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let render_size = renderer.get_render_size();
        match &mut self.gpu_driven {
            None => {
                let gpu_driven = GpuDriven::new(
                    renderer.get_device(),
                    &Scene::new(SEED),
                    renderer.get_hdr_format(),
                    DEPTH_TARGET_FORMAT,
                    hiz_size(render_size),
                );
                with_settings(|settings| settings.total = gpu_driven.instance_count());
                self.gpu_driven = Some(gpu_driven);
            }
            Some(gpu_driven) if self.render_size != render_size => {
                gpu_driven.resize(renderer.get_device(), hiz_size(render_size));
            }
            Some(_) => {}
        }
        self.render_size = render_size;
        let gpu_driven = self.gpu_driven.as_ref().expect("built above");

        let (camera, transform) = world.active_camera()?;
        let render_state = world.get_global::<RenderStateComponent>()?;
        let view_matrix = Mat4::look_to_rh(transform.position, camera_forward(transform), Vec3::Y);
        let hiz = gpu_driven.hiz_size();
        let view = |projection: Mat4| View {
            view_projection: (projection * view_matrix).to_cols_array_2d(),
            light_direction: Vec3::from_array(render_state.lights[0].direction)
                .normalize()
                .extend(0.0)
                .to_array(),
            camera_position: transform.position.extend(1.0).to_array(),
        };
        let use_indirect = with_settings(|settings| settings.use_indirect);
        if use_indirect {
            // The occluders and the culling see the scene with the Hi-Z's aspect
            let occlusion = Mat4::perspective_rh(FOV.to_radians(), hiz.0 as f32 / hiz.1 as f32, NEAR, FAR);
            gpu_driven.cull(renderer.get_queue(), encoder, &view(occlusion));
        } else {
            with_settings(|settings| settings.drawn = Some(settings.total));
        }

        let hdr_view = renderer
            .get_render_target_view(render_state.hdr_target())
            .ok_or("GpuDrivenPass: HDR target missing")?;
        let depth_view = renderer
            .get_render_target_view(render_state.depth_target())
            .ok_or("GpuDrivenPass: depth target missing")?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gdr_main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let main = view(Mat4::perspective_rh(
                camera.fov.to_radians(),
                camera.aspect.get_value(),
                camera.range.start,
                camera.range.end,
            ));
            gpu_driven.draw(renderer.get_queue(), &mut pass, &main, use_indirect);
        }
        let _ = renderer.get_device().poll(wgpu::PollType::Poll);
        self.update_readback(renderer.get_device(), encoder, use_indirect);
        Ok(())
    }
}

/// bgfx's "Settings" window: the draw-indirect toggle, plus the drawn instance count.
fn settings_ui(context: &egui::Context) {
    egui::Window::new("Settings").show(context, |ui| {
        with_settings(|settings| {
            ui.checkbox(&mut settings.use_indirect, "Use Draw Indirect");
            let drawn = settings
                .drawn
                .map_or("—".to_string(), |drawn| drawn.to_string());
            ui.label(format!("drawn: {drawn} / {}", settings.total));
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_of_eye_of_round_trip() {
        let eye = eye_of(orbit_of(START_EYE));
        assert!((eye - START_EYE).length() < 1e-3, "{eye}");
    }
}
