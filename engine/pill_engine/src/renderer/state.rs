#![allow(clippy::too_many_arguments)]
use crate::{
    ecs::EguiClient,
    graphics::{
        BufferDesc, Pass, PassEgui, PassScene, PillRenderer, PipelineV2, PipelineV2Desc,
        RendererCameraHandle, RendererTargetDesc, RendererTextureHandle, WorldQuery,
    },
    internal::{
        get_renderer_resource_handle_from_camera_component, CameraComponent, ComponentStorage,
        EntityHandle, RenderQueueItem, TransformComponent,
    },
    renderer::{
        config::MAX_INSTANCE_PER_DRAWCALL_COUNT,
        drawers::mesh_drawer::MeshDrawer,
        instance::Instance,
        resources::{
            EngineParameters, RendererCamera, RendererMesh, RendererShader, RendererTexture, Vertex,
        },
    },
    resources::ResourceManager,
};

use indexmap::IndexMap;
use pill_core::{debug, info, LogContext, PillSlotMap, PillSlotMapKey, PillStyle, RendererError, Timer};
use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Error, Result};

pub struct Renderer {
    pub state: State,
}

impl Renderer {
    pub async fn new_async(
        window: Arc<winit::window::Window>,
        config: config::Config,
    ) -> Result<Self> {
        info!(LogContext::Rendering => "Initializing {}", "Renderer".module_object_style());
        let state = State::new(window, config).await?;
        Ok(Self { state })
    }
}

impl PillRenderer for Renderer {
    #[cfg(not(target_arch = "wasm32"))]
    fn new(window: Arc<winit::window::Window>, config: config::Config) -> Result<Self> {
        info!(LogContext::Rendering => "Initializing {}", "Renderer".module_object_style());
        let state = pollster::block_on(State::new(window, config))?;
        Ok(Self { state })
    }

    #[cfg(target_arch = "wasm32")]
    fn new(_window: Arc<winit::window::Window>, _config: config::Config) -> Result<Self> {
        panic!("Use Renderer::new_async on WASM")
    }

    // --- Create ---

    fn create_shader_struct(
        &mut self,
        name: &str,
        vertex_wgsl: &str,
        fragment_wgsl: &str,
        texture_slots: &HashMap<String, crate::resources::ShaderTextureSlot>,
        parameter_slots: &IndexMap<String, crate::resources::ShaderParameterSlot>,
        pass_engine_parameters: bool,
        pass_camera_parameters: bool,
    ) -> Result<RendererShader> {
        RendererShader::new(
            name,
            &self.state.device,
            self.state.color_format,
            Some(self.state.depth_format),
            &[
                RendererMesh::data_layout_descriptor(),
                Instance::data_layout_descriptor(),
            ],
            vertex_wgsl,
            fragment_wgsl,
            parameter_slots,
            texture_slots,
            &self.state.engine_parameters.bind_group_layout,
            &self.state.camera_bind_group_layout,
            pass_engine_parameters,
            pass_camera_parameters,
        )
    }

    fn create_camera(&mut self) -> Result<RendererCameraHandle> {
        let camera = RendererCamera::new(
            &self.state.device,
            self.state.camera_bind_group_layout.clone(),
        )?;
        let handle = self.state.cameras.insert(camera);
        Ok(handle)
    }

    // --- Destroy ---

    fn destroy_camera(&mut self, renderer_camera_handle: RendererCameraHandle) -> Result<()> {
        self.state.cameras.remove(renderer_camera_handle).unwrap();
        Ok(())
    }

    // --- Other ---

    fn resize(&mut self, new_window_size: winit::dpi::PhysicalSize<u32>) {
        info!(LogContext::Rendering => "Resizing {} resources", "Renderer".module_object_style());
        self.state.resize(new_window_size)
    }

    fn pass_input_to_egui(&mut self, event: &winit::event::WindowEvent) -> Result<()> {
        if let Some(c) = &self.state.egui_client {
            c.handle_input(event.clone());
        }
        Ok(())
    }

    fn render(
        &mut self,
        active_camera_entity_handle: EntityHandle,
        render_queue: &[RenderQueueItem],
        camera_component_storage: &ComponentStorage<CameraComponent>,
        transform_component_storage: &ComponentStorage<TransformComponent>,
        delta_time: f32,
        timer: &mut Timer,
        resource_manager: &ResourceManager,
    ) -> Result<()> {
        debug!(LogContext::Frame => "Starting frame render");

        timer.record("Get frame");

        let frame = self.state.surface.get_current_texture();
        let frame = match frame {
            std::result::Result::Ok(frame) => frame,
            std::result::Result::Err(error) => match error {
                wgpu::SurfaceError::Lost => return Err(RendererError::SurfaceLost.into()),
                wgpu::SurfaceError::OutOfMemory => {
                    return Err(RendererError::SurfaceOutOfMemory.into())
                }
                _ => return Err(RendererError::SurfaceOther.into()),
            },
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render_encoder"),
            });

        let world = WorldQuery {
            active_camera: active_camera_entity_handle,
            render_queue,
            camera_components: camera_component_storage,
            transform_components: transform_component_storage,
            delta_time,
            resources: resource_manager,
        };

        timer.begin_context("Scene Passes");

        let mut passes = std::mem::take(&mut self.state.passes);
        for pass in &mut passes {
            pass.draw(&mut encoder, self, &frame, &view, &world)?;
        }
        self.state.passes = passes;

        timer.end_context()?;

        timer.record("Submit commands and present frame");

        self.state.queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        Ok(())
    }

    // --- Pass API ---

    fn set_passes(&mut self, mut passes: Vec<Box<dyn Pass>>) -> Result<()> {
        for pass in &mut passes {
            pass.init(self)?;
        }
        self.state.passes = passes;
        Ok(())
    }

    fn init_default_passes(&mut self, egui_client: Arc<EguiClient>) -> Result<()> {
        self.state.egui_client = Some(egui_client.clone());
        self.set_passes(vec![
            Box::new(PassScene::new()),
            Box::new(PassEgui::new(self.state.window.clone(), egui_client)),
        ])
    }

    fn get_device(&self) -> &wgpu::Device {
        &self.state.device
    }

    fn get_queue(&self) -> &wgpu::Queue {
        &self.state.queue
    }

    fn get_surface_format(&self) -> wgpu::TextureFormat {
        self.state.color_format
    }

    fn create_buffer(&mut self, desc: BufferDesc) -> Result<wgpu::Buffer> {
        let buffer = self.state.device.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label,
            size: desc.byte_size,
            usage: desc.usage,
            mapped_at_creation: false,
        });
        Ok(buffer)
    }

    fn create_pipeline_v2(&mut self, desc: PipelineV2Desc) -> Result<PipelineV2> {
        let bind_group_layouts: Vec<wgpu::BindGroupLayout> = desc
            .bind_groups
            .iter()
            .map(|entries| {
                self.state
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries,
                    })
            })
            .collect();

        let bind_group_layout_refs: Vec<&wgpu::BindGroupLayout> =
            bind_group_layouts.iter().collect();

        let pipeline_layout =
            self.state
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: desc.label,
                    bind_group_layouts: &bind_group_layout_refs,
                    push_constant_ranges: &[],
                });

        let vs_module = self
            .state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(desc.vs.entry_func),
                source: wgpu::ShaderSource::Wgsl(desc.vs.source.into()),
            });
        let fs_module = self
            .state
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(desc.ps.entry_func),
                source: wgpu::ShaderSource::Wgsl(desc.ps.source.into()),
            });

        let pipeline =
            self.state
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: desc.label,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: Some(desc.vs.entry_func),
                        buffers: desc.vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs_module,
                        entry_point: Some(desc.ps.entry_func),
                        targets: desc.targets,
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: desc.depth_stencil,
                    multisample: desc.multisample,
                    multiview: None,
                    cache: None,
                });

        Ok(PipelineV2 {
            pipeline,
            bind_group_layouts,
        })
    }

    fn create_render_target(&mut self, desc: RendererTargetDesc) -> Result<RendererTextureHandle> {
        let texture = RendererTexture::new_render_target(
            &self.state.device,
            &desc.name,
            desc.width,
            desc.height,
            desc.format,
        )?;
        let handle = self.state.pass_textures.insert(texture);
        Ok(handle)
    }

    fn create_depth_texture(&mut self, label: &str) -> Result<RendererTextureHandle> {
        let texture = RendererTexture::new_depth_texture(
            &self.state.device,
            &self.state.surface_configuration,
            label,
        )?;
        let handle = self.state.pass_textures.insert(texture);
        Ok(handle)
    }

    fn get_render_target_view(
        &self,
        handle: RendererTextureHandle,
    ) -> Option<&wgpu::TextureView> {
        self.state
            .pass_textures
            .get(handle)
            .map(|t| &t.texture_view)
    }

    fn record_scene_pass(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let camera_storage = world
            .camera_components
            .data
            .get(world.active_camera.data().index as usize)
            .unwrap();
        let active_camera_component = camera_storage.as_ref().unwrap();

        self.state.engine_parameters.update(
            &self.state.queue,
            world.delta_time,
            active_camera_component.fog_density,
            [
                active_camera_component.fog_color.x,
                active_camera_component.fog_color.y,
                active_camera_component.fog_color.z,
            ],
        );

        let renderer_camera_handle =
            get_renderer_resource_handle_from_camera_component(active_camera_component);
        let renderer_camera = self
            .state
            .cameras
            .get_mut(renderer_camera_handle)
            .ok_or(Error::new(RendererError::RendererResourceNotFound))?;

        let camera_transform_storage = world
            .transform_components
            .data
            .get(world.active_camera.data().index as usize)
            .unwrap();
        let active_camera_transform = camera_transform_storage.as_ref().unwrap();
        renderer_camera.update(
            &self.state.queue,
            active_camera_component,
            active_camera_transform,
        );

        let renderer_camera = self.state.cameras.get(renderer_camera_handle).unwrap();
        let clear_color = active_camera_component.clear_color;

        let color_attachment = wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: clear_color.x as f64,
                    g: clear_color.y as f64,
                    b: clear_color.z as f64,
                    a: 1.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        };
        let depth_stencil_attachment = wgpu::RenderPassDepthStencilAttachment {
            view: &self.state.depth_texture.texture_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        };

        let mut timer = Timer::new();
        self.state.mesh_drawer.record_draw_commands(
            &self.state.queue,
            encoder,
            world.resources,
            &self.state.engine_parameters,
            color_attachment,
            depth_stencil_attachment,
            renderer_camera,
            world.render_queue,
            world.transform_components,
            &mut timer,
        )
    }
}

pub struct State {
    pub(crate) cameras: PillSlotMap<RendererCameraHandle, RendererCamera>,
    pub(crate) engine_parameters: EngineParameters,
    pass_textures: PillSlotMap<RendererTextureHandle, RendererTexture>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_configuration: wgpu::SurfaceConfiguration,
    #[allow(dead_code)]
    window_size: winit::dpi::PhysicalSize<u32>,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    depth_texture: RendererTexture,
    passes: Vec<Box<dyn Pass>>,
    egui_client: Option<Arc<EguiClient>>,
    camera_bind_group_layout: wgpu::BindGroupLayout,
    mesh_drawer: MeshDrawer,
    window: Arc<winit::window::Window>,
}

impl State {
    async fn new(window: Arc<winit::window::Window>, config: config::Config) -> Result<Self> {
        let window_width = config
            .get_int("WINDOW_WIDTH")
            .context("WINDOW_WIDTH is missing from config")? as u32;
        let window_height = config
            .get_int("WINDOW_HEIGHT")
            .context("WINDOW_HEIGHT is missing from config")? as u32;
        let window_size = winit::dpi::PhysicalSize::new(window_width, window_height);

        let (instance, surface) = {
            let backends = match std::env::var("WGPU_BACKENDS").as_deref() {
                std::result::Result::Ok("VULKAN") => wgpu::Backends::VULKAN,
                std::result::Result::Ok("DX12") => wgpu::Backends::DX12,
                std::result::Result::Ok("METAL") => wgpu::Backends::METAL,
                std::result::Result::Ok("GL") => wgpu::Backends::GL,
                std::result::Result::Ok("BROWSER_WEBGPU") => wgpu::Backends::BROWSER_WEBGPU,
                _ => wgpu::Backends::all(),
            };

            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends,
                flags: wgpu::InstanceFlags::from_build_config().with_env(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            });
            let surface = instance
                .create_surface(window.clone())
                .context("Failed to create surface")?;
            (instance, surface)
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to request adapter")?;

        let info = adapter.get_info();
        info!(LogContext::Rendering => "Using GPU: {} ({:?})", info.name, info.backend);

        let (device, queue) = {
            let wanted = wgpu::Features::DEPTH_CLIP_CONTROL
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY;
            let features = wanted & adapter.features();

            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: {
                        let adapter_limits = adapter.limits();
                        wgpu::Limits {
                            max_compute_workgroup_storage_size: adapter_limits.max_compute_workgroup_storage_size,
                            max_compute_invocations_per_workgroup: adapter_limits.max_compute_invocations_per_workgroup,
                            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                            max_buffer_size: adapter_limits.max_buffer_size,
                            ..wgpu::Limits::default()
                        }
                    },
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                })
                .await
                .context("Failed to request device")?
        };

        let (surface_configuration, color_format, depth_format) = {
            let preferred_format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let surface_capabilities = surface.get_capabilities(&adapter);

            #[cfg(target_arch = "wasm32")]
            let present_mode = wgpu::PresentMode::Fifo;
            #[cfg(not(target_arch = "wasm32"))]
            let present_mode = if surface_capabilities
                .present_modes
                .contains(&wgpu::PresentMode::Mailbox)
            {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            };

            let format = if surface_capabilities.formats.contains(&preferred_format) {
                preferred_format
            } else if surface_capabilities
                .formats
                .contains(&wgpu::TextureFormat::Bgra8UnormSrgb)
            {
                wgpu::TextureFormat::Bgra8UnormSrgb
            } else if surface_capabilities
                .formats
                .contains(&wgpu::TextureFormat::Bgra8Unorm)
            {
                wgpu::TextureFormat::Bgra8Unorm
            } else {
                surface_capabilities.formats[0]
            };

            let surface_configuration = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: window_size.width,
                height: window_size.height,
                desired_maximum_frame_latency: 2,
                present_mode,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![format, format.remove_srgb_suffix()],
            };
            surface.configure(&device, &surface_configuration);
            let color_format = surface_configuration.format;
            let depth_format = wgpu::TextureFormat::Depth32Float;
            (surface_configuration, color_format, depth_format)
        };

        let depth_texture =
            RendererTexture::new_depth_texture(&device, &surface_configuration, "depth_texture")
                .context("Failed to create depth texture")?;

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_parameters_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let engine_parameters = EngineParameters::new(&device)?;
        let cameras = PillSlotMap::with_capacity_and_key(10);
        let pass_textures = PillSlotMap::with_capacity_and_key(10);
        let mesh_drawer = MeshDrawer::new(&device, MAX_INSTANCE_PER_DRAWCALL_COUNT as u32);

        Ok(Self {
            cameras,
            engine_parameters,
            pass_textures,
            surface,
            device,
            queue,
            surface_configuration,
            window_size,
            color_format,
            depth_format,
            depth_texture,
            passes: Vec::new(),
            egui_client: None,
            camera_bind_group_layout,
            mesh_drawer,
            window,
        })
    }

    fn resize(&mut self, new_window_size: winit::dpi::PhysicalSize<u32>) {
        if new_window_size.width > 0 && new_window_size.height > 0 {
            self.window_size = new_window_size;
            self.surface_configuration.width = new_window_size.width;
            self.surface_configuration.height = new_window_size.height;
            self.surface
                .configure(&self.device, &self.surface_configuration);
            self.depth_texture = RendererTexture::new_depth_texture(
                &self.device,
                &self.surface_configuration,
                "depth_texture",
            )
            .unwrap();
        }
    }
}
