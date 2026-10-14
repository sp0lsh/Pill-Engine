use crate::config::{
    DDGI_COLOR_COMPUTE_SHADER, DDGI_DEPTH_COMPUTE_SHADER, DDGI_TRACE_COMPUTE_SHADER,
};
use crate::ecs::{MeshRenderingComponent, RenderStateComponent, TransformComponent};
use crate::graphics::ddgi::bvh::{Bvh, Triangle};
use crate::graphics::ddgi::{
    random_rotation, GpuDdgiParams, ProbeGrid, COLOR_RESOLUTION, DEPTH_RESOLUTION,
    MAX_RAYS_PER_PROBE,
};
use crate::graphics::fullscreen::uniform_entry_for;
use crate::graphics::pass_lights::LightsFrame;
use crate::graphics::pass_pbr_opaque::{is_translucent, DrawTransform};
use crate::graphics::{
    BufferDesc, ComputePipeline, ComputePipelineDesc, FrameTarget, Pass, PassTextureHandle,
    PillRenderer, ShaderDesc, WorldQuery,
};
use crate::resources::{Material, Mesh, Texture};

use pill_core::{info, LogContext, Result, Vector3f};
use std::cell::RefCell;
use std::rc::Rc;

/// Bytes of one probe ray (direction + depth, radiance) — `DdgiRay` in the DDGI shaders.
const RAY_BYTES: u64 = 32;
/// Bytes of one probe's L1 SH radiance — `DdgiProbe`.
const PROBE_BYTES: u64 = 64;
/// Bytes of one radiance texel's estimator history — `DdgiTexelHistory`.
const TEXEL_HISTORY_BYTES: u64 = 48;
/// Bytes of one depth texel's moment history (float2).
const DEPTH_HISTORY_BYTES: u64 = 8;

/// What the PBR passes read: the probe-grid uniform, the probes' SH and the depth atlas.
/// `generation` changes when the probe resources are recreated (rebuild bind groups).
pub(crate) struct DdgiFrame {
    pub(crate) params: wgpu::Buffer,
    pub(crate) probes: wgpu::Buffer,
    pub(crate) depth_view: wgpu::TextureView,
    pub(crate) generation: u64,
}

impl DdgiFrame {
    /// Stand-ins until the scene has geometry: zeroed uniform (disabled), one probe, 1×1 atlas.
    pub(crate) fn new(renderer: &mut dyn PillRenderer) -> Result<Self> {
        let params = renderer.create_buffer(BufferDesc {
            label: Some("ddgi_params"),
            byte_size: std::mem::size_of::<GpuDdgiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;
        let probes = storage_buffer(renderer, "ddgi_probes", PROBE_BYTES)?;
        let depth_view = depth_atlas(renderer.get_device(), 1, 1);
        Ok(Self {
            params,
            probes,
            depth_view,
            generation: 0,
        })
    }
}

fn storage_buffer(
    renderer: &mut dyn PillRenderer,
    label: &str,
    byte_size: u64,
) -> Result<wgpu::Buffer> {
    renderer.create_buffer(BufferDesc {
        label: Some(label),
        byte_size: byte_size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Depth-moments atlas: written by the depth update (storage), sampled bilinearly by the queries.
fn depth_atlas(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("ddgi_depth_atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn compute_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    compute_entry(
        binding,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    compute_entry(
        binding,
        wgpu::BindingType::Texture {
            multisampled: false,
            view_dimension: wgpu::TextureViewDimension::D2,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
        },
    )
}

/// Probe grid and its GPU resources; recreated with the scene bounds or grid settings.
struct ProbeResources {
    grid: ProbeGrid,
    rays_per_probe: u32,
    probes_per_frame: u32,
    // The bind groups keep the ray and history buffers alive
    trace_group: wgpu::BindGroup,
    color_group: wgpu::BindGroup,
    depth_group: wgpu::BindGroup,
}

struct Pipelines {
    trace: ComputePipeline,
    color: ComputePipeline,
    depth: ComputePipeline,
    sampler: wgpu::Sampler,
}

/// Scene BVH on the GPU and the signature of the scene it was built from.
struct SceneBvh {
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
    bounds: (Vector3f, Vector3f),
    signature: u64,
}

/// Diffuse GI probes (see `ddgi/mod.rs`): rebuilds the BVH when the set of meshes changes (static
/// geometry), traces and updates `Gi::probes_per_frame` probes per frame and publishes them in
/// `frame` for the PBR passes. Settings: `RenderStateComponent::gi`.
pub(crate) struct PassDdgi {
    frame: Rc<RefCell<DdgiFrame>>,
    lights: Rc<RefCell<LightsFrame>>,
    /// Engine default IBL irradiance, used for ray misses when the game's handle is invalid.
    fallback_sky: PassTextureHandle,
    pipelines: Option<Pipelines>,
    bvh: Option<SceneBvh>,
    probes: Option<ProbeResources>,
    /// Sky irradiance handle the trace bind group was built with.
    bound_sky: Option<PassTextureHandle>,
    frame_index: u32,
    probe_offset: u32,
    triangles: Vec<Triangle>,
}

impl PassDdgi {
    pub(crate) fn new(
        frame: Rc<RefCell<DdgiFrame>>,
        lights: Rc<RefCell<LightsFrame>>,
        fallback_sky: PassTextureHandle,
    ) -> Self {
        Self {
            frame,
            lights,
            fallback_sky,
            pipelines: None,
            bvh: None,
            probes: None,
            bound_sky: None,
            frame_index: 0,
            probe_offset: 0,
            triangles: Vec::new(),
        }
    }

    /// Order-independent hash of the shadow-casting opaque mesh entities (static geometry: their
    /// transforms are not part of it).
    fn scene_signature(world: &WorldQuery<'_>) -> u64 {
        let Ok(meshes) = world.query::<MeshRenderingComponent>() else {
            return 0;
        };
        let mut signature = 0u64;
        for (slot, mesh) in meshes.data.iter().enumerate() {
            let Some(key) = mesh.as_ref().and_then(|mesh| mesh.render_queue_key) else {
                continue;
            };
            let value = key ^ ((slot as u64) << 40);
            signature = signature.wrapping_add(value.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        }
        signature
    }

    /// World-space triangles of the opaque, shadow-casting meshes with their albedo.
    fn gather_triangles(&mut self, world: &WorldQuery<'_>) {
        self.triangles.clear();
        let (Ok(transforms), Ok(meshes)) = (
            world.query::<TransformComponent>(),
            world.query::<MeshRenderingComponent>(),
        ) else {
            return;
        };
        for (transform, mesh_rendering) in transforms.data.iter().zip(&meshes.data) {
            let (Some(transform), Some(mesh_rendering)) = (transform, mesh_rendering) else {
                continue;
            };
            let (Some(key), Some(mesh_handle)) =
                (mesh_rendering.render_queue_key, mesh_rendering.mesh_handle)
            else {
                continue;
            };
            if !mesh_rendering.cast_shadows || is_translucent(world, key) {
                continue;
            }
            let Some(data) = world
                .resources
                .get_resource::<Mesh>(&mesh_handle)
                .ok()
                .and_then(Mesh::data)
            else {
                continue;
            };
            let albedo = mesh_rendering
                .material_handle
                .as_ref()
                .and_then(|handle| world.resources.get_resource::<Material>(handle).ok())
                .map(|material| {
                    let texture_average = material
                        .albedo_texture
                        .as_ref()
                        .and_then(|handle| world.resources.get_resource::<Texture>(handle).ok())
                        .and_then(|texture| texture.average_color)
                        .unwrap_or([1.0; 3]);
                    [
                        material.albedo.x * texture_average[0],
                        material.albedo.y * texture_average[1],
                        material.albedo.z * texture_average[2],
                    ]
                })
                .unwrap_or([0.8; 3]);
            let model = DrawTransform::from(transform).model_matrix();
            let position = |index: u32| {
                model.transform_point3(Vector3f::from_array(
                    data.vertices[index as usize].position(),
                ))
            };
            for corner in data.indices.chunks_exact(3) {
                self.triangles.push(Triangle {
                    vertices: [
                        position(corner[0]),
                        position(corner[1]),
                        position(corner[2]),
                    ],
                    albedo,
                });
            }
        }
    }

    fn create_probes(
        &mut self,
        renderer: &mut dyn PillRenderer,
        grid: ProbeGrid,
        rays_per_probe: u32,
        probes_per_frame: u32,
        sky: &wgpu::TextureView,
    ) -> Result<()> {
        let pipelines = self
            .pipelines
            .as_ref()
            .expect("PassDdgi: init() before draw()");
        let bvh = self
            .bvh
            .as_ref()
            .expect("PassDdgi: BVH built before probes");
        let probe_count = grid.probe_count() as u64;
        let rays = storage_buffer(
            renderer,
            "ddgi_rays",
            probes_per_frame as u64 * rays_per_probe as u64 * RAY_BYTES,
        )?;
        let color_history = storage_buffer(
            renderer,
            "ddgi_color_history",
            probe_count * (COLOR_RESOLUTION * COLOR_RESOLUTION) as u64 * TEXEL_HISTORY_BYTES,
        )?;
        let depth_history = storage_buffer(
            renderer,
            "ddgi_depth_history",
            probe_count * (DEPTH_RESOLUTION * DEPTH_RESOLUTION) as u64 * DEPTH_HISTORY_BYTES,
        )?;
        let probes = storage_buffer(renderer, "ddgi_probes", probe_count * PROBE_BYTES)?;
        let (width, height) = grid.depth_atlas_size();
        let depth_view = depth_atlas(renderer.get_device(), width, height);

        let lights = self.lights.borrow();
        let mut frame = self.frame.borrow_mut();
        let device = renderer.get_device();
        let group = |layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindingResource]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ddgi"),
                layout,
                entries: &entries
                    .iter()
                    .enumerate()
                    .map(|(binding, resource)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: resource.clone(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let trace_group = group(
            &pipelines.trace.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                bvh.nodes.as_entire_binding(),
                bvh.triangles.as_entire_binding(),
                probes.as_entire_binding(),
                wgpu::BindingResource::TextureView(&depth_view),
                wgpu::BindingResource::Sampler(&pipelines.sampler),
                lights.buffers.lights.as_entire_binding(),
                wgpu::BindingResource::TextureView(sky),
                rays.as_entire_binding(),
            ],
        );
        let color_group = group(
            &pipelines.color.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                rays.as_entire_binding(),
                color_history.as_entire_binding(),
                probes.as_entire_binding(),
            ],
        );
        let depth_group = group(
            &pipelines.depth.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                rays.as_entire_binding(),
                depth_history.as_entire_binding(),
                wgpu::BindingResource::TextureView(&depth_view),
            ],
        );
        frame.probes = probes;
        frame.depth_view = depth_view;
        frame.generation += 1;
        drop(frame);
        drop(lights);
        self.probes = Some(ProbeResources {
            grid,
            rays_per_probe,
            probes_per_frame,
            trace_group,
            color_group,
            depth_group,
        });
        self.frame_index = 0;
        self.probe_offset = 0;
        Ok(())
    }

    /// Writes the uniform with GI off, so the PBR passes skip the probes.
    fn disable(&self, renderer: &mut dyn PillRenderer) {
        let frame = self.frame.borrow();
        renderer.get_queue().write_buffer(
            &frame.params,
            0,
            bytemuck::bytes_of(&GpuDdgiParams::default()),
        );
    }
}

impl Pass for PassDdgi {
    fn get_label(&self) -> &str {
        "pass_ddgi"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let uniform = uniform_entry_for(0, wgpu::ShaderStages::COMPUTE);
        let trace = renderer.create_compute_pipeline(ComputePipelineDesc {
            label: Some("ddgi_trace"),
            cs: ShaderDesc {
                source: DDGI_TRACE_COMPUTE_SHADER,
                entry_func: "cs_main",
            },
            bind_groups: vec![vec![
                uniform,
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                texture_entry(4),
                compute_entry(
                    5,
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
                storage_entry(6, true),
                texture_entry(7),
                storage_entry(8, false),
            ]],
        })?;
        let color = renderer.create_compute_pipeline(ComputePipelineDesc {
            label: Some("ddgi_color"),
            cs: ShaderDesc {
                source: DDGI_COLOR_COMPUTE_SHADER,
                entry_func: "cs_main",
            },
            bind_groups: vec![vec![
                uniform,
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
            ]],
        })?;
        let depth = renderer.create_compute_pipeline(ComputePipelineDesc {
            label: Some("ddgi_depth"),
            cs: ShaderDesc {
                source: DDGI_DEPTH_COMPUTE_SHADER,
                entry_func: "cs_main",
            },
            bind_groups: vec![vec![
                uniform,
                storage_entry(1, true),
                storage_entry(2, false),
                compute_entry(
                    3,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ]],
        })?;
        let sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("ddgi_linear_clamp"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
        self.pipelines = Some(Pipelines {
            trace,
            color,
            depth,
            sampler,
        });
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let render_state = world.get_global::<RenderStateComponent>()?;
        let settings = render_state.gi;
        if !settings.enabled {
            self.disable(renderer);
            return Ok(());
        }

        // Static geometry: rebuild the BVH only when the set of meshes changes
        let signature = Self::scene_signature(world);
        if self.bvh.as_ref().map(|bvh| bvh.signature) != Some(signature) {
            let start = web_time::Instant::now();
            self.gather_triangles(world);
            let bvh = Bvh::build(&self.triangles);
            let Some(bounds) = bvh.bounds() else {
                self.bvh = None;
                self.probes = None;
                self.disable(renderer);
                return Ok(());
            };
            info!(LogContext::Rendering =>
                "PassDdgi: BVH of {} triangles, {} nodes in {:.1} ms",
                bvh.triangles.len(),
                bvh.nodes.len(),
                start.elapsed().as_secs_f32() * 1000.0
            );
            let nodes = storage_buffer(
                renderer,
                "ddgi_bvh_nodes",
                std::mem::size_of_val(bvh.nodes.as_slice()) as u64,
            )?;
            let triangles = storage_buffer(
                renderer,
                "ddgi_bvh_triangles",
                std::mem::size_of_val(bvh.triangles.as_slice()) as u64,
            )?;
            renderer
                .get_queue()
                .write_buffer(&nodes, 0, bytemuck::cast_slice(&bvh.nodes));
            renderer
                .get_queue()
                .write_buffer(&triangles, 0, bytemuck::cast_slice(&bvh.triangles));
            self.bvh = Some(SceneBvh {
                nodes,
                triangles,
                bounds,
                signature,
            });
            self.probes = None;
        }
        let Some(bvh) = &self.bvh else {
            self.disable(renderer);
            return Ok(());
        };

        // Probe grid over the scene; resources follow the grid and ray settings
        let grid = ProbeGrid::fit(bvh.bounds, settings.probe_spacing);
        let rays_per_probe = settings.rays_per_probe.clamp(1, MAX_RAYS_PER_PROBE);
        let probes_per_frame = match settings.probes_per_frame {
            0 => grid.probe_count(),
            count => count.min(grid.probe_count()),
        };
        let sky_handle = match renderer.get_texture_view(render_state.ibl_diffuse) {
            Some(_) => render_state.ibl_diffuse,
            None => self.fallback_sky,
        };
        let stale = self.probes.as_ref().is_none_or(|probes| {
            probes.grid != grid
                || probes.rays_per_probe != rays_per_probe
                || probes.probes_per_frame != probes_per_frame
        }) || self.bound_sky != Some(sky_handle);
        if stale {
            let sky = renderer
                .get_texture_view(sky_handle)
                .ok_or("PassDdgi: IBL irradiance missing")?;
            self.create_probes(renderer, grid, rays_per_probe, probes_per_frame, &sky)?;
            self.bound_sky = Some(sky_handle);
        }
        let probes = self
            .probes
            .as_ref()
            .expect("PassDdgi: probes created above");

        let light_count = self.lights.borrow().params.grid[3] as u32;
        let sun = render_state.lights[0];
        let params = GpuDdgiParams {
            grid_min: grid.min.to_array(),
            max_distance: grid.max_distance(),
            cell_size: grid.cell.to_array(),
            intensity: settings.intensity,
            grid_dims: grid.dims,
            probe_count: grid.probe_count(),
            depth_texel_size: {
                let (width, height) = grid.depth_atlas_size();
                [1.0 / width as f32, 1.0 / height as f32]
            },
            enabled: if settings.debug_indirect_only { 2 } else { 1 },
            frame: self.frame_index,
            probe_offset: self.probe_offset,
            probes_this_frame: probes_per_frame,
            rays_per_probe,
            light_count,
            sun_direction: Vector3f::from_array(sun.direction)
                .normalize_or(Vector3f::NEG_Y)
                .to_array(),
            blend_speed: settings.blend_speed,
            sun_radiance: (Vector3f::from_array(sun.color) * sun.intensity).to_array(),
            sky_intensity: render_state.ibl_intensity,
            rotation: random_rotation(self.frame_index),
        };
        renderer.get_queue().write_buffer(
            &self.frame.borrow().params,
            0,
            bytemuck::bytes_of(&params),
        );

        let pipelines = self
            .pipelines
            .as_ref()
            .expect("PassDdgi: init() before draw()");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass_ddgi"),
            timestamp_writes: None,
        });
        for (pipeline, group) in [
            (&pipelines.trace, &probes.trace_group),
            (&pipelines.color, &probes.color_group),
            (&pipelines.depth, &probes.depth_group),
        ] {
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(probes_per_frame, 1, 1);
        }
        drop(pass);

        self.frame_index = self.frame_index.wrapping_add(1);
        self.probe_offset = (self.probe_offset + probes_per_frame) % grid.probe_count();
        Ok(())
    }
}
