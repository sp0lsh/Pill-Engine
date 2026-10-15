use crate::config::{
    DDGI_COLOR_COMPUTE_SHADER, DDGI_DEPTH_COMPUTE_SHADER, DDGI_RELOCATE_COMPUTE_SHADER,
    DDGI_RESET_COMPUTE_SHADER, DDGI_TRACE_COMPUTE_SHADER,
};
use crate::ecs::{MeshRenderingComponent, RenderStateComponent, TransformComponent};
use crate::graphics::ddgi::bvh::{Bvh, Triangle};
use crate::graphics::ddgi::{
    GpuDdgiCascade, GpuDdgiParams, ProbeGrid, CASCADE_UPDATE_EVERY, COLOR_RESOLUTION,
    DEPTH_RESOLUTION, MAX_CASCADES, MAX_RAYS_PER_PROBE, NO_SOURCE,
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
/// Bytes of one probe: L1 SH radiance and relocation offset — `DdgiProbe`.
const PROBE_BYTES: u64 = 80;
/// Frames of faster probe blending after a lighting change (e.g. the time of day), that blend
/// and the rays per probe meanwhile: more rays keep the faster blend from passing ray noise
/// through as jitter.
const CATCH_UP_FRAMES: u32 = 96;
/// Share of rays an alpha-tested triangle stops in the trace (its texture's holes, on average).
const CUTOUT_COVERAGE: f32 = 0.5;
/// Workgroups per row of the trace dispatch (`DISPATCH_WIDTH` in ddgi_trace_compute.hlsl).
const TRACE_DISPATCH_WIDTH: u32 = 32768;
const CATCH_UP_BLEND: f32 = 1.0 / 32.0;
const CATCH_UP_RAYS: u32 = 16;
/// Bytes of one radiance texel's history (its mean) — `ddgi_history` in the DDGI shaders.
const TEXEL_HISTORY_BYTES: u64 = 16;
/// Bytes of one depth texel's moment history (float2).
const DEPTH_HISTORY_BYTES: u64 = 8;

/// What the PBR passes read: the probe-grid uniform, the probes' SH and the depth atlas.
/// `generation` changes when the probe resources are recreated (rebuild bind groups).
pub(crate) struct DdgiFrame {
    pub(crate) params: wgpu::Buffer,
    pub(crate) probes: wgpu::Buffer,
    pub(crate) depth_view: wgpu::TextureView,
    pub(crate) generation: u64,
    /// Probes in `probes` this frame (0 with GI off).
    pub(crate) probe_count: u32,
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
            probe_count: 0,
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

/// Probe data of the cascades' grids, cascade after cascade. Kept while only the ray settings,
/// the sky or the BVH change, so the probes keep their history.
struct ProbeStorage {
    grids: Vec<ProbeGrid>,
    color_history: wgpu::Buffer,
    depth_history: wgpu::Buffer,
    /// Also published in `DdgiFrame`.
    probes: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    /// `[storage index, source]` of the probes to reset this frame (`ProbeGrid::exposed`), and
    /// the reset kernel's bind group.
    reset_list: wgpu::Buffer,
    reset_group: wgpu::BindGroup,
}

/// What the ray buffer and the bind groups were built for.
#[derive(Clone, Copy, PartialEq)]
struct BindingKey {
    rays_per_probe: u32,
    sky: PassTextureHandle,
    bvh_generation: u64,
    storage_generation: u64,
}

struct ProbeBindings {
    key: BindingKey,
    // The bind groups keep the ray buffer alive
    relocate_group: wgpu::BindGroup,
    trace_group: wgpu::BindGroup,
    color_group: wgpu::BindGroup,
    depth_group: wgpu::BindGroup,
}

struct Pipelines {
    reset: ComputePipeline,
    relocate: ComputePipeline,
    trace: ComputePipeline,
    color: ComputePipeline,
    depth: ComputePipeline,
    sampler: wgpu::Sampler,
}

/// Scene BVH on the GPU.
struct SceneBvh {
    nodes: wgpu::Buffer,
    triangles: wgpu::Buffer,
    bounds: (Vector3f, Vector3f),
}

/// Diffuse GI probes (see `ddgi/mod.rs`): rebuilds the BVH when the set of meshes changes (static
/// geometry), traces `Gi::rays_per_probe` rays from every probe each frame (the cost is spread over
/// rays, not probes: the pattern rotates per frame and the estimators average over frames) and
/// publishes the probes in `frame` for the PBR passes. Settings: `RenderStateComponent::gi`.
/// Compiles its pipelines on the first frame with GI on.
pub(crate) struct PassDdgi {
    frame: Rc<RefCell<DdgiFrame>>,
    lights: Rc<RefCell<LightsFrame>>,
    /// Engine default IBL irradiance, used for ray misses when the game's handle is invalid.
    fallback_sky: PassTextureHandle,
    pipelines: Option<Pipelines>,
    /// Signature of the scene the BVH was built from (also when it had no triangles).
    bvh_signature: Option<u64>,
    bvh: Option<SceneBvh>,
    bvh_generation: u64,
    storage: Option<ProbeStorage>,
    storage_generation: u64,
    bindings: Option<ProbeBindings>,
    frame_index: u32,
    /// Sun direction and radiance, sky intensity and light count of the last frame.
    lighting: [f32; 8],
    /// `bvh_generation` the probes were last relocated for.
    relocated_bvh_generation: u64,
    /// Frames left of the fast blend after a lighting change.
    catch_up_frames: u32,
    triangles: Vec<Triangle>,
}

/// Render key of a mesh entity the BVH contains: opaque and shadow-casting, as in
/// `gather_triangles`.
fn bvh_candidate(world: &WorldQuery<'_>, mesh: &MeshRenderingComponent) -> Option<u64> {
    let key = mesh.render_queue_key?;
    (mesh.cast_shadows && mesh.mesh_handle.is_some() && !is_translucent(world, key)).then_some(key)
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
            bvh_signature: None,
            bvh: None,
            bvh_generation: 0,
            storage: None,
            storage_generation: 0,
            bindings: None,
            frame_index: 0,
            lighting: [0.0; 8],
            relocated_bvh_generation: 0,
            catch_up_frames: 0,
            triangles: Vec::new(),
        }
    }

    /// Order-independent hash of the mesh entities the BVH contains (static geometry: their
    /// transforms are not part of it).
    fn scene_signature(world: &WorldQuery<'_>) -> u64 {
        let Ok(meshes) = world.query::<MeshRenderingComponent>() else {
            return 0;
        };
        let mut signature = 0u64;
        for (slot, mesh) in meshes.data.iter().enumerate() {
            let Some(key) = mesh.as_ref().and_then(|mesh| bvh_candidate(world, mesh)) else {
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
            if bvh_candidate(world, mesh_rendering).is_none() {
                continue;
            }
            let Some(data) = mesh_rendering
                .mesh_handle
                .as_ref()
                .and_then(|handle| world.resources.get_resource::<Mesh>(handle).ok())
                .and_then(Mesh::data)
            else {
                continue;
            };
            let material = mesh_rendering
                .material_handle
                .as_ref()
                .and_then(|handle| world.resources.get_resource::<Material>(handle).ok());
            // Alpha-tested (foliage cards): stops a share of the rays instead of all of them
            let coverage = match material {
                Some(material) if material.alpha_cutoff > 0.0 => CUTOUT_COVERAGE,
                _ => 1.0,
            };
            let albedo = material
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
                    coverage,
                });
            }
        }
    }

    /// Rebuilds the BVH when the scene signature changes; `None` when the scene has no triangles.
    fn update_bvh(
        &mut self,
        renderer: &mut dyn PillRenderer,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let signature = Self::scene_signature(world);
        if self.bvh_signature == Some(signature) {
            return Ok(());
        }
        self.bvh_signature = Some(signature);
        let start = web_time::Instant::now();
        self.gather_triangles(world);
        let bvh = Bvh::build(&self.triangles);
        let Some(bounds) = bvh.bounds() else {
            self.bvh = None;
            return Ok(());
        };
        info!(LogContext::Rendering =>
            "PassDdgi: BVH of {} triangles, {} nodes, depth {} in {:.1} ms",
            bvh.triangles.len(),
            bvh.nodes.len(),
            bvh.depth,
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
        });
        self.bvh_generation += 1;
        Ok(())
    }

    /// Probe SH, depth atlas and estimator histories for the cascades' `grids` (same dims),
    /// published in `frame`.
    fn create_storage(
        &mut self,
        renderer: &mut dyn PillRenderer,
        grids: Vec<ProbeGrid>,
    ) -> Result<()> {
        let probe_count = (grids[0].probe_count() as usize * grids.len()) as u64;
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
        // Cascades stacked down the atlas
        let (width, height) = grids[0].depth_atlas_size();
        let depth_view = depth_atlas(renderer.get_device(), width, height * grids.len() as u32);
        let reset_list = storage_buffer(renderer, "ddgi_reset_list", probe_count * 8)?;
        let pipelines = self
            .pipelines
            .as_ref()
            .expect("PassDdgi: pipelines before probe storage");
        let reset_group = renderer
            .get_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ddgi_reset"),
                layout: &pipelines.reset.bind_group_layouts[0],
                entries: &[&reset_list, &probes, &color_history, &depth_history]
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
        let mut frame = self.frame.borrow_mut();
        frame.probes = probes.clone();
        frame.depth_view = depth_view.clone();
        frame.generation += 1;
        // New buffers are zeroed: every probe starts without an estimate
        self.storage = Some(ProbeStorage {
            grids,
            color_history,
            depth_history,
            probes,
            depth_view,
            reset_list,
            reset_group,
        });
        self.storage_generation += 1;
        // Frame 0 starts the estimators from this frame's rays
        self.frame_index = 0;
        Ok(())
    }

    /// Ray buffer and the three bind groups for `key`.
    fn create_bindings(&mut self, renderer: &mut dyn PillRenderer, key: BindingKey) -> Result<()> {
        let storage = self
            .storage
            .as_ref()
            .expect("PassDdgi: probe storage before bindings");
        let probe_count = storage.grids[0].probe_count() * storage.grids.len() as u32;
        let rays = storage_buffer(
            renderer,
            "ddgi_rays",
            probe_count as u64 * key.rays_per_probe as u64 * RAY_BYTES,
        )?;
        let sky = renderer
            .get_texture_view(key.sky)
            .ok_or("PassDdgi: IBL irradiance missing")?;
        let pipelines = self
            .pipelines
            .as_ref()
            .expect("PassDdgi: pipelines before bindings");
        let bvh = self.bvh.as_ref().expect("PassDdgi: BVH before bindings");
        let storage = self
            .storage
            .as_ref()
            .expect("PassDdgi: probe storage before bindings");
        let frame = self.frame.borrow();
        let lights = self.lights.borrow();
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
        let relocate_group = group(
            &pipelines.relocate.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                bvh.nodes.as_entire_binding(),
                bvh.triangles.as_entire_binding(),
                storage.probes.as_entire_binding(),
            ],
        );
        let trace_group = group(
            &pipelines.trace.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                bvh.nodes.as_entire_binding(),
                bvh.triangles.as_entire_binding(),
                storage.probes.as_entire_binding(),
                wgpu::BindingResource::TextureView(&storage.depth_view),
                wgpu::BindingResource::Sampler(&pipelines.sampler),
                lights.buffers.lights.as_entire_binding(),
                wgpu::BindingResource::TextureView(&sky),
                rays.as_entire_binding(),
            ],
        );
        let color_group = group(
            &pipelines.color.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                rays.as_entire_binding(),
                storage.color_history.as_entire_binding(),
                storage.probes.as_entire_binding(),
            ],
        );
        let depth_group = group(
            &pipelines.depth.bind_group_layouts[0],
            &[
                frame.params.as_entire_binding(),
                rays.as_entire_binding(),
                storage.depth_history.as_entire_binding(),
                wgpu::BindingResource::TextureView(&storage.depth_view),
            ],
        );
        drop(frame);
        drop(lights);
        self.bindings = Some(ProbeBindings {
            key,
            relocate_group,
            trace_group,
            color_group,
            depth_group,
        });
        Ok(())
    }

    /// Writes the uniform with GI off, so the PBR passes skip the probes.
    fn disable(&self, renderer: &mut dyn PillRenderer) {
        let mut frame = self.frame.borrow_mut();
        frame.probe_count = 0;
        renderer.get_queue().write_buffer(
            &frame.params,
            0,
            bytemuck::bytes_of(&GpuDdgiParams::default()),
        );
    }
}

fn create_pipelines(renderer: &mut dyn PillRenderer) -> Result<Pipelines> {
    let uniform = uniform_entry_for(0, wgpu::ShaderStages::COMPUTE);
    let reset = renderer.create_compute_pipeline(ComputePipelineDesc {
        label: Some("ddgi_reset"),
        cs: ShaderDesc {
            source: DDGI_RESET_COMPUTE_SHADER,
            entry_func: "cs_main",
        },
        bind_groups: vec![vec![
            storage_entry(0, true),
            storage_entry(1, false),
            storage_entry(2, false),
            storage_entry(3, false),
        ]],
    })?;
    let relocate = renderer.create_compute_pipeline(ComputePipelineDesc {
        label: Some("ddgi_relocate"),
        cs: ShaderDesc {
            source: DDGI_RELOCATE_COMPUTE_SHADER,
            entry_func: "cs_main",
        },
        bind_groups: vec![vec![
            uniform,
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
        ]],
    })?;
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
    Ok(Pipelines {
        reset,
        relocate,
        trace,
        color,
        depth,
        sampler,
    })
}

impl Pass for PassDdgi {
    fn get_label(&self) -> &str {
        "pass_ddgi"
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
        let render_state = world.get_global::<RenderStateComponent>()?;
        let settings = render_state.gi;
        if !settings.enabled {
            self.disable(renderer);
            return Ok(());
        }
        if self.pipelines.is_none() {
            self.pipelines = Some(create_pipelines(renderer)?);
        }

        // Static geometry: rebuild the BVH only when the set of meshes changes
        self.update_bvh(renderer, world)?;
        let Some(bvh) = &self.bvh else {
            self.disable(renderer);
            return Ok(());
        };

        // Each resource follows only what it depends on: the probe data the grid, the ray
        // buffer and bind groups the ray settings, sky, BVH and probe data
        // A camera-following grid moves in whole cells and keeps the probes it still covers
        // Cascades: grids around the camera, `cascade_spacings` apart
        let (_, camera) = world.active_camera()?;
        let camera_position = camera.position;
        let grids: Vec<ProbeGrid> = if settings.follow_camera {
            (0..settings.cascades.clamp(1, MAX_CASCADES as u32) as usize)
                .map(|cascade| {
                    ProbeGrid::around(camera_position, settings.cascade_spacings[cascade])
                })
                .collect()
        } else {
            vec![ProbeGrid::fit(
                settings.bounds.unwrap_or(bvh.bounds),
                settings.probe_spacing,
            )]
        };
        let cascade_probes = grids[0].probe_count();
        let mut reset: Vec<[u32; 2]> = Vec::new();
        // Cascades with exposed probes this frame: they update this frame, so the depth pass
        // rewrites the copied probes' atlas tiles (the copy only fills their history)
        let mut reset_mask = 0u32;
        match &mut self.storage {
            Some(storage)
                if storage.grids.len() == grids.len()
                    && storage
                        .grids
                        .iter()
                        .zip(&grids)
                        .all(|(old, new)| old.same_storage(new)) =>
            {
                for (cascade, (old, new)) in storage.grids.iter_mut().zip(&grids).enumerate() {
                    if old.origin != new.origin {
                        // Exposed probes start from their nearest surviving neighbour; indices
                        // over all cascades
                        let offset = cascade as u32 * cascade_probes;
                        reset.extend(new.exposed(old.origin).into_iter().map(|[target, source]| {
                            [
                                target + offset,
                                if source == NO_SOURCE {
                                    NO_SOURCE
                                } else {
                                    source + offset
                                },
                            ]
                        }));
                        *old = *new;
                        reset_mask |= 1 << cascade;
                    }
                }
            }
            _ => self.create_storage(renderer, grids.clone())?,
        }
        let probe_count = cascade_probes * grids.len() as u32;
        let key = BindingKey {
            // The ray buffer holds the catch-up ray count too
            rays_per_probe: settings
                .rays_per_probe
                .max(CATCH_UP_RAYS)
                .clamp(1, MAX_RAYS_PER_PROBE),
            sky: if renderer.has_texture(render_state.ibl_diffuse) {
                render_state.ibl_diffuse
            } else {
                self.fallback_sky
            },
            bvh_generation: self.bvh_generation,
            storage_generation: self.storage_generation,
        };
        if self
            .bindings
            .as_ref()
            .is_none_or(|bindings| bindings.key != key)
        {
            self.create_bindings(renderer, key)?;
        }
        let storage = self
            .storage
            .as_mut()
            .expect("PassDdgi: probe storage created above");
        let bindings = self
            .bindings
            .as_ref()
            .expect("PassDdgi: bindings created above");

        if !reset.is_empty() {
            renderer
                .get_queue()
                .write_buffer(&storage.reset_list, 0, bytemuck::cast_slice(&reset));
        }
        let light_count = self.lights.borrow().params.grid[3] as u32;
        let sun = render_state.lights[0];
        let sun_direction = Vector3f::from_array(sun.direction).normalize_or(Vector3f::NEG_Y);
        let sun_radiance = Vector3f::from_array(sun.color) * sun.intensity;
        // A long blend averages few-ray noise but follows lighting changes slowly: when the
        // lighting changes (e.g. the time of day), blend fast for a few frames
        let lighting = [
            sun_direction.x,
            sun_direction.y,
            sun_direction.z,
            sun_radiance.x,
            sun_radiance.y,
            sun_radiance.z,
            render_state.ibl_intensity,
            light_count as f32,
        ];
        if lighting != self.lighting {
            self.lighting = lighting;
            self.catch_up_frames = CATCH_UP_FRAMES;
        }
        let catching_up = self.catch_up_frames > 0;
        let (blend_speed, rays) = if catching_up {
            self.catch_up_frames -= 1;
            (
                settings.blend_speed.max(CATCH_UP_BLEND),
                settings
                    .rays_per_probe
                    .max(CATCH_UP_RAYS)
                    .clamp(1, MAX_RAYS_PER_PROBE),
            )
        } else {
            (
                settings.blend_speed,
                settings.rays_per_probe.clamp(1, MAX_RAYS_PER_PROBE),
            )
        };
        // Probe relocation depends on the geometry: re-evaluate every probe when the BVH changes
        let relocate_all = (self.relocated_bvh_generation != self.bvh_generation) as u32;
        self.relocated_bvh_generation = self.bvh_generation;
        // Coarser cascades update every few frames
        let update_mask = (0..grids.len())
            .filter(|&cascade| {
                self.frame_index
                    .is_multiple_of(CASCADE_UPDATE_EVERY[cascade])
            })
            .fold(reset_mask, |mask, cascade| mask | 1 << cascade);
        let mut cascades = [GpuDdgiCascade::default(); MAX_CASCADES];
        for (cascade, grid) in grids.iter().enumerate() {
            cascades[cascade] = GpuDdgiCascade::new(grid);
        }
        let params = GpuDdgiParams {
            grid_dims: grids[0].dims,
            probe_count: cascade_probes,
            depth_texel_size: {
                let (width, height) = grids[0].depth_atlas_size();
                [
                    1.0 / width as f32,
                    1.0 / (height * grids.len() as u32) as f32,
                ]
            },
            // New probes are weighted in per probe (`DdgiProbe::sh[0].w`)
            // 3: probe-cell debug colours, 2: indirect light only, 1: on
            enabled: if settings.debug_probe_cells {
                3
            } else if settings.debug_indirect_only {
                2
            } else {
                1
            },
            frame: self.frame_index,
            relocate_all,
            cascade_count: grids.len() as u32,
            rays_per_probe: rays,
            light_count,
            sun_direction: sun_direction.to_array(),
            blend_speed,
            sun_radiance: sun_radiance.to_array(),
            sky_intensity: render_state.ibl_intensity,
            blend_origin: camera_position.to_array(),
            intensity: settings.intensity,
            update_mask,
            _padding: [0; 3],
            cascades,
        };
        renderer.get_queue().write_buffer(
            &{
                let mut frame = self.frame.borrow_mut();
                frame.probe_count = probe_count;
                frame
            }
            .params,
            0,
            bytemuck::bytes_of(&params),
        );

        let pipelines = self
            .pipelines
            .as_ref()
            .expect("PassDdgi: pipelines created above");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pass_ddgi"),
            timestamp_writes: None,
        });
        if !reset.is_empty() {
            pass.set_pipeline(&pipelines.reset.pipeline);
            pass.set_bind_group(0, &storage.reset_group, &[]);
            pass.dispatch_workgroups(reset.len() as u32, 1, 1);
        }
        // Probes not relocated yet (new or exposed), before they trace from their positions
        pass.set_pipeline(&pipelines.relocate.pipeline);
        pass.set_bind_group(0, &bindings.relocate_group, &[]);
        pass.dispatch_workgroups(probe_count.div_ceil(64), 1, 1);
        // Trace: one thread per ray of every probe (128 per workgroup, `THREADS` in the shader),
        // in rows of TRACE_DISPATCH_WIDTH workgroups: 4 cascades × 256 rays exceed the 65535
        // workgroups one dimension allows
        let ray_groups = (probe_count * rays).div_ceil(128);
        let trace_size = (
            ray_groups.min(TRACE_DISPATCH_WIDTH),
            ray_groups.div_ceil(TRACE_DISPATCH_WIDTH),
        );
        // Color and depth: one workgroup per probe, at most 4 × 14336
        for (pipeline, group, (width, height)) in [
            (&pipelines.trace, &bindings.trace_group, trace_size),
            (&pipelines.color, &bindings.color_group, (probe_count, 1)),
            (&pipelines.depth, &bindings.depth_group, (probe_count, 1)),
        ] {
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width, height, 1);
        }
        drop(pass);

        self.frame_index = self.frame_index.wrapping_add(1);
        Ok(())
    }
}
