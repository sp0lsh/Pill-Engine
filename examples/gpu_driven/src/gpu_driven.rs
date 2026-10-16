//! bgfx's GPU-driven pipeline on a wgpu device: the occluders' depth → a Hi-Z max pyramid →
//! per-instance frustum and occlusion culling → stream compaction into per-prop indirect draws.
//! The CPU only writes the camera; which instances are drawn is decided on the GPU.

use crate::scene::{Instance, Scene, CUBE_INDICES, CUBE_VERTICES};
use pill_engine::game::wgpu;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

// Generated from res/shaders/gdr_*.hlsl by pill_assets
const VERTEX_SHADER: &str = include_str!("../res/shaders/gdr_vertex.wgsl");
const FRAGMENT_SHADER: &str = include_str!("../res/shaders/gdr_fragment.wgsl");
const COPY_Z_SHADER: &str = include_str!("../res/shaders/gdr_copy_z_compute.wgsl");
const DOWNSCALE_SHADER: &str = include_str!("../res/shaders/gdr_downscale_compute.wgsl");
const OCCLUDE_SHADER: &str = include_str!("../res/shaders/gdr_occlude_compute.wgsl");
const COMPACT_SHADER: &str = include_str!("../res/shaders/gdr_compact_compute.wgsl");

/// Instance capacity of the single-workgroup compaction (256 threads × 8).
pub const MAX_INSTANCES: usize = 2048;
/// Colors in `gdr_fragment.hlsl`'s uniform array.
pub const MAX_PROPS: usize = 8;
/// Indirect draw arguments per prop: index count, instance count, first index, base vertex,
/// first instance.
pub const DRAW_ARGS: usize = 5;
/// Stride of the per-draw dynamic uniform (WebGPU's minimum uniform offset alignment).
const DRAW_UNIFORM_STRIDE: u64 = 256;
const HIZ_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct View {
    pub view_projection: [[f32; 4]; 4],
    pub light_direction: [f32; 4],
    pub camera_position: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Cull {
    view_projection: [[f32; 4]; 4],
    hiz_size: [f32; 2],
    mip_count: u32,
    instance_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Compact {
    instance_count: u32,
    prop_count: u32,
    index_count: u32,
    _padding: u32,
}

/// The largest power of two ≤ each side of `size` (bgfx's Hi-Z size), at least 1.
pub fn hiz_size((width, height): (u32, u32)) -> (u32, u32) {
    let floor_pow2 = |v: u32| if v == 0 { 1 } else { 1 << (31 - v.leading_zeros()) };
    (floor_pow2(width), floor_pow2(height))
}

fn mip_count((width, height): (u32, u32)) -> u32 {
    32 - width.max(height).leading_zeros()
}

fn storage(device: &wgpu::Device, label: &str, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE | usage,
        mapped_at_creation: false,
    })
}

fn uniform<T: bytemuck::Pod>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: std::mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Layout entries in binding order; `kinds` per binding: 'u' uniform, 'd' dynamic uniform,
/// 'r' read-only storage, 'w' read-write storage, 't' unfilterable float texture (also the depth),
/// 's' R32F write-only storage texture.
fn layout(device: &wgpu::Device, label: &str, visibility: wgpu::ShaderStages, kinds: &str) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = kinds
        .chars()
        .enumerate()
        .map(|(binding, kind)| {
            let buffer = |ty, has_dynamic_offset| wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size: None,
            };
            let ty = match kind {
                'u' => buffer(wgpu::BufferBindingType::Uniform, false),
                'd' => buffer(wgpu::BufferBindingType::Uniform, true),
                'r' => buffer(wgpu::BufferBindingType::Storage { read_only: true }, false),
                'w' => buffer(wgpu::BufferBindingType::Storage { read_only: false }, false),
                't' => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                's' => wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                other => panic!("unknown binding kind {other}"),
            };
            wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility,
                ty,
                count: None,
            }
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn bind_group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, resources: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

fn compute_pipeline(device: &wgpu::Device, label: &str, source: &str, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Bind group layouts of the Hi-Z kernels.
struct HiZLayouts {
    copy: wgpu::BindGroupLayout,
    downscale: wgpu::BindGroupLayout,
    occlude: wgpu::BindGroupLayout,
}

/// The Hi-Z resources of one render size.
struct HiZ {
    size: (u32, u32),
    mip_count: u32,
    depth_view: wgpu::TextureView,
    /// (level size, bind group): mip 0 from the depth, then each level from the previous
    levels: Vec<((u32, u32), wgpu::BindGroup)>,
    occlude_group: wgpu::BindGroup,
}

impl HiZ {
    /// The occluder depth target and Hi-Z pyramid at `size`; `buffers` from
    /// `GpuDriven::occlude_buffers`.
    fn new(device: &wgpu::Device, layouts: &HiZLayouts, buffers: &[&wgpu::Buffer; 5], size: (u32, u32)) -> Self {
        let mip_count = mip_count(size);
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gdr_occluder_depth"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        let hiz = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gdr_hiz"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip_view = |mip: u32| {
            hiz.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let level_size = |mip: u32| ((size.0 >> mip).max(1), (size.1 >> mip).max(1));
        let mut levels = vec![(
            size,
            bind_group(
                device,
                "gdr_copy_z",
                &layouts.copy,
                &[
                    wgpu::BindingResource::TextureView(&depth_view),
                    wgpu::BindingResource::TextureView(&mip_view(0)),
                ],
            ),
        )];
        for mip in 1..mip_count {
            levels.push((
                level_size(mip),
                bind_group(
                    device,
                    "gdr_downscale",
                    &layouts.downscale,
                    &[
                        wgpu::BindingResource::TextureView(&mip_view(mip - 1)),
                        wgpu::BindingResource::TextureView(&mip_view(mip)),
                    ],
                ),
            ));
        }
        let occlude_group = bind_group(
            device,
            "gdr_occlude",
            &layouts.occlude,
            &[
                buffers[0].as_entire_binding(),
                wgpu::BindingResource::TextureView(&hiz.create_view(&Default::default())),
                buffers[1].as_entire_binding(),
                buffers[2].as_entire_binding(),
                buffers[3].as_entire_binding(),
                buffers[4].as_entire_binding(),
            ],
        );
        Self {
            size,
            mip_count,
            depth_view,
            levels,
            occlude_group,
        }
    }
}


pub struct GpuDriven {
    prop_count: u32,
    instance_count: u32,
    /// (first, count) of each prop's instances, and whether it is an occluder
    ranges: Vec<(u32, u32, bool)>,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    instances: wgpu::Buffer,
    bounds: wgpu::Buffer,
    predicates: wgpu::Buffer,
    counts: wgpu::Buffer,
    draws: wgpu::Buffer,
    occluder_view: wgpu::Buffer,
    main_view: wgpu::Buffer,
    cull: wgpu::Buffer,
    /// Main pass instances: compacted (indirect) and full list (fallback)
    culled_group: wgpu::BindGroup,
    full_group: wgpu::BindGroup,
    occluder_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
    occluder_pipeline: wgpu::RenderPipeline,
    main_pipeline: wgpu::RenderPipeline,
    copy_pipeline: wgpu::ComputePipeline,
    downscale_pipeline: wgpu::ComputePipeline,
    occlude_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    hiz_layouts: HiZLayouts,
    compact_group: wgpu::BindGroup,
    hiz: HiZ,
}

impl GpuDriven {
    /// Uploads `scene`; the main pass draws into `color_format` / `depth_format` targets, the
    /// culling runs at `hiz_size`.
    pub fn new(
        device: &wgpu::Device,
        scene: &Scene,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        hiz_size: (u32, u32),
    ) -> Self {
        assert!(scene.instances.len() <= MAX_INSTANCES && scene.props.len() <= MAX_PROPS);
        let init = |label: &str, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let vertex_buffer = init("gdr_vertices", bytemuck::cast_slice(&CUBE_VERTICES), wgpu::BufferUsages::VERTEX);
        let index_buffer = init("gdr_indices", bytemuck::cast_slice(&CUBE_INDICES), wgpu::BufferUsages::INDEX);
        let instances = init("gdr_instances", bytemuck::cast_slice(&scene.instances), wgpu::BufferUsages::STORAGE);
        let bounds = init("gdr_bounds", bytemuck::cast_slice(&scene.bounds), wgpu::BufferUsages::STORAGE);
        let mut colors = [[0.0f32; 4]; MAX_PROPS];
        for (color, prop) in colors.iter_mut().zip(&scene.props) {
            *color = [prop.color[0], prop.color[1], prop.color[2], 1.0];
        }
        let colors = init("gdr_colors", bytemuck::cast_slice(&colors), wgpu::BufferUsages::UNIFORM);
        let zero_bases = init("gdr_zero_bases", &[0u8; MAX_PROPS * 4], wgpu::BufferUsages::STORAGE);
        let mut draw_uniforms = vec![0u8; DRAW_UNIFORM_STRIDE as usize * MAX_PROPS];
        for prop in 0..MAX_PROPS {
            let offset = prop * DRAW_UNIFORM_STRIDE as usize;
            draw_uniforms[offset..offset + 4].copy_from_slice(&(prop as u32).to_le_bytes());
        }
        let draw_uniforms = init("gdr_draw", &draw_uniforms, wgpu::BufferUsages::UNIFORM);
        let compact_config = Compact {
            instance_count: scene.instances.len() as u32,
            prop_count: scene.props.len() as u32,
            index_count: CUBE_INDICES.len() as u32,
            _padding: 0,
        };
        let compact_config = init("gdr_compact", bytemuck::bytes_of(&compact_config), wgpu::BufferUsages::UNIFORM);

        let predicates = storage(device, "gdr_predicates", MAX_INSTANCES * 4, wgpu::BufferUsages::empty());
        let counts = storage(device, "gdr_counts", MAX_PROPS * 4, wgpu::BufferUsages::empty());
        let culled = storage(device, "gdr_culled", MAX_INSTANCES * std::mem::size_of::<Instance>(), wgpu::BufferUsages::empty());
        let draws = storage(
            device,
            "gdr_draws",
            MAX_PROPS * DRAW_ARGS * 4,
            wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
        );
        let bases = storage(device, "gdr_bases", MAX_PROPS * 4, wgpu::BufferUsages::empty());
        let occluder_view = uniform::<View>(device, "gdr_occluder_view");
        let main_view = uniform::<View>(device, "gdr_main_view");
        let cull = uniform::<Cull>(device, "gdr_cull");

        // Render pipelines
        let render_layout = layout(device, "gdr_render", wgpu::ShaderStages::VERTEX_FRAGMENT, "urru");
        let draw_layout = layout(device, "gdr_draw", wgpu::ShaderStages::VERTEX, "d");
        let module = |label, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
        };
        let (vertex_module, fragment_module) = (
            module("gdr_vertex", VERTEX_SHADER),
            module("gdr_fragment", FRAGMENT_SHADER),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gdr_render"),
            bind_group_layouts: &[&render_layout, &draw_layout],
            push_constant_ranges: &[],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };
        let render_pipeline = |label, format, fragment: Option<wgpu::ColorTargetState>| {
            let targets = [fragment];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[vertex_layout.clone()],
                },
                fragment: targets[0].is_some().then(|| wgpu::FragmentState {
                    module: &fragment_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };
        let occluder_pipeline = render_pipeline("gdr_occluder", HIZ_DEPTH_FORMAT, None);
        let main_pipeline = render_pipeline(
            "gdr_main",
            depth_format,
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        );
        let render_group = |label, view: &wgpu::Buffer, source: &wgpu::Buffer, bases: &wgpu::Buffer| {
            bind_group(
                device,
                label,
                &render_layout,
                &[
                    view.as_entire_binding(),
                    source.as_entire_binding(),
                    bases.as_entire_binding(),
                    colors.as_entire_binding(),
                ],
            )
        };
        let culled_group = render_group("gdr_culled", &main_view, &culled, &bases);
        let full_group = render_group("gdr_full", &main_view, &instances, &zero_bases);
        let occluder_group = render_group("gdr_occluder", &occluder_view, &instances, &zero_bases);
        let draw_group = bind_group(
            device,
            "gdr_draw",
            &draw_layout,
            &[wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &draw_uniforms,
                offset: 0,
                size: NonZeroU64::new(16),
            })],
        );

        // Compute pipelines
        let compute = wgpu::ShaderStages::COMPUTE;
        let copy_layout = layout(device, "gdr_copy_z", compute, "ts");
        let downscale_layout = layout(device, "gdr_downscale", compute, "ts");
        let occlude_layout = layout(device, "gdr_occlude", compute, "utrrww");
        let compact_layout = layout(device, "gdr_compact", compute, "urrwwww");
        let copy_pipeline = compute_pipeline(device, "gdr_copy_z", COPY_Z_SHADER, &copy_layout);
        let downscale_pipeline = compute_pipeline(device, "gdr_downscale", DOWNSCALE_SHADER, &downscale_layout);
        let occlude_pipeline = compute_pipeline(device, "gdr_occlude", OCCLUDE_SHADER, &occlude_layout);
        let compact_pipeline = compute_pipeline(device, "gdr_compact", COMPACT_SHADER, &compact_layout);
        let compact_group = bind_group(
            device,
            "gdr_compact",
            &compact_layout,
            &[
                compact_config.as_entire_binding(),
                instances.as_entire_binding(),
                predicates.as_entire_binding(),
                counts.as_entire_binding(),
                culled.as_entire_binding(),
                draws.as_entire_binding(),
                bases.as_entire_binding(),
            ],
        );

        let hiz_layouts = HiZLayouts {
            copy: copy_layout,
            downscale: downscale_layout,
            occlude: occlude_layout,
        };
        let hiz = HiZ::new(
            device,
            &hiz_layouts,
            &[&cull, &bounds, &instances, &predicates, &counts],
            hiz_size,
        );
        Self {
            prop_count: scene.props.len() as u32,
            instance_count: scene.instances.len() as u32,
            ranges: scene.props.iter().map(|prop| (prop.first, prop.count, prop.occluder)).collect(),
            vertex_buffer,
            index_buffer,
            instances,
            bounds,
            predicates,
            counts,
            draws,
            occluder_view,
            main_view,
            cull,
            culled_group,
            full_group,
            occluder_group,
            draw_group,
            occluder_pipeline,
            main_pipeline,
            copy_pipeline,
            downscale_pipeline,
            occlude_pipeline,
            compact_pipeline,
            hiz_layouts,
            compact_group,
            hiz,
        }
    }

    /// Hi-Z size of the culling.
    pub fn hiz_size(&self) -> (u32, u32) {
        self.hiz.size
    }

    /// Rebuilds the Hi-Z targets at `size`.
    pub fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        self.hiz = HiZ::new(device, &self.hiz_layouts, &self.occlude_buffers(), size);
    }

    /// cull, bounds, instances, predicates, counts: the buffers of the occlusion kernel.
    fn occlude_buffers(&self) -> [&wgpu::Buffer; 5] {
        [&self.cull, &self.bounds, &self.instances, &self.predicates, &self.counts]
    }

    /// Records the culling: the occluders' depth from `occluder_view` (a projection with the Hi-Z
    /// aspect), the Hi-Z pyramid, the occlusion test and the compaction into `draws()`.
    pub fn cull(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, occluder_view: &View) {
        queue.write_buffer(&self.occluder_view, 0, bytemuck::bytes_of(occluder_view));
        queue.write_buffer(
            &self.cull,
            0,
            bytemuck::bytes_of(&Cull {
                view_projection: occluder_view.view_projection,
                hiz_size: [self.hiz.size.0 as f32, self.hiz.size.1 as f32],
                mip_count: self.hiz.mip_count,
                instance_count: self.instance_count,
            }),
        );
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gdr_occluders"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.hiz.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.occluder_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.set_bind_group(0, &self.occluder_group, &[]);
            pass.set_bind_group(1, &self.draw_group, &[0]);
            for &(first, count, occluder) in &self.ranges {
                if occluder {
                    pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, first..first + count);
                }
            }
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gdr_cull"),
            timestamp_writes: None,
        });
        for (mip, ((width, height), group)) in self.hiz.levels.iter().enumerate() {
            pass.set_pipeline(if mip == 0 {
                &self.copy_pipeline
            } else {
                &self.downscale_pipeline
            });
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        pass.set_pipeline(&self.occlude_pipeline);
        pass.set_bind_group(0, &self.hiz.occlude_group, &[]);
        pass.dispatch_workgroups(self.instance_count.div_ceil(64), 1, 1);
        pass.set_pipeline(&self.compact_pipeline);
        pass.set_bind_group(0, &self.compact_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Draws the scene from `view` into `pass`: the culled instances with one indirect draw per
    /// prop, or (`indirect` off) every instance with direct draws.
    pub fn draw(&self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass, view: &View, indirect: bool) {
        queue.write_buffer(&self.main_view, 0, bytemuck::bytes_of(view));
        pass.set_pipeline(&self.main_pipeline);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(
            0,
            if indirect {
                &self.culled_group
            } else {
                &self.full_group
            },
            &[],
        );
        for (prop, &(first, count, _)) in self.ranges.iter().enumerate() {
            pass.set_bind_group(1, &self.draw_group, &[prop as u32 * DRAW_UNIFORM_STRIDE as u32]);
            if indirect {
                pass.draw_indexed_indirect(&self.draws, (prop * DRAW_ARGS * 4) as u64);
            } else {
                pass.draw_indexed(0..CUBE_INDICES.len() as u32, 0, first..first + count);
            }
        }
    }

    /// The indirect draw arguments (`DRAW_ARGS` u32 per prop), for read-back.
    pub fn draws(&self) -> &wgpu::Buffer {
        &self.draws
    }

    pub fn prop_count(&self) -> u32 {
        self.prop_count
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    #[test]
    fn hiz_size_largest_powers_of_two_below() {
        assert_eq!(hiz_size((1280, 720)), (1024, 512));
        assert_eq!(hiz_size((1024, 1)), (1024, 1));
        assert_eq!(mip_count((1024, 512)), 11);
    }

    fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("device")
    }

    /// Culls `scene` from `eye` looking at `target` (60°, 2:1, the example's ranges); returns the
    /// per-prop instance counts of the indirect draws.
    fn drawn(scene: &Scene, eye: Vec3, target: Vec3) -> Vec<u32> {
        let (device, queue) = device();
        let gpu_driven = GpuDriven::new(
            &device,
            scene,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Depth32Float,
            (512, 256),
        );
        let projection = Mat4::perspective_rh(60f32.to_radians(), 2.0, 0.1, 500.0);
        let view = View {
            view_projection: (projection * Mat4::look_at_rh(eye, target, Vec3::Y)).to_cols_array_2d(),
            light_direction: [0.0, -1.0, 0.0, 0.0],
            camera_position: eye.extend(1.0).to_array(),
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        gpu_driven.cull(&queue, &mut encoder, &view);
        let size = (gpu_driven.prop_count() as usize * DRAW_ARGS * 4) as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(gpu_driven.draws(), 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let args: Vec<u32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
        args.chunks(DRAW_ARGS)
            .map(|draw| {
                assert_eq!(draw[0], CUBE_INDICES.len() as u32);
                draw[1]
            })
            .collect()
    }

    #[test]
    fn cull_looking_away_draws_nothing() {
        let counts = drawn(&Scene::new(1), Vec3::new(0.0, 5.0, 200.0), Vec3::new(0.0, 5.0, 400.0));
        assert_eq!(counts.iter().sum::<u32>(), 0, "{counts:?}");
    }

    #[test]
    fn cull_overview_from_above_draws_most_props() {
        let scene = Scene::new(1);
        let counts = drawn(&scene, Vec3::new(0.0, 150.0, 1.0), Vec3::ZERO);
        // Seen from above the walls hide little: everything is in view
        assert_eq!(counts[0], 1, "ground");
        assert_eq!(counts[1], 25, "walls");
        assert!(counts[2] + counts[3] > 450, "{counts:?}");
    }

    #[test]
    fn cull_behind_a_wall_hides_most_boxes() {
        // A 1 × 1 km wall right in front of the camera hides everything behind it
        let mut scene = Scene::new(1);
        let wall = Mat4::from_scale_rotation_translation(
            Vec3::new(1000.0, 1000.0, 0.1),
            glam::Quat::IDENTITY,
            Vec3::new(0.0, 5.0, 60.0),
        );
        let first_wall = scene.props[1].first as usize;
        scene.instances[first_wall].world = wall.to_cols_array_2d();
        scene.bounds[first_wall] = crate::scene::cube_bounds(&wall);
        let counts = drawn(&scene, Vec3::new(0.0, 5.0, 62.0), Vec3::new(0.0, 5.0, 0.0));
        assert!(counts[2] + counts[3] < 10, "{counts:?}");
        assert!(counts[1] >= 1, "the near wall itself is drawn");
    }
}
