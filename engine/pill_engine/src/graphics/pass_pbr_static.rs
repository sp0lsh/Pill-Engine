use crate::{
    ecs::{CameraComponent, ComponentStorage, TransformComponent},
    graphics::{
        decompose_render_queue_key, BufferDesc, Pass, PillRenderer, PipelineV2, PipelineV2Desc,
        RendererMaterialHandle, RendererMeshHandle, RendererTextureHandle, ShaderDesc, WorldQuery,
    },
    renderer::resources::{RendererMaterial, RendererMesh},
};
use glam::{Mat3, Mat4, Quat, Vec3};
use pill_core::{PillSlotMapKey, Result};
use std::num::NonZeroU32;

/// Camera uniform layout: position (vec4) + view-projection matrix (mat4x4).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    position: [f32; 4],
    view_projection_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            position: [0.0; 4],
            view_projection_matrix: Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    /// Recomputes view and projection from ECS components; called once per frame before GPU upload.
    fn update_data(
        &mut self,
        camera_component: &CameraComponent,
        transform_component: &TransformComponent,
    ) {
        self.position = [
            transform_component.position.x,
            transform_component.position.y,
            transform_component.position.z,
            0.0,
        ];

        let eye = Vec3::new(
            transform_component.position.x,
            transform_component.position.y,
            transform_component.position.z,
        );
        let view = if let Some(t) = camera_component.look_at {
            Mat4::look_at_rh(eye, Vec3::new(t.x, t.y, t.z), Vec3::Y)
        } else {
            let roll_matrix = Mat3::from_rotation_z(transform_component.rotation.z.to_radians());
            let yaw_matrix = Mat3::from_rotation_y(transform_component.rotation.y.to_radians());
            let pitch_matrix = Mat3::from_rotation_x(transform_component.rotation.x.to_radians());
            let rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix;
            let direction = rotation_matrix * Vec3::Z;
            Mat4::look_to_rh(eye, direction, Vec3::Y)
        };

        let proj = Mat4::perspective_rh(
            camera_component.fov.to_radians(),
            camera_component.aspect.get_value(),
            camera_component.range.start,
            camera_component.range.end,
        );

        self.view_projection_matrix = (proj * view).to_cols_array_2d();
    }
}

/// Per-draw uniform: MVP and model matrix packed into a std140-aligned dynamic UBO.
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct PerDrawStd140 {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}
unsafe impl bytemuck::Zeroable for PerDrawStd140 {}
unsafe impl bytemuck::Pod for PerDrawStd140 {}

// Preallocated dynamic-UBO ring: 256-byte stride to satisfy min UBO offset alignment.
const MAX_EXPECTED_PER_DRAW_INSTANCES: usize = 100_000;
const UNIFORM_OFFSET_ALIGNMENT: usize = 256;
const PER_DRAW_STRIDE_BYTES: usize = ((std::mem::size_of::<PerDrawStd140>()
    + (UNIFORM_OFFSET_ALIGNMENT - 1))
    / UNIFORM_OFFSET_ALIGNMENT)
    * UNIFORM_OFFSET_ALIGNMENT;

pub const MATERIAL_BIND_GROUP_GLOBALS: usize = 0;
pub const MATERIAL_BIND_GROUP_TEXTURES: usize = 1;
pub const MATERIAL_BIND_GROUP_PARAMS: usize = 2;
pub const MATERIAL_BIND_GROUP_PERDRAW: usize = 3;

/// Visible entity ready for batching into a draw call.
pub(crate) struct VisiblePreDraw {
    pub(crate) material_handle: RendererMaterialHandle,
    pub(crate) mesh_handle: RendererMeshHandle,
    pub(crate) entity_index: u32,
    pub(crate) mvp: [[f32; 4]; 4],
    pub(crate) model: [[f32; 4]; 4],
}

/// Mesh batch within a material group: same mesh handle, consecutive per-draw entries in the ring.
pub(crate) struct MeshBatch {
    pub(crate) mesh_handle: RendererMeshHandle,
    pub(crate) instances: Vec<PerDrawStd140>,
    pub(crate) base_offset_u32: u32,
}

/// Draw group: one pipeline + one material, containing one or more mesh batches.
pub(crate) struct GroupCmd {
    pub(crate) pipeline: *const wgpu::RenderPipeline,
    pub(crate) material_handle: RendererMaterialHandle,
    pub(crate) batches: Vec<MeshBatch>,
}

/// GPU-side pass state initialized in `Pass::init`, read every `Pass::draw`.
struct PassPBRStaticState {
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    pipeline: PipelineV2,
    ibl_sampler: wgpu::Sampler,
    ibl_fallback_texture: wgpu::Texture,
    ibl_fallback_view: wgpu::TextureView,
    // Studio IBL textures loaded from embedded cooked assets.
    _ibl_irradiance_texture: wgpu::Texture,
    _ibl_prefilter_texture: wgpu::Texture,
    _ibl_brdf_lut_texture: wgpu::Texture,
    per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
}

/// Default PBR pass: full GGX shading with optional IBL, per-draw dynamic UBO ring.
/// No MeshDrawer — each entity is drawn with a direct indexed draw call.
pub struct PassPBRStatic {
    color_target: Option<RendererTextureHandle>,
    ibl_irradiance_tex: Option<RendererTextureHandle>,
    ibl_prefilter_tex: Option<RendererTextureHandle>,
    ibl_brdf_lut_tex: Option<RendererTextureHandle>,
    depth_texture: Option<RendererTextureHandle>,
    per_draw_stride: u64,
    per_draw_capacity: u64,
    visible_pre_draw_buffer: Vec<VisiblePreDraw>,
    groups_buffer: Vec<GroupCmd>,
    staging_buffer: Vec<u8>,
    state: Option<PassPBRStaticState>,
}

impl PassPBRStatic {
    /// Creates the pass with optional IBL textures; `Pass::init` must run before the first frame.
    pub fn new(
        color_target: Option<RendererTextureHandle>,
        ibl_irradiance_tex: Option<RendererTextureHandle>,
        ibl_prefilter_tex: Option<RendererTextureHandle>,
        ibl_brdf_lut_tex: Option<RendererTextureHandle>,
    ) -> Self {
        Self {
            color_target,
            ibl_irradiance_tex,
            ibl_prefilter_tex,
            ibl_brdf_lut_tex,
            depth_texture: None,
            per_draw_stride: PER_DRAW_STRIDE_BYTES as u64,
            per_draw_capacity: 0,
            visible_pre_draw_buffer: Vec::with_capacity(MAX_EXPECTED_PER_DRAW_INSTANCES),
            groups_buffer: Vec::with_capacity(2000),
            staging_buffer: Vec::with_capacity(
                MAX_EXPECTED_PER_DRAW_INSTANCES * PER_DRAW_STRIDE_BYTES,
            ),
            state: None,
        }
    }
}

/// Decode an RTEX header: returns (rgba_bytes, width, height).
fn decode_rtex(bytes: &[u8]) -> (&[u8], u32, u32, u32, u32) {
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let w = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if version == 3 || version == 4 {
        let mip_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        (&bytes[20..], w, h, mip_count, version)
    } else {
        (&bytes[16..], w, h, 1, version)
    }
}

/// Upload an RTEX blob as a 2D texture.
/// v1/v3 = Rgba8UnormSrgb (LDR); v4 = Rgba32Float HDR mip chain.
fn load_ibl_texture(
    renderer: &dyn PillRenderer,
    label: &str,
    bytes: &[u8],
    is_srgb: bool,
) -> (wgpu::Texture, wgpu::TextureView) {
    let (data, w, h, mip_count, version) = decode_rtex(bytes);
    let size = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    let (format, bytes_per_texel) = if version == 4 {
        (wgpu::TextureFormat::Rgba32Float, 16usize) // 4 × f32
    } else if is_srgb {
        (wgpu::TextureFormat::Rgba8UnormSrgb, 4usize)
    } else {
        (wgpu::TextureFormat::Rgba8Unorm, 4usize)
    };
    let tex = renderer
        .get_device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    let mut offset = 0usize;
    for mip in 0..mip_count {
        let mip_w = (w >> mip).max(1);
        let mip_h = (h >> mip).max(1);
        let mip_bytes = (mip_w * mip_h) as usize * bytes_per_texel;
        renderer.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data[offset..offset + mip_bytes],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(mip_w * bytes_per_texel as u32),
                rows_per_image: Some(mip_h),
            },
            wgpu::Extent3d {
                width: mip_w,
                height: mip_h,
                depth_or_array_layers: 1,
            },
        );
        offset += mip_bytes;
    }
    let view = tex.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(mip_count),
        ..Default::default()
    });
    (tex, view)
}

/// Returns the initialized pass state; panics in debug if `init` was not called.
fn get_state(pass: &mut PassPBRStatic) -> &mut PassPBRStaticState {
    debug_assert!(pass.state.is_some());
    // SAFETY: initialized once in init(), read every draw
    unsafe { pass.state.as_mut().unwrap_unchecked() }
}

impl Pass for PassPBRStatic {
    fn get_label(&self) -> &str {
        "pass_pbr_static"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let vertex_wgsl = include_str!("../../res/shaders/pbr_static_vertex.wgsl");

        let fragment_wgsl = include_str!("../../res/shaders/pbr_static_fragment.wgsl");

        let bind_groups: Vec<Vec<wgpu::BindGroupLayoutEntry>> = vec![
            // 0: globals (camera + IBL irradiance + prefilter + BRDF LUT)
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            // 1: material textures (base_color, normal, metallic_roughness)
            vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            // 2: material params
            vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            // 3: per-draw dynamic UBO (MVP + model)
            vec![wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(std::mem::size_of::<PerDrawStd140>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        ];

        let desc = PipelineV2Desc {
            label: Some("pass_pbr_static_pipeline"),
            vs: ShaderDesc {
                source: vertex_wgsl,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: fragment_wgsl,
                entry_func: "fs_main",
            },
            vertex_buffers: &[
                <crate::renderer::resources::RendererMesh as crate::renderer::resources::Vertex>::data_layout_descriptor(),
            ],
            bind_groups,
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
        };

        let pipeline = renderer.create_pipeline_v2(desc)?;

        let camera_buffer = {
            let device = renderer.get_device();
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass_pbr_static_camera_buffer"),
                size: std::mem::size_of::<CameraUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        self.per_draw_capacity = MAX_EXPECTED_PER_DRAW_INSTANCES as u64;
        let per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_static_per_draw_ring"),
            byte_size: self.per_draw_stride * self.per_draw_capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;
        let per_draw_bind_group = {
            let layout_ptr: *const wgpu::BindGroupLayout =
                &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_PERDRAW] as *const _;
            let device = renderer.get_device();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pass_pbr_static_per_draw_bind_group"),
                layout: unsafe { &*layout_ptr },
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &per_draw_buffer,
                        offset: 0,
                        size: Some(
                            std::num::NonZeroU64::new(std::mem::size_of::<PerDrawStd140>() as u64)
                                .unwrap(),
                        ),
                    }),
                }],
            })
        };

        // 1×1 black fallback texture used when IBL handles are None — zero IBL contribution.
        let ibl_fallback_texture = {
            let device = renderer.get_device();
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ibl_fallback"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        renderer.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &ibl_fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let ibl_fallback_view =
            ibl_fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let ibl_sampler = {
            let device = renderer.get_device();
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            })
        };

        // Load studio IBL textures from embedded cooked assets.
        static DIFFUSE_IBL: &[u8] =
            include_bytes!("../../res/textures/studio_diffuse_ibl.cooked_tex");
        static SPECULAR_IBL: &[u8] =
            include_bytes!("../../res/textures/studio_specular_ibl.cooked_tex");
        static BRDF_LUT: &[u8] = include_bytes!("../../res/textures/brdf_lut.cooked_tex");

        let (ibl_irradiance_texture, irradiance_view) =
            load_ibl_texture(renderer, "studio_diffuse_ibl", DIFFUSE_IBL, true);
        let (ibl_prefilter_texture, prefilter_view) =
            load_ibl_texture(renderer, "studio_specular_ibl", SPECULAR_IBL, true);
        let (ibl_brdf_lut_texture, brdf_lut_view) =
            load_ibl_texture(renderer, "brdf_lut", BRDF_LUT, false); // linear

        let prefilter_sampler = {
            let device = renderer.get_device();
            device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                lod_min_clamp: 0.0,
                lod_max_clamp: 16.0,
                ..Default::default()
            })
        };

        let globals_bind_group = {
            let layout_ptr: *const wgpu::BindGroupLayout =
                &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_GLOBALS] as *const _;
            let device = renderer.get_device();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pass_pbr_static_globals_bind_group"),
                layout: unsafe { &*layout_ptr },
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&irradiance_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&ibl_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&prefilter_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&prefilter_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&brdf_lut_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&ibl_sampler),
                    },
                ],
            })
        };

        self.depth_texture = Some(renderer.create_depth_texture("pass_pbr_static_depth")?);

        self.state = Some(PassPBRStaticState {
            camera_uniform: CameraUniform::new(),
            camera_buffer,
            globals_bind_group,
            pipeline,
            ibl_sampler,
            ibl_fallback_texture,
            ibl_fallback_view,
            _ibl_irradiance_texture: ibl_irradiance_texture,
            _ibl_prefilter_texture: ibl_prefilter_texture,
            _ibl_brdf_lut_texture: ibl_brdf_lut_texture,
            per_draw_buffer,
            per_draw_bind_group,
        });

        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame: &wgpu::SurfaceTexture,
        view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        // Read active camera and transform.
        let active_camera_index = world.active_camera.data().index as usize;
        let active_camera_component = world
            .camera_components
            .data
            .get(active_camera_index)
            .unwrap()
            .as_ref()
            .unwrap();
        let active_camera_transform = world
            .transform_components
            .data
            .get(active_camera_index)
            .unwrap()
            .as_ref()
            .unwrap();

        // Update camera uniform and write to GPU buffer.
        let vp_mat: Mat4 = {
            let state = get_state(self);
            state
                .camera_uniform
                .update_data(active_camera_component, active_camera_transform);
            renderer.get_queue().write_buffer(
                &state.camera_buffer,
                0,
                bytemuck::bytes_of(&state.camera_uniform),
            );
            Mat4::from_cols_array_2d(&state.camera_uniform.view_projection_matrix)
        };

        let _clear_color = active_camera_component.clear_color;

        // Build visible set from the render queue.
        self.visible_pre_draw_buffer.clear();
        for render_queue_item in world.render_queue.iter() {
            let key_fields = decompose_render_queue_key(render_queue_item.key);
            let mesh_handle = RendererMeshHandle::new(
                key_fields.mesh_index.into(),
                NonZeroU32::new(key_fields.mesh_version.into()).unwrap(),
            );
            let material_handle = RendererMaterialHandle::new(
                key_fields.material_index.into(),
                NonZeroU32::new(key_fields.material_version.into()).unwrap(),
            );

            let entity_index = render_queue_item.entity_index as usize;
            let transform = world
                .transform_components
                .data
                .get(entity_index)
                .unwrap()
                .as_ref()
                .unwrap();
            // Rotation stored in radians (game convention); match old shader's X*Y*Z quaternion order.
            let rotation = Quat::from_rotation_x(transform.rotation.x)
                * Quat::from_rotation_y(transform.rotation.y)
                * Quat::from_rotation_z(transform.rotation.z);
            let model_mat: Mat4 = Mat4::from_scale_rotation_translation(
                transform.scale,
                rotation,
                transform.position,
            );
            let mvp: [[f32; 4]; 4] = (vp_mat * model_mat).to_cols_array_2d();
            let model: [[f32; 4]; 4] = model_mat.to_cols_array_2d();

            self.visible_pre_draw_buffer.push(VisiblePreDraw {
                material_handle,
                mesh_handle,
                entity_index: render_queue_item.entity_index,
                mvp,
                model,
            });
        }

        // Sort visible set by material (Pipeline → Material → Mesh).
        self.visible_pre_draw_buffer.sort_by_key(|visible| {
            ((visible.material_handle.data().version.get() as u64) << 32)
                | (visible.material_handle.data().index as u64)
        });

        // Build group/batch command list.
        self.groups_buffer.clear();
        let pipeline_ptr: *const wgpu::RenderPipeline = {
            let state = get_state(self);
            &state.pipeline.pipeline as *const _
        };
        for visible in &self.visible_pre_draw_buffer {
            let need_new_group = self
                .groups_buffer
                .last()
                .map(|group| group.material_handle != visible.material_handle)
                .unwrap_or(true);
            if need_new_group {
                self.groups_buffer.push(GroupCmd {
                    pipeline: pipeline_ptr,
                    material_handle: visible.material_handle,
                    batches: Vec::new(),
                });
            }
            let group = self.groups_buffer.last_mut().unwrap();
            if let Some(batch) = group
                .batches
                .iter_mut()
                .find(|batch| batch.mesh_handle == visible.mesh_handle)
            {
                batch.instances.push(PerDrawStd140 {
                    mvp: visible.mvp,
                    model: visible.model,
                });
            } else {
                group.batches.push(MeshBatch {
                    mesh_handle: visible.mesh_handle,
                    instances: vec![PerDrawStd140 {
                        mvp: visible.mvp,
                        model: visible.model,
                    }],
                    base_offset_u32: 0,
                });
            }
        }

        // Upload per-draw data to the ring buffer.
        let needed: u64 = self
            .groups_buffer
            .iter()
            .map(|group| {
                group
                    .batches
                    .iter()
                    .map(|batch| batch.instances.len() as u64)
                    .sum::<u64>()
            })
            .sum();
        if self.per_draw_capacity < needed {
            log::error!(
                "PassPBRStatic: per-draw capacity exceeded (needed={}, capacity={})",
                needed,
                self.per_draw_capacity
            );
        }
        self.staging_buffer.clear();
        let mut next_offset_u32: u32 = 0;
        for group in self.groups_buffer.iter_mut() {
            for batch in group.batches.iter_mut() {
                batch.base_offset_u32 = next_offset_u32;
                for per_draw in &batch.instances {
                    self.staging_buffer
                        .extend_from_slice(bytemuck::bytes_of(per_draw));
                    let pad =
                        (self.per_draw_stride as usize) - std::mem::size_of::<PerDrawStd140>();
                    self.staging_buffer.extend(std::iter::repeat(0u8).take(pad));
                    next_offset_u32 = next_offset_u32.wrapping_add(self.per_draw_stride as u32);
                }
            }
        }
        {
            let state_ref: &PassPBRStaticState = unsafe { self.state.as_ref().unwrap_unchecked() };
            renderer
                .get_queue()
                .write_buffer(&state_ref.per_draw_buffer, 0, &self.staging_buffer);
        }

        let depth_view = renderer
            .get_render_target_view(self.depth_texture.unwrap())
            .unwrap();
        let color_view = self
            .color_target
            .and_then(|h| renderer.get_render_target_view(h))
            .unwrap_or(view);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_pbr_static_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
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

        // Record draw commands: bind pipeline → globals → material → per-draw per instance.
        let state_ref: &PassPBRStaticState = unsafe { self.state.as_ref().unwrap_unchecked() };
        for group in &self.groups_buffer {
            render_pass.set_pipeline(unsafe { &*group.pipeline });
            render_pass.set_bind_group(
                MATERIAL_BIND_GROUP_GLOBALS as u32,
                &state_ref.globals_bind_group,
                &[],
            );

            let mat = world
                .resources
                .get_resource::<RendererMaterial>(&group.material_handle)
                .unwrap();

            // Skip materials that don't have PBR-compatible bind groups.
            let (Some(textures_bg), Some(params_bg)) = (
                mat.textures_bind_group.as_ref(),
                mat.parameters_bind_group.as_ref(),
            ) else {
                continue;
            };
            render_pass.set_bind_group(MATERIAL_BIND_GROUP_TEXTURES as u32, textures_bg, &[]);
            render_pass.set_bind_group(MATERIAL_BIND_GROUP_PARAMS as u32, params_bg, &[]);

            for batch in &group.batches {
                let mesh = world
                    .resources
                    .get_resource::<RendererMesh>(&batch.mesh_handle)
                    .unwrap();
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                for instance_index in 0..batch.instances.len() {
                    let offset = batch
                        .base_offset_u32
                        .wrapping_add((instance_index as u32) * (self.per_draw_stride as u32));
                    render_pass.set_bind_group(
                        MATERIAL_BIND_GROUP_PERDRAW as u32,
                        &state_ref.per_draw_bind_group,
                        &[offset],
                    );
                    render_pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                }
            }
        }

        Ok(())
    }
}
