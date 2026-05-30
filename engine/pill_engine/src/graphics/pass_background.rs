use crate::{
    ecs::{CameraComponent, ComponentStorage, TransformComponent},
    graphics::{
        Pass, PillRenderer, PipelineV2, PipelineV2Desc, RendererTextureHandle, ShaderDesc,
        WorldQuery,
    },
};
use glam::{Mat3, Vec3};
use pill_core::{PillSlotMapKey, Result};

static VS: &str = include_str!("../../res/shaders/background_vertex.wgsl");
static FS: &str = include_str!("../../res/shaders/background_fragment.wgsl");

static EQUIRECT_BYTES: &[u8] = include_bytes!("../../res/textures/studio_equirect.cooked_tex");

pub struct PassBackground {
    hdr_target: RendererTextureHandle,
    state: Option<BgState>,
}

struct BgState {
    pipeline: PipelineV2,
    bind_group: wgpu::BindGroup,
    camera_buffer: wgpu::Buffer,
    _texture: wgpu::Texture,
    _sampler: wgpu::Sampler,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BgCameraUbo {
    right: [f32; 3],
    tan_half_fov: f32,
    up: [f32; 3],
    aspect: f32,
    fwd: [f32; 3],
    _pad: f32,
}

impl PassBackground {
    pub fn new(hdr_target: RendererTextureHandle) -> Self {
        Self {
            hdr_target,
            state: None,
        }
    }
}

fn decode_rtex(bytes: &[u8]) -> (&[u8], u32, u32, u32) {
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let w = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let h = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    (&bytes[16..], w, h, version)
}

impl Pass for PassBackground {
    fn get_label(&self) -> &str {
        "pass_background"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let bind_groups = vec![vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
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
        ]];

        let pipeline = renderer.create_pipeline_v2(PipelineV2Desc {
            label: Some("pass_background"),
            vs: ShaderDesc {
                source: VS,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: FS,
                entry_func: "fs_main",
            },
            vertex_buffers: &[],
            bind_groups,
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
        })?;

        let (raw, w, h, version) = decode_rtex(EQUIRECT_BYTES);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let (tex_format, bytes_per_channel) = if version == 2 {
            (wgpu::TextureFormat::Rgba32Float, 4u32)
        } else {
            (wgpu::TextureFormat::Rgba8UnormSrgb, 1u32)
        };
        let device = renderer.get_device();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("studio_equirect"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: tex_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        renderer.get_queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            raw,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * bytes_per_channel * w),
                rows_per_image: Some(h),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::Repeat, // equirect wraps in U
                address_mode_v: wgpu::AddressMode::ClampToEdge, // poles clamp in V
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        let camera_buffer = renderer
            .get_device()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass_background_camera"),
                size: std::mem::size_of::<BgCameraUbo>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        let bind_group = {
            let layout = &pipeline.bind_group_layouts[0];
            renderer
                .get_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("pass_background_bind_group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: camera_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
        };

        self.state = Some(BgState {
            pipeline,
            bind_group,
            camera_buffer,
            _texture: texture,
            _sampler: sampler,
        });
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame: &wgpu::SurfaceTexture,
        _view: &wgpu::TextureView,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let state = self.state.as_ref().unwrap();

        // Compute inv_view_proj from active camera — same matrices as PBR pass.
        let active_idx = world.active_camera.data().index as usize;
        let cam = world
            .camera_components
            .data
            .get(active_idx)
            .unwrap()
            .as_ref()
            .unwrap();
        let tfm = world
            .transform_components
            .data
            .get(active_idx)
            .unwrap()
            .as_ref()
            .unwrap();

        let eye = Vec3::new(tfm.position.x, tfm.position.y, tfm.position.z);
        let fwd = if let Some(t) = cam.look_at {
            (Vec3::new(t.x, t.y, t.z) - eye).normalize()
        } else {
            let roll = Mat3::from_rotation_z(tfm.rotation.z.to_radians());
            let yaw = Mat3::from_rotation_y(tfm.rotation.y.to_radians());
            let pitch = Mat3::from_rotation_x(tfm.rotation.x.to_radians());
            (yaw * pitch * roll) * Vec3::Z
        };
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);

        let ubo = BgCameraUbo {
            right: right.to_array(),
            tan_half_fov: (cam.fov.to_radians() / 2.0).tan(),
            up: up.to_array(),
            aspect: cam.aspect.get_value(),
            fwd: fwd.to_array(),
            _pad: 0.0,
        };
        renderer
            .get_queue()
            .write_buffer(&state.camera_buffer, 0, bytemuck::bytes_of(&ubo));

        let hdr_view = renderer.get_render_target_view(self.hdr_target).unwrap();
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_background_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&state.pipeline.pipeline);
        rp.set_bind_group(0, &state.bind_group, &[]);
        rp.draw(0..3, 0..1);
        Ok(())
    }
}
