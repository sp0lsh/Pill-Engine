//! Draws the camera video over the whole frame (web build only): each frame is copied from the
//! `<video>` into the pass's texture (`copy_external_image_to_texture`), then drawn into the HDR
//! target scaled to cover the screen and mirrored. The engine tonemap is `Tonemap::None`, so it
//! only sRGB-encodes the feed.

use crate::camera;
use crate::game::cover_uv_scale;
use pill_engine::game::*;

// Generated from res/shaders/*.hlsl by pill_assets
const VERTEX_SHADER: &str = include_str!("../res/shaders/video_vertex.wgsl");
const FRAGMENT_SHADER: &str = include_str!("../res/shaders/video_fragment.wgsl");
const VIDEO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VideoParams {
    uv_scale: [f32; 2],
    _padding: [f32; 2],
}

struct VideoTexture {
    texture: wgpu::Texture,
    size: (u32, u32),
    bind_group: Option<wgpu::BindGroup>,
}

struct CameraState {
    pipeline: Pipeline,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    video: Option<VideoTexture>,
}

pub struct CameraPass {
    state: Option<CameraState>,
}

impl CameraPass {
    pub fn new() -> Self {
        Self { state: None }
    }
}

fn layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
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
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

impl Pass for CameraPass {
    fn get_label(&self) -> &str {
        "camera_pass"
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let hdr_format = renderer.get_hdr_format();
        let pipeline = renderer.create_pipeline(PipelineDesc {
            label: Some("camera_pass"),
            vs: ShaderDesc {
                source: VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            fragment_constants: &[],
            vertex_buffers: &[],
            bind_groups: vec![layout_entries()],
            targets: &[Some(wgpu::ColorTargetState {
                format: hdr_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState::default(),
        })?;
        let device = renderer.get_device();
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("camera_pass"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_pass_params"),
            size: std::mem::size_of::<VideoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.state = Some(CameraState {
            pipeline,
            sampler,
            params,
            video: None,
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
        let Some(video) = camera::video() else {
            return Ok(());
        };
        let state = self
            .state
            .as_mut()
            .expect("CameraPass: init() must run before draw()");
        let size = (video.video_width(), video.video_height());
        if state.video.as_ref().is_none_or(|texture| texture.size != size) {
            let texture = renderer
                .get_device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("camera_video"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: VIDEO_FORMAT,
                    // WebGPU's copyExternalImageToTexture needs RENDER_ATTACHMENT on the destination
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
            state.video = Some(VideoTexture {
                texture,
                size,
                bind_group: None,
            });
        }
        let video_texture = state.video.as_mut().expect("video texture created above");
        renderer.get_queue().copy_external_image_to_texture(
            &wgpu::CopyExternalImageSourceInfo {
                source: wgpu::ExternalImageSource::HTMLVideoElement(video),
                origin: wgpu::Origin2d::ZERO,
                flip_y: false,
            },
            wgpu::CopyExternalImageDestInfo {
                texture: &video_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
                color_space: wgpu::PredefinedColorSpace::Srgb,
                premultiplied_alpha: false,
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );

        let (render_width, render_height) = renderer.get_render_size();
        let uv_scale = cover_uv_scale(
            Vector2f::new(size.0 as f32, size.1 as f32),
            Vector2f::new(render_width as f32, render_height as f32),
        );
        renderer.get_queue().write_buffer(
            &state.params,
            0,
            bytemuck::bytes_of(&VideoParams {
                uv_scale: uv_scale.to_array(),
                _padding: [0.0; 2],
            }),
        );

        if video_texture.bind_group.is_none() {
            let view = video_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let group = renderer
                .get_device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("camera_pass"),
                    layout: &state.pipeline.bind_group_layouts[0],
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&state.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: state.params.as_entire_binding(),
                        },
                    ],
                });
            video_texture.bind_group = Some(group);
        }
        let group = video_texture
            .bind_group
            .as_ref()
            .expect("bind group built above");

        let hdr_view = renderer
            .get_render_target_view(world.get_global::<RenderStateComponent>()?.hdr_target())
            .ok_or("CameraPass: HDR target missing")?;
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("camera_pass"),
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
        render_pass.set_pipeline(&state.pipeline.pipeline);
        render_pass.set_bind_group(0, group, &[]);
        render_pass.draw(0..3, 0..1);
        Ok(())
    }
}
