use crate::game::{ACTIVE_SCENE, AUTO_POSES, SCENE_NAMES, ScenePose};
use crate::gaussian_cloud::{GaussianCloud, GaussianCloudSource};
use pill_engine::game::{Pass, PillSlotMapKey};
use pill_engine::internal::{PillRenderer, WorldQuery};
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
#[cfg(target_arch = "wasm32")]
use std::io::BufReader;
use wgpu_3dgs_viewer::core::{Gaussians, GaussiansSource, IterGaussian};

// ── GPU-side types ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplat {
    position: [f32; 3],
    _pad0: f32,
    scale: [f32; 3],
    _pad1: f32,
    color: [f32; 4],
    quat: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

#[derive(Clone, Default)]
pub struct DiagData {
    pub frame_ms:   f32,
    pub sort_ms:    f32,
    pub upload_ms:  f32,
    pub total_n:    usize,
    pub visible_n:  usize,
    pub sort_skip:  bool,
    pub scene_name: &'static str,
    pub scene_idx:  usize,
    pub memory_mb:  f32,
}

pub static DIAG: Mutex<DiagData> = Mutex::new(DiagData {
    frame_ms:   0.0,
    sort_ms:    0.0,
    upload_ms:  0.0,
    total_n:    0,
    visible_n:  0,
    sort_skip:  false,
    scene_name: "",
    scene_idx:  0,
    memory_mb:  0.0,
});

// ── Shader ────────────────────────────────────────────────────────────────────

const SHADER: &str = r#"
struct CameraUniform {
    view     : mat4x4<f32>,
    proj     : mat4x4<f32>,
    viewport : vec2<f32>,
    _pad     : vec2<f32>,
}

struct Splat {
    position : vec3<f32>,
    _pad0    : f32,
    scale    : vec3<f32>,
    _pad1    : f32,
    color    : vec4<f32>,
    quat     : vec4<f32>,
}

@group(0) @binding(0) var<uniform>         cam    : CameraUniform;
@group(0) @binding(1) var<storage, read>   splats : array<Splat>;

struct VOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       uv    : vec2<f32>,
    @location(1)       color : vec4<f32>,
}

fn quat_to_mat3(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y*y+z*z),  2.0*(x*y+z*w),      2.0*(x*z-y*w)),
        vec3<f32>(2.0*(x*y-z*w),         1.0-2.0*(x*x+z*z),  2.0*(y*z+x*w)),
        vec3<f32>(2.0*(x*z+y*w),         2.0*(y*z-x*w),      1.0-2.0*(x*x+y*y)),
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    let iid = vid / 6u;
    let cid = vid % 6u;
    var out: VOut;

    let s = splats[iid];

    let cp4  = cam.view * vec4<f32>(s.position, 1.0);
    let t    = cp4.xyz;

    if t.z > -0.1 {
        out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let R      = quat_to_mat3(s.quat);
    let M      = mat3x3<f32>(R[0]*s.scale.x, R[1]*s.scale.y, R[2]*s.scale.z);
    let sigma3 = M * transpose(M);

    let W = mat3x3<f32>(cam.view[0].xyz, cam.view[1].xyz, cam.view[2].xyz);

    let fx  = cam.proj[0][0] * cam.viewport.x * 0.5;
    let fy  = cam.proj[1][1] * cam.viewport.y * 0.5;
    let iz  = 1.0 / t.z;
    let iz2 = iz * iz;

    let J = mat3x2<f32>(
        vec2<f32>(-fx * iz,        0.0        ),
        vec2<f32>( 0.0,           -fy * iz    ),
        vec2<f32>( fx * t.x*iz2,  fy * t.y*iz2),
    );

    let T      = J * W;
    let sigma2 = (T * sigma3) * transpose(T);

    let a = sigma2[0][0];
    let b = sigma2[0][1];
    let c = sigma2[1][1];

    let mid = 0.5 * (a + c);
    let rr  = sqrt(max(0.0, mid*mid - (a*c - b*b)));
    let l1  = mid + rr;
    let l2  = mid - rr;

    var ev1: vec2<f32>;
    if abs(b) > 1e-4 { ev1 = normalize(vec2<f32>(b, l1 - a)); }
    else              { ev1 = vec2<f32>(1.0, 0.0); }
    let ev2 = vec2<f32>(-ev1.y, ev1.x);

    let to_ndc = vec2<f32>(2.0 / cam.viewport.x, 2.0 / cam.viewport.y);
    let ax1    = ev1 * sqrt(abs(l1)) * 3.0 * to_ndc;
    let ax2    = ev2 * sqrt(abs(l2)) * 3.0 * to_ndc;

    let dc = array<vec2<f32>, 6>(
        -ax1-ax2,  ax1-ax2, -ax1+ax2,
        -ax1+ax2,  ax1-ax2,  ax1+ax2,
    );
    let uvs = array<vec2<f32>, 6>(
        vec2(-1.,-1.), vec2(1.,-1.), vec2(-1., 1.),
        vec2(-1., 1.), vec2(1.,-1.), vec2( 1., 1.),
    );

    let clip = cam.proj * cp4;
    let ndc  = clip.xy / clip.w;

    out.pos   = vec4<f32>(ndc + dc[cid], clip.z / clip.w, 1.0);
    out.uv    = uvs[cid] * 3.0;
    out.color = s.color;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let alpha = in.color.a * exp(-0.5 * dot(in.uv, in.uv));
    if alpha < 1.0 / 255.0 { discard; }
    return vec4<f32>(in.color.rgb, alpha);
}
"#;

// ── Minimal wgpu-only egui renderer ──────────────────────────────────────────

struct LocalEguiRenderer {
    context:          egui::Context,
    renderer:         egui_wgpu::Renderer,
    pixels_per_point: f32,
}

impl LocalEguiRenderer {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions {
                depth_stencil_format: None,
                msaa_samples: 1,
                dithering: false,
                ..Default::default()
            },
        );
        Self { context: egui::Context::default(), renderer, pixels_per_point: 2.0 }
    }

    /// Renders `run_ui` into `view` (LoadOp::Load, compositing over existing content).
    fn draw(
        &mut self,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        size_px: [u32; 2],
        mut run_ui: impl FnMut(&egui::Context),
    ) {
        let ppp = self.pixels_per_point;
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size_px,
            pixels_per_point: ppp,
        };
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(size_px[0] as f32 / ppp, size_px[1] as f32 / ppp),
            )),
            ..Default::default()
        };

        let full_output = self.context.run(raw_input, |ctx| run_ui(ctx));

        let tris = self.context.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer.update_buffers(device, queue, encoder, &tris, &screen_desc);

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass_splat_egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
            });

            // SAFETY: rpass borrows encoder, which outlives this function.
            let rpass: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut rpass) };
            self.renderer.render(rpass, &tris, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

// ── Pipeline state ────────────────────────────────────────────────────────────

struct SplatState {
    pipeline:     wgpu::RenderPipeline,
    camera_buf:   wgpu::Buffer,
    splat_buf:    wgpu::Buffer,
    bind_group:   wgpu::BindGroup,
    splats:       Vec<GpuSplat>,
    visible:      Vec<GpuSplat>,
    last_pos:     Vec3,
    last_forward: Vec3,
}

// ── Public pass ───────────────────────────────────────────────────────────────

pub struct PassSplat {
    cloud_name:    String,
    current_idx:   usize,
    state:         Option<SplatState>,
    egui_renderer: Option<LocalEguiRenderer>,
    last_draw:     Option<Instant>,
}

impl PassSplat {
    pub fn new(cloud_name: &str) -> Self {
        Self {
            cloud_name:    cloud_name.to_string(),
            current_idx:   0,
            state:         None,
            egui_renderer: None,
            last_draw:     None,
        }
    }

    fn load_splats(source: &GaussianCloudSource) -> Result<(Vec<GpuSplat>, ScenePose)> {
        let gaussians = match source {
            GaussianCloudSource::Path(path) => {
                [GaussiansSource::Ply, GaussiansSource::Spz]
                    .into_iter()
                    .find_map(|fmt| Gaussians::read_from_file(path, fmt).ok())
                    .ok_or_else(|| anyhow!("cannot load {:?} as PLY or SPZ", path))?
            }
            #[cfg(target_arch = "wasm32")]
            GaussianCloudSource::Bytes(bytes) => {
                let mut r = BufReader::new(bytes.as_slice());
                Gaussians::read_from(&mut r, GaussiansSource::Ply)
                    .or_else(|_| {
                        let mut r = BufReader::new(bytes.as_slice());
                        Gaussians::read_from(&mut r, GaussiansSource::Spz)
                    })
                    .map_err(|e| anyhow!("cannot load gaussian bytes: {e}"))?
            }
        };

        let splats: Vec<GpuSplat> = gaussians
            .iter_gaussian()
            .map(|g| {
                let rgba = g.color.as_vec4() / 255.0;
                // Y-axis flip to match wgpu NDC; conjugate quaternion accordingly
                GpuSplat {
                    position: [g.pos.x, -g.pos.y, g.pos.z],
                    _pad0: 0.0,
                    scale: g.scale.to_array(),
                    _pad1: 0.0,
                    color: rgba.to_array(),
                    quat: [-g.rot.x, g.rot.y, -g.rot.z, g.rot.w],
                }
            })
            .collect();

        let n = splats.len() as f32;
        let centroid = splats.iter().fold(Vec3::ZERO, |s, g| s + Vec3::from(g.position)) / n;
        let radius = splats
            .iter()
            .map(|g| (Vec3::from(g.position) - centroid).length())
            .fold(0.0_f32, f32::max);

        // Place camera at the scene centroid at eye level.
        // Works for both indoor captures (start inside the room) and outdoor scenes.
        let cam_pos = Vec3::new(centroid.x, centroid.y + radius * 0.1, centroid.z);

        Ok((splats, ScenePose { position: cam_pos.to_array(), yaw: 180.0, pitch: 0.0 }))
    }

    fn init_pipeline(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        splats:         Vec<GpuSplat>,
    ) -> Result<SplatState> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("pass_splat_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("pass_splat_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("pass_splat_layout"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let linear_format = surface_format.remove_srgb_suffix();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("pass_splat_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: linear_format,
                    blend:  Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview:     None,
            cache:         None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pass_splat_camera"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let splat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("pass_splat_data"),
            size:               (splats.len() * std::mem::size_of::<GpuSplat>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("pass_splat_bg"),
            layout:  &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: splat_buf.as_entire_binding() },
            ],
        });

        let visible = Vec::with_capacity(splats.len());

        Ok(SplatState {
            pipeline,
            camera_buf,
            splat_buf,
            bind_group,
            splats,
            visible,
            last_pos:     Vec3::splat(f32::NAN),
            last_forward: Vec3::splat(f32::NAN),
        })
    }
}

impl Pass for PassSplat {
    fn get_label(&self) -> &str { "pass_splat" }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> { Ok(()) }

    fn draw(
        &mut self,
        encoder:  &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame:    &wgpu::SurfaceTexture,
        view:     &wgpu::TextureView,
        world:    &WorldQuery<'_>,
    ) -> Result<()> {
        let requested_idx = ACTIVE_SCENE.load(Ordering::Relaxed).min(SCENE_NAMES.len() - 1);
        if self.state.is_none() || requested_idx != self.current_idx {
            self.current_idx = requested_idx;
            self.cloud_name  = SCENE_NAMES[requested_idx].to_string();
            self.state       = None;

            let cloud = world
                .resources
                .get_resource_by_name::<GaussianCloud>(&self.cloud_name)
                .map_err(|_| anyhow!("GaussianCloud '{}' not found", self.cloud_name))?;

            let (splats, pose) = Self::load_splats(&cloud.source)?;
            AUTO_POSES.lock().unwrap()[requested_idx] = Some(pose);

            self.state = Some(Self::init_pipeline(
                renderer.get_device(),
                renderer.get_surface_format(),
                splats,
            )?);
        }

        let state = self.state.as_mut().unwrap();
        let queue = renderer.get_queue();

        let entity_idx = world.active_camera.data().index as usize;

        let cam = world
            .camera_components
            .data
            .get(entity_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| anyhow!("active camera component not found"))?;

        let xform = world
            .transform_components
            .data
            .get(entity_idx)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| anyhow!("active camera transform not found"))?;

        let tex_size = frame.texture.size();
        let aspect   = tex_size.width as f32 / tex_size.height as f32;

        let pitch   = Mat3::from_rotation_x(xform.rotation.x.to_radians());
        let yaw     = Mat3::from_rotation_y(xform.rotation.y.to_radians());
        let forward = (yaw * pitch) * Vec3::Z;
        let view_m  = Mat4::look_to_rh(xform.position, forward, Vec3::Y);
        let proj    = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.range.start, cam.range.end);

        // ── Sort-skip when camera hasn't moved ────────────────────────────────
        let pos = xform.position;
        let stationary = (pos - state.last_pos).length_squared()
            + (forward - state.last_forward).length_squared()
            < 1e-8;

        let (sort_ms, upload_ms);

        if stationary {
            sort_ms   = 0.0_f32;
            upload_ms = 0.0_f32;
        } else {
            state.last_pos     = pos;
            state.last_forward = forward;

            // ── Frustum culling (Gribb-Hartmann, wgpu [0,1] depth range) ─────
            let vp = proj * view_m;
            // column-major: cols[c][r] is element at column c, row r
            let cols = vp.to_cols_array_2d();
            let row = |r: usize| Vec4::new(cols[0][r], cols[1][r], cols[2][r], cols[3][r]);
            let r0 = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
            let planes = [
                r3 + r0, // left
                r3 - r0, // right
                r3 + r1, // bottom
                r3 - r1, // top
                r2,      // near  (wgpu uses [0,1] NDC depth, so near = r2 alone)
                r3 - r2, // far
            ];

            state.visible.clear();
            for splat in &state.splats {
                let p = Vec4::new(splat.position[0], splat.position[1], splat.position[2], 1.0);
                if planes.iter().all(|pl| pl.dot(p) >= 0.0) {
                    state.visible.push(*splat);
                }
            }

            // ── CPU sort (back-to-front along view axis) ──────────────────────
            let view_z = -forward;
            let t0 = Instant::now();
            state.visible.sort_unstable_by(|a, b| {
                let da = Vec3::from(a.position).dot(view_z);
                let db = Vec3::from(b.position).dot(view_z);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            sort_ms = t0.elapsed().as_secs_f32() * 1000.0;

            let t1 = Instant::now();
            queue.write_buffer(&state.splat_buf, 0, bytemuck::cast_slice(&state.visible));
            upload_ms = t1.elapsed().as_secs_f32() * 1000.0;
        }

        // ── Camera uniform (always updated) ──────────────────────────────────
        let uniform = CameraUniform {
            view:     view_m.to_cols_array_2d(),
            proj:     proj.to_cols_array_2d(),
            viewport: [tex_size.width as f32, tex_size.height as f32],
            _pad:     [0.0; 2],
        };
        queue.write_buffer(&state.camera_buf, 0, bytemuck::bytes_of(&uniform));

        // ── Update diagnostics ────────────────────────────────────────────────
        let now      = Instant::now();
        let frame_ms = self.last_draw.map(|t| t.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
        self.last_draw = Some(now);

        let total_n   = state.splats.len();
        let visible_n = state.visible.len();
        let mem_mb    = (total_n * std::mem::size_of::<GpuSplat>()) as f32 / (1024.0 * 1024.0);
        let diag = DiagData {
            frame_ms,
            sort_ms,
            upload_ms,
            total_n,
            visible_n,
            sort_skip:  stationary,
            scene_name: SCENE_NAMES[requested_idx],
            scene_idx:  requested_idx,
            memory_mb:  mem_mb,
        };
        if let Ok(mut guard) = DIAG.try_lock() { *guard = diag.clone(); }

        // ── Splat render pass ─────────────────────────────────────────────────
        let linear_fmt  = renderer.get_surface_format().remove_srgb_suffix();
        let render_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(linear_fmt),
            ..Default::default()
        });

        let draw_count = state.visible.len() as u32;

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pass_splat"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &render_view,
                    resolve_target: None,
                    depth_slice:    None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05, g: 0.05, b: 0.07, a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set:      None,
                timestamp_writes:         None,
            });

            rpass.set_pipeline(&state.pipeline);
            rpass.set_bind_group(0, &state.bind_group, &[]);
            rpass.draw(0..draw_count * 6, 0..1);
        }

        // ── Stats overlay (rendered directly, bypasses egui_client) ──────────
        let surface_format = renderer.get_surface_format();
        let device = renderer.get_device();
        let queue  = renderer.get_queue();

        let egui_rend = self.egui_renderer.get_or_insert_with(|| {
            LocalEguiRenderer::new(device, surface_format)
        });

        let d = diag;
        let mut copy_text: Option<String> = None;
        egui_rend.draw(device, queue, encoder, view, [tex_size.width, tex_size.height], |ctx| {
            egui::Window::new("Splat Stats")
                .resizable(false)
                .anchor(egui::Align2::LEFT_TOP, [6.0, 6.0])
                .show(ctx, |ui| {
                    if d.frame_ms > 0.5 {
                        let fps = 1000.0 / d.frame_ms;
                        ui.label(format!("Frame  {:.1} ms  ({:.0} fps)", d.frame_ms, fps));
                    } else {
                        ui.label("Frame  --- ms");
                    }

                    if d.sort_skip {
                        ui.label("Sort   skipped (stationary)");
                    } else {
                        ui.label(format!(
                            "Sort   {:.1} ms   Upload {:.1} ms",
                            d.sort_ms, d.upload_ms
                        ));
                    }

                    ui.separator();

                    ui.label(format!(
                        "Scene  {} [{}/{}]",
                        d.scene_name, d.scene_idx + 1, SCENE_NAMES.len()
                    ));
                    ui.label(format!(
                        "Total  {:>10}  ({:.0} MB)",
                        fmt_n(d.total_n), d.memory_mb
                    ));
                    let pct = if d.total_n > 0 {
                        d.visible_n as f32 / d.total_n as f32 * 100.0
                    } else {
                        0.0
                    };
                    ui.label(format!("Vis    {:>10}  ({:.1}%)", fmt_n(d.visible_n), pct));

                    ui.separator();

                    let fps = if d.frame_ms > 0.5 { 1000.0 / d.frame_ms } else { 0.0 };
                    let text = format!(
                        "Frame: {:.1}ms ({:.0}fps)\nSort: {:.1}ms  Upload: {:.1}ms  Skip: {}\nScene: {} [{}/{}]\nTotal: {} ({:.0}MB)\nVis: {} ({:.1}%)",
                        d.frame_ms, fps,
                        d.sort_ms, d.upload_ms, d.sort_skip,
                        d.scene_name, d.scene_idx + 1, SCENE_NAMES.len(),
                        fmt_n(d.total_n), d.memory_mb,
                        fmt_n(d.visible_n), pct,
                    );
                    if ui.button("Copy").clicked() {
                        copy_text = Some(text);
                    }
                });
        });

        if let Some(text) = copy_text {
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(text);
            }
        }

        Ok(())
    }
}

fn fmt_n(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f32 / 1_000.0)
    } else {
        format!("{n}")
    }
}
