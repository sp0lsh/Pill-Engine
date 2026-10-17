use crate::config::{
    AMBIENT_TARGET_FORMAT, BACKGROUND_FRAGMENT_SHADER, BACKGROUND_VERTEX_SHADER,
    DEFAULT_EQUIRECT_FALLBACK_PIXEL, DEPTH_TARGET_FORMAT, PBR_ALPHA_TEST_CONSTANT,
    PBR_OPAQUE_FAST_FRAGMENT_SHADER, PBR_OPAQUE_FRAGMENT_SHADER, PBR_OPAQUE_VERTEX_SHADER,
    SSR_NORMAL_TARGET_FORMAT, SSR_SURFACE_TARGET_FORMAT, VELOCITY_TARGET_FORMAT,
};
use crate::graphics::shadow_casters::{box_corners, frustum_planes, outside_frustum};
use crate::{
    ecs::{CameraComponent, MeshRenderingComponent, RenderStateComponent, TransformComponent},
    graphics::ddgi::DdgiFrame,
    graphics::fullscreen::linear_clamp_sampler,
    graphics::ibl::IblFrame,
    graphics::pass_lights::{LightBuffers, LightsFrame},
    graphics::shader_bindings::{
        background_fragment, background_vertex, merge_layouts, pbr_opaque_fragment,
        pbr_opaque_vertex,
        structs::{BackgroundCamera, Camera, Scene},
    },
    graphics::shadow::ShadowFrame,
    graphics::{
        camera_forward, camera_view_projection, decompose_render_queue_key, BufferDesc,
        FrameTarget, Pass, PassTextureHandle, PillRenderer, Pipeline, PipelineDesc, RenderQueueKey,
        RendererMaterialHandle, RendererMeshHandle, RendererTargetDesc, ShaderDesc, TargetSize,
        TextureData, WorldQuery,
    },
    renderer::resources::{RendererMaterial, RendererMesh, Vertex},
};
use pill_core::{error, LogContext, Matrix4f, Result, Vector3f};
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::rc::Rc;
// Motion history (post_fx): entity handles and their index/version
#[cfg(feature = "post_fx")]
use {crate::ecs::EntityHandle, pill_core::PillSlotMapKey};

pub(crate) use crate::graphics::shader_bindings::structs::DrawTransform;

impl Camera {
    fn new() -> Self {
        Self {
            position: [0.0; 4],
            view_projection: Matrix4f::IDENTITY.to_cols_array_2d(),
            previous_view_projection: Matrix4f::IDENTITY.to_cols_array_2d(),
            flags: [0.0; 4],
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

        self.view_projection =
            camera_view_projection(camera_component, transform_component).to_cols_array_2d();
    }
}

impl DrawTransform {
    pub(crate) fn from(transform: &TransformComponent) -> Self {
        Self {
            position: [
                transform.position.x,
                transform.position.y,
                transform.position.z,
                0.0,
            ],
            // TransformComponent stores degrees; the vertex shader expects radians
            rotation: [
                transform.rotation.x.to_radians(),
                transform.rotation.y.to_radians(),
                transform.rotation.z.to_radians(),
                0.0,
            ],
            scale: [transform.scale.x, transform.scale.y, transform.scale.z, 0.0],
        }
    }

    /// Stores material slot `index` in `position.w`: the shaders read the draw's parameters from
    /// the material array there (draw_transform.hlsl `draw_material_index`).
    pub(crate) fn with_material_index(mut self, index: impl Into<u32>) -> Self {
        self.position[3] = index.into() as f32;
        self
    }

    /// Model matrix T · (Rx·Ry·Rz) · S, as the vertex shaders build it (draw_transform.hlsl).
    pub(crate) fn model_matrix(&self) -> Matrix4f {
        let [x, y, z, _] = self.position;
        let [pitch, yaw, roll, _] = self.rotation;
        let [scale_x, scale_y, scale_z, _] = self.scale;
        Matrix4f::from_translation(Vector3f::new(x, y, z))
            * Matrix4f::from_rotation_x(pitch)
            * Matrix4f::from_rotation_y(yaw)
            * Matrix4f::from_rotation_z(roll)
            * Matrix4f::from_scale(Vector3f::new(scale_x, scale_y, scale_z))
    }
}

/// Last frame's transform of an entity slot, the entity version that drew it and the frame.
#[cfg_attr(not(feature = "post_fx"), allow(dead_code))]
#[derive(Clone, Copy)]
struct PreviousTransform {
    transform: DrawTransform,
    version: NonZeroU32,
    frame: u64,
}

// Per-draw transforms live in storage buffers indexed by instance id (no per-draw dynamic offsets):
// binding 0 this frame's, binding 1 last frame's (motion vectors only).
pub(crate) const PER_DRAW_STRIDE_BYTES: usize = std::mem::size_of::<DrawTransform>();

pub const MATERIAL_BIND_GROUP_GLOBALS: usize = pbr_opaque_fragment::SCENE.group as usize;
pub const MATERIAL_BIND_GROUP_TEXTURES: usize =
    pbr_opaque_fragment::BASE_COLOR_TEXTURE.group as usize;
pub const MATERIAL_BIND_GROUP_ARRAY: usize = pbr_opaque_fragment::MATERIAL_ARRAY.group as usize;
pub const MATERIAL_BIND_GROUP_PERDRAW: usize = pbr_opaque_vertex::PER_DRAW_ARRAY.group as usize;

/// One entity's draw before batching.
struct Draw {
    key: RenderQueueKey,
    // To the camera, for the translucent back-to-front sort
    distance_squared: f32,
    current: DrawTransform,
    previous: DrawTransform,
}

/// Scene pipeline variant: with the PassSsr outputs, the alpha-test shader, no back-face culling.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct SceneVariant {
    ssr: bool,
    alpha_test: bool,
    double_sided: bool,
    additive: bool,
}

impl SceneVariant {
    fn of(material: &RendererMaterial, ssr: bool) -> Self {
        Self {
            ssr,
            alpha_test: material.alpha_tested,
            double_sided: material.double_sided,
            additive: material.additive,
        }
    }
}

/// One instanced draw: a run of equal render keys (same material and mesh) in the sorted
/// per-draw array; instances `first..first + count` of the storage buffer.
struct DrawBatch {
    material_handle: RendererMaterialHandle,
    mesh_handle: RendererMeshHandle,
    first: u32,
    count: u32,
}

/// GPU-side state for the background sub-draw within the PBR render pass.
struct BgSubState {
    pipeline: Pipeline,
    // With the PassSsr outputs; built on the first frame with SSR on
    ssr_pipeline: Option<Pipeline>,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    ibl_sampler: wgpu::Sampler,
    // Equirect handle the current bind_group was built from; rebuild only when it changes (the
    // IBL specular map it also binds is allocated once).
    cached_bg: PassTextureHandle,
    // White 1×1 equirect drawn (tinted by background_color) when no environment image is set
    fallback_bg: PassTextureHandle,
}

/// GPU-side pass state initialized in `Pass::init`, read every `Pass::draw`.
struct PassPBROpaqueState {
    camera_uniform: Camera,
    camera_buffer: wgpu::Buffer,
    scene_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    // `SceneVariant::default()`; its layouts build the bind groups
    pipeline: Pipeline,
    // Other variants, built on the first frame that draws them
    variants: HashMap<SceneVariant, Pipeline>,
    ibl_sampler: wgpu::Sampler,
    prefilter_sampler: wgpu::Sampler,
    brdf_lut_sampler: wgpu::Sampler,
    // Bound when no shadow map is published (the shader then skips it), and the comparison sampler
    shadow_stand_in: PassTextureHandle,
    shadow_sampler: wgpu::Sampler,
    light_buffers: LightBuffers,
    // Bound until PassPunctualShadows publishes its texture array
    punctual_shadow_stand_in: wgpu::TextureView,
    // Bound without PassDdgi (its uniform stays zeroed: GI off)
    ddgi_stand_in: DdgiFrame,
    per_draw_buffer: wgpu::Buffer,
    // Last frame's transforms; a single entry when the pass writes no motion vectors (never read)
    previous_per_draw_buffer: wgpu::Buffer,
    per_draw_bind_group: wgpu::BindGroup,
    material_array_bind_group: wgpu::BindGroup,
    bg: BgSubState,
    // IBL handles and shadow map the current globals_bind_group was built from; rebuild only when
    // these change.
    // BRDF LUT handle the current globals_bind_group was built from (the IBL maps are fixed views)
    cached_brdf_lut: PassTextureHandle,
    cached_shadow_map: Option<PassTextureHandle>,
    cached_punctual_shadows: u64,
    cached_ddgi: u64,
}

/// View of a game-set RenderStateComponent texture; an invalid handle logs and uses `fallback`.
/// Callers cache by handle, so the error is logged once per handle value.
fn view_or_fallback(
    renderer: &dyn PillRenderer,
    handle: PassTextureHandle,
    fallback: PassTextureHandle,
    label: &str,
) -> Result<wgpu::TextureView> {
    if let Some(view) = renderer.get_texture_view(handle) {
        return Ok(view);
    }
    error!(LogContext::Rendering => "RenderStateComponent.{} is not a valid texture; using the default", label);
    renderer
        .get_texture_view(fallback)
        .ok_or_else(|| format!("PassPBROpaque: default {label} texture missing").into())
}

/// Builds the globals bind group (camera UBO + IBL irradiance/prefilter/BRDF-LUT views + scene UBO).
/// Shared by init() and the per-frame rebuild-on-change path in draw().
#[allow(clippy::too_many_arguments)]
fn build_globals_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    scene_buffer: &wgpu::Buffer,
    irradiance_view: &wgpu::TextureView,
    prefilter_view: &wgpu::TextureView,
    brdf_lut_view: &wgpu::TextureView,
    ibl_sampler: &wgpu::Sampler,
    prefilter_sampler: &wgpu::Sampler,
    brdf_lut_sampler: &wgpu::Sampler,
    shadow_view: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
    light_buffers: &LightBuffers,
    punctual_shadow_view: &wgpu::TextureView,
    ddgi: &DdgiFrame,
) -> wgpu::BindGroup {
    pbr_opaque_fragment::Group0 {
        camera: camera_buffer,
        irradiance_texture: irradiance_view,
        irradiance_sampler: ibl_sampler,
        prefilter_texture: prefilter_view,
        prefilter_sampler,
        brdf_lut_texture: brdf_lut_view,
        brdf_lut_sampler,
        scene: scene_buffer,
        shadow_map: shadow_view,
        shadow_sampler,
        punctual_lights: &light_buffers.lights,
        froxel_entries: &light_buffers.entries,
        froxel_records: &light_buffers.records,
        punctual_shadows: &light_buffers.shadows,
        punctual_shadow_maps: punctual_shadow_view,
        ddgi_probes: &ddgi.probes,
        ddgi_depth: &ddgi.depth_view,
        ddgi: &ddgi.params,
    }
    .create(device, layout)
}

/// Builds the background bind group (camera UBO + equirect view + sampler + the prefiltered
/// environment its blur reads).
fn build_bg_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    camera_buffer: &wgpu::Buffer,
    equirect_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    ibl_specular: &wgpu::TextureView,
    ibl_sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    background_fragment::Group0 {
        camera: camera_buffer,
        equirect_texture: equirect_view,
        equirect_sampler: sampler,
        ibl_specular,
        ibl_sampler,
    }
    .create(device, layout)
}

/// Default PBR pass: GGX + IBL, one instanced draw per material/mesh batch.
/// Background, IBL and fog are read from RenderStateComponent every frame.
/// Optional extra outputs: screen-space motion vectors (`with_velocity_target`, UV units per frame)
/// and the indirect (IBL) term alone (`with_ambient_target`, for SSAO).
pub struct PassPBROpaque {
    // Translucent mode (PassPBRTranslucent): draws only materials with opacity < 1, back to front,
    // blended over the color target, depth-tested against the opaque depth without writing it
    translucent: bool,
    // Opaque mode with a PassPBRTranslucent after it: leave opacity < 1 materials to that pass
    #[cfg_attr(not(feature = "post_fx"), allow(dead_code))]
    exclude_translucent: bool,
    color_target: PassTextureHandle,
    depth_target: PassTextureHandle,
    velocity_target: Option<PassTextureHandle>,
    ambient_target: Option<PassTextureHandle>,
    // PassSsr inputs: [specular weight + roughness, octahedral normal]
    ssr_targets: Option<[PassTextureHandle; 2]>,
    // Per-draw storage capacity; at most one instance per entity
    max_instances: usize,
    // Key light shadow published by PassShadow each frame
    shadow: Option<Rc<RefCell<ShadowFrame>>>,
    // Point/spot lights and froxel lists published by PassLights each frame
    lights: Option<Rc<RefCell<LightsFrame>>>,
    // Diffuse GI probes published by PassDdgi
    ddgi: Option<Rc<RefCell<DdgiFrame>>>,
    // Engine BRDF LUT: the first frame's, and the fallback for an invalid game handle
    default_brdf_lut: PassTextureHandle,
    // Octahedral irradiance and specular maps baked by PassIblBake
    ibl: Rc<RefCell<IblFrame>>,
    capacity_warned: bool,
    // Motion history (only with a velocity target): last frame's transform per entity slot, the
    // scene it belongs to and the camera view-projection
    frame: u64,
    previous_transforms: Vec<Option<PreviousTransform>>,
    previous_view_projection: Option<Matrix4f>,
    motion_history_epoch: u32,
    history_scene: usize,
    // Per-frame scratch, capacity reused across frames; f32 = squared camera distance (translucent sort)
    draws: Vec<Draw>,
    batches: Vec<DrawBatch>,
    staging_buffer: Vec<u8>,
    previous_staging_buffer: Vec<u8>,
    state: Option<PassPBROpaqueState>,
}

impl PassPBROpaque {
    /// `max_instances` sizes the per-draw buffer; `default_brdf_lut` is the engine's, also the
    /// fallback for an invalid game handle; `ibl` holds the maps `PassIblBake` bakes. Call
    /// `Pass::init` before drawing.
    pub(crate) fn new(
        color_target: PassTextureHandle,
        depth_target: PassTextureHandle,
        max_instances: usize,
        default_brdf_lut: PassTextureHandle,
        ibl: Rc<RefCell<IblFrame>>,
    ) -> Self {
        let max_instances = max_instances.max(1);
        Self {
            translucent: false,
            exclude_translucent: false,
            color_target,
            depth_target,
            velocity_target: None,
            ambient_target: None,
            ssr_targets: None,
            max_instances,
            shadow: None,
            lights: None,
            ddgi: None,
            default_brdf_lut,
            ibl,
            capacity_warned: false,
            frame: 0,
            previous_transforms: Vec::new(),
            previous_view_projection: None,
            motion_history_epoch: 0,
            history_scene: 0,
            draws: Vec::with_capacity(max_instances),
            batches: Vec::new(),
            staging_buffer: Vec::with_capacity(max_instances * PER_DRAW_STRIDE_BYTES),
            previous_staging_buffer: Vec::new(),
            state: None,
        }
    }

    /// Motion vectors (and with them the per-entity history) only when a velocity target is set.
    fn writes_motion_vectors(&self) -> bool {
        self.velocity_target.is_some()
    }

    /// Last frame's transform of `entity`, recording `current` for the next frame. An entity not
    /// drawn last frame, or a different entity in a reused slot (version changed), has no motion.
    #[cfg(feature = "post_fx")]
    fn previous_transform(
        &mut self,
        entity: EntityHandle,
        current: DrawTransform,
    ) -> DrawTransform {
        let slot = entity.data().index as usize;
        let version = entity.data().version;
        if self.previous_transforms.len() <= slot {
            self.previous_transforms.resize(slot + 1, None);
        }
        let previous = match self.previous_transforms[slot] {
            Some(previous) if previous.frame + 1 == self.frame && previous.version == version => {
                previous.transform
            }
            _ => current,
        };
        self.previous_transforms[slot] = Some(PreviousTransform {
            transform: current,
            version,
            frame: self.frame,
        });
        previous
    }

    #[cfg(feature = "post_fx")]
    pub fn max_instances(&self) -> usize {
        self.max_instances
    }

    #[cfg(feature = "post_fx")]
    pub(crate) fn default_brdf_lut(&self) -> PassTextureHandle {
        self.default_brdf_lut
    }

    #[cfg(feature = "post_fx")]
    /// Skips materials with opacity < 1, for chains where a `PassPBRTranslucent` draws them.
    pub fn without_translucent(mut self) -> Self {
        self.exclude_translucent = true;
        self
    }

    #[cfg(feature = "post_fx")]
    /// Also writes motion vectors (VELOCITY_TARGET_FORMAT): UV offset from last frame's position.
    pub fn with_velocity_target(mut self, target: PassTextureHandle) -> Self {
        self.velocity_target = Some(target);
        self
    }

    /// Shades the point/spot lights `PassLights` publishes in `frame`.
    pub(crate) fn with_lights(mut self, frame: Rc<RefCell<LightsFrame>>) -> Self {
        self.lights = Some(frame);
        self
    }

    /// Takes indirect diffuse light from the DDGI probes `PassDdgi` publishes in `frame`.
    pub(crate) fn with_ddgi(mut self, frame: Rc<RefCell<DdgiFrame>>) -> Self {
        self.ddgi = Some(frame);
        self
    }

    /// Receives the key light's shadow from the `PassShadow` publishing `frame`.
    pub(crate) fn with_shadow(mut self, frame: Rc<RefCell<ShadowFrame>>) -> Self {
        self.shadow = Some(frame);
        self
    }

    #[cfg(feature = "post_fx")]
    /// Also writes the fogged IBL term alone (AMBIENT_TARGET_FORMAT), which SSAO darkens.
    pub fn with_ambient_target(mut self, target: PassTextureHandle) -> Self {
        self.ambient_target = Some(target);
        self
    }

    #[cfg(feature = "post_fx")]
    /// Also writes the inputs of `PassSsr` (SSR_SURFACE_TARGET_FORMAT, SSR_NORMAL_TARGET_FORMAT)
    /// and, while `RenderStateComponent::ssr` is on, leaves the specular IBL term to it.
    pub fn with_ssr_targets(
        mut self,
        surface: PassTextureHandle,
        normal: PassTextureHandle,
    ) -> Self {
        self.ssr_targets = Some([surface, normal]);
        self
    }

    /// Color targets in shader output order (color, velocity, ambient, SSR surface, SSR normal);
    /// absent outputs are `None`, the SSR ones also while `ssr` is false.
    fn color_targets(
        &self,
        color_format: wgpu::TextureFormat,
        ssr: bool,
        additive: bool,
    ) -> [Option<wgpu::ColorTargetState>; 5] {
        let blend = if self.translucent && additive {
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }
        } else if self.translucent {
            wgpu::BlendState::ALPHA_BLENDING
        } else {
            wgpu::BlendState::REPLACE
        };
        let ssr_targets = self.ssr_targets.filter(|_| ssr);
        [
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            }),
            self.velocity_target.map(|_| wgpu::ColorTargetState {
                format: VELOCITY_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            self.ambient_target.map(|_| wgpu::ColorTargetState {
                format: AMBIENT_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            ssr_targets.map(|_| wgpu::ColorTargetState {
                format: SSR_SURFACE_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            ssr_targets.map(|_| wgpu::ColorTargetState {
                format: SSR_NORMAL_TARGET_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ]
    }

    /// Scene pipeline of `variant`: `ssr` adds the `PassSsr` outputs (`with_ssr_targets`),
    /// `alpha_test` the shader with discard, `double_sided` turns culling off.
    fn create_scene_pipeline(
        &self,
        renderer: &mut dyn PillRenderer,
        variant: SceneVariant,
    ) -> Result<Pipeline> {
        // 0: globals, 1: material textures, 2: material array, 3: this and last frame's per-draw
        // transforms (storage arrays: no dynamic offset, no 256-B Metal alignment)
        let bind_groups =
            merge_layouts(&[pbr_opaque_vertex::layout(), pbr_opaque_fragment::layout()]);

        let color_targets =
            self.color_targets(renderer.get_hdr_format(), variant.ssr, variant.additive);
        let desc = PipelineDesc {
            label: Some("pass_pbr_opaque_pipeline"),
            vs: ShaderDesc {
                source: PBR_OPAQUE_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                // 16-bit BRDF and wave-scalarized light loop where the device supports both
                source: if renderer.fast_shader_tier() {
                    PBR_OPAQUE_FAST_FRAGMENT_SHADER
                } else {
                    PBR_OPAQUE_FRAGMENT_SHADER
                },
                entry_func: "fs_main",
            },
            fragment_constants: &[(
                PBR_ALPHA_TEST_CONSTANT,
                if variant.alpha_test { 1.0 } else { 0.0 },
            )],
            vertex_buffers: &RendererMesh::data_layout_descriptors(),
            bind_groups,
            targets: &color_targets,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TARGET_FORMAT,
                // Translucent surfaces also write depth (drawn back to front), so the passes after
                // them (DoF, motion blur) see the glass rather than what is behind it
                // Additive ones don't: what they add to stays visible to those passes
                depth_write_enabled: !variant.additive,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: (!variant.double_sided).then_some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
        };

        renderer.create_pipeline(desc)
    }

    /// Background sub-draw pipeline over the same attachments as the scene pipeline.
    fn create_bg_pipeline(&self, renderer: &mut dyn PillRenderer, ssr: bool) -> Result<Pipeline> {
        let bg_bind_groups =
            merge_layouts(&[background_vertex::layout(), background_fragment::layout()]);
        // Same attachments as the scene draw; the sky writes color and velocity, ambient and
        // the SSR inputs stay 0
        let mut bg_targets = self.color_targets(renderer.get_hdr_format(), ssr, false);
        for target in bg_targets[2..].iter_mut().flatten() {
            target.write_mask = wgpu::ColorWrites::empty();
        }
        renderer.create_pipeline(PipelineDesc {
            label: Some("pass_pbr_opaque_bg_pipeline"),
            vs: ShaderDesc {
                source: BACKGROUND_VERTEX_SHADER,
                entry_func: "vs_main",
            },
            ps: ShaderDesc {
                source: BACKGROUND_FRAGMENT_SHADER,
                entry_func: "fs_main",
            },
            fragment_constants: &[],
            vertex_buffers: &[],
            bind_groups: bg_bind_groups,
            targets: &bg_targets,
            // LessEqual + write disabled: fills only pixels where depth == 1.0 (far plane).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_TARGET_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
        })
    }
}

/// Returns the initialized pass state; panics in debug if `init` was not called.
fn get_state(pass: &mut PassPBROpaque) -> &mut PassPBROpaqueState {
    debug_assert!(pass.state.is_some());
    pass.state
        .as_mut()
        .expect("PassPBROpaque: state not initialized — call init() before draw()")
}

/// Whether the material of a render key is translucent (opacity < 1 alpha-blended, or additive);
/// a missing material counts as opaque.
pub(crate) fn is_translucent(world: &WorldQuery<'_>, key: RenderQueueKey) -> bool {
    let key_fields = decompose_render_queue_key(key);
    let Some(version) = NonZeroU32::new(key_fields.material_version.into()) else {
        return false;
    };
    let handle = RendererMaterialHandle::new(key_fields.material_index.into(), version);
    world
        .resources
        .get_resource::<RendererMaterial>(&handle)
        .is_ok_and(|material| material.translucent)
}

#[cfg(feature = "post_fx")]
/// Translucent PBR: materials with opacity < 1, sorted back to front and alpha-blended over the
/// HDR target after the opaque pass, depth-tested against the opaque depth. Also writes depth and,
/// with `with_velocity_target`, motion vectors, so depth of field and motion blur treat the glass
/// surface as the nearest one. Shares PassPBROpaque's shaders.
pub struct PassPBRTranslucent(PassPBROpaque);

#[cfg(feature = "post_fx")]
impl PassPBRTranslucent {
    /// Same targets and IBL inputs as the opaque pass it follows.
    pub(crate) fn new(
        color_target: PassTextureHandle,
        depth_target: PassTextureHandle,
        max_instances: usize,
        default_brdf_lut: PassTextureHandle,
        ibl: Rc<RefCell<IblFrame>>,
    ) -> Self {
        let mut pass = PassPBROpaque::new(
            color_target,
            depth_target,
            max_instances,
            default_brdf_lut,
            ibl,
        );
        pass.translucent = true;
        Self(pass)
    }

    /// Writes motion vectors (replacing the opaque ones behind) into `target`.
    pub fn with_velocity_target(self, target: PassTextureHandle) -> Self {
        Self(self.0.with_velocity_target(target))
    }

    /// Takes indirect diffuse light from the DDGI probes, like `PassPBROpaque::with_ddgi`.
    pub(crate) fn with_ddgi(self, frame: Rc<RefCell<DdgiFrame>>) -> Self {
        Self(self.0.with_ddgi(frame))
    }

    /// Shades the point/spot lights, like `PassPBROpaque::with_lights`.
    pub(crate) fn with_lights(self, frame: Rc<RefCell<LightsFrame>>) -> Self {
        Self(self.0.with_lights(frame))
    }

    /// Receives the key light's shadow, like `PassPBROpaque::with_shadow`.
    pub(crate) fn with_shadow(self, frame: Rc<RefCell<ShadowFrame>>) -> Self {
        Self(self.0.with_shadow(frame))
    }
}

#[cfg(feature = "post_fx")]
impl Pass for PassPBRTranslucent {
    fn get_label(&self) -> &str {
        self.0.get_label()
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        self.0.init(renderer)
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        self.0.draw(encoder, renderer, frame_target, world)
    }
}

impl Pass for PassPBROpaque {
    fn get_label(&self) -> &str {
        if self.translucent {
            "pass_pbr_translucent"
        } else {
            "pass_pbr_opaque"
        }
    }

    fn init(&mut self, renderer: &mut dyn PillRenderer) -> Result<()> {
        let pipeline = self.create_scene_pipeline(renderer, SceneVariant::default())?;

        let camera_buffer = {
            let device = renderer.get_device();
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pass_pbr_opaque_camera_buffer"),
                size: std::mem::size_of::<Camera>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let scene_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_opaque_scene_buffer"),
            byte_size: std::mem::size_of::<Scene>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })?;

        let per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_opaque_per_draw"),
            byte_size: (PER_DRAW_STRIDE_BYTES * self.max_instances) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })?;
        let previous_capacity = if self.writes_motion_vectors() {
            self.max_instances
        } else {
            1
        };
        let previous_per_draw_buffer = renderer.create_buffer(BufferDesc {
            label: Some("pass_pbr_opaque_previous_per_draw"),
            byte_size: (PER_DRAW_STRIDE_BYTES * previous_capacity) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })?;
        let per_draw_bind_group = pbr_opaque_vertex::Group3 {
            per_draw_array: &per_draw_buffer,
            previous_per_draw_array: &previous_per_draw_buffer,
        }
        .create(
            renderer.get_device(),
            &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_PERDRAW],
        );
        let material_array_bind_group = pbr_opaque_fragment::Group2 {
            material_array: renderer.get_material_buffer(),
        }
        .create(
            renderer.get_device(),
            &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_ARRAY],
        );

        // The oct maps carry their own seam borders: clamp, and pick mips explicitly
        // (oct_ibl.hlsl), so no mip filtering
        let ibl_sampler = linear_clamp_sampler(renderer);
        let prefilter_sampler = linear_clamp_sampler(renderer);
        let (irradiance_view, prefilter_view) = {
            let ibl = self.ibl.borrow();
            (ibl.irradiance_view.clone(), ibl.specular_view.clone())
        };
        // Initial BRDF LUT; draw() rebuilds the bind group when RenderStateComponent's changes
        let brdf_lut_view = renderer
            .get_texture_view(self.default_brdf_lut)
            .ok_or("PassPBROpaque: default BRDF LUT texture missing")?;

        // The LUT's U axis is NdotV, not an angle: clamp instead of wrapping
        let brdf_lut_sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });

        // Until PassShadow publishes a map (and whenever shadows are off) a 1×1 depth target fills
        // the binding; the shader skips it
        let shadow_stand_in = renderer.create_render_target(RendererTargetDesc {
            name: "pass_pbr_shadow_stand_in".to_string(),
            format: DEPTH_TARGET_FORMAT,
            size: TargetSize::Fixed(1, 1),
        })?;
        let shadow_view = renderer
            .get_texture_view(shadow_stand_in)
            .ok_or("PassPBROpaque: shadow stand-in missing")?;
        let shadow_sampler = renderer
            .get_device()
            .create_sampler(&wgpu::SamplerDescriptor {
                label: Some("pass_pbr_shadow_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                // Linear + compare: each tap is a hardware 2×2 PCF
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            });

        // Without PassLights, minimal buffers fill the bindings (the light count stays 0)
        let light_buffers = match &self.lights {
            Some(frame) => frame.borrow().buffers.clone(),
            None => LightBuffers::new(renderer, 1, 1, 2, 1)?,
        };
        let ddgi_stand_in = DdgiFrame::new(renderer)?;
        let punctual_shadow_stand_in = renderer
            .get_device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("pass_pbr_punctual_shadow_stand_in"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_TARGET_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });

        let globals_bind_group = build_globals_bind_group(
            renderer.get_device(),
            &pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_GLOBALS],
            &camera_buffer,
            &scene_buffer,
            &irradiance_view,
            &prefilter_view,
            &brdf_lut_view,
            &ibl_sampler,
            &prefilter_sampler,
            &brdf_lut_sampler,
            &shadow_view,
            &shadow_sampler,
            &light_buffers,
            &punctual_shadow_stand_in,
            &ddgi_stand_in,
        );
        let cached_brdf_lut = self.default_brdf_lut;

        // Background sub-draw; draw() rebinds its equirect when RenderStateComponent.background
        // changes, starting from a 1px fallback.
        let bg = {
            let bg_pipeline = self.create_bg_pipeline(renderer, false)?;
            let equirect_fallback_h = renderer.create_texture_from_pixels(
                "bg_equirect_fallback",
                TextureData {
                    mips: &[DEFAULT_EQUIRECT_FALLBACK_PIXEL],
                    width: 1,
                    height: 1,
                    format: wgpu::TextureFormat::Rgba32Float,
                },
            )?;
            let equirect_view = renderer
                .get_texture_view(equirect_fallback_h)
                .ok_or("PassPBROpaque: background fallback texture missing")?;
            // The oct IBL map carries its own seam borders: clamp (oct_ibl.hlsl)
            let bg_ibl_sampler = linear_clamp_sampler(renderer);
            let bg_sampler = renderer
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
            let bg_camera_buf = renderer
                .get_device()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pass_pbr_opaque_bg_camera"),
                    size: std::mem::size_of::<BackgroundCamera>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            let bg_bind_group = build_bg_bind_group(
                renderer.get_device(),
                &bg_pipeline.bind_group_layouts[0],
                &bg_camera_buf,
                &equirect_view,
                &bg_sampler,
                &self.ibl.borrow().specular_view,
                &bg_ibl_sampler,
            );
            BgSubState {
                pipeline: bg_pipeline,
                ssr_pipeline: None,
                camera_buf: bg_camera_buf,
                bind_group: bg_bind_group,
                sampler: bg_sampler,
                ibl_sampler: bg_ibl_sampler,
                cached_bg: equirect_fallback_h,
                fallback_bg: equirect_fallback_h,
            }
        };

        self.state = Some(PassPBROpaqueState {
            camera_uniform: Camera::new(),
            camera_buffer,
            scene_buffer,
            globals_bind_group,
            pipeline,
            variants: HashMap::new(),
            ibl_sampler,
            prefilter_sampler,
            brdf_lut_sampler,
            shadow_stand_in,
            shadow_sampler,
            light_buffers,
            punctual_shadow_stand_in,
            ddgi_stand_in,
            per_draw_buffer,
            previous_per_draw_buffer,
            per_draw_bind_group,
            material_array_bind_group,
            bg,
            cached_brdf_lut,
            cached_shadow_map: None,
            cached_punctual_shadows: 0,
            cached_ddgi: 0,
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
        // Query per-entity storages + the global render state (bg/IBL/fog) this pass needs.
        let render_state = world.get_global::<RenderStateComponent>()?;

        let shadow_frame = self
            .shadow
            .as_ref()
            .map(|frame| *frame.borrow())
            .unwrap_or_default();
        let (lights_params, punctual_shadows) = match &self.lights {
            Some(frame) => {
                let frame = frame.borrow();
                (
                    frame.params,
                    frame
                        .shadow_view
                        .clone()
                        .map(|view| (view, frame.shadow_generation)),
                )
            }
            None => (Default::default(), None),
        };
        let punctual_generation = punctual_shadows
            .as_ref()
            .map_or(0, |(_, generation)| *generation);
        let ddgi = self.ddgi.clone();
        let ddgi_generation = ddgi.as_ref().map_or(0, |frame| frame.borrow().generation);

        // Rebuild the globals bind group only when the component's IBL handles or the published
        // shadow map change (steady state: 0).
        {
            let brdf_lut = render_state.ibl_brdf_lut;
            let state = get_state(self);
            if state.cached_brdf_lut != brdf_lut
                || state.cached_shadow_map != shadow_frame.map
                || state.cached_punctual_shadows != punctual_generation
                || state.cached_ddgi != ddgi_generation
            {
                let shadow_view = renderer
                    .get_texture_view(shadow_frame.map.unwrap_or(state.shadow_stand_in))
                    .ok_or("PassPBROpaque: shadow map missing")?;
                let (irradiance_view, prefilter_view) = {
                    let ibl = self.ibl.borrow();
                    (ibl.irradiance_view.clone(), ibl.specular_view.clone())
                };
                let brdf_lut_view =
                    view_or_fallback(renderer, brdf_lut, self.default_brdf_lut, "ibl_brdf_lut")?;
                let device = renderer.get_device();
                let state = get_state(self);
                let ddgi_frame = ddgi.as_ref().map(|frame| frame.borrow());
                state.globals_bind_group = build_globals_bind_group(
                    device,
                    &state.pipeline.bind_group_layouts[MATERIAL_BIND_GROUP_GLOBALS],
                    &state.camera_buffer,
                    &state.scene_buffer,
                    &irradiance_view,
                    &prefilter_view,
                    &brdf_lut_view,
                    &state.ibl_sampler,
                    &state.prefilter_sampler,
                    &state.brdf_lut_sampler,
                    &shadow_view,
                    &state.shadow_sampler,
                    &state.light_buffers,
                    punctual_shadows
                        .as_ref()
                        .map_or(&state.punctual_shadow_stand_in, |(view, _)| view),
                    ddgi_frame.as_deref().unwrap_or(&state.ddgi_stand_in),
                );
                state.cached_ddgi = ddgi_generation;
                state.cached_brdf_lut = brdf_lut;
                state.cached_shadow_map = shadow_frame.map;
                state.cached_punctual_shadows = punctual_generation;
            }
        }

        // Rebuild the background bind group only when the component's equirect handle changes.
        {
            let bg_h = render_state
                .background
                .unwrap_or(get_state(self).bg.fallback_bg);
            if get_state(self).bg.cached_bg != bg_h {
                let fallback = get_state(self).bg.fallback_bg;
                let equirect_view = view_or_fallback(renderer, bg_h, fallback, "background")?;
                let ibl_specular = self.ibl.borrow().specular_view.clone();
                let device = renderer.get_device();
                let state = get_state(self);
                state.bg.bind_group = build_bg_bind_group(
                    device,
                    &state.bg.pipeline.bind_group_layouts[0],
                    &state.bg.camera_buf,
                    &equirect_view,
                    &state.bg.sampler,
                    &ibl_specular,
                    &state.bg.ibl_sampler,
                );
                state.bg.cached_bg = bg_h;
            }
        }

        let (active_camera_component, active_camera_transform) = world.active_camera()?;
        // SSR on: the pipelines and attachments with the PassSsr outputs, else without them
        let ssr = self.ssr_targets.is_some() && render_state.ssr.enabled;
        if ssr && get_state(self).bg.ssr_pipeline.is_none() {
            let bg = self.create_bg_pipeline(renderer, true)?;
            get_state(self).bg.ssr_pipeline = Some(bg);
        }

        // Motion history: a reset (camera cut) or another scene makes this frame's previous
        // transforms = current
        let motion_vectors = self.writes_motion_vectors();
        self.frame += 1;
        if self.motion_history_epoch != render_state.motion_history_epoch
            || self.history_scene != world.scene_id()
        {
            self.motion_history_epoch = render_state.motion_history_epoch;
            self.history_scene = world.scene_id();
            self.previous_transforms.clear();
            self.previous_view_projection = None;
        }
        let view_projection =
            camera_view_projection(active_camera_component, active_camera_transform);
        let previous_view_projection = self
            .previous_view_projection
            .replace(view_projection)
            .unwrap_or(view_projection);

        // Update camera and scene uniforms. MVP = viewProjection·model is computed in the vertex
        // shader, so the CPU never multiplies per-entity matrices.
        {
            let state = get_state(self);
            state
                .camera_uniform
                .update_data(active_camera_component, active_camera_transform);
            state.camera_uniform.previous_view_projection =
                previous_view_projection.to_cols_array_2d();
            state.camera_uniform.flags = [motion_vectors as u32 as f32, 0.0, 0.0, 0.0];
            renderer.get_queue().write_buffer(
                &state.camera_buffer,
                0,
                bytemuck::bytes_of(&state.camera_uniform),
            );
            renderer.get_queue().write_buffer(
                &state.scene_buffer,
                0,
                bytemuck::bytes_of(&Scene::from_render_state(
                    render_state,
                    &shadow_frame,
                    &lights_params,
                    ssr,
                )),
            );
        }

        // Collect (render key, per-draw) pairs of this pass's materials. Opaque: sorting by key groups
        // them by material, then mesh (depth-tested, so order doesn't matter). Translucent: back to
        // front; equal neighbouring keys still share an instanced draw, which keeps that order.
        self.draws.clear();
        // Main-view frustum culling by each mesh's world-space bounding box
        let planes = frustum_planes(&view_projection);
        // Motion history needs entity versions; without it plain storage iteration is enough
        #[cfg(feature = "post_fx")]
        let entities = world.query_two::<TransformComponent, MeshRenderingComponent>()?;
        #[cfg(not(feature = "post_fx"))]
        let entities = world
            .query::<TransformComponent>()?
            .data
            .iter()
            .zip(world.query::<MeshRenderingComponent>()?.data.iter())
            .filter_map(|(transform, mesh_rendering)| {
                Some(((), transform.as_ref()?, mesh_rendering.as_ref()?))
            });
        for (entity, transform, mesh_rendering) in entities {
            let Some(key) = mesh_rendering.render_queue_key else {
                continue;
            };
            #[cfg(feature = "post_fx")]
            if (self.translucent || self.exclude_translucent)
                && is_translucent(world, key) != self.translucent
            {
                continue;
            }
            if self.draws.len() == self.max_instances {
                if !self.capacity_warned {
                    log::error!(
                        "PassPBROpaque: per-draw capacity {} reached; extra meshes are not drawn",
                        self.max_instances
                    );
                    self.capacity_warned = true;
                }
                break;
            }
            let fields = decompose_render_queue_key(key);
            let current = DrawTransform::from(transform).with_material_index(fields.material_index);
            #[cfg(feature = "post_fx")]
            let previous = if motion_vectors {
                self.previous_transform(entity, current)
            } else {
                current
            };
            #[cfg(not(feature = "post_fx"))]
            let (previous, _) = (current, entity);
            // After the motion history update, so an entity entering the view has a valid one
            let mesh_handle = NonZeroU32::new(fields.mesh_version.into())
                .map(|version| RendererMeshHandle::new(fields.mesh_index.into(), version));
            if let Some(mesh) = mesh_handle
                .as_ref()
                .and_then(|handle| world.resources.get_resource::<RendererMesh>(handle).ok())
            {
                let model = current.model_matrix();
                let corners =
                    box_corners(mesh.local_bounds).map(|corner| model.transform_point3(corner));
                if outside_frustum(&planes, &corners) {
                    continue;
                }
            }
            let distance_squared =
                (transform.position - active_camera_transform.position).length_squared();
            self.draws.push(Draw {
                key,
                distance_squared,
                current,
                previous,
            });
        }
        if self.translucent {
            self.draws
                .sort_unstable_by(|a, b| b.distance_squared.total_cmp(&a.distance_squared));
        } else {
            self.draws.sort_unstable_by_key(|draw| draw.key);
        }

        // One batch per run of equal keys; the key is decoded once per batch
        self.batches.clear();
        self.staging_buffer.clear();
        self.previous_staging_buffer.clear();
        for (index, draw) in self.draws.iter().enumerate() {
            let key = &draw.key;
            self.staging_buffer
                .extend_from_slice(bytemuck::bytes_of(&draw.current));
            if motion_vectors {
                self.previous_staging_buffer
                    .extend_from_slice(bytemuck::bytes_of(&draw.previous));
            }
            if index > 0 && self.draws[index - 1].key == *key {
                if let Some(batch) = self.batches.last_mut() {
                    batch.count += 1;
                }
                continue;
            }
            let key_fields = decompose_render_queue_key(*key);
            self.batches.push(DrawBatch {
                material_handle: RendererMaterialHandle::new(
                    key_fields.material_index.into(),
                    NonZeroU32::new(key_fields.material_version.into()).unwrap(),
                ),
                mesh_handle: RendererMeshHandle::new(
                    key_fields.mesh_index.into(),
                    NonZeroU32::new(key_fields.mesh_version.into()).unwrap(),
                ),
                first: index as u32,
                count: 1,
            });
        }
        {
            let state_ref = self
                .state
                .as_ref()
                .expect("PassPBROpaque: state not initialized — call init() before draw()");
            renderer
                .get_queue()
                .write_buffer(&state_ref.per_draw_buffer, 0, &self.staging_buffer);
            if motion_vectors {
                renderer.get_queue().write_buffer(
                    &state_ref.previous_per_draw_buffer,
                    0,
                    &self.previous_staging_buffer,
                );
            }
        }

        // Pipelines for the variants this frame's materials need
        let mut missing: Vec<SceneVariant> = Vec::new();
        for batch in &self.batches {
            if let Ok(material) = world
                .resources
                .get_resource::<RendererMaterial>(&batch.material_handle)
            {
                let variant = SceneVariant::of(material, ssr);
                let built = self
                    .state
                    .as_ref()
                    .is_some_and(|state| state.variants.contains_key(&variant));
                if variant != SceneVariant::default() && !built && !missing.contains(&variant) {
                    missing.push(variant);
                }
            }
        }
        for variant in missing {
            let pipeline = self.create_scene_pipeline(renderer, variant)?;
            get_state(self).variants.insert(variant, pipeline);
        }

        let depth_view = renderer
            .get_render_target_view(self.depth_target)
            .expect("PassPBROpaque: depth target missing");
        // The translucent pass draws over the opaque pass's color and depth
        let load = if self.translucent {
            wgpu::LoadOp::Load
        } else {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        };
        let cleared = |handle: PassTextureHandle| {
            let view = renderer
                .get_render_target_view(handle)
                .expect("PassPBROpaque: color target missing");
            wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            }
        };
        let color_attachments = [
            Some(cleared(self.color_target)),
            self.velocity_target.map(cleared),
            self.ambient_target.map(cleared),
            self.ssr_targets
                .filter(|_| ssr)
                .map(|[surface, _]| cleared(surface)),
            self.ssr_targets
                .filter(|_| ssr)
                .map(|[_, normal]| cleared(normal)),
        ];

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pass_pbr_opaque_render_pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: if self.translucent {
                        wgpu::LoadOp::Load
                    } else {
                        wgpu::LoadOp::Clear(1.0)
                    },
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        let state_ref = self
            .state
            .as_ref()
            .expect("PassPBROpaque: state not initialized — call init() before draw()");
        // Each variant was built above; set per material (batches are grouped by material)
        let mut bound_variant: Option<SceneVariant> = None;
        // Per-draw array bound once; each batch's instance range selects its entries.
        render_pass.set_bind_group(
            MATERIAL_BIND_GROUP_PERDRAW as u32,
            &state_ref.per_draw_bind_group,
            &[],
        );
        render_pass.set_bind_group(
            MATERIAL_BIND_GROUP_GLOBALS as u32,
            &state_ref.globals_bind_group,
            &[],
        );
        // Every material's parameters; draws select theirs by index (DrawTransform)
        render_pass.set_bind_group(
            MATERIAL_BIND_GROUP_ARRAY as u32,
            &state_ref.material_array_bind_group,
            &[],
        );
        // Batches are sorted by material, so each material is bound once
        let mut bound_material: Option<RendererMaterialHandle> = None;
        for batch in &self.batches {
            if bound_material != Some(batch.material_handle) {
                // A material destroyed this frame can still be referenced until its components'
                // deferred updates run; skip it instead of failing the frame
                let Ok(mat) = world
                    .resources
                    .get_resource::<RendererMaterial>(&batch.material_handle)
                else {
                    continue;
                };
                let variant = SceneVariant::of(mat, ssr);
                if bound_variant != Some(variant) {
                    let pipeline = match state_ref.variants.get(&variant) {
                        Some(pipeline) => pipeline,
                        None => &state_ref.pipeline,
                    };
                    render_pass.set_pipeline(&pipeline.pipeline);
                    bound_variant = Some(variant);
                }
                render_pass.set_bind_group(
                    MATERIAL_BIND_GROUP_TEXTURES as u32,
                    &mat.textures_bind_group,
                    &[],
                );
                bound_material = Some(batch.material_handle);
            }

            let Ok(mesh) = world
                .resources
                .get_resource::<RendererMesh>(&batch.mesh_handle)
            else {
                continue;
            };
            mesh.set_vertex_buffers(&mut render_pass);
            render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // One instanced draw per batch: instance_index in [first, first+count) selects per-draw data.
            render_pass.draw_indexed(
                0..mesh.index_count,
                0,
                batch.first..(batch.first + batch.count),
            );
        }

        // Background sub-draw: within the same render pass, after opaque geometry.
        // LessEqual depth test + write_enabled=false: fills only pixels where depth == 1.0 (far plane = no geometry).
        if !self.translucent {
            let bg = &state_ref.bg;
            let fwd = camera_forward(active_camera_transform);
            let right = fwd.cross(Vector3f::Y).normalize();
            let up = right.cross(fwd);
            let ubo = BackgroundCamera {
                right: right.to_array(),
                tangent_half_fov: (active_camera_component.fov.to_radians() / 2.0).tan(),
                up: up.to_array(),
                aspect: active_camera_component.aspect.get_value(),
                forward: fwd.to_array(),
                rotation: render_state.environment_rotation_turns(),
                // Environment images are drawn scaled by `background_intensity`; the white fallback
                // is tinted to the solid color
                background_color: match render_state.background {
                    Some(_) => [render_state.background_intensity; 3],
                    None => render_state.background_color,
                },
                blur: render_state.background_blur.clamp(0.0, 1.0),
                previous_view_projection: previous_view_projection.to_cols_array_2d(),
            };
            renderer
                .get_queue()
                .write_buffer(&bg.camera_buf, 0, bytemuck::bytes_of(&ubo));
            let pipeline = match (&bg.ssr_pipeline, ssr) {
                (Some(ssr_pipeline), true) => ssr_pipeline,
                _ => &bg.pipeline,
            };
            render_pass.set_pipeline(&pipeline.pipeline);
            render_pass.set_bind_group(0, &bg.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
