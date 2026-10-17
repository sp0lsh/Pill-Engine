use crate::{
    ecs::components::{GlobalComponent, GlobalComponentStorage},
    graphics::PassTextureHandle,
};

use pill_core::{PillTypeMapKey, Vector3f};

/// HDR → display mapping applied by the tonemap pass.
#[derive(Clone, Copy, Debug)]
pub enum Tonemap {
    /// Linear pass-through; values above 1 clip.
    None,
    /// Extended Reinhard c·(1 + c/W²)/(1 + c): `white_point` W maps to 1.0; INFINITY = plain c/(1 + c).
    Reinhard { white_point: f32 },
    /// Lottes' filmic curve x^a / (x^(a·d)·b + c) per channel (GDC 2016, "Advanced Techniques and
    /// Optimization of VDR Color Pipelines"); b and c are solved so `mid_in` maps to `mid_out` and
    /// `hdr_max` to 1. `Tonemap::lottes()` gives the demo's values.
    Lottes {
        /// a: toe steepness.
        contrast: f32,
        /// d: shoulder speed.
        shoulder: f32,
        /// Input mapped to white.
        hdr_max: f32,
        /// Middle grey in and out.
        mid_in: f32,
        mid_out: f32,
    },
}

impl Tonemap {
    /// Lottes curve with contrast 1.6, shoulder 0.977, white at 8, grey 0.18 → 0.267.
    pub fn lottes() -> Self {
        Self::Lottes {
            contrast: 1.6,
            shoulder: 0.977,
            hdr_max: 8.0,
            mid_in: 0.18,
            mid_out: 0.267,
        }
    }
}

/// Pass chain the engine builds on the first frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PassChain {
    /// Opaque PBR → game passes → tonemap.
    #[default]
    Default,
    /// Default chain plus the post-processing passes (see `config::post_fx_pass_chain`); needs
    /// pill_engine's `post_fx` feature.
    #[cfg(feature = "post_fx")]
    PostFx,
}

/// PostFx chain bloom.
#[derive(Clone, Copy, Debug)]
pub struct Bloom {
    /// Bloom added to the image (0 = off).
    pub strength: f32,
    /// Upsample tap offset as a fraction of the screen width; larger = wider glow.
    pub radius: f32,
}

/// PostFx chain depth of field. Circle of confusion = (1/focus_distance − 1/distance) · focus_scale,
/// clamped to ±1; negative is in front of the focal plane (near layer).
#[derive(Clone, Copy, Debug)]
pub struct DepthOfField {
    /// View-space distance in focus.
    pub focus_distance: f32,
    /// Blur gain (0 = off).
    pub focus_scale: f32,
    /// Near-layer blur width multiplier (0 = no near blur).
    pub near_blur: f32,
    /// Near CoC at which the near layer becomes fully opaque (lower = more foreground blurred).
    pub near_cut: f32,
    /// Horizontal bokeh squeeze: 1 = round, 0.5 = 2:1 anamorphic oval.
    pub anamorphic_squeeze: f32,
}

/// What the PBR passes show: the lit image, only the light reaching the surfaces (every surface a
/// white diffuse material without specular or emission, for auditing light levels), or only the
/// surfaces' colors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DebugView {
    #[default]
    Lit,
    /// Geometric normals.
    LightingOnly,
    /// As `LightingOnly`, with the materials' normal maps.
    DetailLighting,
    /// The materials' base color (factor × albedo texture, alpha test kept) without any light,
    /// emission or fog.
    Unlit,
}

/// PostFx chain screen-space ambient occlusion; darkens only the indirect (IBL) light.
#[derive(Clone, Copy, Debug)]
pub struct Ssao {
    /// Blend of the occlusion (0 = off, 1 = full).
    pub strength: f32,
    /// World-space radius of the horizon search.
    pub radius: f32,
    /// Scales the occlusion: visibility = 1 − (1 − v) · intensity.
    pub intensity: f32,
    /// Contrast exponent applied to the visibility.
    pub power: f32,
}

/// PostFx chain lens and film emulation; the defaults leave the image unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Lens {
    /// k1 of r² distortion: negative = barrel, positive = pincushion.
    pub distortion: f32,
    /// Red/blue split at the frame corner in UV units, falling off as r² to 0 at the center.
    pub chromatic_aberration: f32,
    /// Vignette falloff scale (0 = off, ~0.6 subtle).
    pub vignette: f32,
    /// Film grain amplitude (0 = off, 0.03 barely visible, 0.12 medium film stock).
    pub grain_strength: f32,
    /// Grain texel size in pixels at 1280 px width (scales with resolution).
    pub grain_size: f32,
    /// Grain pattern updates per second (24 = filmic, 0 = every frame).
    pub grain_fps: f32,
    /// Grade: per-channel gamma, then smoothstep(edge0, edge1) (edge0 > edge1 inverts), blended in
    /// by `grade_mix`.
    pub grade_gamma: [f32; 3],
    pub grade_edge0: f32,
    pub grade_edge1: f32,
    /// 0 = no grade.
    pub grade_mix: f32,
}

impl Default for Lens {
    fn default() -> Self {
        Self {
            distortion: 0.0,
            chromatic_aberration: 0.0,
            vignette: 0.0,
            grain_strength: 0.0,
            grain_size: 4.0,
            grain_fps: 24.0,
            grade_gamma: [1.0; 3],
            grade_edge0: 0.0,
            grade_edge1: 1.0,
            grade_mix: 0.0,
        }
    }
}

/// Shadow map of the key light (`lights[0]`). `constant_bias` and `slope_bias` are pipeline state,
/// read when the pass chain is built (first frame); the rest every frame. Translucent materials
/// (opacity < 1) don't cast.
#[derive(Clone, Copy, Debug)]
pub struct Shadow {
    pub enabled: bool,
    /// Shadow map width and height in texels, clamped to the device limit (8192 by default).
    pub map_size: u32,
    /// View distance covered by the shadow map; shadows fade out over its last 10% and surfaces
    /// beyond it are lit.
    pub distance: f32,
    /// Hardware depth bias of the caster pass, in depth-buffer units.
    pub constant_bias: i32,
    /// Hardware depth bias scaled by the caster's depth slope.
    pub slope_bias: f32,
    /// Receiver offset along the surface normal, in shadow-map texels.
    pub normal_bias: f32,
}

impl Default for Shadow {
    fn default() -> Self {
        Self {
            enabled: false,
            map_size: 2048,
            distance: 30.0,
            constant_bias: 2,
            slope_bias: 2.0,
            normal_bias: 1.5,
        }
    }
}

/// Diffuse global illumination from a grid of DDGI probes over the scene's opaque, shadow-casting
/// meshes (traced against a BVH built once per set of meshes: static geometry, so moving them or
/// editing material colors is not picked up). Replaces the diffuse IBL term once every probe
/// has been updated (the IBL until then); ray misses see the IBL. Read every frame.
#[derive(Clone, Copy, Debug)]
pub struct Gi {
    pub enabled: bool,
    /// Distance (m) between probes of the grid fitted over the scene (`cascade_spacings` with
    /// `follow_camera`); the grid is capped at 32 × 16 × 32 probes.
    pub probe_spacing: f32,
    /// Rays traced from every probe each frame (at most 256); the pattern rotates per frame.
    pub rays_per_probe: u32,
    /// How fast probes follow lighting changes: the fraction of each update blended into a probe
    /// once it has more updates than 1 / `blend_speed` (before that, a running mean). Lower
    /// averages more frames: less noise with few rays, slower response (a change of the sun, sky
    /// or lights blends faster for a few frames).
    pub blend_speed: f32,
    /// Scales the indirect diffuse light.
    pub intensity: f32,
    /// Shows only the indirect diffuse light.
    pub debug_indirect_only: bool,
    /// Draws every probe as a small sphere shaded with its irradiance (magenta: no weight yet).
    pub debug_probes: bool,
    /// Colours surfaces by the probe cell that lights them (darker outside the probe grid).
    pub debug_probe_cells: bool,
    /// World-space box (min, max) the probe grid covers; `None` = the whole scene. A box around
    /// the playable area gives denser probes in large scenes (the grid is capped per axis).
    /// Ignored with `follow_camera`.
    pub bounds: Option<(Vector3f, Vector3f)>,
    /// The grid follows the camera instead: the largest grid the caps allow, `probe_spacing`
    /// apart, centred on the camera and moving in whole cells; probes it keeps keep their
    /// history, newly covered ones are updated first and lit by the IBL until they are.
    pub follow_camera: bool,
    /// With `follow_camera`: grids around the camera, 1–4, each with the same probe counts and
    /// its `cascade_spacings` spacing. Coarser ones light what lies past
    /// the finer ones, which blend into them toward their edges.
    pub cascades: u32,
    /// With `follow_camera`: probe spacing (m) of each cascade, finest first (`probe_spacing` is
    /// the fitted grid's). Default 1.5, 4.5, 9, 15.
    pub cascade_spacings: [f32; 4],
}

impl Default for Gi {
    fn default() -> Self {
        Self {
            enabled: false,
            probe_spacing: 1.5,
            rays_per_probe: 64,
            blend_speed: 0.08,
            intensity: 1.0,
            debug_indirect_only: false,
            debug_probes: false,
            debug_probe_cells: false,
            bounds: None,
            follow_camera: false,
            cascades: 1,
            cascade_spacings: crate::graphics::ddgi::CASCADE_SPACINGS,
        }
    }
}

/// PostFx chain: screen-space reflections of the opaque scene, by a ray march against the depth
/// buffer along the mirror direction (no history, so no noise and no TAA). Fades out with
/// roughness; rays that miss or leave the screen fall back to the specular IBL. While on, SSAO
/// darkens only the diffuse ambient light, not the specular term. Its three full-screen targets
/// stay allocated in the PostFx chain when off. Read every frame.
#[derive(Clone, Copy, Debug)]
pub struct Ssr {
    pub enabled: bool,
    /// Evenly spaced depth tests along the ray to find a first crossing.
    pub linear_steps: u32,
    /// Binary-search steps refining the crossing.
    pub bisection_steps: u32,
    /// Assumed thickness (m) of the surfaces in the depth buffer; rays farther behind pass on.
    pub thickness: f32,
    /// Roughness range over which reflections fade to the IBL (full below start, none above end).
    pub roughness_fade: (f32, f32),
    /// Screen-border width (fraction of the screen) over which hits fade to the IBL.
    pub edge_fade: f32,
    /// Shows only the reflected light.
    pub debug_reflections_only: bool,
}

impl Default for Ssr {
    fn default() -> Self {
        Self {
            enabled: false,
            linear_steps: 16,
            bisection_steps: 6,
            thickness: 0.25,
            roughness_fade: (0.3, 0.5),
            edge_fade: 0.05,
            debug_reflections_only: false,
        }
    }
}

/// Clustered shading of the scene's `LightComponent`s (point and spot lights): the view is split
/// into screen tiles × exponential depth slices ("froxels"), each with the list of lights reaching
/// it. Read every frame.
#[derive(Clone, Copy, Debug)]
pub struct PunctualLights {
    /// View distance (m) the froxel grid covers; lights beyond it are not shaded.
    pub light_far: f32,
    /// Far end (m) of the first depth slice; the remaining slices are spaced exponentially from
    /// it to `light_far`.
    pub first_slice_depth: f32,
    /// Replaces the shaded color with a heatmap of the light count per froxel.
    pub debug_heatmap: bool,
    /// Shadow maps of lights with `cast_shadows`; off skips rendering them.
    pub shadows_enabled: bool,
    /// Width and height of each point/spot shadow map in texels.
    pub shadow_map_size: u32,
    /// Shadow maps rendered per frame (spot light: 1, point light: 6), given to the lights
    /// nearest the camera; at most 64.
    pub max_shadow_maps: u32,
    /// Hardware depth bias of the shadow casters, in depth-buffer units; read when the pass
    /// chain is built.
    pub shadow_constant_bias: i32,
    /// Hardware depth bias scaled by the caster's depth slope; read when the pass chain is built.
    pub shadow_slope_bias: f32,
    /// Receiver offset along the surface normal, in shadow-map texels at the receiver's distance.
    pub shadow_normal_bias: f32,
}

impl Default for PunctualLights {
    fn default() -> Self {
        Self {
            light_far: 100.0,
            first_slice_depth: 5.0,
            debug_heatmap: false,
            shadows_enabled: true,
            shadow_map_size: 1024,
            max_shadow_maps: 16,
            shadow_constant_bias: 2,
            shadow_slope_bias: 2.0,
            shadow_normal_bias: 1.5,
        }
    }
}

/// Directional light; `direction` points from the light toward the scene.
#[derive(Clone, Copy, Debug)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    /// Linear RGB, multiplied by `intensity`.
    pub color: [f32; 3],
    pub intensity: f32,
}

/// Engine-owned (always present, cannot be removed or constructed by games); games set the
/// public environment, IBL, fog and tonemap fields.
pub struct RenderStateComponent {
    pub(crate) boot_done: bool,
    /// Chain built on the first frame; set it from `start()`, later changes have no effect.
    pub pass_chain: PassChain,
    /// Scene color target (`PillRenderer::get_hdr_format`) shared by scene passes; resolved by the tonemap pass.
    pub(crate) hdr_target: PassTextureHandle,
    /// Scene depth target (DEPTH_TARGET_FORMAT) shared by scene passes.
    pub(crate) depth_target: PassTextureHandle,
    /// Engine default environment (1×1 grey equirect) baked when `environment` is unset or invalid.
    pub(crate) default_environment: PassTextureHandle,
    /// Engine BRDF LUT, used when `ibl_brdf_lut` is invalid.
    pub(crate) default_brdf_lut: PassTextureHandle,
    /// Equirect radiance the image-based lighting is baked from on the GPU (`PassIblBake`) when it
    /// changes; `None` uses a neutral grey. Usually the same image as `background`.
    pub environment: Option<PassTextureHandle>,
    /// Environment (equirect) image drawn behind the scene; `None` draws `background_color`.
    pub background: Option<PassTextureHandle>,
    /// Solid background color used when `background` is `None` (linear RGB).
    pub background_color: [f32; 3],
    /// Scales the `background` environment image (e.g. a daytime sky dimmed at night).
    pub background_intensity: f32,
    /// Turns the environment (its image-based lighting and the `background` image) about +Y, in
    /// radians; a change rebakes the IBL maps.
    pub environment_rotation: f32,
    /// Blurs the `background` image toward the prefiltered environment: 0 = sharp, 1 = the
    /// roughest IBL mip.
    pub background_blur: f32,
    pub ibl_brdf_lut: PassTextureHandle,
    /// Exponential-squared fog: color is mixed toward `fog_color` by 1 - exp(-density² · distance²);
    /// density 0 disables fog.
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    /// Scales both IBL terms (diffuse irradiance and specular reflection); 0 = direct light only.
    pub ibl_intensity: f32,
    /// Rain on every surface, 0 = dry, 1 = soaked: porous surfaces darken, a water film lowers
    /// roughness (fully on upward-facing surfaces, partly on walls).
    pub wetness: f32,
    /// Lit image or lighting-only debug views.
    pub debug_view: DebugView,
    /// Key, rim and bounce lights of the PBR passes.
    pub lights: [DirectionalLight; 3],
    /// Shadow map of the key light (`lights[0]`); off by default.
    pub shadow: Shadow,
    /// Point and spot lights (`LightComponent`).
    pub punctual_lights: PunctualLights,
    /// Diffuse global illumination (DDGI probes); off by default.
    pub gi: Gi,
    /// PostFx chain: screen-space reflections; off by default.
    pub ssr: Ssr,
    pub tonemap: Tonemap,
    /// PostFx chain: motion blur length as a fraction of the frame's motion (0.5 = 180° shutter,
    /// 0 = off).
    pub motion_blur: f32,
    /// PostFx chain: contrast-adaptive sharpening of the tonemapped image (0 = off, 1 = strongest).
    pub sharpness: f32,
    pub ssao: Ssao,
    pub depth_of_field: DepthOfField,
    pub bloom: Bloom,
    pub lens: Lens,
    /// Bumped by `reset_motion_history`; passes drop last frame's transforms when it changes.
    pub(crate) motion_history_epoch: u32,
}

impl RenderStateComponent {
    /// `environment_rotation` in turns, 0–1: the IBL bake and the background both use it.
    pub(crate) fn environment_rotation_turns(&self) -> f32 {
        (self.environment_rotation / std::f32::consts::TAU).rem_euclid(1.0)
    }

    /// Creates the component in its pre-boot state; handles are replaced by `create_default_resources`.
    pub(crate) fn new() -> Self {
        Self {
            boot_done: false,
            pass_chain: PassChain::Default,
            hdr_target: PassTextureHandle::default(),
            depth_target: PassTextureHandle::default(),
            default_environment: PassTextureHandle::default(),
            default_brdf_lut: PassTextureHandle::default(),
            environment: None,
            background: None,
            background_color: [0.15, 0.15, 0.15],
            background_intensity: 1.0,
            environment_rotation: 0.0,
            background_blur: 0.0,
            ibl_brdf_lut: PassTextureHandle::default(),
            fog_color: [0.0, 0.0, 0.0],
            fog_density: 0.0,
            ibl_intensity: 1.0,
            wetness: 0.0,
            debug_view: DebugView::Lit,
            // Key behind the camera upper-left, cool rim front upper-right, neutral bounce from below
            lights: [
                DirectionalLight {
                    direction: [0.38, -0.38, -0.84],
                    color: [1.0, 0.98, 0.95],
                    intensity: 2.2,
                },
                DirectionalLight {
                    direction: [-0.50, 0.50, 0.71],
                    color: [0.6, 0.65, 1.0],
                    intensity: 0.8,
                },
                DirectionalLight {
                    direction: [0.0, -1.0, 0.0],
                    color: [0.8, 0.8, 0.8],
                    intensity: 0.3,
                },
            ],
            shadow: Shadow::default(),
            punctual_lights: PunctualLights::default(),
            gi: Gi::default(),
            ssr: Ssr::default(),
            tonemap: Tonemap::Reinhard {
                white_point: f32::INFINITY,
            },
            motion_blur: 0.5,
            sharpness: 0.0,
            ssao: Ssao {
                strength: 1.0,
                radius: 0.5,
                intensity: 1.0,
                power: 1.5,
            },
            depth_of_field: DepthOfField {
                focus_distance: 6.0,
                focus_scale: 0.0,
                near_blur: 1.0,
                near_cut: 0.9,
                anamorphic_squeeze: 1.0,
            },
            bloom: Bloom {
                strength: 0.04,
                radius: 0.005,
            },
            lens: Lens::default(),
            motion_history_epoch: 0,
        }
    }
}

impl RenderStateComponent {
    /// Scene color target (`PillRenderer::get_hdr_format`) for game-defined passes.
    pub fn hdr_target(&self) -> PassTextureHandle {
        self.hdr_target
    }

    /// Scene depth target (DEPTH_TARGET_FORMAT) for game-defined passes.
    pub fn depth_target(&self) -> PassTextureHandle {
        self.depth_target
    }

    /// Next frame renders without motion vectors (previous transforms = current); call on camera
    /// cuts and teleports so motion blur does not smear across them.
    pub fn reset_motion_history(&mut self) {
        self.motion_history_epoch = self.motion_history_epoch.wrapping_add(1);
    }
}

impl PillTypeMapKey for RenderStateComponent {
    type Storage = GlobalComponentStorage<RenderStateComponent>;
}

impl GlobalComponent for RenderStateComponent {}
