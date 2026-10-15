// PBR fragment shader body, included by pbr_opaque_fragment.hlsl and its variants.
// Full GGX microfacet BRDF with 3 directional lights, clustered point/spot lights and IBL.

struct Camera {
    float4                position;
    column_major float4x4 view_projection;
    column_major float4x4 previous_view_projection;
    float4                flags; // x = 1 when the pass writes motion vectors
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

// IBL resources in globals bind group (set 0, bindings 1-6).
[[vk::binding(1, 0)]] Texture2D    irradiance_texture;
[[vk::binding(2, 0)]] SamplerState irradiance_sampler;
[[vk::binding(3, 0)]] Texture2D    prefilter_texture;
[[vk::binding(4, 0)]] SamplerState prefilter_sampler;
[[vk::binding(5, 0)]] Texture2D    brdf_lut_texture;
[[vk::binding(6, 0)]] SamplerState brdf_lut_sampler;

// Scene globals (set 0, binding 7) — mirrors SceneUniform in scene_uniform.rs; all float4 rows.
struct Scene {
    float4 fog_color_density; // xyz = fog color, w = fog density
    float4 ibl_intensity;     // x = IBL scale, y = 1 when PassSsr adds the specular IBL term, z = wetness,
                              // w = debug view (0 lit, 1 lighting only, 2 lighting only with normal maps)
    float4 light_direction0;  // xyz = direction from the light toward the scene
    float4 light_color0;      // rgb = color, w = intensity
    float4 light_direction1;
    float4 light_color1;
    float4 light_direction2;
    float4 light_color2;
    column_major float4x4 shadow_view_projection; // world → light clip space (key light)
    float4 shadow_params; // x = normal offset (world), y = shadow distance (0 = none), z = 1 / map size, w = map size
    float4 froxel_grid;   // x, y = froxel columns, rows; z = tile size (px); w = light count
    float4 froxel_depth;  // x = 1 / slice linearizer, y = light far, z = slice count, w = 1: heatmap
    float4 camera_forward; // xyz
};
[[vk::binding(7, 0)]] ConstantBuffer<Scene> scene;

// Wet surfaces: albedo darkening of a fully porous surface, film on walls relative to upward
// surfaces, roughness of a full water film
static const float WET_DARKENING = 0.5;
static const float WET_WALL_FILM = 0.35;
static const float WET_ROUGHNESS = 0.08;

// Point/spot lights and their froxel lists (froxelizer.rs) — mirrors GpuLight in pass_lights.rs
struct PunctualLight {
    float4 position_falloff;  // xyz = world position, w = 1 / range²
    float4 direction;         // xyz = direction the spot shines in
    float4 color_intensity;   // rgb = color, w = intensity
    // x, y: angle term saturate(cos · x + y) (point light: 0, 1); z: first shadow layer (−1 = no
    // shadow); w: 1 = point light (6 cube-face layers from z)
    float4 spot_scale_offset;
};
[[vk::binding(10, 0)]] StructuredBuffer<PunctualLight> punctual_lights;
[[vk::binding(11, 0)]] StructuredBuffer<uint2>         froxel_entries; // (offset, count) per froxel
[[vk::binding(12, 0)]] StructuredBuffer<uint>          froxel_records; // u16 light indices, 2 per uint

// Point/spot shadow maps (punctual_shadows.rs): one per spot light, six cube faces per point
// light, all layers of one depth array; sampled with shadow_sampler — mirrors GpuShadowMap
struct PunctualShadow {
    column_major float4x4 view_projection; // world → light clip space (depth 0..1)
    float4 params; // x = normal offset per meter of light distance, y = map size, z = 1 / map size
};
[[vk::binding(13, 0)]] StructuredBuffer<PunctualShadow> punctual_shadows;
[[vk::binding(14, 0)]] DepthTexture2DArray              punctual_shadow_maps;

// Diffuse GI probes (ddgi/pass_ddgi.rs); `ddgi.enabled` 0 = off (use the IBL), 2 = indirect-only
// view, 3 = probe-cell colours
#define DDGI_SAMPLER brdf_lut_sampler
#include "ddgi_common.hlsl"
[[vk::binding(15, 0)]] StructuredBuffer<DdgiProbe>  ddgi_probes;
[[vk::binding(16, 0)]] Texture2D                    ddgi_depth;
[[vk::binding(17, 0)]] ConstantBuffer<DdgiParams>   ddgi;

// 3×3 tent filter from 4 bilinear comparison taps (Castaño 2013, Filament PCF_Low): tap
// coordinates and weights (sum 16) around `uv` in a map of `size` texels (`texel` = 1 / size).
struct TentTaps {
    float2 uv[4];
    float  weight[4];
};
TentTaps tent_taps(float2 uv, float size, float texel) {
    float2 scaled = uv * size + 0.5;
    float2 base   = (floor(scaled) - 0.5) * texel;
    float2 st     = frac(scaled);
    float2 uw     = float2(3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x);
    float2 vw     = float2(3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y);
    float2 u      = float2((2.0 - st.x) / uw.x - 1.0, st.x / uw.y + 1.0) * texel;
    float2 v      = float2((2.0 - st.y) / vw.x - 1.0, st.y / vw.y + 1.0) * texel;
    TentTaps taps;
    taps.uv[0] = base + float2(u.x, v.x);  taps.weight[0] = uw.x * vw.x;
    taps.uv[1] = base + float2(u.y, v.x);  taps.weight[1] = uw.y * vw.x;
    taps.uv[2] = base + float2(u.x, v.y);  taps.weight[2] = uw.x * vw.y;
    taps.uv[3] = base + float2(u.y, v.y);  taps.weight[3] = uw.y * vw.y;
    return taps;
}

// Key light shadow map; the comparison sampler (LessEqual, linear) makes every tap a 2×2 PCF
[[vk::binding(8, 0)]] DepthTexture2D         shadow_map;
[[vk::binding(9, 0)]] SamplerComparisonState shadow_sampler;

// Fraction of the key light reaching `world_position` (1 = lit). The receiver is pushed along its
// geometric normal by a texel-sized offset against acne. 3×3 tent filter from 4 bilinear compare
// taps (Castaño 2013, Filament PCF_Low). Outside the map, or with shadows off, it is lit; it fades
// to lit over the last 10% of the shadow distance.
float shadow_visibility(float3 world_position, float3 geometric_normal) {
    float shadow_distance = scene.shadow_params.y;
    if (shadow_distance <= 0.0) {
        return 1.0; // uniform branch: shadows off
    }

    float3 receiver = world_position + geometric_normal * scene.shadow_params.x;
    float4 clip     = mul(scene.shadow_view_projection, float4(receiver, 1.0));
    float2 uv       = clip.xy * float2(0.5, -0.5) + 0.5; // NDC y up, texture v down
    float  depth    = clip.z;

    // SampleCmpLevelZero (textureSampleCompareLevel) is valid in non-uniform control flow
    TentTaps taps = tent_taps(uv, scene.shadow_params.w, scene.shadow_params.z);
    float    sum  = 0.0;
    for (uint tap = 0; tap < 4; tap++) {
        sum += taps.weight[tap] * shadow_map.SampleCmpLevelZero(shadow_sampler, taps.uv[tap], depth);
    }
    bool  outside = any(uv < 0.0) || any(uv > 1.0) || depth > 1.0;
    float fade    = saturate((length(world_position - camera.position.xyz) - 0.9 * shadow_distance) / (0.1 * shadow_distance));
    return outside ? 1.0 : lerp(sum / 16.0, 1.0, fade);
}

// PBR material textures (set 1) — bindings match RendererMaterial texture layout (0-7).
[[vk::binding(0, 1)]] Texture2D    base_color_texture;
[[vk::binding(1, 1)]] SamplerState base_color_sampler;
[[vk::binding(2, 1)]] Texture2D    normal_texture;
[[vk::binding(3, 1)]] SamplerState normal_sampler;
[[vk::binding(4, 1)]] Texture2D    metallic_roughness_texture;
[[vk::binding(5, 1)]] SamplerState metallic_roughness_sampler;
[[vk::binding(6, 1)]] Texture2D    emissive_texture;
[[vk::binding(7, 1)]] SamplerState emissive_sampler;

// Material array (set 2), indexed by the draw's material slot
#define MATERIAL_ARRAY_SET 2
#include "material.hlsl"

static const float PI                 = 3.14159265359;
static const float MAX_REFLECTION_LOD = 4.0;
// Least cosine between the shading normal and the view direction
static const float MIN_NORMAL_DOT_VIEW = 0.02;

#include "oct_ibl.hlsl"

// BRDF math in `real`: 16-bit floats in the PILL_F16 variants (double rate on most GPUs), else
// 32-bit. Both evaluate the same formulas, arranged so every intermediate and every divisor's
// reciprocal stays in normal fp16 range; radiances (light intensities can exceed it) stay float.
// Bounds: D ≤ 891 (the 1e-7 term), the Smith visibility ≤ 1/(4·k²) ≈ 11.4.
#ifdef PILL_F16
typedef half  real;
typedef half3 real3;
// Kelemen's L·H² floor: 1e-7 is below fp16's smallest normal (6.1e-5)
static const real MIN_DIVISOR = real(6.1e-5);
#else
typedef float  real;
typedef float3 real3;
static const real MIN_DIVISOR = 1e-7;
#endif
static const real REAL_PI = real(3.14159265359);

// GGX D = α² / (π·d² + 1e-7), d = NoH²(α² − 1) + 1 = |N×H|² + (NoH·α)² (Lagrange's identity:
// no 1 − NoH² cancellation near highlights in low precision, Filament). Numerator and
// denominator are scaled by 32² = 1024: the 1e-7 term and α² (≥ 6.25e-6) become normal fp16,
// and the denominator stays ≤ π·32² ≈ 3217, so its reciprocal (GPUs divide by multiplying with
// it) does not fall below fp16's smallest normal and flush to zero.
real distribution_ggx(real3 normal, real3 half_vector, real roughness) {
    real  alpha_32      = max(roughness * roughness, real(0.0025)) * real(32.0);
    real3 n_cross_h     = cross(normal, half_vector);
    real  n_dot_h_alpha = max(dot(normal, half_vector), real(0.0)) * alpha_32 * real(0.17677670); // / √32
    real  d_32          = dot(n_cross_h, n_cross_h) * real(32.0) + n_dot_h_alpha * n_dot_h_alpha;
    return alpha_32 * alpha_32 / (REAL_PI * d_32 * d_32 + real(0.0001024));
}

// Smith-Schlick G / (4·NoV·NoL + 1e-4) (Heitz's k for direct light), with G's NoV·NoL factored
// out of its denominators so the quotient stays bounded
real visibility_smith(real normal_dot_view, real normal_dot_light, real roughness) {
    real roughness_plus_one = roughness + real(1.0);
    real k_factor           = roughness_plus_one * roughness_plus_one / real(8.0);
    real view_light         = normal_dot_view * normal_dot_light;
    // ≤ 0.25 over a divisor ≥ k² ≈ 0.022: no fp16 subnormals on either side
    real normalized         = view_light / (real(4.0) * view_light + real(0.0001));
    return normalized / ((normal_dot_light * (real(1.0) - k_factor) + k_factor)
                       * (normal_dot_view * (real(1.0) - k_factor) + k_factor));
}

// Schlick Fresnel; an F0 of exactly zero (the lighting-only views) means no specular at all, so
// it also drops the grazing term
real specular_present(real3 base_reflectivity) {
    return max(base_reflectivity.r, max(base_reflectivity.g, base_reflectivity.b)) > real(0.0) ? real(1.0) : real(0.0);
}

real3 fresnel_schlick(real cos_theta, real3 base_reflectivity) {
    real grazing = pow(real(1.0) - cos_theta, real(5.0)) * specular_present(base_reflectivity);
    return base_reflectivity + (real3(1.0, 1.0, 1.0) - base_reflectivity) * grazing;
}

real3 fresnel_schlick_roughness(real cos_theta, real3 base_reflectivity, real roughness) {
    real grazing = pow(real(1.0) - cos_theta, real(5.0)) * specular_present(base_reflectivity);
    real smooth  = real(1.0) - roughness;
    return base_reflectivity + (max(real3(smooth, smooth, smooth), base_reflectivity) - base_reflectivity) * grazing;
}

// Kelemen visibility: cheaper than Smith, enough for a thin dielectric coat (Filament)
real visibility_kelemen(real light_dot_half) {
    return real(0.25) / max(light_dot_half * light_dot_half, MIN_DIVISOR);
}

// Clear-coat specular lobe of one light: GGX D, Kelemen V, Schlick F with F0 = coat weight
float3 clearcoat_directional_light(
    float3 normal, float3 view_direction, float weight, float roughness,
    float3 light_direction_to_surface, float4 light_color
) {
    float3 light_direction  = normalize(-light_direction_to_surface);
    float  normal_dot_light = max(dot(normal, light_direction), 0.0);
    real3  half_vector      = normalize(real3(view_direction) + real3(light_direction));
    real   light_dot_half   = max(dot(real3(light_direction), half_vector), real(0.0));
    real   coat             = real(weight);
    real   distribution     = distribution_ggx(real3(normal), half_vector, real(roughness));
    real   fresnel          = coat + (real(1.0) - coat) * pow(real(1.0) - light_dot_half, real(5.0));
    // D (≤ 5100) · V (≤ 4096 at grazing L·H) can exceed fp16: their product in float. The
    // grazing Fresnel term does not vanish at weight 0, so the weight also gates the lobe
    float  lobe = float(distribution) * float(visibility_kelemen(light_dot_half)) * float(fresnel * coat);
    return lobe * light_color.w * light_color.xyz * normal_dot_light;
}

float3 accumulate_directional_light(
    float3 normal, float3 view_direction, float3 base_reflectivity,
    float3 albedo, float roughness, float metallic,
    float3 light_direction_to_surface, float4 light_color
) {
    // light_direction_to_surface is direction from light to surface; incoming light is opposite.
    float3 light_direction  = normalize(-light_direction_to_surface);
    float  normal_dot_light = max(dot(normal, light_direction), 0.0);
    real3  n                = real3(normal);
    real3  v                = real3(view_direction);
    real3  half_vector      = normalize(v + real3(light_direction));
    real   rough            = real(roughness);
    real   distribution     = distribution_ggx(n, half_vector, rough);
    real   visibility       = visibility_smith(max(dot(n, v), real(0.0)), real(normal_dot_light), rough);
    real3  fresnel          = fresnel_schlick(max(dot(half_vector, v), real(0.0)), real3(base_reflectivity));
    real3  diffuse          = (real3(1.0, 1.0, 1.0) - fresnel) * (real(1.0) - real(metallic)) * real3(albedo) / REAL_PI;
    real3  specular         = distribution * visibility * fresnel;
    // BRDF in `real`, radiance in float
    return float3(diffuse + specular) * light_color.w * light_color.xyz * normal_dot_light;
}

// Targets 1–4 exist only when the pass has velocity / ambient / SSR targets; otherwise ignored.
// Froxel of a fragment: its screen tile and the exponential depth slice of its view depth
// (FroxelLayout::slice in froxelizer.rs).
uint froxel_index(float2 pixel, float3 world_position) {
    uint   columns = uint(scene.froxel_grid.x);
    uint   rows    = uint(scene.froxel_grid.y);
    uint2  tile    = min(uint2(pixel / scene.froxel_grid.z), uint2(columns - 1, rows - 1));
    float  depth   = max(dot(world_position - camera.position.xyz, scene.camera_forward.xyz), 1e-6);
    float  slices  = scene.froxel_depth.z;
    float  slice   = clamp(floor(log2(depth / scene.froxel_depth.y) * scene.froxel_depth.x + slices), 0.0, slices - 1.0);
    return tile.x + tile.y * columns + uint(slice) * columns * rows;
}

// Cube face of a light-to-surface direction: its dominant axis, in +X −X +Y −Y +Z −Z order
// (cube_face in punctual_shadows.rs).
uint cube_face(float3 direction) {
    float3 magnitude = abs(direction);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return direction.x > 0.0 ? 0u : 1u;
    }
    if (magnitude.y >= magnitude.z) {
        return direction.y > 0.0 ? 2u : 3u;
    }
    return direction.z > 0.0 ? 4u : 5u;
}

// Fraction of a point/spot light reaching `world_position` (1 = lit, or the light has no shadow
// map this frame). Same normal offset and tent PCF as the key light. A point light's face is
// chosen for the offset receiver; faces carry a texel border (cube_face_fov), so the taps of a
// receiver on a face edge stay inside that face's map.
float punctual_shadow_visibility(PunctualLight light, float3 world_position, float3 geometric_normal, float distance) {
    int layer = int(light.spot_scale_offset.z);
    if (layer < 0) {
        return 1.0;
    }
    // All maps of a frame share the params (map size); the normal offset per meter as well
    float3 receiver = world_position + geometric_normal * (punctual_shadows[layer].params.x * distance);
    if (light.spot_scale_offset.w > 0.5) {
        layer += int(cube_face(receiver - light.position_falloff.xyz));
    }
    PunctualShadow shadow = punctual_shadows[layer];
    float4 clip     = mul(shadow.view_projection, float4(receiver, 1.0));
    float3 ndc      = clip.xyz / max(clip.w, 1e-6);
    float2 uv       = ndc.xy * float2(0.5, -0.5) + 0.5;
    TentTaps taps   = tent_taps(uv, shadow.params.y, shadow.params.z);
    float    sum    = 0.0;
    for (uint tap = 0; tap < 4; tap++) {
        sum += taps.weight[tap] * punctual_shadow_maps.SampleCmpLevelZero(shadow_sampler, float3(taps.uv[tap], float(layer)), ndc.z);
    }
    bool outside = clip.w <= 0.0 || any(uv < 0.0) || any(uv > 1.0) || ndc.z > 1.0;
    return outside ? 1.0 : sum / 16.0;
}

// Point/spot lights of the fragment's froxel: Filament's inverse-square falloff with a smooth
// window to zero at the light's range, and the spot angle term. Returns the light count in `count`.
float3 accumulate_punctual_lights(
    float2 pixel, float3 world_position, float3 normal, float3 geometric_normal, float3 view_direction,
    float3 base_reflectivity, float3 albedo, float roughness, float metallic,
    float clearcoat, float coat_roughness, out uint count
) {
    count = 0;
    float3 radiance = float3(0.0, 0.0, 0.0);
    if (scene.froxel_grid.w < 0.5) {
        return radiance; // uniform branch: no lights
    }
    uint2 entry = froxel_entries[froxel_index(pixel, world_position)];
    count = entry.y;
    for (uint index = 0; index < entry.y; index++) {
        uint record = entry.x + index;
        uint light_index = (froxel_records[record >> 1] >> ((record & 1u) * 16u)) & 0xFFFFu;
        PunctualLight light = punctual_lights[light_index];
        float3 to_light        = light.position_falloff.xyz - world_position;
        float  distance_square = dot(to_light, to_light);
        float  window          = saturate(1.0 - pow(distance_square * light.position_falloff.w, 2.0));
        float3 incoming        = to_light * rsqrt(max(distance_square, 1e-8));
        float  angle           = saturate(dot(light.direction.xyz, -incoming) * light.spot_scale_offset.x + light.spot_scale_offset.y);
        float  attenuation     = window * window / max(distance_square, 1e-4) * angle * angle;
        if (attenuation > 0.0) {
            attenuation *= punctual_shadow_visibility(light, world_position, geometric_normal, sqrt(distance_square));
        }
        float4 light_color     = float4(light.color_intensity.rgb, light.color_intensity.w * attenuation);
        radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, -incoming, light_color);
        if (clearcoat > 0.0) {
            radiance += clearcoat_directional_light(normal, view_direction, clearcoat, coat_roughness, -incoming, light_color);
        }
    }
    return radiance;
}

// Light count → blue (1) … green … red (≥ 32); black for none
float3 froxel_heatmap(uint count) {
    if (count == 0) {
        return float3(0.0, 0.0, 0.0);
    }
    float t = saturate(float(count) / 32.0);
    return saturate(float3(t * 2.0 - 1.0, 1.0 - abs(t * 2.0 - 1.0), 1.0 - t * 2.0)) + 0.05;
}

struct FragmentOutput {
    [[vk::location(0)]] float4 color    : SV_TARGET0;
    [[vk::location(1)]] float2 velocity : SV_TARGET1; // UV offset to last frame's position
    [[vk::location(2)]] float4 ambient  : SV_TARGET2; // fogged IBL term alone, darkened by SSAO
    // PassSsr inputs: rgb = fogged specular weight F·A + B of the IBL term, a = roughness
    [[vk::location(3)]] float4 ssr_surface : SV_TARGET3;
    [[vk::location(4)]] float2 ssr_normal  : SV_TARGET4; // octahedral-encoded shading normal
};

[shader("fragment")]
FragmentOutput fs_main(
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0,
    [[vk::location(1)]] float3 world_position      : TEXCOORD1,
    [[vk::location(2)]] float3 input_normal        : TEXCOORD2,
    [[vk::location(3)]] float3 input_tangent       : TEXCOORD3,
    [[vk::location(4)]] float3 input_bitangent     : TEXCOORD4,
    [[vk::location(5)]] float4 clip_position          : TEXCOORD5,
    [[vk::location(6)]] float4 previous_clip_position : TEXCOORD6,
    [[vk::location(7)]] nointerpolation uint material_index : TEXCOORD7,
                        float4 frag_coord             : SV_Position
) {
    MaterialParams material = load_material(material_index);
    float4 base_color         = base_color_texture.Sample(base_color_sampler, texture_coordinates);
#ifdef ALPHA_TEST
    // Separate shader variant: discard turns off early depth rejection for the whole pipeline
    if (base_color.a < material.alpha_cutoff) {
        discard;
    }
#endif
    // Double-sided surfaces are lit from the side the camera sees: flipped by the vertex normal
    // vs the view, not by triangle winding (foliage cards often wind against their normals)
    float3 view_direction     = normalize(camera.position.xyz - world_position);
    float  face_sign          = material.double_sided > 0.5 && dot(input_normal, view_direction) < 0.0 ? -1.0 : 1.0;
    float3 vertex_normal      = normalize(input_normal) * face_sign;
    float3 albedo             = base_color.rgb * material.base_color_factor;
    float2 metallic_roughness = metallic_roughness_texture.Sample(metallic_roughness_sampler, texture_coordinates).gb;
    // metallic_roughness.x = G channel (roughness 0=smooth, 1=rough); metallic_roughness.y = B channel (metallic 0=dielectric, 1=metal)
    float  roughness = clamp(metallic_roughness.x * material.roughness_factor, 0.045, 0.99);
    float  metallic  = metallic_roughness.y * material.metallic_factor;
    // Tangent-space normal map; the default texel (0.5, 0.5, 1) keeps the vertex normal
    float3 tangent_normal    = normal_texture.Sample(normal_sampler, texture_coordinates).xyz * 2.0 - 1.0;
    // z rebuilt from x and y (BC5-style): two-channel maps leave blue empty, and mips filtered
    // between a two-channel and a full level would otherwise mix z toward the surface. For a
    // unit-length map this is its own z.
    tangent_normal.z = sqrt(saturate(1.0 - dot(tangent_normal.xy, tangent_normal.xy)));
    tangent_normal.y        *= material.normal_y_sign;
    // Meshes without a usable tangent frame (degenerate UVs) keep the vertex normal
    bool   has_tangent_frame = dot(input_tangent, input_tangent) > 1e-6;
    float3 normal            = has_tangent_frame
        ? normalize(tangent_normal.x * input_tangent + tangent_normal.y * input_bitangent + tangent_normal.z * input_normal) * face_sign
        : vertex_normal;
    // A normal-mapped normal can tilt past the horizon at grazing angles; facing away it would
    // get full Fresnel and no diffuse (a silver sheen). Keep it in the visible hemisphere.
    float  normal_dot_view   = dot(normal, view_direction);
    if (normal_dot_view < MIN_NORMAL_DOT_VIEW) {
        normal = normalize(normal + view_direction * (MIN_NORMAL_DOT_VIEW - normal_dot_view));
    }
    // Rain (`RenderStateComponent::wetness`): water fills the pores of rough dielectrics (darker
    // albedo) and covers surfaces with a film (lower roughness), fully on upward-facing ones
    // (ground, ledges, roofs), partly on walls where it runs off
    float wetness = scene.ibl_intensity.z;
    if (wetness > 0.0) {
        float porosity = (1.0 - metallic) * roughness;
        albedo   *= 1.0 - WET_DARKENING * wetness * porosity;
        float film = wetness * lerp(WET_WALL_FILM, 1.0, saturate(vertex_normal.y));
        roughness = lerp(roughness, WET_ROUGHNESS, film);
    }
    // Lighting-only views (light audits): every surface white, diffuse only, unlit by itself; the
    // light reaching it is all that shows (method: Unreal's Lighting Only / Detail Lighting views)
    float debug_view    = scene.ibl_intensity.w;
    bool  lighting_only = debug_view > 0.5;
    if (lighting_only) {
        albedo   = float3(1.0, 1.0, 1.0);
        metallic = 0.0;
        if (debug_view < 1.5) {
            normal = vertex_normal;
        }
    }
    float  clearcoat         = lighting_only ? 0.0 : material.clearcoat;
    float3 base_reflectivity = lighting_only
        ? float3(0.0, 0.0, 0.0)
        : lerp(float3(0.04, 0.04, 0.04), albedo, float3(metallic, metallic, metallic));
    float3 outgoing_radiance = float3(0.0, 0.0, 0.0);
    // Key light (0) shadow; point/spot shadows are in accumulate_punctual_lights
    float  key_visibility    = shadow_visibility(world_position, vertex_normal);
    float4 key_light_color   = float4(scene.light_color0.rgb, scene.light_color0.w * key_visibility);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction0.xyz, key_light_color);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction1.xyz, scene.light_color1);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction2.xyz, scene.light_color2);
    float coat_roughness = clamp(material.clearcoat_roughness, 0.045, 0.99);
    uint  punctual_count;
    outgoing_radiance += accumulate_punctual_lights(frag_coord.xy, world_position, normal, vertex_normal, view_direction, base_reflectivity, albedo, roughness, metallic, clearcoat, coat_roughness, punctual_count);
    if (clearcoat > 0.0) {
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, clearcoat, coat_roughness, scene.light_direction0.xyz, key_light_color);
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, clearcoat, coat_roughness, scene.light_direction1.xyz, scene.light_color1);
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, clearcoat, coat_roughness, scene.light_direction2.xyz, scene.light_color2);
    }
    // Diffuse IBL ambient.
    float3 specular_factor = float3(fresnel_schlick(real(max(dot(normal, view_direction), 0.0)), real3(base_reflectivity)));
    float3 diffuse_factor  = (float3(1.0, 1.0, 1.0) - specular_factor) * (1.0 - metallic);
    // Indirect diffuse: DDGI probes when on (their rays that miss already see the IBL), else IBL
    float3 sky_irradiance  = sample_ibl_irradiance(irradiance_texture, irradiance_sampler, normal) * scene.ibl_intensity.x;
    float3 irradiance      = sky_irradiance;
    // Specular occlusion: the sky reflection is occluded as much as the probes find the sky light
    // occluded here (inside foliage, under awnings), which the IBL alone can't know
    float  specular_occlusion = 1.0;
    if (ddgi.enabled > 0) {
        // Probes without an estimate yet (newly exposed by a moving grid) fall back to the IBL
        float4 probe_irradiance = ddgi_irradiance(world_position, normal);
        irradiance = lerp(irradiance, probe_irradiance.rgb * ddgi.intensity, probe_irradiance.a);
        const float3 luminance = float3(0.2126, 0.7152, 0.0722);
        float sky = dot(sky_irradiance, luminance);
        specular_occlusion = sky > 1e-4 ? saturate(dot(irradiance, luminance) / sky) : 1.0;
    }
    float3 ambient_diffuse = diffuse_factor * irradiance * albedo;
    // Specular IBL.
    float3 reflection        = reflect(-view_direction, normal);
    float3 prefiltered_color = sample_ibl_specular(prefilter_texture, prefilter_sampler, reflection, roughness * MAX_REFLECTION_LOD);
    float2 environment_brdf  = brdf_lut_texture.Sample(brdf_lut_sampler, float2(max(dot(normal, view_direction), 0.0), roughness)).rg;
    float3 fresnel           = float3(fresnel_schlick_roughness(real(max(dot(normal, view_direction), 0.0)), real3(base_reflectivity), real(roughness)));
    float3 specular_weight   = lighting_only
        ? float3(0.0, 0.0, 0.0)
        : (fresnel * environment_brdf.x + environment_brdf.y) * specular_occlusion;
    float3 specular_ibl      = prefiltered_color * specular_weight * scene.ibl_intensity.x;
    float3 emissive = lighting_only
        ? float3(0.0, 0.0, 0.0)
        : emissive_texture.Sample(emissive_sampler, texture_coordinates).rgb * material.emissive_factor;
    // With SSR (ibl_intensity.y) PassSsr adds the specular term: reflection where its ray hits, IBL elsewhere
    bool   ssr   = scene.ibl_intensity.y > 0.5;
    float3 color = outgoing_radiance + ambient_diffuse + (ssr ? float3(0.0, 0.0, 0.0) : specular_ibl) + emissive;
    float  distance = length(camera.position.xyz - world_position);
    float  fog_density = scene.fog_color_density.w;
    float  fog      = 1.0 - exp(-fog_density * fog_density * distance * distance);
    FragmentOutput output;
    output.color    = float4(lerp(color, scene.fog_color_density.xyz, fog), material.opacity * base_color.a);
    if (ddgi.enabled == 2) {
        output.color.rgb = ambient_diffuse;
    }
    if (ddgi.enabled == 3) {
        output.color.rgb = ddgi_cell_debug_color(world_position);
    }
    if (scene.froxel_depth.w > 0.5) {
        output.color.rgb = froxel_heatmap(punctual_count);
    }
    // Motion vector in UV units (NDC y points up, UV v points down); w clamped for points that were
    // at or behind the camera last frame
    float2 ndc          = clip_position.xy / max(clip_position.w, 1e-4);
    float2 previous_ndc = previous_clip_position.xy / max(previous_clip_position.w, 1e-4);
    output.velocity = (ndc - previous_ndc) * float2(0.5, -0.5);
    // What SSAO takes back must be in `color`; with SSR, PassSsr adds the specular term instead
    output.ambient  = float4((ambient_diffuse + (ssr ? float3(0.0, 0.0, 0.0) : specular_ibl)) * (1.0 - fog), 0.0);
    bool no_reflections = ddgi.enabled >= 2 || scene.froxel_depth.w > 0.5 || lighting_only;
    output.ssr_surface = float4(ssr && !no_reflections ? specular_weight * (1.0 - fog) : float3(0.0, 0.0, 0.0), roughness);
    output.ssr_normal  = oct_encode(normal);
    return output;
}
