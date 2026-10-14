// PBR fragment shader body, included by pbr_opaque_fragment.hlsl and (with POST_FX, which adds
// the velocity and ambient outputs) pbr_opaque_post_fx_fragment.hlsl.
// Full GGX microfacet BRDF with 3 directional lights and optional IBL fallback.

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
    float4 ibl_intensity;     // x
    float4 light_direction0;  // xyz = direction from the light toward the scene
    float4 light_color0;      // rgb = color, w = intensity
    float4 light_direction1;
    float4 light_color1;
    float4 light_direction2;
    float4 light_color2;
    column_major float4x4 shadow_view_projection; // world → light clip space (key light)
    float4 shadow_params; // x = normal offset (world), y = shadow distance (0 = none), z = 1 / map size, w = map size
};
[[vk::binding(7, 0)]] ConstantBuffer<Scene> scene;

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

    float  texel = scene.shadow_params.z;
    float2 scaled = uv * scene.shadow_params.w + 0.5;
    float2 base   = (floor(scaled) - 0.5) * texel;
    float2 st     = frac(scaled);
    float2 uw     = float2(3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x);
    float2 vw     = float2(3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y);
    float2 u      = float2((2.0 - st.x) / uw.x - 1.0, st.x / uw.y + 1.0) * texel;
    float2 v      = float2((2.0 - st.y) / vw.x - 1.0, st.y / vw.y + 1.0) * texel;
    // SampleCmpLevelZero (textureSampleCompareLevel) is valid in non-uniform control flow
    float sum = uw.x * vw.x * shadow_map.SampleCmpLevelZero(shadow_sampler, base + float2(u.x, v.x), depth)
              + uw.y * vw.x * shadow_map.SampleCmpLevelZero(shadow_sampler, base + float2(u.y, v.x), depth)
              + uw.x * vw.y * shadow_map.SampleCmpLevelZero(shadow_sampler, base + float2(u.x, v.y), depth)
              + uw.y * vw.y * shadow_map.SampleCmpLevelZero(shadow_sampler, base + float2(u.y, v.y), depth);
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

// PBR params UBO (set 2) — 48 bytes, mirrors MaterialParams in renderer_material.rs.
// Each float3 is followed by a float so all 16-byte rows are fully packed (std140).
struct MaterialParams {
    float3 base_color_factor;
    float  roughness_factor;
    float3 emissive_factor;
    float  metallic_factor;
    float  clearcoat;           // coat weight, 0 = no coat
    float  clearcoat_roughness; // perceptual
    float  opacity;             // < 1: drawn blended by the translucent pass
    float  _padding;
};
[[vk::binding(0, 2)]] ConstantBuffer<MaterialParams> material;

static const float PI                 = 3.14159265359;
static const float MAX_REFLECTION_LOD = 4.0;

#include "equirect.hlsl"

float distribution_ggx(float3 normal, float3 half_vector, float roughness) {
    // Add epsilon to avoid singularities at very low roughness.
    float alpha           = max(roughness * roughness, 0.0025);
    float alpha_squared   = alpha * alpha;
    float normal_dot_half = max(dot(normal, half_vector), 0.0);
    float denominator     = (normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0);
    return alpha_squared / (PI * denominator * denominator + 1e-7);
}

float geometry_schlick_ggx(float normal_dot_view, float roughness) {
    // Heitz's k for direct lighting approximation.
    float roughness_plus_one = roughness + 1.0;
    float k_factor           = (roughness_plus_one * roughness_plus_one) / 8.0;
    return normal_dot_view / (normal_dot_view * (1.0 - k_factor) + k_factor);
}

float geometry_smith(float3 normal, float3 view_direction, float3 light_direction, float roughness) {
    float normal_dot_view  = max(dot(normal, view_direction), 0.0);
    float normal_dot_light = max(dot(normal, light_direction), 0.0);
    return geometry_schlick_ggx(normal_dot_light, roughness) * geometry_schlick_ggx(normal_dot_view, roughness);
}

float3 fresnel_schlick(float cos_theta, float3 base_reflectivity) {
    return base_reflectivity + (float3(1.0, 1.0, 1.0) - base_reflectivity) * pow(1.0 - cos_theta, 5.0);
}

float3 fresnel_schlick_roughness(float cos_theta, float3 base_reflectivity, float roughness) {
    return base_reflectivity + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), base_reflectivity) - base_reflectivity) * pow(1.0 - cos_theta, 5.0);
}


// Kelemen visibility: cheaper than Smith, enough for a thin dielectric coat (Filament)
float visibility_kelemen(float light_dot_half) {
    return 0.25 / max(light_dot_half * light_dot_half, 1e-7);
}

// Clear-coat specular lobe of one light: GGX D, Kelemen V, Schlick F with F0 = coat weight
float3 clearcoat_directional_light(
    float3 normal, float3 view_direction, float weight, float roughness,
    float3 light_direction_to_surface, float4 light_color
) {
    float3 light_direction  = normalize(-light_direction_to_surface);
    float3 half_vector      = normalize(view_direction + light_direction);
    float  normal_dot_light = max(dot(normal, light_direction), 0.0);
    float  light_dot_half   = max(dot(light_direction, half_vector), 0.0);
    float  distribution     = distribution_ggx(normal, half_vector, roughness);
    float  fresnel          = weight + (1.0 - weight) * pow(1.0 - light_dot_half, 5.0);
    // The grazing Fresnel term does not vanish at weight 0, so the weight also gates the lobe
    return distribution * visibility_kelemen(light_dot_half) * fresnel * weight
         * light_color.w * light_color.xyz * normal_dot_light;
}

float3 accumulate_directional_light(
    float3 normal, float3 view_direction, float3 base_reflectivity,
    float3 albedo, float roughness, float metallic,
    float3 light_direction_to_surface, float4 light_color
) {
    // light_direction_to_surface is direction from light to surface; incoming light is opposite.
    float3 light_direction     = normalize(-light_direction_to_surface);
    float3 half_vector         = normalize(view_direction + light_direction);
    float3 radiance            = light_color.w * light_color.xyz;
    float  normal_distribution = distribution_ggx(normal, half_vector, roughness);
    float  geometry            = geometry_smith(normal, view_direction, light_direction, roughness);
    float3 fresnel             = fresnel_schlick(max(dot(half_vector, view_direction), 0.0), base_reflectivity);
    float3 diffuse_factor      = (float3(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    float  denominator         = 4.0 * max(dot(normal, view_direction), 0.0) * max(dot(normal, light_direction), 0.0) + 0.0001;
    float3 specular            = (normal_distribution * geometry * fresnel) / float3(denominator, denominator, denominator);
    return (diffuse_factor * (albedo / PI) + specular) * radiance * max(dot(normal, light_direction), 0.0);
}

// POST_FX: targets 1 and 2 exist only when the pass has velocity / ambient targets; otherwise ignored.
struct FragmentOutput {
    [[vk::location(0)]] float4 color    : SV_TARGET0;
#ifdef POST_FX
    [[vk::location(1)]] float2 velocity : SV_TARGET1; // UV offset to last frame's position
    [[vk::location(2)]] float4 ambient  : SV_TARGET2; // fogged IBL term alone, darkened by SSAO
#endif
};

[shader("fragment")]
FragmentOutput fs_main(
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0,
    [[vk::location(1)]] float3 world_position      : TEXCOORD1,
    [[vk::location(2)]] float3 input_normal        : TEXCOORD2,
    [[vk::location(3)]] float3 input_tangent       : TEXCOORD3,
    [[vk::location(4)]] float3 input_bitangent     : TEXCOORD4
#ifdef POST_FX
  , [[vk::location(5)]] float4 clip_position          : TEXCOORD5,
    [[vk::location(6)]] float4 previous_clip_position : TEXCOORD6
#endif
) {
    float4 base_color         = base_color_texture.Sample(base_color_sampler, texture_coordinates);
    float3 albedo             = base_color.rgb * material.base_color_factor;
    float2 metallic_roughness = metallic_roughness_texture.Sample(metallic_roughness_sampler, texture_coordinates).gb;
    // metallic_roughness.x = G channel (roughness 0=smooth, 1=rough); metallic_roughness.y = B channel (metallic 0=dielectric, 1=metal)
    float  roughness = clamp(metallic_roughness.x * material.roughness_factor, 0.045, 0.99);
    float  metallic  = metallic_roughness.y * material.metallic_factor;
    // Tangent-space normal map; the default texel (0.5, 0.5, 1) keeps the vertex normal
    float3 tangent_normal    = normal_texture.Sample(normal_sampler, texture_coordinates).xyz * 2.0 - 1.0;
    // Meshes without a usable tangent frame (degenerate UVs) keep the vertex normal
    bool   has_tangent_frame = dot(input_tangent, input_tangent) > 1e-6;
    float3 normal            = has_tangent_frame
        ? normalize(tangent_normal.x * input_tangent + tangent_normal.y * input_bitangent + tangent_normal.z * input_normal)
        : normalize(input_normal);
    float3 view_direction    = normalize(camera.position.xyz - world_position);
    float3 base_reflectivity = lerp(float3(0.04, 0.04, 0.04), albedo, float3(metallic, metallic, metallic));
    float3 outgoing_radiance = float3(0.0, 0.0, 0.0);
    // Only the key light (0) casts shadows
    float  key_visibility    = shadow_visibility(world_position, normalize(input_normal));
    float4 key_light_color   = float4(scene.light_color0.rgb, scene.light_color0.w * key_visibility);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction0.xyz, key_light_color);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction1.xyz, scene.light_color1);
    outgoing_radiance += accumulate_directional_light(normal, view_direction, base_reflectivity, albedo, roughness, metallic, scene.light_direction2.xyz, scene.light_color2);
    if (material.clearcoat > 0.0) {
        float coat_roughness = clamp(material.clearcoat_roughness, 0.045, 0.99);
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, material.clearcoat, coat_roughness, scene.light_direction0.xyz, key_light_color);
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, material.clearcoat, coat_roughness, scene.light_direction1.xyz, scene.light_color1);
        outgoing_radiance += clearcoat_directional_light(normal, view_direction, material.clearcoat, coat_roughness, scene.light_direction2.xyz, scene.light_color2);
    }
    // Diffuse IBL ambient.
    float3 specular_factor = fresnel_schlick(max(dot(normal, view_direction), 0.0), base_reflectivity);
    float3 diffuse_factor  = (float3(1.0, 1.0, 1.0) - specular_factor) * (1.0 - metallic);
    float3 irradiance      = irradiance_texture.Sample(irradiance_sampler, dir_to_equirect_uv(normal)).rgb;
    float3 ambient_diffuse = diffuse_factor * irradiance * albedo * scene.ibl_intensity.x;
    // Specular IBL.
    float3 reflection        = reflect(-view_direction, normal);
    float3 prefiltered_color = prefilter_texture.SampleLevel(prefilter_sampler, dir_to_equirect_uv(reflection), roughness * MAX_REFLECTION_LOD).rgb;
    float2 environment_brdf  = brdf_lut_texture.Sample(brdf_lut_sampler, float2(max(dot(normal, view_direction), 0.0), roughness)).rg;
    float3 fresnel           = fresnel_schlick_roughness(max(dot(normal, view_direction), 0.0), base_reflectivity, roughness);
    float3 specular_ibl      = prefiltered_color * (fresnel * environment_brdf.x + environment_brdf.y) * scene.ibl_intensity.x;
    float3 emissive = emissive_texture.Sample(emissive_sampler, texture_coordinates).rgb * material.emissive_factor;
    float3 color = outgoing_radiance + ambient_diffuse + specular_ibl + emissive;
    float  distance = length(camera.position.xyz - world_position);
    float  fog_density = scene.fog_color_density.w;
    float  fog      = 1.0 - exp(-fog_density * fog_density * distance * distance);
    FragmentOutput output;
    output.color    = float4(lerp(color, scene.fog_color_density.xyz, fog), material.opacity * base_color.a);
#ifdef POST_FX
    // Motion vector in UV units (NDC y points up, UV v points down); w clamped for points that were
    // at or behind the camera last frame
    float2 ndc          = clip_position.xy / max(clip_position.w, 1e-4);
    float2 previous_ndc = previous_clip_position.xy / max(previous_clip_position.w, 1e-4);
    output.velocity = (ndc - previous_ndc) * float2(0.5, -0.5);
    output.ambient  = float4((ambient_diffuse + specular_ibl) * (1.0 - fog), 0.0);
#endif
    return output;
}
