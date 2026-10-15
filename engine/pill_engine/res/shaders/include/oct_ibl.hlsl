#ifndef OCT_IBL_HLSL
#define OCT_IBL_HLSL
// Sampling of the octahedral IBL maps baked by PassIblBake (ibl.rs): each mip is a `size`² map
// with a one-texel border, so bilinear taps never cross the octahedral seams.

#include "octahedral.hlsl"

static const float IBL_SPECULAR_SIZE   = 128.0; // ibl.rs IBL_SPECULAR_SIZE
static const float IBL_SPECULAR_MAX_LOD = 4.0;  // IBL_SPECULAR_MIPS - 1
static const float IBL_IRRADIANCE_SIZE = 32.0;  // ibl.rs IBL_IRRADIANCE_SIZE

// UV of `direction` in a `size`² oct map whose border texels sit outside the mapped area
float2 oct_ibl_uv(float3 direction, float size) {
    float2 unit = oct_encode(normalize(direction)) * 0.5 + 0.5;
    return (1.0 + unit * (size - 2.0)) / size;
}

float3 sample_ibl_irradiance(Texture2D map, SamplerState linear_clamp, float3 normal) {
    return map.SampleLevel(linear_clamp, oct_ibl_uv(normal, IBL_IRRADIANCE_SIZE), 0.0).rgb;
}

// Prefiltered radiance at `lod` (roughness · IBL_SPECULAR_MAX_LOD): the two nearest mips, each
// with its own border-aware UV, blended (hardware trilinear would mix mismatched UVs)
float3 sample_ibl_specular(Texture2D map, SamplerState linear_clamp, float3 direction, float lod) {
    float lod_clamped = clamp(lod, 0.0, IBL_SPECULAR_MAX_LOD);
    float lower       = floor(lod_clamped);
    float upper       = min(lower + 1.0, IBL_SPECULAR_MAX_LOD);
    float3 a = map.SampleLevel(linear_clamp, oct_ibl_uv(direction, IBL_SPECULAR_SIZE / exp2(lower)), lower).rgb;
    float3 b = map.SampleLevel(linear_clamp, oct_ibl_uv(direction, IBL_SPECULAR_SIZE / exp2(upper)), upper).rgb;
    return lerp(a, b, lod_clamped - lower);
}
#endif
