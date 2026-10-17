// PassIblBake: one octahedral IBL map mip per draw (ibl.rs). Each texel, border included, gets
// the direction of its inner texel and integrates the environment around it: GGX-prefiltered
// radiance (specular mips) or the cosine-weighted average radiance E/π (irradiance). Samples
// read the source equirect mip whose texels match the sample's solid angle (filtered importance
// sampling), so few samples stay free of fireflies.

#include "include/octahedral.hlsl"
#include "include/equirect.hlsl"

struct IblBakeParams {
    float size;         // target mip width and height, border included
    float roughness;    // GGX roughness (specular)
    float mode;         // 0 = specular, 1 = irradiance
    float sample_count;
    float2 source_size; // source equirect mip 0 in texels
    float source_max_lod;
    float rotation;     // environment turned about +Y, in turns
};

[[vk::binding(0, 0)]] Texture2D                        source;
[[vk::binding(1, 0)]] SamplerState                     source_sampler;
[[vk::binding(2, 0)]] ConstantBuffer<IblBakeParams>    params;

static const float PI = 3.14159265359;

// Inner texel a texel of an `inner`² map with a one-texel border holds (ibl.rs
// oct_border_inner_texel): border rows and columns mirror the opposite inner edge
int2 oct_ibl_inner_texel(int2 texel, int inner) {
    int2 c = texel - 1;
    if (c.x < 0 || c.x >= inner) {
        c.x = clamp(c.x, 0, inner - 1);
        c.y = inner - 1 - c.y;
    }
    if (c.y < 0 || c.y >= inner) {
        c.y = clamp(c.y, 0, inner - 1);
        c.x = inner - 1 - c.x;
    }
    return c;
}

float radical_inverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float3 tangent_to_world(float3 local, float3 normal) {
    float3 up        = abs(normal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 tangent   = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return tangent * local.x + bitangent * local.y + normal * local.z;
}

// Source mip whose texel solid angle matches a sample of density `pdf` among `count`
float source_lod(float pdf, float count) {
    float sample_solid_angle = 1.0 / (count * pdf + 1e-6);
    float texel_solid_angle  = 4.0 * PI / (params.source_size.x * params.source_size.y);
    return clamp(0.5 * log2(sample_solid_angle / texel_solid_angle) + 1.0, 0.0, params.source_max_lod);
}

float3 environment(float3 direction, float lod) {
    float2 uv = dir_to_equirect_uv(direction);
    uv.x += params.rotation; // the sampler repeats in u
    return source.SampleLevel(source_sampler, uv, lod).rgb;
}

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    int    inner     = int(params.size) - 2;
    int2   texel     = oct_ibl_inner_texel(int2(frag_coord.xy), inner);
    float3 normal    = oct_decode(((float2(texel) + 0.5) / float(inner)) * 2.0 - 1.0);
    uint   count     = uint(params.sample_count);

    float3 sum    = float3(0.0, 0.0, 0.0);
    float  weight = 0.0;
    for (uint i = 0u; i < count; i++) {
        float2 xi = float2(float(i) / float(count), radical_inverse(i));
        float  phi = 2.0 * PI * xi.x;
        if (params.mode > 0.5) {
            // Cosine-weighted hemisphere: the plain average is E/π
            float  cos_theta = sqrt(1.0 - xi.y);
            float  sin_theta = sqrt(xi.y);
            float3 direction = tangent_to_world(float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta), normal);
            sum    += environment(direction, source_lod(cos_theta / PI, float(count)));
            weight += 1.0;
        } else {
            // GGX half vectors around the normal, view = normal (split-sum prefilter)
            float  alpha     = params.roughness * params.roughness;
            float  cos_theta = sqrt((1.0 - xi.y) / (1.0 + (alpha * alpha - 1.0) * xi.y));
            float  sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
            float3 half_vector = tangent_to_world(float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta), normal);
            float3 light       = 2.0 * dot(normal, half_vector) * half_vector - normal;
            float  n_dot_l     = dot(normal, light);
            if (n_dot_l > 0.0) {
                // pdf of the light direction = D(h) · NoH / (4 · VoH), with V = N: D(h) / 4
                float denominator = cos_theta * cos_theta * (alpha * alpha - 1.0) + 1.0;
                float ggx         = alpha * alpha / (PI * denominator * denominator);
                sum    += environment(light, source_lod(ggx * 0.25, float(count))) * n_dot_l;
                weight += n_dot_l;
            }
        }
    }
    return float4(sum / max(weight, 1e-6), 1.0);
}
