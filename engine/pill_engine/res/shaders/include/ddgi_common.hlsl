#ifndef DDGI_COMMON_HLSL
#define DDGI_COMMON_HLSL
// DDGI probe grid: layout, octahedral mapping, L1 SH and the 8-probe irradiance query, after
// WickedEngine's ShaderInterop_DDGI.h. The includer declares `ddgi` (ConstantBuffer<DdgiParams>),
// and, for the irradiance query, `ddgi_probes` (StructuredBuffer<DdgiProbe>), `ddgi_depth`
// (Texture2D) and DDGI_SAMPLER (a linear clamp sampler, defined before the include).

#include "octahedral.hlsl"

static const float DDGI_PI = 3.14159265;
static const uint  DDGI_COLOR_RESOLUTION = 8;
static const uint  DDGI_DEPTH_RESOLUTION = 16;
static const uint  DDGI_DEPTH_TEXELS     = DDGI_DEPTH_RESOLUTION + 2; // with border

// Mirrors GpuDdgiParams in ddgi/mod.rs
struct DdgiParams {
    float3 grid_min;
    float  max_distance;
    float3 cell_size;
    float  intensity;
    uint3  grid_dims;
    uint   probe_count;
    float2 depth_texel_size;
    uint   enabled;
    uint   frame;
    uint   probe_offset;
    uint   probes_this_frame;
    uint   rays_per_probe;
    uint   light_count;
    float3 sun_direction;
    float  blend_speed;
    float3 sun_radiance;
    float  sky_intensity;
    float4 rotation[3]; // columns
};

// L1 SH radiance of one probe: sh[0] = L0, sh[1..3] = L1 (y, z, x); rgb in xyz
struct DdgiProbe {
    float4 sh[4];
};

uint3 ddgi_probe_coord(uint probe_index) {
    uint3 dims = ddgi.grid_dims;
    return uint3(probe_index % dims.x, (probe_index / dims.x) % dims.y, probe_index / (dims.x * dims.y));
}

uint ddgi_probe_index(uint3 coord) {
    return coord.x + coord.y * ddgi.grid_dims.x + coord.z * ddgi.grid_dims.x * ddgi.grid_dims.y;
}

float3 ddgi_probe_position(uint3 coord) {
    return ddgi.grid_min + float3(coord) * ddgi.cell_size;
}

// Top-left inner texel of a probe's depth tile (ProbeGrid::depth_pixel)
uint2 ddgi_depth_pixel(uint3 coord) {
    return uint2(coord.x * DDGI_DEPTH_TEXELS + coord.y * ddgi.grid_dims.x * DDGI_DEPTH_TEXELS + 1,
                 coord.z * DDGI_DEPTH_TEXELS + 1);
}


float2 ddgi_depth_uv(uint3 coord, float3 direction) {
    float2 pixel = float2(ddgi_depth_pixel(coord)) + (oct_encode(normalize(direction)) * 0.5 + 0.5) * DDGI_DEPTH_RESOLUTION;
    return pixel * ddgi.depth_texel_size;
}

#ifdef DDGI_SAMPLER
// Irradiance / π (same scale as the IBL irradiance map: × albedo gives diffuse radiance) at P
// with normal N, from the 8 probes around P: trilinear weights, smooth backface weight and
// Chebyshev visibility from the depth moments (ddgi_sample_irradiance in Wicked).
float3 ddgi_irradiance(float3 P, float3 N) {
    uint3  max_coord   = ddgi.grid_dims - 1;
    float3 cell_coord  = clamp((P - ddgi.grid_min) / ddgi.cell_size, 0.0, float3(max_coord));
    uint3  base_coord  = min(uint3(floor(cell_coord)), max(max_coord, 1u) - 1u);
    float3 alpha       = saturate(cell_coord - float3(base_coord));

    float4 sum_sh[4] = { float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0) };
    float  sum_weight = 0.0;
    for (uint i = 0; i < 8; i++) {
        uint3  offset          = uint3(i, i >> 1, i >> 2) & 1u;
        uint3  probe_coord     = min(base_coord + offset, max_coord);
        float3 probe_pos       = ddgi_probe_position(probe_coord);
        float3 probe_to_point  = P - probe_pos + N * 0.001;
        float3 dir             = normalize(-probe_to_point);
        float3 trilinear       = lerp(1.0 - alpha, alpha, float3(offset));
        float3 true_direction  = normalize(probe_pos - P);
        // Wrap-shaded backface weight: never fully zero, so small details keep some probes
        float  weight          = pow(max(0.0001, (dot(true_direction, N) + 1.0) * 0.5), 2.0) + 0.2;

        // Chebyshev visibility from the probe's depth moments (variance shadow maps, eq. 5)
        float2 moments    = ddgi_depth.SampleLevel(DDGI_SAMPLER, ddgi_depth_uv(probe_coord, -dir), 0.0).xy;
        float  distance   = length(probe_to_point);
        float  variance   = abs(moments.x * moments.x - moments.y);
        float  difference = max(distance - moments.x, 0.0);
        float  chebyshev  = variance / (variance + difference * difference);
        chebyshev = max(chebyshev * chebyshev * chebyshev, 0.0);
        weight *= distance <= moments.x ? 1.0 : chebyshev;

        weight = max(0.01, weight);
        // Crush tiny weights (light is visible even at low values), keeping the curve continuous
        const float crush_threshold = 0.2;
        if (weight < crush_threshold) {
            weight *= weight * weight / (crush_threshold * crush_threshold);
        }
        weight *= trilinear.x * trilinear.y * trilinear.z;

        DdgiProbe probe = ddgi_probes[ddgi_probe_index(probe_coord)];
        for (uint c = 0; c < 4; c++) {
            sum_sh[c] += probe.sh[c] * weight;
        }
        sum_weight += weight;
    }
    if (sum_weight <= 0.0) {
        return float3(0.0, 0.0, 0.0);
    }
    // Convolve with the cosine lobe and evaluate in N (SH::CalculateIrradiance), divided by π
    const float a0 = DDGI_PI;
    const float a1 = 2.0 * DDGI_PI / 3.0;
    const float basis_l0 = 0.2820948;
    const float basis_l1 = 0.4886025;
    float3 irradiance = basis_l0 * a0 * sum_sh[0].xyz
                      + basis_l1 * a1 * (N.y * sum_sh[1].xyz + N.z * sum_sh[2].xyz + N.x * sum_sh[3].xyz);
    return max(irradiance / (sum_weight * DDGI_PI), 0.0);
}
#endif // DDGI_SAMPLER
#endif
