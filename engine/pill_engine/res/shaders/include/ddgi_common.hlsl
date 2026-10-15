#ifndef DDGI_COMMON_HLSL
#define DDGI_COMMON_HLSL
// DDGI probe grids: layout, octahedral mapping, L1 SH and the 8-probe irradiance query (method:
// WickedEngine ShaderInterop_DDGI.h; cascades: Flax Engine DDGI). The includer declares `ddgi`
// (ConstantBuffer<DdgiParams>),
// and, for the irradiance query, `ddgi_probes` (StructuredBuffer<DdgiProbe>), `ddgi_depth`
// (Texture2D) and DDGI_SAMPLER (a linear clamp sampler, defined before the include).

#include "octahedral.hlsl"

static const float DDGI_PI = 3.14159265;
static const uint  DDGI_COLOR_RESOLUTION = 8;
static const uint  DDGI_DEPTH_RESOLUTION = 16;
static const uint  DDGI_DEPTH_TEXELS     = DDGI_DEPTH_RESOLUTION + 2; // with border

// One cascade's grid: min.xyz world position of logical probe (0, 0, 0), cell.xyz probe spacing,
// origin.xyz world cell of logical probe (0, 0, 0) (storage is toroidal around it)
struct DdgiCascade {
    float4 min;
    float4 cell;
    int4   origin;
};

static const uint DDGI_MAX_CASCADES = 4;

// Mirrors GpuDdgiParams in ddgi/mod.rs
struct DdgiParams {
    uint3  grid_dims;     // probes per axis, the same for every cascade
    uint   probe_count;   // per cascade
    float2 depth_texel_size;
    uint   enabled;
    uint   frame;
    uint   relocate_all;  // 1: re-evaluate every probe's relocation (the geometry changed)
    uint   cascade_count;
    uint   rays_per_probe;
    uint   light_count;
    float3 sun_direction;
    float  blend_speed;
    float3 sun_radiance;
    float  sky_intensity;
    float3 blend_origin;  // camera position: where the cascade blend weights are centred
    float  intensity;
    uint   update_mask;   // bit c: cascade c updates this frame
    uint   _padding0;
    uint   _padding1;
    uint   _padding2;
    DdgiCascade cascades[DDGI_MAX_CASCADES];
};

// L1 SH radiance of one probe: sh[0] = L0, sh[1..3] = L1 (y, z, x); rgb in xyz. sh[0].w = the
// probe's weight, 0 after a reset (the grid moved and exposed it) rising to 1 over its first updates;
// sh[1].w = updates since the reset; sh[2].w = 1 when it started from a neighbour's lighting
struct DdgiProbe {
    float4 sh[4];
    // xyz: relocation offset from the grid position (m); w: 0 = not relocated yet, 1 = relocated
    // — ddgi_relocate_compute.hlsl
    float4 offset;
};

// Probe index within a cascade ↔ storage coordinate
uint3 ddgi_probe_coord(uint probe_index) {
    uint3 dims = ddgi.grid_dims;
    return uint3(probe_index % dims.x, (probe_index / dims.x) % dims.y, probe_index / (dims.x * dims.y));
}

uint ddgi_probe_index(uint3 coord) {
    return coord.x + coord.y * ddgi.grid_dims.x + coord.z * ddgi.grid_dims.x * ddgi.grid_dims.y;
}

// Probes of all cascades live in one array, cascade after cascade
uint ddgi_global_index(uint cascade, uint probe_index) {
    return cascade * ddgi.probe_count + probe_index;
}

bool ddgi_cascade_updates(uint cascade) {
    return ((ddgi.update_mask >> cascade) & 1u) != 0u;
}

// Probe data lives at `(logical + origin) mod dims` (ProbeGrid in ddgi/mod.rs): a grid that moves
// in whole cells keeps the probes it still covers in place
uint3 ddgi_wrap(int3 value) {
    int3 dims = int3(ddgi.grid_dims);
    return uint3(((value % dims) + dims) % dims);
}

uint3 ddgi_storage_coord(uint3 logical, uint cascade) {
    return ddgi_wrap(int3(logical) + ddgi.cascades[cascade].origin.xyz);
}

uint3 ddgi_logical_coord(uint3 storage, uint cascade) {
    return ddgi_wrap(int3(storage) - ddgi.cascades[cascade].origin.xyz);
}

// World position of the probe stored at `storage` (without its relocation offset)
float3 ddgi_probe_position(uint3 storage, uint cascade) {
    DdgiCascade grid = ddgi.cascades[cascade];
    return grid.min.xyz + float3(ddgi_logical_coord(storage, cascade)) * grid.cell.xyz;
}

// Ray length: the depth range of a probe, a cell diagonal and a half
float ddgi_max_distance(uint cascade) {
    return length(ddgi.cascades[cascade].cell.xyz) * 1.5;
}

// Top-left inner texel of a probe's depth tile (ProbeGrid::depth_pixel); cascades are stacked
// down the atlas
uint2 ddgi_depth_pixel(uint3 coord, uint cascade) {
    return uint2(coord.x * DDGI_DEPTH_TEXELS + coord.y * ddgi.grid_dims.x * DDGI_DEPTH_TEXELS + 1,
                 (coord.z + cascade * ddgi.grid_dims.z) * DDGI_DEPTH_TEXELS + 1);
}

float2 ddgi_depth_uv(uint3 coord, uint cascade, float3 direction) {
    float2 pixel = float2(ddgi_depth_pixel(coord, cascade)) + (oct_encode(normalize(direction)) * 0.5 + 0.5) * DDGI_DEPTH_RESOLUTION;
    return pixel * ddgi.depth_texel_size;
}

// Signed distance to a box of half-size `b` with corners rounded by `r` (Inigo Quilez)
float ddgi_sd_round_box(float3 p, float3 b, float r) {
    float3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// How much cascade `cascade` lights P, 0–1: 1 inside its grid less a
// cell, fading over 2 cells toward its edge. Centred on the camera position (`blend_origin`), not
// on the grid snapped to whole cells, so the fade slides with the camera instead of jumping a
// cell when the grid scrolls
float ddgi_cascade_weight(uint cascade, float3 P) {
    float  spacing = ddgi.cascades[cascade].cell.x;
    float3 extent  = float3(ddgi.grid_dims - 1) * ddgi.cascades[cascade].cell.xyz * 0.5;
    float  fade    = spacing * 2.0;
    float  distance = ddgi_sd_round_box(P - ddgi.blend_origin, extent - spacing, spacing * 2.0) + fade;
    return 1.0 - saturate(distance / fade);
}

// Debug colour of the probe cell lighting P (`Gi::debug_probe_cells`): a colour per world cell
// of the base probe of the cascade lighting P (the finest with weight ≥ 0.5), with
// ddgi_irradiance's clamp, so a surface keeps its colour while the grid scrolls and changes it
// when other probes light it; coarser cascades are darker, the area outside every grid darkest
float3 ddgi_cell_debug_color(float3 P) {
    uint cascade = 0;
    while (cascade + 1 < ddgi.cascade_count && ddgi_cascade_weight(cascade, P) < 0.5) {
        cascade++;
    }
    DdgiCascade grid  = ddgi.cascades[cascade];
    uint3  max_coord  = ddgi.grid_dims - 1;
    float3 cell_coord = clamp((P - grid.min.xyz) / grid.cell.xyz, 0.0, float3(max_coord));
    int3   cell       = int3(min(uint3(floor(cell_coord)), max(max_coord, 1u) - 1u)) + grid.origin.xyz;
    uint   hash       = (uint(cell.x) * 73856093u) ^ (uint(cell.y) * 19349663u) ^ (uint(cell.z) * 83492791u) ^ (cascade * 2654435761u);
    hash = hash * 747796405u + 2891336453u;
    float3 color      = float3(hash & 255u, (hash >> 8) & 255u, (hash >> 16) & 255u) / 255.0;
    float3 grid_max   = grid.min.xyz + float3(max_coord) * grid.cell.xyz;
    bool   outside    = any(P < grid.min.xyz) || any(P > grid_max);
    color *= cascade == 0 ? 1.0 : 0.6;
    return outside ? color * 0.25 : color;
}

#ifdef DDGI_SAMPLER
// Irradiance / π (same scale as the IBL irradiance map: × albedo gives diffuse radiance) at P
// with normal N, from the 8 probes around P: trilinear weights, smooth backface weight and
// Chebyshev visibility from the depth moments. w = coverage:
// 4 × the trilinear weight of the probes with weight, saturated (1 unless most of the cell's probes
// are new); callers blend toward their fallback by it.
float4 ddgi_irradiance_cascade(float3 P, float3 N, uint cascade) {
    DdgiCascade grid   = ddgi.cascades[cascade];
    uint3  max_coord   = ddgi.grid_dims - 1;
    float3 cell_coord  = clamp((P - grid.min.xyz) / grid.cell.xyz, 0.0, float3(max_coord));
    uint3  base_coord  = min(uint3(floor(cell_coord)), max(max_coord, 1u) - 1u);
    float3 alpha       = saturate(cell_coord - float3(base_coord));

    float4 sum_sh[4] = { float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0), float4(0.0, 0.0, 0.0, 0.0) };
    float  sum_weight = 0.0;
    float  coverage   = 0.0;
    for (uint i = 0; i < 8; i++) {
        uint3  offset          = uint3(i, i >> 1, i >> 2) & 1u;
        uint3  logical_coord   = min(base_coord + offset, max_coord);
        uint3  probe_coord     = ddgi_storage_coord(logical_coord, cascade);
        DdgiProbe probe        = ddgi_probes[ddgi_global_index(cascade, ddgi_probe_index(probe_coord))];
        float  valid           = probe.sh[0].w;
        float3 probe_pos       = grid.min.xyz + float3(logical_coord) * grid.cell.xyz + probe.offset.xyz;
        float3 probe_to_point  = P - probe_pos + N * 0.001;
        float3 dir             = normalize(-probe_to_point);
        float3 trilinear       = lerp(1.0 - alpha, alpha, float3(offset));
        float3 true_direction  = normalize(probe_pos - P);
        // Wrap-shaded backface weight: never fully zero, so small details keep some probes
        float  weight          = pow(max(0.0001, (dot(true_direction, N) + 1.0) * 0.5), 2.0) + 0.2;

        // Chebyshev visibility from the probe's depth moments (variance shadow maps, eq. 5)
        float2 moments    = ddgi_depth.SampleLevel(DDGI_SAMPLER, ddgi_depth_uv(probe_coord, cascade, -dir), 0.0).xy;
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
        float trilinear_weight = trilinear.x * trilinear.y * trilinear.z;
        weight   *= trilinear_weight * valid;
        coverage += trilinear_weight * valid;
        for (uint c = 0; c < 4; c++) {
            sum_sh[c] += probe.sh[c] * weight;
        }
        sum_weight += weight;
    }
    if (sum_weight <= 0.0) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    // Convolve with the cosine lobe and evaluate in N (SH::CalculateIrradiance), divided by π
    const float a0 = DDGI_PI;
    const float a1 = 2.0 * DDGI_PI / 3.0;
    const float basis_l0 = 0.2820948;
    const float basis_l1 = 0.4886025;
    float3 irradiance = basis_l0 * a0 * sum_sh[0].xyz
                      + basis_l1 * a1 * (N.y * sum_sh[1].xyz + N.z * sum_sh[2].xyz + N.x * sum_sh[3].xyz);
    // The probes with weight already renormalize the irradiance above; the caller's fallback (the
    // unoccluded sky) only takes over as the weight of the 8 probes nears 0. Taking the summed
    // weight as is let one new (warming up) probe pull sky light into its cell: bright flashes
    // where a grid move exposed probes
    return float4(max(irradiance / (sum_weight * DDGI_PI), 0.0), saturate(coverage * 4.0));
}

// Irradiance at P from the cascades: the finest
// cascade lights P alone where its weight is 1; toward its edge it blends into the next coarser
// one; the last cascade clamps to its edge beyond it
float4 ddgi_irradiance(float3 P, float3 N) {
    uint cascade = 0;
    float weight = 1.0;
    for (; cascade + 1 < ddgi.cascade_count; cascade++) {
        weight = ddgi_cascade_weight(cascade, P);
        if (weight > 0.0) {
            break;
        }
    }
    float4 result = ddgi_irradiance_cascade(P, N, cascade);
    if (cascade + 1 < ddgi.cascade_count && weight < 0.99) {
        result = lerp(ddgi_irradiance_cascade(P, N, cascade + 1), result, weight);
    }
    return result;
}
#endif // DDGI_SAMPLER
#endif
