// DDGI depth update (port of WickedEngine ddgi_updateCS with DDGI_UPDATE_DEPTH): one workgroup per
// probe; each of the 16×16 octahedral texels averages the ray distances (and their squares)
// around its direction with a sharp cos⁶⁴ lobe, blends with history and writes the atlas tile plus
// its 1-texel border, so bilinear filtering in the PBR passes stays inside the tile.

#include "include/ddgi_common.hlsl"

struct DdgiRay {
    float4 direction_depth;
    float4 radiance;
};

[[vk::binding(0, 0)]] ConstantBuffer<DdgiParams>        ddgi;
[[vk::binding(1, 0)]] StructuredBuffer<DdgiRay>         ddgi_rays;
// History of the moments (mean, mean²) per inner texel; the atlas can't be read back in place
[[vk::binding(2, 0)]] RWStructuredBuffer<float2>        ddgi_depth_history;
[[vk::binding(3, 0)]] [format("rgba16f")] WTexture2D<float4> ddgi_depth_out;
[[vk::binding(4, 0)]] StructuredBuffer<uint>            ddgi_probe_list;

static const uint TEXELS = DDGI_DEPTH_RESOLUTION * DDGI_DEPTH_RESOLUTION;
groupshared float4 ray_cache[TEXELS]; // xyz = direction, w = distance
groupshared float2 tile[TEXELS];

[shader("compute")]
[numthreads(DDGI_DEPTH_RESOLUTION, DDGI_DEPTH_RESOLUTION, 1)]
void cs_main(uint3 group : SV_GroupID, uint2 thread : SV_GroupThreadID, uint thread_index : SV_GroupIndex) {
    uint  slot         = group.x;
    uint  probe_index  = ddgi_probe_list[slot];
    uint3 probe_coord  = ddgi_probe_coord(probe_index);
    uint  ray_count    = ddgi.rays_per_probe;
    float max_distance = ddgi.max_distance;
    float3 texel_direction = oct_decode(((float2(thread) + 0.5) / DDGI_DEPTH_RESOLUTION) * 2.0 - 1.0);

    float2 sum = float2(0.0, 0.0);
    float  total_weight = 0.0;
    for (uint first = 0; first < ray_count; first += TEXELS) {
        uint count = min(TEXELS, ray_count - first);
        if (thread_index < count) {
            ray_cache[thread_index] = ddgi_rays[slot * ray_count + first + thread_index].direction_depth;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0; r < count; r++) {
            float4 ray    = ray_cache[r];
            float  depth  = ray.w > 0.0 ? clamp(ray.w - 0.01, 0.0, max_distance) : max_distance;
            float  weight = pow(saturate(dot(texel_direction, ray.xyz)), 64.0);
            if (weight > 0.0001) {
                sum += float2(depth, depth * depth) * weight;
                total_weight += weight;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float2 result = total_weight > 0.0001 ? sum / total_weight : float2(max_distance, max_distance * max_distance);

    uint   history_index = probe_index * TEXELS + thread_index;
    float2 previous = ddgi_depth_history[history_index];
    // Zeroed history: start from this frame's value; else Wicked's 2 % blend
    if (previous.y > 0.0) {
        result = lerp(previous, result, 0.02);
    }
    ddgi_depth_history[history_index] = result;
    tile[thread_index] = result;

    uint2 inner = ddgi_depth_pixel(probe_coord);
    ddgi_depth_out.Store(inner + thread, float4(result, 0.0, 0.0));
    GroupMemoryBarrierWithGroupSync();

    // Border: each edge texel mirrors the opposite-flipped edge of the octahedral tile, corners
    // the diagonal corner (Wicked's DDGI_DEPTH_BORDER_OFFSETS, generated instead of tabulated)
    uint2 origin = inner - 1;
    const uint last = DDGI_DEPTH_RESOLUTION - 1;
    if (thread.y == 0) {
        // Top row (y = 0) from inner row 0 mirrored in x; bottom row (y = 17) from inner row 15
        uint x = thread.x;
        ddgi_depth_out.Store(origin + uint2(x + 1, 0), float4(tile[0 * DDGI_DEPTH_RESOLUTION + (last - x)], 0.0, 0.0));
        ddgi_depth_out.Store(origin + uint2(x + 1, DDGI_DEPTH_RESOLUTION + 1), float4(tile[last * DDGI_DEPTH_RESOLUTION + (last - x)], 0.0, 0.0));
    } else if (thread.y == 1) {
        // Left column (x = 0) from inner column 0 mirrored in y; right column from inner column 15
        uint y = thread.x;
        ddgi_depth_out.Store(origin + uint2(0, y + 1), float4(tile[(last - y) * DDGI_DEPTH_RESOLUTION + 0], 0.0, 0.0));
        ddgi_depth_out.Store(origin + uint2(DDGI_DEPTH_RESOLUTION + 1, y + 1), float4(tile[(last - y) * DDGI_DEPTH_RESOLUTION + last], 0.0, 0.0));
    } else if (thread_index == 2 * DDGI_DEPTH_RESOLUTION) {
        ddgi_depth_out.Store(origin + uint2(0, 0), float4(tile[last * DDGI_DEPTH_RESOLUTION + last], 0.0, 0.0));
        ddgi_depth_out.Store(origin + uint2(DDGI_DEPTH_RESOLUTION + 1, 0), float4(tile[last * DDGI_DEPTH_RESOLUTION + 0], 0.0, 0.0));
        ddgi_depth_out.Store(origin + uint2(0, DDGI_DEPTH_RESOLUTION + 1), float4(tile[0 * DDGI_DEPTH_RESOLUTION + last], 0.0, 0.0));
        ddgi_depth_out.Store(origin + uint2(DDGI_DEPTH_RESOLUTION + 1, DDGI_DEPTH_RESOLUTION + 1), float4(tile[0], 0.0, 0.0));
    }
}
