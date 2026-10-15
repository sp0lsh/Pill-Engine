// DDGI probe relocation (method: NVIDIA RTXGI ProbeRelocationCS): one thread per probe, for probes not evaluated yet (`DdgiProbe::offset.w`
// 0: new storage, a grid move exposed it, or `relocate_all` after the geometry changed). From the
// probe's grid position it traces RELOCATION_RAYS fixed (unrotated, so the result is stable)
// spherical Fibonacci rays and, up to RELOCATION_PASSES times, applies the rule:
// - more than BACKFACE_THRESHOLD of the rays hit backfaces: the probe is inside geometry, move it
//   through the closest backface, half the minimum frontface distance past it
// - the closest frontface is nearer than the minimum frontface distance: move toward the farthest
//   frontface (at most 1 m), if that points away from the closest one
// - otherwise move back toward the grid position
// An offset is only taken while it stays inside the 0.45-cell ellipsoid around the grid position.
// Probes still inside geometry are not marked inactive: weight 0 lets the sky fallback light
// their cells.

#include "include/ddgi_common.hlsl"
#include "include/bvh_trace.hlsl"

[[vk::binding(0, 0)]] ConstantBuffer<DdgiParams>        ddgi;
[[vk::binding(1, 0)]] StructuredBuffer<BvhNode>         bvh_nodes;
[[vk::binding(2, 0)]] StructuredBuffer<BvhTriangle>     bvh_triangles;
[[vk::binding(3, 0)]] RWStructuredBuffer<DdgiProbe>     ddgi_probes_out;

static const uint  THREADS            = 64;
// Fixed rays, backface threshold, minimum frontface distance (m)
static const uint  RELOCATION_RAYS    = 32;
static const float BACKFACE_THRESHOLD = 0.25;
static const float MIN_FRONTFACE      = 1.0;
static const uint  RELOCATION_PASSES  = 3;

float3 fixed_ray_direction(uint i) {
    const float golden = 1.61803398875;
    float phi       = 2.0 * DDGI_PI * frac(float(i) * (golden - 1.0));
    float cos_theta = 1.0 - (2.0 * float(i) + 1.0) / float(RELOCATION_RAYS);
    float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint probe_index = id.x;
    if (probe_index >= ddgi.probe_count * ddgi.cascade_count) {
        return;
    }
    DdgiProbe probe = ddgi_probes_out[probe_index];
    if (probe.offset.w != 0.0 && ddgi.relocate_all == 0) {
        return;
    }
    uint   cascade       = probe_index / ddgi.probe_count;
    float3 cell_size     = ddgi.cascades[cascade].cell.xyz;
    float  max_distance  = ddgi_max_distance(cascade);
    float3 grid_position = ddgi_probe_position(ddgi_probe_coord(probe_index % ddgi.probe_count), cascade);
    // Within the probe's own cell: the minimum frontface distance, kept reachable at small
    // spacings
    float  min_frontface = min(MIN_FRONTFACE, 0.45 * min(cell_size.x, min(cell_size.y, cell_size.z)));
    float3 offset = float3(0.0, 0.0, 0.0);
    float  backface_ratio = 0.0;
    // The last pass only classifies the final position
    for (uint pass = 0; pass <= RELOCATION_PASSES; pass++) {
        float3 origin = grid_position + offset;
        float  backfaces = 0.0;
        float  closest_backface = 1.0e27;
        float3 closest_backface_direction = float3(0.0, 0.0, 0.0);
        float  closest_frontface = 1.0e27;
        float3 closest_frontface_direction = float3(0.0, 0.0, 0.0);
        float  farthest_frontface = 0.0;
        float3 farthest_frontface_direction = float3(0.0, 0.0, 0.0);
        for (uint ray = 0; ray < RELOCATION_RAYS; ray++) {
            float3 direction = fixed_ray_direction(ray);
            // Fixed seed per probe and ray: the same leaves block it every time it is relocated
            BvhHit hit = bvh_trace_closest(origin, direction, max_distance, probe_index * RELOCATION_RAYS + ray);
            // A miss counts as a frontface at the longest ray distance
            float distance = hit.found ? hit.distance : max_distance;
            if (hit.found && hit.backface) {
                backfaces += 1.0;
                if (distance < closest_backface) {
                    closest_backface = distance;
                    closest_backface_direction = direction;
                }
            } else {
                if (distance < closest_frontface) {
                    closest_frontface = distance;
                    closest_frontface_direction = direction;
                }
                if (distance > farthest_frontface) {
                    farthest_frontface = distance;
                    farthest_frontface_direction = direction;
                }
            }
        }
        backface_ratio = backfaces / float(RELOCATION_RAYS);
        if (pass == RELOCATION_PASSES) {
            break;
        }
        float3 full_offset = offset;
        if (backfaces > 0.0 && backface_ratio > BACKFACE_THRESHOLD) {
            full_offset = offset + closest_backface_direction * (closest_backface + min_frontface * 0.5);
        } else if (closest_frontface < min_frontface) {
            if (dot(closest_frontface_direction, farthest_frontface_direction) <= 0.0) {
                full_offset = offset + farthest_frontface_direction * min(farthest_frontface, 1.0);
            }
        } else if (length(offset) > 0.0) {
            float move_back = min(closest_frontface - min_frontface, length(offset));
            full_offset = offset - normalize(offset) * move_back;
        }
        float3 normalized = full_offset / cell_size;
        if (dot(normalized, normalized) < 0.2025) { // 0.45²
            offset = full_offset;
        }
    }
    probe.offset = float4(offset, 1.0);
    ddgi_probes_out[probe_index] = probe;
}
