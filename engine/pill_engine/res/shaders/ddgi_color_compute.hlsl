// DDGI radiance update (method: WickedEngine ddgi_updateCS without DDGI_UPDATE_DEPTH): one
// workgroup per probe; each of the 8×8 octahedral texels takes the cosine-weighted mean of this
// frame's rays around its direction and averages it over frames (a running mean over the probe's
// first updates, then an exponential blend of `blend_speed`), then thread 0 projects the 64
// texels onto L1 SH for the probe.

#include "include/ddgi_common.hlsl"

struct DdgiRay {
    float4 direction_depth;
    float4 radiance;
};

[[vk::binding(0, 0)]] ConstantBuffer<DdgiParams>              ddgi;
[[vk::binding(1, 0)]] StructuredBuffer<DdgiRay>               ddgi_rays;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4>              ddgi_history; // texel mean (rgb)
[[vk::binding(3, 0)]] RWStructuredBuffer<DdgiProbe>           ddgi_probes_out;

static const uint TEXELS = DDGI_COLOR_RESOLUTION * DDGI_COLOR_RESOLUTION;
static const float DDGI_WARMUP_UPDATES = 32.0;
// Share of backface hits above which a probe is taken as inside geometry
static const float RANDOM_RAY_BACKFACE_THRESHOLD = 0.1;
groupshared float3 ray_directions[TEXELS];
groupshared float4 ray_radiance[TEXELS]; // w: backface hit
groupshared float3 texel_radiance[TEXELS];

[shader("compute")]
[numthreads(DDGI_COLOR_RESOLUTION, DDGI_COLOR_RESOLUTION, 1)]
void cs_main(uint3 group : SV_GroupID, uint2 thread : SV_GroupThreadID, uint thread_index : SV_GroupIndex) {
    uint   probe_index = group.x; // over all cascades
    // Coarser cascades update every few frames (`update_mask`); the whole group returns together
    if (!ddgi_cascade_updates(probe_index / ddgi.probe_count)) {
        return;
    }
    uint   ray_count   = ddgi.rays_per_probe;
    float3 texel_direction = oct_decode(((float2(thread) + 0.5) / DDGI_COLOR_RESOLUTION) * 2.0 - 1.0);

    // Updates since the probe's reset (written by thread 0 at the end, after every thread read it)
    float  updates = ddgi_probes_out[probe_index].sh[1].w;
    // Relocation offset and state (ddgi_relocate_compute.hlsl), kept as they are
    float4 relocation = ddgi_probes_out[probe_index].offset;
    float3 sum = float3(0.0, 0.0, 0.0);
    uint   backfaces = 0;
    for (uint first = 0; first < ray_count; first += TEXELS) {
        uint count = min(TEXELS, ray_count - first);
        if (thread_index < count) {
            DdgiRay ray = ddgi_rays[probe_index * ray_count + first + thread_index];
            ray_directions[thread_index] = ray.direction_depth.xyz;
            ray_radiance[thread_index]   = ray.radiance;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0; r < count; r++) {
            if (ray_radiance[r].w > 0.0) {
                backfaces++;
                continue;
            }
            float weight = saturate(dot(texel_direction, ray_directions[r]));
            if (weight > 0.0001) {
                sum += ray_radiance[r].rgb * weight;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // Backface hits are ignored, and a probe whose rays hit backfaces RANDOM_RAY_BACKFACE_THRESHOLD
    // of the time is probably inside geometry: it keeps its lighting this frame instead of
    // averaging the unlit inside of a wall. Every thread counts the same rays, so the whole group
    // returns together
    uint max_backfaces = uint(float(ray_count) * RANDOM_RAY_BACKFACE_THRESHOLD);
    if (backfaces > 0 && backfaces >= max_backfaces) {
        return;
    }
    uint history_index = probe_index * TEXELS + thread_index;
    float3 history = ddgi_history[history_index].rgb;
    // Cosine-weighted mean of the radiance around the texel, normalized by the weight N uniformly
    // spread rays sum to on average (∫ max(0, cos) dω / 4π = 1/4 per ray), not by this frame's
    // weight: with few rays per probe the ratio sum / Σw returns whichever rays happened to land
    // near the texel at any angle (grazing ones count as much as head-on ones: biased), while
    // this estimate stays unbiased at any ray count
    float3 result = sum * (4.0 / float(ray_count - backfaces));
    // Running mean while the probe is new (unbiased from its first update, whatever the ray
    // count), then the exponential blend
    float  blend = max(1.0 / (updates + 1.0), ddgi.blend_speed);
    float3 mean  = lerp(history, result, blend);
    ddgi_history[history_index] = float4(mean, 0.0);
    texel_radiance[thread_index] = mean;
    GroupMemoryBarrierWithGroupSync();

    if (thread_index == 0) {
        float3 sh0 = float3(0.0, 0.0, 0.0);
        float3 sh1 = float3(0.0, 0.0, 0.0);
        float3 sh2 = float3(0.0, 0.0, 0.0);
        float3 sh3 = float3(0.0, 0.0, 0.0);
        for (uint y = 0; y < DDGI_COLOR_RESOLUTION; y++) {
            for (uint x = 0; x < DDGI_COLOR_RESOLUTION; x++) {
                float3 direction = oct_decode(((float2(x, y) + 0.5) / DDGI_COLOR_RESOLUTION) * 2.0 - 1.0);
                float3 value = texel_radiance[y * DDGI_COLOR_RESOLUTION + x];
                sh0 += 0.2820948 * value;
                sh1 += 0.4886025 * direction.y * value;
                sh2 += 0.4886025 * direction.z * value;
                sh3 += 0.4886025 * direction.x * value;
            }
        }
        // Monte Carlo over the sphere: × 4π / texel count
        float scale = 4.0 * DDGI_PI / float(TEXELS);
        // Updates since the probe's reset (sh[1].w); its weight (sh[0].w) ramps in over the first
        // DDGI_WARMUP_UPDATES, so a probe a grid move exposed fades in from few-ray estimates
        // while its older neighbours carry the lighting, instead of popping in
        float count = min(updates + 1.0, 65535.0);
        DdgiProbe probe;
        // A probe started from a neighbour's lighting (sh[2].w = 1, ddgi_reset_compute.hlsl) has
        // full weight from the start
        float copied = ddgi_probes_out[probe_index].sh[2].w;
        float weight = max(min(count / DDGI_WARMUP_UPDATES, 1.0), copied);
        probe.sh[0] = float4(sh0 * scale, weight);
        probe.sh[1] = float4(sh1 * scale, count);
        probe.sh[2] = float4(sh2 * scale, copied);
        probe.sh[3] = float4(sh3 * scale, 0.0);
        probe.offset = relocation;
        ddgi_probes_out[probe_index] = probe;
    }
}
