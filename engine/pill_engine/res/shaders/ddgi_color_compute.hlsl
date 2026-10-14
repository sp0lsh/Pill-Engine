// DDGI radiance update (port of WickedEngine ddgi_updateCS without DDGI_UPDATE_DEPTH): one
// workgroup per probe; each of the 8×8 octahedral texels averages this frame's rays around its
// direction (a texel no ray reached keeps its history), is blended with its history by Wicked's multiscale mean estimator, then thread 0
// projects the 64 texels onto L1 SH for the probe.

#include "include/ddgi_common.hlsl"

struct DdgiRay {
    float4 direction_depth;
    float4 radiance;
};
// Per texel: mean.xyz + variance-based blend reduction, short mean.xyz + inconsistency, variance.xyz
struct DdgiTexelHistory {
    float4 mean_vbbr;
    float4 short_mean_inconsistency;
    float4 variance;
};

[[vk::binding(0, 0)]] ConstantBuffer<DdgiParams>              ddgi;
[[vk::binding(1, 0)]] StructuredBuffer<DdgiRay>               ddgi_rays;
[[vk::binding(2, 0)]] RWStructuredBuffer<DdgiTexelHistory>    ddgi_history;
[[vk::binding(3, 0)]] RWStructuredBuffer<DdgiProbe>           ddgi_probes_out;

static const uint TEXELS = DDGI_COLOR_RESOLUTION * DDGI_COLOR_RESOLUTION;
static const float DDGI_WARMUP_UPDATES = 32.0;
groupshared float3 ray_directions[TEXELS];
groupshared float3 ray_radiance[TEXELS];
groupshared float3 texel_radiance[TEXELS];

// Wicked's MultiscaleMeanEstimator (float instead of half)
void multiscale_mean_estimator(float3 y, inout DdgiTexelHistory data, float short_window_blend) {
    float3 mean          = data.mean_vbbr.xyz;
    float  vbbr          = data.mean_vbbr.w;
    float3 short_mean    = data.short_mean_inconsistency.xyz;
    float  inconsistency = data.short_mean_inconsistency.w;
    float3 variance      = data.variance.xyz;
    const float3 luminance = float3(0.299, 0.587, 0.114);

    // Suppress fireflies
    float3 deviation = sqrt(max(1e-5, variance));
    float3 high_threshold = 0.1 + short_mean + deviation * 8.0;
    y -= max(0.0, y - high_threshold);

    float3 delta = y - short_mean;
    short_mean = lerp(short_mean, y, short_window_blend);
    float3 delta2 = y - short_mean;
    variance = lerp(variance, delta * delta2, short_window_blend * 0.5);
    deviation = sqrt(max(1e-5, variance));

    float3 short_difference = mean - short_mean;
    float relative_difference = dot(luminance, abs(short_difference) / max(1e-5, deviation));
    inconsistency = lerp(inconsistency, relative_difference, 0.08);
    float variance_blend_reduction = clamp(dot(luminance, 0.5 * short_mean / max(1e-5, deviation)), 1.0 / 32.0, 1.0);
    float catch_up_blend = clamp(smoothstep(0.0, 1.0, relative_difference * max(0.02, inconsistency - 0.2)), 1.0 / 256.0, 1.0) * vbbr;
    vbbr = lerp(vbbr, variance_blend_reduction, 0.1);
    mean = lerp(mean, y, saturate(catch_up_blend));

    data.mean_vbbr = float4(mean, vbbr);
    data.short_mean_inconsistency = float4(short_mean, inconsistency);
    data.variance = float4(variance, 0.0);
}

[shader("compute")]
[numthreads(DDGI_COLOR_RESOLUTION, DDGI_COLOR_RESOLUTION, 1)]
void cs_main(uint3 group : SV_GroupID, uint2 thread : SV_GroupThreadID, uint thread_index : SV_GroupIndex) {
    uint   probe_index = group.x;
    uint   ray_count   = ddgi.rays_per_probe;
    float3 texel_direction = oct_decode(((float2(thread) + 0.5) / DDGI_COLOR_RESOLUTION) * 2.0 - 1.0);

    float3 sum = float3(0.0, 0.0, 0.0);
    float  total_weight = 0.0;
    for (uint first = 0; first < ray_count; first += TEXELS) {
        uint count = min(TEXELS, ray_count - first);
        if (thread_index < count) {
            DdgiRay ray = ddgi_rays[probe_index * ray_count + first + thread_index];
            ray_directions[thread_index] = ray.direction_depth.xyz;
            ray_radiance[thread_index]   = ray.radiance.rgb;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0; r < count; r++) {
            float weight = saturate(dot(texel_direction, ray_directions[r]));
            if (weight > 0.0001) {
                sum += ray_radiance[r] * weight;
                total_weight += weight;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint history_index = probe_index * TEXELS + thread_index;
    DdgiTexelHistory history = ddgi_history[history_index];
    // No ray near this texel this frame (few rays per probe): keep its history
    if (total_weight > 0.0001) {
        float3 result = sum / total_weight;
        // Zeroed history (vbbr is never 0 once written): start from this frame's value
        if (history.mean_vbbr.w == 0.0) {
            history.mean_vbbr = float4(result, 0.0);
            history.short_mean_inconsistency = float4(result, 1.0);
            history.variance = float4(0.0, 0.0, 0.0, 0.0);
        }
        multiscale_mean_estimator(result, history, ddgi.blend_speed);
        ddgi_history[history_index] = history;
    }
    texel_radiance[thread_index] = history.mean_vbbr.xyz;
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
        float updates = min(ddgi_probes_out[probe_index].sh[1].w + 1.0, DDGI_WARMUP_UPDATES);
        DdgiProbe probe;
        probe.sh[0] = float4(sh0 * scale, updates / DDGI_WARMUP_UPDATES);
        probe.sh[1] = float4(sh1 * scale, updates);
        probe.sh[2] = float4(sh2 * scale, 0.0);
        probe.sh[3] = float4(sh3 * scale, 0.0);
        ddgi_probes_out[probe_index] = probe;
    }
}
