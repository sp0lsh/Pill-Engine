// DDGI probe reset: clears the probes a moving grid exposes (ProbeGrid::exposed), one workgroup
// per probe. Zeroed SH marks the probe as without an estimate (sh[0].w = 0, so queries fall back
// to the sky); zeroed histories make its next update start from that frame's rays.

// Mirror ddgi_common.hlsl (included without its `ddgi` uniform, which this kernel doesn't bind)
static const uint DDGI_COLOR_RESOLUTION = 8;
static const uint DDGI_DEPTH_RESOLUTION = 16;
struct DdgiProbe {
    float4 sh[4];
};

// Mirrors DdgiTexelHistory in ddgi_color_compute.hlsl
struct DdgiTexelHistory {
    float4 mean_vbbr;
    float4 short_mean_inconsistency;
    float4 variance;
};

[[vk::binding(0, 0)]] StructuredBuffer<uint>                  ddgi_reset_list;
[[vk::binding(1, 0)]] RWStructuredBuffer<DdgiProbe>           ddgi_probes_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<DdgiTexelHistory>    ddgi_history;
[[vk::binding(3, 0)]] RWStructuredBuffer<float2>              ddgi_depth_history;

static const uint THREADS       = 64;
static const uint COLOR_TEXELS  = DDGI_COLOR_RESOLUTION * DDGI_COLOR_RESOLUTION;
static const uint DEPTH_TEXELS  = DDGI_DEPTH_RESOLUTION * DDGI_DEPTH_RESOLUTION;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    uint probe_index = ddgi_reset_list[group.x];
    if (thread == 0) {
        DdgiProbe probe = (DdgiProbe)0;
        ddgi_probes_out[probe_index] = probe;
    }
    for (uint color_texel = thread; color_texel < COLOR_TEXELS; color_texel += THREADS) {
        ddgi_history[probe_index * COLOR_TEXELS + color_texel] = (DdgiTexelHistory)0;
    }
    for (uint depth_texel = thread; depth_texel < DEPTH_TEXELS; depth_texel += THREADS) {
        ddgi_depth_history[probe_index * DEPTH_TEXELS + depth_texel] = float2(0.0, 0.0);
    }
}
