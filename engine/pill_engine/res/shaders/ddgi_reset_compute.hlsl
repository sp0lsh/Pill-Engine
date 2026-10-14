// DDGI probe reset: one workgroup per probe a moving grid exposes (ProbeGrid::exposed). The probe
// copies its nearest surviving neighbour — SH with weight and warm-up count, radiance estimator
// history, depth moments — so it starts from the neighbour's lighting instead of an unweighted
// sky fallback (Godot's SDFGI fills scrolled probes from the coarser cascade). Without a
// neighbour (a jump of a grid size or more) it is zeroed: weight 0, warm-up restarts, and the
// histories start from its next frame's rays.

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

[[vk::binding(0, 0)]] StructuredBuffer<uint2>                 ddgi_reset_list; // target, source
[[vk::binding(1, 0)]] RWStructuredBuffer<DdgiProbe>           ddgi_probes_out;
[[vk::binding(2, 0)]] RWStructuredBuffer<DdgiTexelHistory>    ddgi_history;
[[vk::binding(3, 0)]] RWStructuredBuffer<float2>              ddgi_depth_history;

static const uint THREADS       = 64;
static const uint NO_SOURCE     = 0xFFFFFFFFu; // mod.rs NO_SOURCE
static const uint COLOR_TEXELS  = DDGI_COLOR_RESOLUTION * DDGI_COLOR_RESOLUTION;
static const uint DEPTH_TEXELS  = DDGI_DEPTH_RESOLUTION * DDGI_DEPTH_RESOLUTION;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    uint2 entry       = ddgi_reset_list[group.x];
    uint  probe_index = entry.x;
    uint  source      = entry.y;
    // Sources are never exposed probes, so reading them here doesn't race with the writes
    bool  copy        = source != NO_SOURCE;
    if (thread == 0) {
        ddgi_probes_out[probe_index] = copy ? ddgi_probes_out[source] : (DdgiProbe)0;
    }
    for (uint color_texel = thread; color_texel < COLOR_TEXELS; color_texel += THREADS) {
        ddgi_history[probe_index * COLOR_TEXELS + color_texel] =
            copy ? ddgi_history[source * COLOR_TEXELS + color_texel] : (DdgiTexelHistory)0;
    }
    for (uint depth_texel = thread; depth_texel < DEPTH_TEXELS; depth_texel += THREADS) {
        ddgi_depth_history[probe_index * DEPTH_TEXELS + depth_texel] =
            copy ? ddgi_depth_history[source * DEPTH_TEXELS + depth_texel] : float2(0.0, 0.0);
    }
}
