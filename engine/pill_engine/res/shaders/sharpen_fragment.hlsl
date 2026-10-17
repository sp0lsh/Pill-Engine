// Contrast-adaptive sharpening (after AMD FidelityFX CAS, simplified to the 5-tap cross) over the
// tonemapped, still linear image: each pixel moves away from its neighbours' average, less where
// the neighbourhood is already near black or white (no ringing or clipping there).

struct SharpenParams {
    float2 rcp_frame; // 1/width, 1/height of the source
    float  sharpness; // 0 = copy, 1 = strongest
    float  _padding;
};

[[vk::binding(0, 0)]] Texture2D<float4>             t_src;
[[vk::binding(1, 0)]] SamplerState                  s_src;
[[vk::binding(2, 0)]] ConstantBuffer<SharpenParams> params;

float3 fetch(float2 uv, float2 offset) {
    return t_src.SampleLevel(s_src, uv + offset * params.rcp_frame, 0.0).rgb;
}

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    float2 uv     = frag_coord.xy * params.rcp_frame;
    float3 center = fetch(uv, float2(0.0, 0.0));
    if (params.sharpness <= 0.0) {
        return float4(center, 1.0);
    }
    float3 up    = fetch(uv, float2(0.0, -1.0));
    float3 left  = fetch(uv, float2(-1.0, 0.0));
    float3 right = fetch(uv, float2(1.0, 0.0));
    float3 down  = fetch(uv, float2(0.0, 1.0));
    float3 low   = min(center, min(min(up, left), min(right, down)));
    float3 high  = max(center, max(max(up, left), max(right, down)));
    // Headroom to black and white, relative to the local maximum
    float3 amount = sqrt(saturate(min(low, 1.0 - high) / max(high, 1e-4)));
    float3 weight = amount * (-0.2 * saturate(params.sharpness));
    float3 color  = (center + weight * (up + left + right + down)) / (1.0 + 4.0 * weight);
    return float4(saturate(color), 1.0);
}
