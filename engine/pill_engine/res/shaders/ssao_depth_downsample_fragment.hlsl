// SSAO depth chain: view-space depth at ½, ¼, ⅛ and 1/16 resolution. Each texel is the median of
// the 2×2 texels below it (the mean of the middle two): a min or max snaps a floor texel onto a
// wall texel at an edge and moves occlusion, an average invents a surface that exists nowhere.
// The first level reads the Depth32Float scene depth and linearizes it.

// Mirrors SsaoDownsampleParams in pass_ssao.rs
struct SsaoDownsampleParams {
    float depth_mul;  // view distance = depth_mul / (depth_add - depth)
    float depth_add;
    float linearize;  // 1: the source is the scene depth buffer, 0: a level of this chain
    float _pad;
};

[[vk::binding(0, 0)]] Texture2D<float> t_source;
[[vk::binding(1, 0)]] ConstantBuffer<SsaoDownsampleParams> params;

float source_depth(int2 texel, int2 max_texel) {
    float depth = t_source.Load(int3(clamp(texel, int2(0, 0), max_texel), 0));
    return params.linearize > 0.5 ? params.depth_mul / (params.depth_add - depth) : depth;
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    uint width, height;
    t_source.GetDimensions(width, height);
    int2  max_texel = int2((int)width - 1, (int)height - 1);
    int2  size      = max(int2((int)width / 2, (int)height / 2), int2(1, 1));
    int2  base      = clamp((int2)(uv * (float2)size), int2(0, 0), size - int2(1, 1)) * 2;
    float d0 = source_depth(base + int2(0, 0), max_texel);
    float d1 = source_depth(base + int2(1, 0), max_texel);
    float d2 = source_depth(base + int2(0, 1), max_texel);
    float d3 = source_depth(base + int2(1, 1), max_texel);
    float lowest  = min(min(d0, d1), min(d2, d3));
    float highest = max(max(d0, d1), max(d2, d3));
    return float4(0.5 * (d0 + d1 + d2 + d3 - lowest - highest), 0.0, 0.0, 0.0);
}
