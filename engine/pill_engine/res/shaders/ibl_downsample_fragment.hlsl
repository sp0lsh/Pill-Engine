// PassIblBake source chain: resamples the environment equirect into mip 0, then each mip from
// the one above (a bilinear tap at the texel center of a half-size target = 2×2 box filter).

[[vk::binding(0, 0)]] Texture2D    source;
[[vk::binding(1, 0)]] SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    return float4(source.SampleLevel(source_sampler, uv, 0.0).rgb, 1.0);
}
