// @variant copy COPY
// Bevy's default display transform for a Camera3d: Tony McMapface (48³ LUT indexed by c / (c + 1),
// Bevy tonemapping.wesl `sample_tony_mc_mapface_lut`), then the deband dither Bevy applies in gamma
// 2.2 before its sRGB target encodes (`screen_space_dither`, pbr_functions.wesl DEBAND_DITHER).
// The base shader writes the result to a scratch target; the COPY variant copies it back into the
// engine's HDR target, which the engine tonemap (Tonemap::None) then only sRGB-encodes.

[[vk::binding(0, 0)]] Texture2D<float4> source;

#ifdef COPY
[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    return source.Load(int3(int2(frag_coord.xy), 0));
}
#else
[[vk::binding(1, 0)]] Texture3D<float4> lut;
[[vk::binding(2, 0)]] SamplerState     lut_sampler;
static const float LUT_SIZE = 48.0;

float3 screen_space_dither(float2 frag_coord) {
    float3 dither = dot(float2(171.0, 231.0), frag_coord).xxx;
    dither = frac(dither / float3(103.0, 71.0, 97.0));
    return (dither - 0.5) / 255.0;
}

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    float3 stimulus = max(source.Load(int3(int2(frag_coord.xy), 0)).rgb, 0.0);
    float3 uvw = saturate(stimulus / (stimulus + 1.0) * ((LUT_SIZE - 1.0) / LUT_SIZE) + 0.5 / LUT_SIZE);
    float3 mapped = lut.SampleLevel(lut_sampler, uvw, 0.0).rgb;
    float3 dithered = pow(max(mapped, 0.0), 1.0 / 2.2) + screen_space_dither(frag_coord.xy);
    return float4(pow(max(dithered, 0.0), 2.2), 1.0);
}
#endif
