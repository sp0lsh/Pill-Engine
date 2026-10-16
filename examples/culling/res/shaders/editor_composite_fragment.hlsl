// @variant copy COPY
// EditorPass: the base shader draws `source` scaled into the viewport (linear filtering); the COPY
// variant copies it texel for texel.

[[vk::binding(0, 0)]] Texture2D<float4> source;

#ifdef COPY
[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    return source.Load(int3(int2(frag_coord.xy), 0));
}
#else
[[vk::binding(1, 0)]] SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position, [[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target {
    return source.SampleLevel(source_sampler, uv, 0.0);
}
#endif
