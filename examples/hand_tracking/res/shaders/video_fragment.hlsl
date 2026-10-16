// CameraPass: the camera frame scaled to cover the screen, mirrored (pill_assets generates the .wgsl)

[[vk::binding(0, 0)]] Texture2D<float4> video;
[[vk::binding(1, 0)]] SamplerState video_sampler;

struct VideoParams {
    float2 uv_scale; // screen uv → video uv around the centre; x < 0 mirrors
    float2 _padding;
};
[[vk::binding(2, 0)]] ConstantBuffer<VideoParams> params;

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position, [[vk::location(0)]] float2 uv : TEXCOORD0) : SV_Target {
    float2 video_uv = 0.5 + (uv - 0.5) * params.uv_scale;
    return float4(video.SampleLevel(video_sampler, video_uv, 0.0).rgb, 1.0);
}
