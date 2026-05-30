struct SkyCam {
    float3 right;
    float  tan_half_fov;
    float3 up;
    float  aspect;
    float3 fwd;
    float  _pad;
};
[[vk::binding(0, 0)]] ConstantBuffer<SkyCam> UCam;
[[vk::binding(1, 0)]] Texture2D    texEquirect;
[[vk::binding(2, 0)]] SamplerState smpEquirect;

static const float PI = 3.14159265359;

float2 dir_to_equirect_uv(float3 dir) {
    float3 d = normalize(dir);
    float  u = 0.5 + atan2(d.z, d.x) / (2.0 * PI);
    float  v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
    return float2(frac(u), clamp(v, 0.0, 1.0));
}

[shader("fragment")]
float4 fs_main(float2 ndc : TEXCOORD0) : SV_Target {
    float3 dir = normalize(
        UCam.right * (ndc.x * UCam.tan_half_fov * UCam.aspect)
      + UCam.up    * (ndc.y * UCam.tan_half_fov)
      + UCam.fwd
    );
    float2 uv = dir_to_equirect_uv(dir);
    return texEquirect.Sample(smpEquirect, uv);
}
