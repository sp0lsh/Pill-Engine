[[vk::binding(0, 0)]] Texture2D    hdr_texture;
[[vk::binding(1, 0)]] SamplerState hdr_sampler;

// From RenderStateComponent::tonemap
struct TonemapParams {
    uint  mode;                  // 0 = pass-through, 1 = extended Reinhard, 2 = Lottes
    float inverse_white_squared; // Reinhard: 1/W², 0 for an infinite white point (plain c/(1 + c))
    float contrast;              // Lottes: x^a / (x^(a·d)·b + c)
    float shoulder;
    float b;
    float c;
    float _padding0, _padding1;
};
[[vk::binding(2, 0)]] ConstantBuffer<TonemapParams> params;
// [1]: exposure multiplier from the exposure pass (auto exposure / RenderStateComponent::exposure)
[[vk::binding(3, 0)]] StructuredBuffer<float> exposure_state;

[shader("fragment")]
float4 fs_main(float2 texture_coordinates : TEXCOORD0) : SV_Target {
    float3 hdr_color = hdr_texture.Sample(hdr_sampler, texture_coordinates).rgb * exposure_state[1];
    if (params.mode == 1) {
        hdr_color = hdr_color * (1.0 + hdr_color * params.inverse_white_squared) / (1.0 + hdr_color);
    } else if (params.mode == 2) {
        float3 powered = pow(max(hdr_color, 0.0), params.contrast);
        hdr_color = powered / (pow(powered, params.shoulder) * params.b + params.c);
    }
    return float4(hdr_color, 1.0);
}
