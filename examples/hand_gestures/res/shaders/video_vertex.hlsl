// Full-screen triangle for CameraPass at the far plane (depth 1, the engine's depth clear value),
// so a depth test keeps the scene in front of it (pill_assets generates the .wgsl)

struct VSOut {
    float4 position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOut vs_main(uint vertex_index : SV_VertexID) {
    float2 uv = float2(float((vertex_index << 1) & 2), float(vertex_index & 2));
    VSOut output;
    output.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 1.0, 1.0);
    output.uv = uv;
    return output;
}
