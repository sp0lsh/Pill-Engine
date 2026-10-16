// Full-screen triangle for TonyMcMapfacePass (pill_assets generates the .wgsl)

[shader("vertex")]
float4 vs_main(uint vertex_index : SV_VertexID) : SV_Position {
    float2 uv = float2(float((vertex_index << 1) & 2), float(vertex_index & 2));
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
