// Clears one tile of the point/spot shadow atlas inside a pass that keeps the rest: a triangle
// covering the viewport (set to the tile) at the far plane, written with depth test Always.

[shader("vertex")]
float4 vs_main(uint vertex_id : SV_VertexID) : SV_Position {
    float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 1.0, 1.0);
}
