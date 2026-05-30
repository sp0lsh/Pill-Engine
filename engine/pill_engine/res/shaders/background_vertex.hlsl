struct VOut {
    float4 pos : SV_Position;
    float2 ndc : TEXCOORD0;  // NDC XY passed to fragment for ray reconstruction
};

// Full-screen triangle: 3 vertices cover NDC [-1,1]² without a vertex buffer.
VOut vs_main(uint vi : SV_VertexID) {
    float2 p[3] = {
        float2(-1.0, -3.0),
        float2( 3.0,  1.0),
        float2(-1.0,  1.0)
    };
    VOut o;
    o.pos = float4(p[vi], 0.0, 1.0);
    o.ndc = p[vi];
    return o;
}
