// Wireframe preview fragment shader. Edit here — pill_assets regenerates the .wgsl.
// Flat per-object color; no lighting, no tonemapping (the pass draws straight to the LDR
// swapchain after PassTonemap, so colors are written as-is).

struct PSIn {
    [[vk::location(0)]] float4 color : COLOR0;
};

struct PSOut {
    float4 color : SV_TARGET;
};

[shader("fragment")]
PSOut fs_main(PSIn input) {
    PSOut output;
    output.color = input.color;
    return output;
}
