// One Hi-Z level from the previous: the farthest depth of each 2×2 block, reads clamped at odd
// edges (bgfx cs_gdr_downscale_hi_z)

[[vk::binding(0, 0)]] Texture2D<float> source;
// Write-only (WTexture2D): baseline WebGPU has no read-write storage textures
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] WTexture2D<float> destination;

[shader("compute")]
[numthreads(16, 16, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint width, height, source_width, source_height;
    destination.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) {
        return;
    }
    source.GetDimensions(source_width, source_height);
    int2 last = int2(source_width, source_height) - 1;
    int2 base = int2(id.xy) * 2;
    float a = source.Load(int3(min(base, last), 0));
    float b = source.Load(int3(min(base + int2(1, 0), last), 0));
    float c = source.Load(int3(min(base + int2(0, 1), last), 0));
    float d = source.Load(int3(min(base + int2(1, 1), last), 0));
    destination.Store(id.xy, max(max(a, b), max(c, d)));
}
