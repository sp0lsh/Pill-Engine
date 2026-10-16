// Occluder depth → mip 0 of the Hi-Z pyramid (bgfx cs_gdr_copy_z)

[[vk::binding(0, 0)]] Texture2D<float> depth;
// Write-only (WTexture2D): baseline WebGPU has no read-write storage textures
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] WTexture2D<float> hiz;

[shader("compute")]
[numthreads(16, 16, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint width, height;
    hiz.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) {
        return;
    }
    hiz.Store(id.xy, depth.Load(int3(id.xy, 0)));
}
