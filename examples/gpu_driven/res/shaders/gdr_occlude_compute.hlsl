// Per-instance frustum and Hi-Z occlusion test (bgfx cs_gdr_occlude_props): the AABB's screen
// rect and nearest depth against the farthest Hi-Z depth at the level where the rect covers
// about 2 × 2 texels. Writes the predicate and counts the visible instances of each prop.

#include "include/gdr_common.hlsl"

struct Cull {
    column_major float4x4 view_projection;
    float2 hiz_size;
    uint mip_count;
    uint instance_count;
};
[[vk::binding(0, 0)]] ConstantBuffer<Cull> cull;
[[vk::binding(1, 0)]] Texture2D<float> hiz;
[[vk::binding(2, 0)]] StructuredBuffer<Bounds> bounds;
[[vk::binding(3, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> predicates;
[[vk::binding(5, 0)]] RWStructuredBuffer<Atomic<uint>> counts;

float depth_at(float2 uv, uint mip) {
    uint width, height, levels;
    hiz.GetDimensions(mip, width, height, levels);
    int2 size = int2(width, height);
    int2 texel = clamp(int2(uv * float2(size)), int2(0, 0), size - 1);
    return hiz.Load(int3(texel, mip));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    if (i >= cull.instance_count) {
        return;
    }
    Bounds box = bounds[i];
    float3 ndc_min = float3(1e9, 1e9, 1e9);
    float3 ndc_max = float3(-1e9, -1e9, -1e9);
    uint behind = 0;
    for (uint corner = 0; corner < 8; corner++) {
        float3 p = float3((corner & 1) != 0 ? box.max_corner.x : box.min_corner.x,
                          (corner & 2) != 0 ? box.max_corner.y : box.min_corner.y,
                          (corner & 4) != 0 ? box.max_corner.z : box.min_corner.z);
        float4 clip = mul(cull.view_projection, float4(p, 1.0));
        if (clip.w <= 0.0) {
            behind++;
            continue;
        }
        float3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc);
        ndc_max = max(ndc_max, ndc);
    }
    // Entirely behind the camera: culled; across the camera plane: kept (no screen rect)
    bool visible = behind < 8;
    if (behind == 0) {
        bool outside = ndc_max.x < -1.0 || ndc_max.y < -1.0 || ndc_min.x > 1.0 || ndc_min.y > 1.0 || ndc_min.z > 1.0;
        // Screen rect in uv (y down)
        float2 a = clamp(ndc_min.xy, -1.0, 1.0);
        float2 b = clamp(ndc_max.xy, -1.0, 1.0);
        float2 uv_min = float2(a.x, -b.y) * 0.5 + 0.5;
        float2 uv_max = float2(b.x, -a.y) * 0.5 + 0.5;
        float2 extent = (uv_max - uv_min) * cull.hiz_size;
        uint mip = uint(clamp(ceil(log2(max(max(extent.x, extent.y), 1.0))), 0.0, float(cull.mip_count - 1)));
        // One level finer when the rect still covers at most 2 × 2 of its texels
        int lower = max(int(mip) - 1, 0);
        float2 scale = cull.hiz_size / exp2(float(lower));
        float2 texels = ceil(uv_max * scale) - floor(uv_min * scale);
        if (texels.x <= 2.0 && texels.y <= 2.0) {
            mip = uint(lower);
        }
        float farthest = max(max(depth_at(uv_min, mip), depth_at(float2(uv_max.x, uv_min.y), mip)),
                             max(depth_at(float2(uv_min.x, uv_max.y), mip), depth_at(uv_max, mip)));
        visible = !outside && max(ndc_min.z, 0.0) <= farthest;
    }
    predicates[i] = visible ? 1 : 0;
    if (visible) {
        counts[instances[i].prop].add(1);
    }
}
