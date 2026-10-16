// Occluder and main draws (bgfx vs_gdr_*). instances[bases[draw.prop] + instance]: the main pass
// reads the compacted instances at each prop's base (WebGPU's indirect draws can't offset the
// instance index without the optional indirect-first-instance feature); the occluder pass and
// the no-indirect fallback read the full instance list with zero bases and direct draws.

#include "include/gdr_common.hlsl"

struct View {
    column_major float4x4 view_projection;
    float4 light_direction;
    float4 camera_position;
};
struct Draw {
    uint prop;
    uint padding0;
    uint padding1;
    uint padding2;
};
[[vk::binding(0, 0)]] ConstantBuffer<View> view;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(2, 0)]] StructuredBuffer<uint> bases;
[[vk::binding(0, 1)]] ConstantBuffer<Draw> draw;

struct VertexOut {
    float4 clip : SV_Position;
    [[vk::location(0)]] float3 world : TEXCOORD0;
    [[vk::location(1)]] nointerpolation uint prop : TEXCOORD1;
};

[shader("vertex")]
VertexOut vs_main([[vk::location(0)]] float3 position : POSITION, uint instance : SV_InstanceID) {
    Instance data = instances[bases[draw.prop] + instance];
    float4 world = mul(data.world, float4(position, 1.0));
    VertexOut output;
    output.clip = mul(view.view_projection, world);
    output.world = world.xyz;
    output.prop = data.prop;
    return output;
}
