// Shadow caster of alpha-tested or double-sided materials: positions and texture coordinates,
// transformed like the PBR vertex shader into light clip space.

#include "include/draw_transform.hlsl"

struct ShadowCamera {
    column_major float4x4 view_projection; // world → light clip space
};
[[vk::binding(0, 0)]] ConstantBuffer<ShadowCamera> shadow_camera;
[[vk::binding(0, 1)]] StructuredBuffer<DrawTransform> per_draw_array;

struct VertexOutput {
    float4 position                                : SV_Position;
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0;
};

// Vertex buffer locations of RendererMesh (as pbr_opaque_vertex.hlsl); Slang keeps explicit
// locations on struct members, not on bare entry-point parameters
struct VertexInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(4)]] float2 texture_coordinates;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input, uint instance_id : SV_InstanceID) {
    DrawTransform transform = per_draw_array[instance_id];
    float3 world_position = draw_world_position(transform, rotation_matrix(transform), input.position);
    VertexOutput output;
    output.position            = mul(shadow_camera.view_projection, float4(world_position, 1.0));
    output.texture_coordinates = input.texture_coordinates;
    return output;
}
