// Shadow caster: positions only, transformed like the PBR vertex shader into light clip space.

#include "include/bindings.hlsl"
#include "include/draw_transform.hlsl"

struct ShadowCamera {
    column_major float4x4 view_projection; // world → light clip space
};
[[vk::binding(0, 0)]] [DynamicOffset] ConstantBuffer<ShadowCamera> shadow_camera; // one per shadow map
[[vk::binding(0, 1)]] StructuredBuffer<DrawTransform> per_draw_array;

[shader("vertex")]
float4 vs_main([[vk::location(0)]] float3 position : POSITION, uint instance_id : SV_InstanceID) : SV_Position {
    DrawTransform transform = per_draw_array[instance_id];
    float3 world_position = draw_world_position(transform, rotation_matrix(transform), position);
    return mul(shadow_camera.view_projection, float4(world_position, 1.0));
}
