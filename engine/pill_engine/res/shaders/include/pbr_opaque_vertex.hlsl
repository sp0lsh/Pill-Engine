// PBR vertex shader body, included by pbr_opaque_vertex.hlsl. The motion-vector outputs are
// always declared; without post_fx `camera.flags.x` is 0 and the pass has no velocity target.

#include "draw_transform.hlsl"

struct Camera {
    float4                position;
    column_major float4x4 view_projection;
    column_major float4x4 previous_view_projection;
    float4                flags; // x = 1 when the pass writes motion vectors
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

// This frame's and last frame's transforms (motion vectors only)
[[vk::binding(0, 3)]] StructuredBuffer<DrawTransform> per_draw_array;
[[vk::binding(1, 3)]] StructuredBuffer<DrawTransform> previous_per_draw_array;

struct VertexOutput {
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0;
    [[vk::location(1)]] float3 world_position      : TEXCOORD1;
    [[vk::location(2)]] float3 world_normal        : TEXCOORD2;
    [[vk::location(3)]] float3 world_tangent       : TEXCOORD3;
    [[vk::location(4)]] float3 world_bitangent     : TEXCOORD4;
    // Clip positions of this and last frame; the fragment shader divides per pixel (dividing per
    // vertex and interpolating is wrong across a triangle, and for vertices behind the camera)
    [[vk::location(5)]] float4 clip_position          : TEXCOORD5;
    [[vk::location(6)]] float4 previous_clip_position : TEXCOORD6;
    // Material slot of the draw, read by the fragment shader from the material array
    [[vk::location(7)]] nointerpolation uint material_index : TEXCOORD7;
                        float4 sv_position         : SV_POSITION;
};

// RendererMesh streams: slot 0 position; slot 1 packed attributes. Normal and tangent arrive as
// Unorm10_10_10_2 of v * 0.5 + 0.5; the tangent's alpha codes the bitangent sign (mesh.rs
// TANGENT_CODE_*: 0 = -1, 1/3 = no tangent frame, 1 = +1).
struct VertexInput {
    [[vk::location(0)]] float3 position;
    [[vk::location(4)]] float2 texture_coordinates;
    [[vk::location(5)]] float4 packed_normal;
    [[vk::location(6)]] float4 packed_tangent;
};

// Zero for (near) zero-length input, so degenerate tangents don't turn into NaN
float3 safe_normalize(float3 value) {
    float length_squared = dot(value, value);
    return length_squared > 1e-12 ? value * rsqrt(length_squared) : float3(0.0, 0.0, 0.0);
}

[shader("vertex")]
VertexOutput vs_main(VertexInput input, uint instance_id : SV_InstanceID) {
    DrawTransform transform = per_draw_array[instance_id];

    float3x3 rotation = rotation_matrix(transform);
    float3 world_position = draw_world_position(transform, rotation, input.position);
    float4 clip_position  = mul(camera.view_projection, float4(world_position, 1.0));

    VertexOutput output;
    output.sv_position            = clip_position;
    output.clip_position          = clip_position;
    output.previous_clip_position = clip_position;
    if (camera.flags.x > 0.5) {
        DrawTransform previous = previous_per_draw_array[instance_id];
        float3 previous_world_position = draw_world_position(previous, rotation_matrix(previous), input.position);
        output.previous_clip_position  = mul(camera.previous_view_projection, float4(previous_world_position, 1.0));
    }
    float3 normal       = input.packed_normal.xyz * 2.0 - 1.0;
    int    tangent_code = (int)round(input.packed_tangent.w * 3.0);
    // No tangent frame: zero tangent and bitangent, the fragment shader keeps the vertex normal
    float3 tangent      = tangent_code == 1 ? float3(0.0, 0.0, 0.0) : input.packed_tangent.xyz * 2.0 - 1.0;
    float3 bitangent    = cross(normal, tangent) * (tangent_code == 0 ? -1.0 : 1.0);

    output.texture_coordinates = input.texture_coordinates;
    output.material_index      = draw_material_index(transform);
    output.world_position      = world_position;
    // Inverse-transpose of R*S is R*S^-1: divide by scale so non-uniform scale keeps normals perpendicular
    output.world_normal        = normalize(mul(rotation, normal / transform.scale.xyz));
    // Tangents lie in the surface, so they transform like positions (R*S)
    output.world_tangent       = safe_normalize(mul(rotation, tangent * transform.scale.xyz));
    output.world_bitangent     = safe_normalize(mul(rotation, bitangent * transform.scale.xyz));
    return output;
}
