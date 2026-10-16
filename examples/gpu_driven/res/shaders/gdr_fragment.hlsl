// Main draw (bgfx fs_gdr_instanced_indirect_rendering): the prop's flat color with a Lambert term
// from the face normal (screen-space derivatives, faced to the camera)

struct View {
    column_major float4x4 view_projection;
    float4 light_direction;
    float4 camera_position;
};
[[vk::binding(0, 0)]] ConstantBuffer<View> view;
[[vk::binding(3, 0)]] ConstantBuffer<float4[8]> colors;

[shader("fragment")]
float4 fs_main(float4 clip : SV_Position,
               [[vk::location(0)]] float3 world : TEXCOORD0,
               [[vk::location(1)]] nointerpolation uint prop : TEXCOORD1) : SV_Target {
    float3 normal = normalize(cross(ddx(world), ddy(world)));
    if (dot(normal, view.camera_position.xyz - world) < 0.0) {
        normal = -normal;
    }
    float lambert = max(dot(normal, -view.light_direction.xyz), 0.0);
    return float4(colors[prop].rgb * (0.35 + 0.65 * lambert), 1.0);
}
