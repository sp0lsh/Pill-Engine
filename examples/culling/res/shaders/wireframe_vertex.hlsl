// Wireframe preview vertex shader. Edit here — pill_assets regenerates the .wgsl.
// Mirrors pbr_opaque_vertex's per-draw model-matrix build (T * Rx*Ry*Rz * S, rotation in
// radians) so the wireframe lines align exactly with the shaded geometry; adds a per-object
// color carried through to the fragment. Only vertex position (location 0) is consumed.

struct Camera {
    column_major float4x4 viewProjection;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> UCamera;

struct PerDraw {
    float4 position; // xyz
    float4 rotation; // xyz, radians
    float4 scale;    // xyz
    float4 color;    // rgb (HDR, see display_to_hdr), a = opacity
};
[[vk::binding(0, 1)]] StructuredBuffer<PerDraw> UPerDrawArray;

struct VSOut {
    [[vk::location(0)]] float4 color      : COLOR0;
                        float4 sv_position : SV_POSITION;
};

struct VSIn {
    [[vk::location(0)]] float3 pos;
};

// Row-major float3x3 constructors; mul(R, v) applies standard right-handed rotation.
float3x3 rot_x(float a) { float c = cos(a), s = sin(a); return float3x3(1, 0, 0,  0, c, -s,  0, s, c); }
float3x3 rot_y(float a) { float c = cos(a), s = sin(a); return float3x3(c, 0, s,  0, 1, 0,  -s, 0, c); }
float3x3 rot_z(float a) { float c = cos(a), s = sin(a); return float3x3(c, -s, 0,  s, c, 0,  0, 0, 1); }

[shader("vertex")]
VSOut vs_main(VSIn input, uint instance_id : SV_InstanceID) {
    PerDraw per_draw = UPerDrawArray[instance_id];

    // model = T * (Rx*Ry*Rz) * S — identical to the PBR pass so overlays match.
    float3x3 rotation = mul(rot_x(per_draw.rotation.x), mul(rot_y(per_draw.rotation.y), rot_z(per_draw.rotation.z)));
    float3 scaled = input.pos * per_draw.scale.xyz;
    float3 worldPos = per_draw.position.xyz + mul(rotation, scaled);

    VSOut output;
    output.sv_position = mul(UCamera.viewProjection, float4(worldPos, 1.0));
    output.color       = per_draw.color;
    return output;
}
