// Per-draw transform of the vertex shaders (PBR, shadow caster), filled in pass_pbr_opaque.rs.
// The model matrix is built here from position / rotation (radians) / scale,
// so the CPU uploads raw transforms.

struct DrawTransform {
    float4 position; // xyz
    float4 rotation; // xyz, radians
    float4 scale;    // xyz
};

// Row-major float3x3 constructors; mul(R, v) applies standard right-handed rotation.
float3x3 rot_x(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(1, 0, 0,  0, cosine, -sine,  0, sine, cosine); }
float3x3 rot_y(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(cosine, 0, sine,  0, 1, 0,  -sine, 0, cosine); }
float3x3 rot_z(float angle) { float cosine = cos(angle), sine = sin(angle); return float3x3(cosine, -sine, 0,  sine, cosine, 0,  0, 0, 1); }

// model = T * (Rx*Ry*Rz) * S — matches glam from_scale_rotation_translation(Quat::x*y*z) and
// DrawTransform::model_matrix.
float3x3 rotation_matrix(DrawTransform transform) {
    return mul(rot_x(transform.rotation.x), mul(rot_y(transform.rotation.y), rot_z(transform.rotation.z)));
}

float3 draw_world_position(DrawTransform transform, float3x3 rotation, float3 position) {
    return transform.position.xyz + mul(rotation, position * transform.scale.xyz);
}

// Material slot of the draw: index into the material array (material.hlsl), set by
// DrawTransform::with_material_index
uint draw_material_index(DrawTransform transform) {
    return (uint)transform.position.w;
}
