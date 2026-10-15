#ifndef MATERIAL_HLSL
#define MATERIAL_HLSL
// Material array: every material's parameters as 16 half floats in two uint4 rows, indexed by
// material slot (renderer_material.rs PackedMaterial). The including shader defines
// MATERIAL_ARRAY_SET, the bind group of the array.

static const uint MATERIAL_ARRAY_CAPACITY = 1024; // config.rs MATERIAL_ARRAY_CAPACITY

struct MaterialArray {
    uint4 rows[MATERIAL_ARRAY_CAPACITY * 2];
};
[[vk::binding(0, MATERIAL_ARRAY_SET)]] ConstantBuffer<MaterialArray> material_array;

struct MaterialParams {
    float3 base_color_factor;
    float  roughness_factor;
    float3 emissive_factor;
    float  metallic_factor;
    float  clearcoat;           // coat weight, 0 = no coat
    float  clearcoat_roughness; // perceptual
    float  opacity;             // < 1: drawn blended by the translucent pass
    float  alpha_cutoff;        // > 0: texels with base-color alpha below it are discarded
    float  normal_y_sign;       // -1: DirectX normal map (green down)
    float  double_sided;        // 1: back faces drawn, lit with the flipped normal
};

float2 unpack_halves(uint word) {
    return float2(f16tof32(word & 0xffff), f16tof32(word >> 16));
}

MaterialParams load_material(uint index) {
    uint4 row0 = material_array.rows[index * 2];
    uint4 row1 = material_array.rows[index * 2 + 1];
    float2 rg = unpack_halves(row0.x), b_roughness = unpack_halves(row0.y);
    float2 emissive_rg = unpack_halves(row0.z), emissive_b_metallic = unpack_halves(row0.w);
    float2 clearcoat = unpack_halves(row1.x), opacity_cutoff = unpack_halves(row1.y);
    float2 normal_sign_double_sided = unpack_halves(row1.z);

    MaterialParams material;
    material.base_color_factor   = float3(rg, b_roughness.x);
    material.roughness_factor    = b_roughness.y;
    material.emissive_factor     = float3(emissive_rg, emissive_b_metallic.x);
    material.metallic_factor     = emissive_b_metallic.y;
    material.clearcoat           = clearcoat.x;
    material.clearcoat_roughness = clearcoat.y;
    material.opacity             = opacity_cutoff.x;
    material.alpha_cutoff        = opacity_cutoff.y;
    material.normal_y_sign       = normal_sign_double_sided.x;
    material.double_sided        = normal_sign_double_sided.y;
    return material;
}
#endif
