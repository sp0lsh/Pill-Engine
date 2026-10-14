// Shadow caster of alpha-tested materials: texels the PBR pass discards cast no shadow.

// Mirrors MaterialParams in include/pbr_opaque_fragment.hlsl (renderer_material.rs)
struct MaterialParams {
    float3 base_color_factor;
    float  roughness_factor;
    float3 emissive_factor;
    float  metallic_factor;
    float  clearcoat;
    float  clearcoat_roughness;
    float  opacity;
    float  alpha_cutoff;
    float  normal_y_sign;
    float  double_sided;
    float2 _padding;
};
[[vk::binding(0, 2)]] Texture2D                       base_color_texture;
[[vk::binding(1, 2)]] SamplerState                    base_color_sampler;
[[vk::binding(0, 3)]] ConstantBuffer<MaterialParams>  material;

[shader("fragment")]
void fs_main([[vk::location(0)]] float2 texture_coordinates : TEXCOORD0) {
    if (base_color_texture.Sample(base_color_sampler, texture_coordinates).a < material.alpha_cutoff) {
        discard;
    }
}
