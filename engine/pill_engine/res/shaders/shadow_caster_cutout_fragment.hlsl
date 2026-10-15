// Shadow caster of alpha-tested materials: texels the PBR pass discards cast no shadow.

[[vk::binding(0, 2)]] Texture2D    base_color_texture;
[[vk::binding(1, 2)]] SamplerState base_color_sampler;
// Material array (set 3), indexed by the draw's material slot
#define MATERIAL_ARRAY_SET 3
#include "include/material.hlsl"

[shader("fragment")]
void fs_main(
    [[vk::location(0)]] float2 texture_coordinates : TEXCOORD0,
    [[vk::location(1)]] nointerpolation uint material_index : TEXCOORD1
) {
    MaterialParams material = load_material(material_index);
    if (base_color_texture.Sample(base_color_sampler, texture_coordinates).a < material.alpha_cutoff) {
        discard;
    }
}
