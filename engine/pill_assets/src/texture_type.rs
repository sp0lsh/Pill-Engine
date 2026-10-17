/// What a material texture is used as; decides how it's cooked (sRGB color, normal map, linear).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureType {
    Color,
    Normal,
    MetallicRoughness,
    Emissive,
}
