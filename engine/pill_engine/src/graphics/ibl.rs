//! Image-based lighting maps baked on the GPU by `PassIblBake` from the environment equirect:
//! octahedral maps with a one-texel border per mip, so hardware bilinear filtering is seamless
//! (oct_ibl.hlsl samples them).

use crate::graphics::PillRenderer;

/// Width and height of the specular map's mip 0.
pub(crate) const IBL_SPECULAR_SIZE: u32 = 128;
/// Specular mips, one per roughness in `IBL_SPECULAR_MIP_ROUGHNESS`.
pub(crate) const IBL_SPECULAR_MIPS: u32 = 5;
/// GGX roughness prefiltered into each specular mip; the PBR shader picks lod = roughness · 4.
pub(crate) const IBL_SPECULAR_MIP_ROUGHNESS: [f32; IBL_SPECULAR_MIPS as usize] =
    [0.04, 0.25, 0.5, 0.75, 1.0];
/// Width and height of the diffuse irradiance map.
pub(crate) const IBL_IRRADIANCE_SIZE: u32 = 32;
pub(crate) const IBL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The baked maps the PBR, SSR and DDGI passes sample. `generation` changes on every bake, so
/// consumers rebuild their bind groups.
pub(crate) struct IblFrame {
    pub(crate) irradiance_texture: wgpu::Texture,
    pub(crate) specular_texture: wgpu::Texture,
    pub(crate) irradiance_view: wgpu::TextureView,
    pub(crate) specular_view: wgpu::TextureView,
    pub(crate) generation: u64,
}

impl IblFrame {
    /// Allocates both maps; their content is undefined until `PassIblBake` runs (the first pass
    /// of the frame).
    pub(crate) fn new(renderer: &dyn PillRenderer) -> Self {
        let map = |label, size, mips| {
            renderer
                .get_device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: mips,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: IBL_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
        };
        let irradiance_texture = map("ibl_irradiance", IBL_IRRADIANCE_SIZE, 1);
        let specular_texture = map("ibl_specular", IBL_SPECULAR_SIZE, IBL_SPECULAR_MIPS);
        Self {
            irradiance_view: irradiance_texture.create_view(&Default::default()),
            specular_view: specular_texture.create_view(&Default::default()),
            irradiance_texture,
            specular_texture,
            generation: 0,
        }
    }
}

/// Inner texel whose direction a texel of an `inner`² octahedral map with a one-texel border
/// holds: border rows and columns mirror the opposite inner edge, corners the diagonally opposite
/// inner corner (`oct_ibl_inner_texel` in ibl_bake_fragment.hlsl).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn oct_border_inner_texel(texel: [i32; 2], inner: i32) -> [i32; 2] {
    let [mut x, mut y] = [texel[0] - 1, texel[1] - 1];
    if x < 0 || x >= inner {
        x = x.clamp(0, inner - 1);
        y = inner - 1 - y;
    }
    if y < 0 || y >= inner {
        y = y.clamp(0, inner - 1);
        x = inner - 1 - x;
    }
    [x, y]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oct_border_left_column_mirrors_first_inner_column() {
        assert_eq!(oct_border_inner_texel([0, 1], 4), [0, 3]);
        assert_eq!(oct_border_inner_texel([0, 4], 4), [0, 0]);
    }

    #[test]
    fn oct_border_top_row_mirrors_first_inner_row() {
        assert_eq!(oct_border_inner_texel([1, 0], 4), [3, 0]);
    }

    #[test]
    fn oct_border_corner_diagonally_opposite_inner_corner() {
        assert_eq!(oct_border_inner_texel([0, 0], 4), [3, 3]);
        assert_eq!(oct_border_inner_texel([5, 5], 4), [0, 0]);
        assert_eq!(oct_border_inner_texel([5, 0], 4), [0, 3]);
    }

    #[test]
    fn oct_border_inner_texel_unchanged() {
        assert_eq!(oct_border_inner_texel([2, 3], 4), [1, 2]);
    }
}
