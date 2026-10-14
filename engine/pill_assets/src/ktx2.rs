//! KTX2 (Basis Universal, e.g. glTF `KHR_texture_basisu`) → RTEX v5: BC7 with a full mip chain;
//! RTEX v1 (RGBA8, mips built at upload) for base sizes BC7 can't hold (not whole 4×4 blocks).
//!
//! Levels stored in the file are transcoded straight to BC7 (UASTC is a subset of BC7, so that is
//! near lossless and fast). Missing levels are built from the transcoded RGBA8 with a 2×2 box
//! filter and encoded with Intel's ISPC BC7 encoder.

use anyhow::{anyhow, Result};
use basisu::{DecodeFlags, TargetFormat, Transcoder};

/// Changes whenever the cooked output does, so callers can version their caches.
pub const COOK_VERSION: u32 = 2;

/// How a texture's texels are filtered and stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureKind {
    /// sRGB color (base color, emissive): filtered in linear light, sampled as sRGB.
    Color,
    /// Linear data (metallic-roughness, occlusion).
    Linear,
    /// Tangent-space normal map (either green convention): renormalised per mip.
    Normal,
}

/// Cooks a KTX2 file into RTEX bytes: v5 (BC7 + mips), or v1 (RGBA8) when the base size is not
/// in whole 4×4 blocks (GPUs require that for block-compressed textures).
pub fn ktx2_to_rtex(bytes: &[u8], kind: TextureKind) -> Result<Vec<u8>> {
    let transcoder = Transcoder::new(bytes).map_err(|e| anyhow!("KTX2: {e:?}"))?;
    let (width, height) = transcoder.base_dimensions();
    let transcode = |level: u32, target: TargetFormat| {
        transcoder
            .transcode(level, target, DecodeFlags::from_bits(0))
            .map_err(|e| anyhow!("KTX2 level {level} → {target:?}: {e:?}"))
    };
    if width % 4 != 0 || height % 4 != 0 {
        return Ok(crate::convert::rtex_bytes(
            width,
            height,
            &transcode(0, TargetFormat::Rgba32)?,
        ));
    }
    let mip_count = u32::BITS - width.max(height).leading_zeros();
    let stored = transcoder.level_count();

    let mut mips = Vec::with_capacity(mip_count as usize);
    for level in 0..stored.min(mip_count) {
        mips.push(transcode(level, TargetFormat::Bc7Rgba)?);
    }
    // The last stored level in RGBA8 seeds the generated ones; the base level gives the average
    let base = transcode(0, TargetFormat::Rgba32)?;
    let average_color = average_color(&base, kind);
    if mips.len() < mip_count as usize {
        let last = mips.len() as u32 - 1;
        let mut level = if last == 0 {
            base
        } else {
            transcode(last, TargetFormat::Rgba32)?
        };
        let (mut level_width, mut level_height) = ((width >> last).max(1), (height >> last).max(1));
        let settings = if transcoder.has_alpha() {
            intel_tex_2::bc7::alpha_very_fast_settings()
        } else {
            intel_tex_2::bc7::opaque_very_fast_settings()
        };
        while mips.len() < mip_count as usize {
            level = downsample(&level, level_width, level_height, kind);
            (level_width, level_height) = ((level_width / 2).max(1), (level_height / 2).max(1));
            mips.push(encode_bc7(&level, level_width, level_height, &settings));
        }
    }
    Ok(crate::convert::rtex_bc7_bytes(
        width,
        height,
        kind == TextureKind::Color,
        average_color,
        &mips,
    ))
}

/// BC7 blocks of RGBA8 texels; sizes below or between whole blocks are padded by repeating the
/// last row and column (the GPU samples only the real texels).
fn encode_bc7(
    rgba: &[u8],
    width: u32,
    height: u32,
    settings: &intel_tex_2::bc7::EncodeSettings,
) -> Vec<u8> {
    let (padded_width, padded_height) = (width.div_ceil(4) * 4, height.div_ceil(4) * 4);
    let padded;
    let data = if (padded_width, padded_height) == (width, height) {
        rgba
    } else {
        padded = (0..padded_height)
            .flat_map(|y| {
                (0..padded_width).flat_map(move |x| {
                    let index = ((y.min(height - 1) * width + x.min(width - 1)) * 4) as usize;
                    rgba[index..index + 4].iter().copied()
                })
            })
            .collect::<Vec<u8>>();
        &padded
    };
    intel_tex_2::bc7::compress_blocks(
        settings,
        &intel_tex_2::RgbaSurface {
            data,
            width: padded_width,
            height: padded_height,
            stride: padded_width * 4,
        },
    )
}

/// Next mip level: 2×2 box filter (an odd last row/column repeats). Color is averaged in linear
/// light (sRGB approximated as gamma 2); normals are renormalised. Mirrors pill_engine's
/// `renderer_texture::downsample` for RGBA8 uploads.
fn downsample(rgba: &[u8], width: u32, height: u32, kind: TextureKind) -> Vec<u8> {
    let (next_width, next_height) = ((width / 2).max(1), (height / 2).max(1));
    let srgb = kind == TextureKind::Color;
    let texel = |x: u32, y: u32| {
        let index = ((y.min(height - 1) * width + x.min(width - 1)) * 4) as usize;
        &rgba[index..index + 4]
    };
    let mut output = Vec::with_capacity((next_width * next_height * 4) as usize);
    for y in 0..next_height {
        for x in 0..next_width {
            let block = [
                texel(2 * x, 2 * y),
                texel(2 * x + 1, 2 * y),
                texel(2 * x, 2 * y + 1),
                texel(2 * x + 1, 2 * y + 1),
            ];
            let mut average = [0.0_f32; 4];
            for (channel, value) in average.iter_mut().enumerate() {
                let gamma = srgb && channel < 3;
                *value = block
                    .iter()
                    .map(|texel| {
                        let value = texel[channel] as f32 / 255.0;
                        if gamma {
                            value * value
                        } else {
                            value
                        }
                    })
                    .sum::<f32>()
                    / 4.0;
                if gamma {
                    *value = value.sqrt();
                }
            }
            if kind == TextureKind::Normal {
                let mut normal = [0, 1, 2].map(|channel| average[channel] * 2.0 - 1.0);
                // Two-channel maps (empty blue): z rebuilt from x and y, as the PBR shader does
                if normal[2] < -0.9 {
                    normal[2] = (1.0 - normal[0] * normal[0] - normal[1] * normal[1])
                        .max(0.0)
                        .sqrt();
                }
                let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
                if length > 1e-4 {
                    for channel in 0..3 {
                        average[channel] = normal[channel] / length * 0.5 + 0.5;
                    }
                }
            }
            output.extend(average.map(|value| (value * 255.0 + 0.5) as u8));
        }
    }
    output
}

/// Mean linear RGB weighted by alpha, from at most ~64k strided samples; white when transparent.
fn average_color(rgba: &[u8], kind: TextureKind) -> [f32; 3] {
    let stride = (rgba.len() / 4).div_ceil(65_536).max(1);
    let to_linear = |byte: u8| {
        let value = byte as f32 / 255.0;
        match kind {
            TextureKind::Color if value <= 0.04045 => value / 12.92,
            TextureKind::Color => ((value + 0.055) / 1.055).powf(2.4),
            _ => value,
        }
    };
    let mut sum = [0.0f64; 3];
    let mut weight = 0.0f64;
    for pixel in rgba.chunks_exact(4).step_by(stride) {
        let alpha = pixel[3] as f64 / 255.0;
        for (channel, total) in sum.iter_mut().enumerate() {
            *total += to_linear(pixel[channel]) as f64 * alpha;
        }
        weight += alpha;
    }
    if weight <= 0.0 {
        return [1.0; 3];
    }
    sum.map(|total| (total / weight) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Bistro texture cached by examples/bistro (KTX2 UASTC + Zstd, 2048², one level); the
    /// test is skipped when the example has not downloaded it yet.
    fn cached_bistro_ktx2() -> Option<Vec<u8>> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/bistro/build/fetch_cache/Textures/Antenna_Metal_BaseColor.ktx2"
        );
        std::fs::read(path).ok()
    }

    #[test]
    fn bistro_uastc_cooks_to_full_bc7_mip_chain() {
        let Some(bytes) = cached_bistro_ktx2() else {
            eprintln!("skipped: Bistro texture not cached");
            return;
        };
        let rtex = ktx2_to_rtex(&bytes, TextureKind::Color).unwrap();
        let read_u32 =
            |offset: usize| u32::from_le_bytes(rtex[offset..offset + 4].try_into().unwrap());
        assert_eq!(
            (read_u32(4), read_u32(8), read_u32(12), read_u32(16)),
            (5, 2048, 2048, 1)
        );
        assert_eq!(read_u32(32), 12);
        // 2048² down to 1×1: whole 4×4 blocks of 16 bytes per level
        let blocks: usize = (0..12)
            .map(|mip| ((2048usize >> mip).max(1)).div_ceil(4).pow(2))
            .sum();
        assert_eq!(rtex.len(), 36 + blocks * 16);
    }

    #[test]
    fn padded_encode_of_a_2x2_level_is_one_block() {
        let rgba = [255u8; 2 * 2 * 4];
        let blocks = encode_bc7(&rgba, 2, 2, &intel_tex_2::bc7::opaque_very_fast_settings());
        assert_eq!(blocks.len(), 16);
    }

    #[test]
    fn normal_downsample_stays_unit_length() {
        // Two opposite tilts average to straight up after renormalisation
        let rgba = [
            [255, 128, 128, 255],
            [0, 128, 128, 255],
            [255, 128, 128, 255],
            [0, 128, 128, 255],
        ]
        .concat();
        let next = downsample(&rgba, 2, 2, TextureKind::Normal);
        let normal = [0, 1, 2].map(|channel| next[channel] as f32 / 255.0 * 2.0 - 1.0);
        let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 0.02, "{normal:?}");
    }
}
