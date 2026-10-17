#[cfg(not(feature = "headless"))]
use crate::resources::material::set_renderer_textures;
use crate::{
    engine::Engine,
    renderer::resources::RendererTexture,
    resources::{Material, Resource, ResourceLoader, ResourceStorage},
};

use std::borrow::Cow;

use pill_core::{get_type_name, ErrorContext, PillSlotMapKey, PillStyle, PillTypeMapKey, Result};
pill_core::define_new_pill_slotmap_key! {
    pub struct TextureHandle;
}

pub use pill_assets::TextureType;

#[readonly::make]
pub struct Texture {
    #[readonly]
    pub name: String,
    #[readonly]
    pub resource_loader: ResourceLoader,
    #[readonly]
    pub texture_type: TextureType,
    /// Mean linear RGB of a color texture, measured at load (diffuse GI uses it as the surface
    /// albedo at ray hits).
    pub(crate) average_color: Option<[f32; 3]>,
}

impl Texture {
    pub fn new(name: &str, texture_type: TextureType, resource_loader: ResourceLoader) -> Self {
        Self {
            name: name.to_string(),
            resource_loader,
            texture_type,
            average_color: None,
        }
    }

    pub fn from_bytes(name: &str, texture_type: TextureType, bytes: &[u8]) -> Self {
        Self::from_bytes_owned(name, texture_type, bytes.to_vec())
    }

    /// Like `from_bytes`, taking ownership so large runtime-decoded textures aren't copied.
    pub fn from_bytes_owned(name: &str, texture_type: TextureType, bytes: Vec<u8>) -> Self {
        Self::new(
            name,
            texture_type,
            ResourceLoader::Bytes(bytes.into_boxed_slice()),
        )
    }
}

/// Decoded `.cooked_tex`: b"RTEX" | u32 version | u32 width | u32 height (LE); v1 sRGB RGBA8,
/// v2 Rgba32Float, v4 u32 mip_count + Rgba32Float mips, v5 u32 flags (bit 0 = sRGB) + f32×3
/// average linear color + u32 mip_count + BC7 mips (see pill_assets `util/rtex.rs`).
pub(crate) struct CookedTexture<'a> {
    pub(crate) version: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) mips: Vec<&'a [u8]>,
    /// v5: mean linear color stored by the cooker.
    pub(crate) average_color: Option<[f32; 3]>,
}

pub(crate) fn parse_cooked_tex(bytes: &[u8]) -> Result<CookedTexture<'_>> {
    let invalid = |reason: &str| -> pill_core::PillError {
        format!("not a valid .cooked_tex file: {reason}").into()
    };
    if bytes.len() < 16 || &bytes[0..4] != b"RTEX" {
        return Err(invalid("bad magic or truncated header"));
    }
    let read_u32 =
        |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let read_f32 = |offset: usize| f32::from_bits(read_u32(offset));
    let version = read_u32(4);
    let width = read_u32(8);
    let height = read_u32(12);

    let mut average_color = None;
    let (format, mip_count, data_offset) = match version {
        1 => (wgpu::TextureFormat::Rgba8UnormSrgb, 1, 16),
        2 => (wgpu::TextureFormat::Rgba32Float, 1, 16),
        4 => {
            if bytes.len() < 20 {
                return Err(invalid("truncated v4 header"));
            }
            (wgpu::TextureFormat::Rgba32Float, read_u32(16), 20)
        }
        5 => {
            if bytes.len() < 36 {
                return Err(invalid("truncated v5 header"));
            }
            average_color = Some([read_f32(20), read_f32(24), read_f32(28)]);
            let format = if read_u32(16) & 1 != 0 {
                wgpu::TextureFormat::Bc7RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc7RgbaUnorm
            };
            (format, read_u32(32), 36)
        }
        other => return Err(invalid(&format!("unsupported version {other}"))),
    };
    // Validate the header before sizing anything from it (files can be truncated or corrupt)
    if width == 0 || height == 0 {
        return Err(invalid("zero width or height"));
    }
    let max_mip_count = 32 - width.max(height).leading_zeros();
    if mip_count == 0 || mip_count > max_mip_count {
        return Err(invalid(&format!(
            "invalid mip count {mip_count} for {width}x{height}"
        )));
    }
    let mut mips = Vec::with_capacity(mip_count as usize);
    let mut offset: usize = data_offset;
    for mip in 0..mip_count {
        let (blocks_x, blocks_y, block_bytes) =
            crate::graphics::mip_copy_layout(format, width, height, mip)
                .expect("RTEX formats have a block size");
        let mip_data = (blocks_x as usize)
            .checked_mul(blocks_y as usize)
            .and_then(|blocks| blocks.checked_mul(block_bytes as usize))
            .and_then(|mip_bytes| offset.checked_add(mip_bytes))
            .and_then(|end| bytes.get(offset..end))
            .ok_or_else(|| invalid("pixel data shorter than header declares"))?;
        offset += mip_data.len();
        mips.push(mip_data);
    }

    Ok(CookedTexture {
        version,
        width,
        height,
        format,
        mips,
        average_color,
    })
}

/// Mean linear RGB of sRGB RGBA8 pixels weighted by alpha (the color of what is visible, not of
/// cut-out texels), from at most ~64k evenly strided samples; white when fully transparent.
/// Computed for every Color texture, emissive maps included (the type does not tell them apart).
fn average_srgb8_color(rgba: &[u8]) -> [f32; 3] {
    let pixel_count = rgba.len() / 4;
    let stride = pixel_count.div_ceil(65_536).max(1);
    let to_linear = |byte: u8| {
        let value = byte as f32 / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
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

#[cfg(not(target_arch = "wasm32"))]
fn decode_png(bytes: &[u8]) -> Result<(Cow<'static, [u8]>, u32, u32)> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder
        .read_info()
        .map_err(|e| -> pill_core::PillError { e.to_string().into() })?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| -> pill_core::PillError { e.to_string().into() })?;
    let width = info.width;
    let height = info.height;
    let raw = &buf[..info.buffer_size()];
    let rgba = match info.color_type {
        png::ColorType::Rgba => raw.to_vec(),
        png::ColorType::Rgb => raw
            .chunks(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        png::ColorType::Grayscale => raw.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::GrayscaleAlpha => raw
            .chunks(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        _ => {
            let e: pill_core::PillError =
                format!("unsupported PNG color type: {:?}", info.color_type).into();
            return Err(e);
        }
    };
    Ok((Cow::Owned(rgba), width, height))
}

impl PillTypeMapKey for Texture {
    type Storage = ResourceStorage<Texture>;
}

impl Resource for Texture {
    type Handle = TextureHandle;

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn initialize(&mut self, engine: &mut Engine) -> Result<()> {
        let error_message = format!(
            "Initializing {} {} failed",
            "Resource".general_object_style(),
            get_type_name::<Self>().specific_object_style()
        );

        // Source bytes: the mounted asset pack, a cooked file on disk, a PNG, or the loader's bytes
        let source: Cow<'_, [u8]> = match &self.resource_loader {
            ResourceLoader::Path(path) => {
                let base = engine.game_resources_directory_path.join(path);
                let cooked_tex_path = base.with_extension("cooked_tex");
                // The mounted asset pack wins over the cooked file on disk
                if let Some(bytes) = engine.packed_asset(&path.with_extension("cooked_tex")) {
                    bytes
                } else if cooked_tex_path.exists() {
                    Cow::Owned(std::fs::read(&cooked_tex_path).map_err(
                        |e| -> pill_core::PillError {
                            format!("Failed to read texture {cooked_tex_path:?}: {e}").into()
                        },
                    )?)
                } else {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        // Check if path to asset is correct
                        pill_core::validate_asset_path(&base, &["png", "cooked_tex"])?;
                        Cow::Owned(std::fs::read(&base).map_err(|e| -> pill_core::PillError {
                            format!("Failed to read texture {base:?}: {e}").into()
                        })?)
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        return Err(pill_core::PillError::from(format!(
                            "No preprocessed .cooked_tex found for {:?}; run `pill_launcher -a assets`",
                            base
                        )));
                    }
                }
            }
            ResourceLoader::Bytes(bytes) => Cow::Borrowed(&bytes[..]),
        };

        // Material textures are RGBA8 (PNG or RTEX v1, mips built here) or BC7 with mips (RTEX
        // v5); float RTEX files are environment/IBL maps for `Engine::load_cooked_texture`
        #[cfg_attr(feature = "headless", allow(unused_variables))]
        let renderer_texture = if source.starts_with(b"RTEX") {
            let cooked = parse_cooked_tex(&source)?;
            match cooked.version {
                1 => {
                    let rgba = cooked.mips[0];
                    if self.texture_type == TextureType::Color {
                        self.average_color = Some(average_srgb8_color(rgba));
                    }
                    #[cfg(not(feature = "headless"))]
                    RendererTexture::new_texture(
                        engine.renderer.get_device(),
                        engine.renderer.get_queue(),
                        &self.name,
                        rgba,
                        cooked.width,
                        cooked.height,
                        self.texture_type,
                    )
                    .context(error_message)?
                }
                5 => {
                    if self.texture_type == TextureType::Color {
                        self.average_color = cooked.average_color;
                    }
                    #[cfg(not(feature = "headless"))]
                    RendererTexture::new_bc7_texture(
                        engine.renderer.get_device(),
                        engine.renderer.get_queue(),
                        &self.name,
                        (cooked.width, cooked.height),
                        cooked.format == wgpu::TextureFormat::Bc7RgbaUnormSrgb,
                        &cooked.mips,
                    )
                    .context(error_message)?
                }
                other => {
                    return Err(pill_core::PillError::from(format!(
                        "RTEX v{other} is a float texture; load it with Engine::load_cooked_texture into RenderStateComponent"
                    )))
                }
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (rgba, width, height) = decode_png(&source)?;
                if self.texture_type == TextureType::Color {
                    self.average_color = Some(average_srgb8_color(&rgba));
                }
                #[cfg(not(feature = "headless"))]
                RendererTexture::new_texture(
                    engine.renderer.get_device(),
                    engine.renderer.get_queue(),
                    &self.name,
                    &rgba,
                    width,
                    height,
                    self.texture_type,
                )
                .context(error_message)?
            }
            #[cfg(target_arch = "wasm32")]
            {
                return Err(pill_core::PillError::from(
                    "Texture::from_bytes on wasm requires pre-converted COOKED_TEX format",
                ));
            }
        };
        drop(source);
        #[cfg(not(feature = "headless"))]
        engine.resource_manager.add_resource(renderer_texture)?;

        // The pixels live on the GPU now; nothing reads the source bytes again
        if let ResourceLoader::Bytes(bytes) = &mut self.resource_loader {
            *bytes = Box::default();
        }

        Ok(())
    }

    fn destroy<H: PillSlotMapKey>(&mut self, engine: &mut Engine, self_handle: H) -> Result<()> {
        #[cfg(not(feature = "headless"))]
        engine
            .resource_manager
            .remove_resource_by_name::<RendererTexture>(&self.name)?;

        // Clear slots referencing this texture (CPU side), then rebuild their GPU bind groups
        let texture_handle = TextureHandle::from(self_handle.data());
        let affected_material_names: Vec<String> = engine
            .resource_manager
            .get_resource_storage_mut::<Material>()?
            .data
            .iter_mut()
            .filter_map(|(_, slot)| {
                let material = slot.as_mut().expect("Critical: Resource is None");
                material
                    .remove_texture(&texture_handle)
                    .then(|| material.name.clone())
            })
            .collect();

        #[cfg(not(feature = "headless"))]
        for material_name in &affected_material_names {
            let textures = engine
                .resource_manager
                .get_resource_by_name::<Material>(material_name)?
                .renderer_textures(&engine.resource_manager)?;
            set_renderer_textures(engine, material_name, &textures)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{average_srgb8_color, parse_cooked_tex};

    #[test]
    fn average_color_half_transparent_texture_averages_opaque_texels_only() {
        // Opaque red and fully transparent white texels
        let pixels = [255, 0, 0, 255, 255, 255, 255, 0].repeat(8);
        let [r, g, b] = average_srgb8_color(&pixels);
        assert!((r - 1.0).abs() < 1e-6 && g.abs() < 1e-6 && b.abs() < 1e-6);
        assert_eq!(average_srgb8_color(&[10, 20, 30, 0]), [1.0; 3]);
    }

    fn rtex_v4(width: u32, height: u32, mip_count: u32, payload_bytes: usize) -> Vec<u8> {
        let mut bytes = b"RTEX".to_vec();
        for value in [4, width, height, mip_count] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.resize(bytes.len() + payload_bytes, 0);
        bytes
    }

    #[test]
    fn v4_mip_count_0_is_rejected() {
        assert!(parse_cooked_tex(&rtex_v4(4, 4, 0, 256)).is_err());
    }

    #[test]
    fn v4_mip_count_u32_max_is_rejected() {
        assert!(parse_cooked_tex(&rtex_v4(4, 4, u32::MAX, 256)).is_err());
    }

    #[test]
    fn v4_4x4_with_3_mips_parses_all_levels() {
        // 4x4 + 2x2 + 1x1 Rgba32Float texels
        let bytes = rtex_v4(4, 4, 3, (16 + 4 + 1) * 16);
        let cooked = parse_cooked_tex(&bytes).unwrap();
        assert_eq!(cooked.mips.len(), 3);
    }

    #[test]
    fn v4_truncated_payload_is_rejected() {
        assert!(parse_cooked_tex(&rtex_v4(4, 4, 3, 16 * 16)).is_err());
    }
}
