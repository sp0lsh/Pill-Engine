use crate::{
    config::{
        DEFAULT_COLOR_TEXTURE_NAME, DEFAULT_EMISSIVE_TEXTURE_NAME,
        DEFAULT_METALLIC_ROUGHNESS_TEXTURE_NAME, DEFAULT_NORMAL_TEXTURE_NAME,
    },
    graphics::{
        shader_bindings::pbr_opaque_fragment, RendererMaterialHandle, RendererTextureHandle,
    },
    renderer::resources::RendererTexture,
    resources::{Resource, ResourceManager, ResourceStorage},
};
use pill_core::{debug, LogContext, PillError, PillStyle, PillTypeMapKey, RendererError, Result};

/// Material parameters as set by `Material`; stored on the GPU as `PackedMaterial` in the
/// renderer's material array.
#[derive(Copy, Clone)]
pub(crate) struct MaterialParams {
    pub(crate) base_color: [f32; 3],
    pub(crate) roughness: f32,
    pub(crate) emissive: [f32; 3],
    pub(crate) metallic: f32,
    pub(crate) clearcoat: f32,
    pub(crate) clearcoat_roughness: f32,
    pub(crate) opacity: f32,
    /// Alpha test: texels with base-color alpha below it are discarded (0 = off).
    pub(crate) alpha_cutoff: f32,
    /// -1 flips the normal map's green channel (DirectX convention), else 1.
    pub(crate) normal_y_sign: f32,
    /// 1 = no back-face culling; back faces are lit with the flipped normal.
    pub(crate) double_sided: f32,
}

impl MaterialParams {
    pub(crate) fn alpha_tested(&self) -> bool {
        self.alpha_cutoff > 0.0
    }

    pub(crate) fn is_double_sided(&self) -> bool {
        self.double_sided > 0.5
    }

    /// The 14 parameters as half floats, two per u32 (low half first), in the order
    /// `load_material` in include/material.hlsl unpacks them.
    pub(crate) fn packed(&self) -> PackedMaterial {
        let pair = |low: f32, high: f32| {
            half::f16::from_f32(low).to_bits() as u32
                | ((half::f16::from_f32(high).to_bits() as u32) << 16)
        };
        let [r, g, b] = self.base_color;
        let [emissive_r, emissive_g, emissive_b] = self.emissive;
        PackedMaterial([
            pair(r, g),
            pair(b, self.roughness),
            pair(emissive_r, emissive_g),
            pair(emissive_b, self.metallic),
            pair(self.clearcoat, self.clearcoat_roughness),
            pair(self.opacity, self.alpha_cutoff),
            pair(self.normal_y_sign, self.double_sided),
            0,
        ])
    }
}

/// One material in the GPU material array: 16 half floats in two uint4 rows, so a shader loads a
/// whole material with two uniform reads.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackedMaterial(pub(crate) [u32; 8]);

/// Byte offset of material slot `index` in the material array buffer.
pub(crate) fn material_array_offset(index: u32) -> wgpu::BufferAddress {
    index as wgpu::BufferAddress * std::mem::size_of::<PackedMaterial>() as wgpu::BufferAddress
}

/// Renderer textures of a material; `None` falls back to the engine default texture of that slot.
#[derive(Clone, Copy, Default)]
pub(crate) struct MaterialTextures {
    pub(crate) base_color: Option<RendererTextureHandle>,
    pub(crate) normal: Option<RendererTextureHandle>,
    pub(crate) metallic_roughness: Option<RendererTextureHandle>,
    pub(crate) emissive: Option<RendererTextureHandle>,
}

/// Bind group layout of the material textures: (texture, sampler) pairs for base_color, normal,
/// metallic_roughness, emissive. Shared by RendererMaterial and the shadow casters.
pub(crate) fn material_textures_layout_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let group = pbr_opaque_fragment::BASE_COLOR_TEXTURE.group as usize;
    pbr_opaque_fragment::layout().swap_remove(group)
}

pub struct RendererMaterial {
    pub name: String,
    /// Opacity < 1: drawn by the translucent PBR pass (blended, back to front) instead of the opaque one.
    pub translucent: bool,
    /// Alpha cutoff set: drawn with discard; shadow casters use the cutout pipeline.
    pub alpha_tested: bool,
    /// Drawn without back-face culling.
    pub double_sided: bool,
    pub textures_bind_group: wgpu::BindGroup,
}

impl RendererMaterial {
    /// Creates the texture bind group of a material; its parameters go to the material array with
    /// `write_parameters` once the material has a slot.
    pub(crate) fn new(
        device: &wgpu::Device,
        resource_manager: &ResourceManager,
        name: &str,
        parameters: &MaterialParams,
        textures: &MaterialTextures,
    ) -> Result<Self> {
        debug!(LogContext::Rendering => "Creating material {}", name.name_style());

        let textures_bind_group =
            Self::create_textures_bind_group(device, resource_manager, textures)?;

        Ok(Self {
            name: name.to_string(),
            translucent: parameters.opacity < 1.0,
            alpha_tested: parameters.alpha_tested(),
            double_sided: parameters.is_double_sided(),
            textures_bind_group,
        })
    }

    /// Writes the parameters of material slot `index` into the material array.
    pub(crate) fn write_parameters(
        &mut self,
        queue: &wgpu::Queue,
        material_buffer: &wgpu::Buffer,
        index: u32,
        parameters: &MaterialParams,
    ) {
        self.translucent = parameters.opacity < 1.0;
        self.alpha_tested = parameters.alpha_tested();
        self.double_sided = parameters.is_double_sided();
        queue.write_buffer(
            material_buffer,
            material_array_offset(index),
            bytemuck::bytes_of(&parameters.packed()),
        );
    }

    /// Builds the texture bind group, substituting the engine default texture for empty slots.
    pub(crate) fn create_textures_bind_group(
        device: &wgpu::Device,
        resource_manager: &ResourceManager,
        textures: &MaterialTextures,
    ) -> Result<wgpu::BindGroup> {
        let slots = [
            (textures.base_color, DEFAULT_COLOR_TEXTURE_NAME),
            (textures.normal, DEFAULT_NORMAL_TEXTURE_NAME),
            (
                textures.metallic_roughness,
                DEFAULT_METALLIC_ROUGHNESS_TEXTURE_NAME,
            ),
            (textures.emissive, DEFAULT_EMISSIVE_TEXTURE_NAME),
        ];

        // Resolve handles first so their borrows outlive the texture references below
        let mut handles = Vec::with_capacity(slots.len());
        for (handle, default_name) in slots {
            handles.push(match handle {
                Some(handle) => handle,
                None => resource_manager
                    .get_resource_handle::<RendererTexture>(default_name)
                    .map_err(|_| PillError::from(RendererError::RendererResourceNotFound))?,
            });
        }

        let mut renderer_textures = Vec::with_capacity(handles.len());
        for handle in &handles {
            renderer_textures.push(
                resource_manager
                    .get_resource::<RendererTexture>(handle)
                    .map_err(|_| PillError::from(RendererError::RendererResourceNotFound))?,
            );
        }

        let [base_color, normal, metallic_roughness, emissive] = renderer_textures[..] else {
            unreachable!("one texture per slot");
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material_textures_bind_group_layout"),
            entries: &material_textures_layout_entries(),
        });
        Ok(pbr_opaque_fragment::Group1 {
            base_color_texture: &base_color.texture_view,
            base_color_sampler: &base_color.sampler,
            normal_texture: &normal.texture_view,
            normal_sampler: &normal.sampler,
            metallic_roughness_texture: &metallic_roughness.texture_view,
            metallic_roughness_sampler: &metallic_roughness.sampler,
            emissive_texture: &emissive.texture_view,
            emissive_sampler: &emissive.sampler,
        }
        .create(device, &layout))
    }
}

impl PillTypeMapKey for RendererMaterial {
    type Storage = ResourceStorage<RendererMaterial>;
}

impl Resource for RendererMaterial {
    type Handle = RendererMaterialHandle;

    fn get_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(packed: &PackedMaterial) -> Vec<f32> {
        packed
            .0
            .iter()
            .flat_map(|&word| [word & 0xffff, word >> 16])
            .map(|bits| half::f16::from_bits(bits as u16).to_f32())
            .collect()
    }

    #[test]
    fn material_pack_all_fields_round_trip_within_f16_precision() {
        let parameters = MaterialParams {
            base_color: [0.8, 0.5, 0.25],
            roughness: 0.35,
            emissive: [12.0, 0.0, 3.5],
            metallic: 1.0,
            clearcoat: 0.6,
            clearcoat_roughness: 0.1,
            opacity: 0.75,
            alpha_cutoff: 0.5,
            normal_y_sign: -1.0,
            double_sided: 1.0,
        };
        let expected = [
            0.8, 0.5, 0.25, 0.35, 12.0, 0.0, 3.5, 1.0, 0.6, 0.1, 0.75, 0.5, -1.0, 1.0, 0.0, 0.0,
        ];
        for (unpacked, expected) in unpack(&parameters.packed()).iter().zip(expected) {
            assert!((unpacked - expected).abs() <= expected.abs() / 1024.0);
        }
    }

    #[test]
    fn material_index_1023_offset_32736() {
        assert_eq!(material_array_offset(1023), 32736);
    }
}
