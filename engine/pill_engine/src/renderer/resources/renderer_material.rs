#![cfg_attr(debug_assertions, allow(dead_code, unused_imports, unused_variables))]

use indexmap::IndexMap;
use pill_core::{debug, LogContext, PillStyle, PillTypeMapKey, RendererError};
use crate::{
    config::{DEFAULT_COLOR_TEXTURE_NAME, DEFAULT_NORMAL_TEXTURE_NAME},
    graphics::RendererMaterialHandle,
    renderer::resources::{RendererTexture, RendererShader},
    resources::{
        MaterialParameter, MaterialTexture, Resource, ResourceManager, ResourceStorage,
        ShaderParameterSlot, ShaderParameterType, ShaderTextureSlot, Texture,
    },
};
use crate::graphics::RendererShaderHandle;
use crate::graphics::RendererTextureHandle;

use anyhow::{Error, Result};
use std::collections::HashMap;

pub struct RendererMaterial {
    pub name: String,
    pub shader_handle: RendererShaderHandle,
    pub parameters_bind_group: Option<wgpu::BindGroup>,
    pub textures_bind_group: Option<wgpu::BindGroup>,
    pub(crate) parameters_uniform_buffer: Option<wgpu::Buffer>,
}

impl RendererMaterial {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resource_manager: &ResourceManager,
        name: &str,
        shader_handle: RendererShaderHandle,
        resolved_textures: &IndexMap<String, RendererTextureHandle>,
        parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<Self> {
        debug!(LogContext::Rendering => "Creating material {}", name.name_style());

        let shader = resource_manager
            .get_resource::<RendererShader>(&shader_handle)
            .map_err(|_| Error::new(RendererError::RendererResourceNotFound))?;

        let parameter_slots = &shader.parameter_slots;
        let texture_slots = &shader.texture_slots;

        let (parameters_bind_group, parameters_uniform_buffer) = {
            if !parameter_slots.is_empty() {
                let parameters_uniform_buffer_size = Self::calculate_uniform_size(parameter_slots);

                let parameters_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("{}_material_parameters_buffer", name)),
                    size: parameters_uniform_buffer_size as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                Self::write_parameters_to_buffer(
                    queue,
                    &parameters_uniform_buffer,
                    parameter_slots,
                    parameters,
                )?;

                let parameters_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{}_material_parameters_bind_group", name)),
                    layout: shader.parameters_bind_group_layout.as_ref().unwrap(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: parameters_uniform_buffer.as_entire_binding(),
                    }],
                });

                (Some(parameters_bind_group), Some(parameters_uniform_buffer))
            } else {
                (None, None)
            }
        };

        let textures_bind_group = if !texture_slots.is_empty() {
            Some(Self::create_textures_bind_group(
                device,
                resource_manager,
                shader.textures_bind_group_layout.as_ref().unwrap(),
                &format!("{}_textures", name),
                texture_slots,
                resolved_textures,
            )?)
        } else {
            None
        };

        Ok(Self {
            name: name.to_string(),
            shader_handle,
            parameters_bind_group,
            textures_bind_group,
            parameters_uniform_buffer,
        })
    }

    fn calculate_uniform_size(parameter_slots: &IndexMap<String, ShaderParameterSlot>) -> usize {
        parameter_slots.len() * 16
    }

    pub(crate) fn write_parameters_to_buffer(
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        parameter_slots: &IndexMap<String, ShaderParameterSlot>,
        parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<()> {
        let mut data = Vec::new();

        for (slot_name, slot) in parameter_slots {
            match slot.parameter_type {
                ShaderParameterType::Color => {
                    if let Some(MaterialParameter::Color(value)) = parameters.get(slot_name) {
                        data.extend_from_slice(&value.x.to_le_bytes());
                        data.extend_from_slice(&value.y.to_le_bytes());
                        data.extend_from_slice(&value.z.to_le_bytes());
                        data.extend_from_slice(&0.0f32.to_le_bytes());
                    } else {
                        data.extend_from_slice(&[0u8; 16]);
                    }
                }
                ShaderParameterType::Scalar => {
                    if let Some(MaterialParameter::Scalar(value)) = parameters.get(slot_name) {
                        data.extend_from_slice(&value.to_le_bytes());
                        data.extend_from_slice(&[0u8; 12]);
                    } else {
                        data.extend_from_slice(&[0u8; 16]);
                    }
                }
                ShaderParameterType::Bool => {
                    if let Some(MaterialParameter::Bool(value)) = parameters.get(slot_name) {
                        let value: u32 = if *value { 1 } else { 0 };
                        data.extend_from_slice(&value.to_le_bytes());
                        data.extend_from_slice(&[0u8; 12]);
                    } else {
                        data.extend_from_slice(&[0u8; 16]);
                    }
                }
            }
        }

        if !data.is_empty() {
            queue.write_buffer(buffer, 0, &data);
        }

        Ok(())
    }

    pub(crate) fn create_textures_bind_group(
        device: &wgpu::Device,
        resource_manager: &ResourceManager,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        name: &str,
        texture_slots: &HashMap<String, ShaderTextureSlot>,
        resolved_textures: &IndexMap<String, RendererTextureHandle>,
    ) -> Result<wgpu::BindGroup> {
        // Collect handles as owned values first so their borrows outlive the entries vec.
        let mut slot_handles: Vec<(u32, u32, RendererTextureHandle)> = Vec::new();
        for (slot_name, slot) in texture_slots {
            let handle = match resolved_textures.get(slot_name) {
                Some(h) => *h,
                None => {
                    let default_name = match slot.texture_type {
                        crate::resources::TextureType::Color => DEFAULT_COLOR_TEXTURE_NAME,
                        crate::resources::TextureType::Normal => DEFAULT_NORMAL_TEXTURE_NAME,
                    };
                    resource_manager
                        .get_resource_handle::<RendererTexture>(default_name)
                        .map_err(|_| Error::new(RendererError::RendererResourceNotFound))?
                }
            };
            slot_handles.push((slot.texture_binding, slot.sampler_binding, handle));
        }

        let mut entries = Vec::new();
        for (texture_binding, sampler_binding, handle) in &slot_handles {
            let texture = resource_manager
                .get_resource::<RendererTexture>(handle)
                .map_err(|_| Error::new(RendererError::RendererResourceNotFound))?;
            entries.push(wgpu::BindGroupEntry {
                binding: *texture_binding,
                resource: wgpu::BindingResource::TextureView(&texture.texture_view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: *sampler_binding,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            });
        }

        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &entries,
            label: Some(name),
        }))
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
