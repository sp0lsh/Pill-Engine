#![cfg_attr(debug_assertions, allow(dead_code, unused_imports, unused_variables))]

use indexmap::IndexMap;
use pill_core::{debug, LogContext, PillStyle, RendererError};
use crate::{
    config::get_default_texture_handles,
    graphics::RendererMaterialHandle,
    resources::{
        get_renderer_texture_handle_from_material_texture, MaterialParameter, MaterialTexture,
        ShaderParameterSlot, ShaderParameterType, ShaderTextureSlot,
    },
};
use crate::graphics::RendererShaderHandle;

use crate::renderer::resources::RendererResourceStorage;
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
        rendering_resource_storage: &RendererResourceStorage,
        name: &str,
        shader_handle: RendererShaderHandle,
        textures: &IndexMap<String, MaterialTexture>,
        parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<Self> {
        debug!(LogContext::Rendering => "Creating material {}", name.name_style());

        let shader = rendering_resource_storage
            .shaders
            .get(shader_handle)
            .ok_or(Error::new(RendererError::RendererResourceNotFound))?;

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
                rendering_resource_storage,
                shader.textures_bind_group_layout.as_ref().unwrap(),
                &format!("{}_textures", name),
                texture_slots,
                textures,
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

    pub fn update_textures(
        _device: &wgpu::Device,
        _material_renderer_handle: RendererMaterialHandle,
        _rendering_resource_storage: &mut RendererResourceStorage,
        _textures: &IndexMap<String, MaterialTexture>,
    ) -> Result<()> {
        Ok(())
    }

    pub fn update_parameters(
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_renderer_handle: RendererMaterialHandle,
        rendering_resource_storage: &mut RendererResourceStorage,
        parameters: &HashMap<String, MaterialParameter>,
    ) -> Result<()> {
        let material = rendering_resource_storage
            .materials
            .get(material_renderer_handle)
            .ok_or(Error::new(RendererError::RendererResourceNotFound))?;
        let shader_handle = material.shader_handle;
        let shader = rendering_resource_storage
            .shaders
            .get(shader_handle)
            .ok_or(Error::new(RendererError::RendererResourceNotFound))?;

        let parameter_slots = &shader.parameter_slots;

        let material = rendering_resource_storage
            .materials
            .get_mut(material_renderer_handle)
            .ok_or(Error::new(RendererError::RendererResourceNotFound))?;

        if let Some(ref buffer) = material.parameters_uniform_buffer {
            Self::write_parameters_to_buffer(queue, buffer, parameter_slots, parameters)?;
        }

        Ok(())
    }

    fn calculate_uniform_size(parameter_slots: &IndexMap<String, ShaderParameterSlot>) -> usize {
        parameter_slots.len() * 16
    }

    fn write_parameters_to_buffer(
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

    fn create_textures_bind_group(
        device: &wgpu::Device,
        rendering_resource_storage: &RendererResourceStorage,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        name: &str,
        texture_slots: &HashMap<String, ShaderTextureSlot>,
        textures: &IndexMap<String, MaterialTexture>,
    ) -> Result<wgpu::BindGroup> {
        let mut entries = Vec::new();

        for (slot_name, slot) in texture_slots {
            let renderer_texture_handle = match textures.get(slot_name) {
                Some(material_texture) => {
                    get_renderer_texture_handle_from_material_texture(material_texture).unwrap()
                }
                None => get_default_texture_handles(slot.texture_type).1,
            };

            let texture = rendering_resource_storage
                .textures
                .get(renderer_texture_handle)
                .unwrap();

            entries.push(wgpu::BindGroupEntry {
                binding: slot.texture_binding,
                resource: wgpu::BindingResource::TextureView(&texture.texture_view),
            });

            entries.push(wgpu::BindGroupEntry {
                binding: slot.sampler_binding,
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
