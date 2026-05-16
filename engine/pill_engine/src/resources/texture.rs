use crate::{
    engine::Engine,
    renderer::resources::{RendererMaterial, RendererTexture},
    resources::{Material, Resource, ResourceLoader, ResourceStorage},
};

use indexmap::IndexMap;
use pill_core::{get_type_name, PillSlotMapKey, PillStyle, PillTypeMapKey};

use anyhow::{Context, Result};
pill_core::define_new_pill_slotmap_key! {
    pub struct TextureHandle;
}

#[derive(Clone, Copy, Debug)]
pub enum TextureType {
    Color,
    Normal,
}

#[readonly::make]
pub struct Texture {
    #[readonly]
    pub name: String,
    #[readonly]
    pub resource_loader: ResourceLoader,
    #[readonly]
    pub texture_type: TextureType,
}

impl Texture {
    pub fn new(name: &str, texture_type: TextureType, resource_loader: ResourceLoader) -> Self {
        Self {
            name: name.to_string(),
            resource_loader,
            texture_type,
        }
    }

    pub fn from_bytes(name: &str, texture_type: TextureType, bytes: &[u8]) -> Self {
        Self::new(
            name,
            texture_type,
            ResourceLoader::Bytes(bytes.to_vec().into_boxed_slice()),
        )
    }
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

        let image_data = match &self.resource_loader {
            ResourceLoader::Path(path) => {
                let resource_file_path = engine.game_resources_directory_path.join(path);
                pill_core::validate_asset_path(&resource_file_path, &["png", "jpg", "gif", "tif"])?;
                image::open(&resource_file_path)?
            }
            ResourceLoader::Bytes(bytes) => image::load_from_memory(bytes)?,
        };

        #[cfg(not(feature = "headless"))]
        {
            let renderer_texture = RendererTexture::new_texture(
                engine.renderer.get_device(),
                engine.renderer.get_queue(),
                &self.name,
                &image_data,
                self.texture_type,
            )
            .context(error_message)?;
            engine.resource_manager.add_resource(renderer_texture)?;
        }

        Ok(())
    }

    fn destroy<H: PillSlotMapKey>(&mut self, engine: &mut Engine, self_handle: H) -> Result<()> {
        #[cfg(not(feature = "headless"))]
        engine
            .resource_manager
            .remove_resource_by_name::<RendererTexture>(&self.name)?;

        // Phase 1: find and update affected materials (CPU side only)
        let affected_material_names: Vec<String> = {
            let resource_storage = engine
                .resource_manager
                .get_resource_storage_mut::<Material>()?;
            let mut affected = Vec::new();
            for (_, slot) in resource_storage.data.iter_mut() {
                let material = slot.as_mut().expect("Critical: Resource is None");
                let before = material.textures.len();
                material
                    .textures
                    .retain(|_, t| t.texture_handle.data() != self_handle.data());
                if material.textures.len() < before {
                    affected.push(material.name.clone());
                }
            }
            affected
        };

        // Phase 2: rebuild RendererMaterial textures bind group for affected materials
        #[cfg(not(feature = "headless"))]
        for mat_name in &affected_material_names {
            let new_bind_group = {
                let renderer_mat = engine
                    .resource_manager
                    .get_resource_by_name::<RendererMaterial>(mat_name)?;
                let renderer_shader = engine
                    .resource_manager
                    .get_resource::<crate::renderer::resources::RendererShader>(
                        &renderer_mat.shader_handle,
                    )?;
                let material = engine
                    .resource_manager
                    .get_resource_by_name::<Material>(mat_name)?;
                let mut resolved = IndexMap::new();
                for (slot_name, mat_tex) in &material.textures {
                    let tex = engine
                        .resource_manager
                        .get_resource::<crate::resources::Texture>(&mat_tex.texture_handle)?;
                    let h = engine
                        .resource_manager
                        .get_resource_handle::<RendererTexture>(&tex.name)?;
                    resolved.insert(slot_name.clone(), h);
                }
                RendererMaterial::create_textures_bind_group(
                    engine.renderer.get_device(),
                    &engine.resource_manager,
                    renderer_shader.textures_bind_group_layout.as_ref().unwrap(),
                    &format!("{}_textures", mat_name),
                    &renderer_shader.texture_slots,
                    &resolved,
                )?
            };
            engine
                .resource_manager
                .get_resource_by_name_mut::<RendererMaterial>(mat_name)?
                .textures_bind_group = Some(new_bind_group);
        }

        Ok(())
    }
}
