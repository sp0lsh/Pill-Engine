use pill_engine::game::{Engine, PillSlotMapKey, PillTypeMapKey, Resource, ResourceStorage};

use anyhow::Result;
use std::path::PathBuf;

pill_engine::game::define_new_pill_slotmap_key! {
    pub struct GaussianCloudHandle;
}

pub enum GaussianCloudSource {
    Path(PathBuf),
    #[cfg(target_arch = "wasm32")]
    Bytes(Vec<u8>),
}

pub struct GaussianCloud {
    pub name:   String,
    pub source: GaussianCloudSource,
}

impl GaussianCloud {
    pub fn from_path(name: &str, path: impl Into<PathBuf>) -> Self {
        Self { name: name.to_string(), source: GaussianCloudSource::Path(path.into()) }
    }

    #[cfg(target_arch = "wasm32")]
    pub fn from_bytes(name: &str, bytes: &[u8]) -> Self {
        Self { name: name.to_string(), source: GaussianCloudSource::Bytes(bytes.to_vec()) }
    }
}

impl PillTypeMapKey for GaussianCloud {
    type Storage = ResourceStorage<GaussianCloud>;
}

impl Resource for GaussianCloud {
    type Handle = GaussianCloudHandle;

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn initialize(&mut self, engine: &mut Engine) -> Result<()> {
        #[cfg(not(target_arch = "wasm32"))]
        let GaussianCloudSource::Path(ref mut path) = self.source;
        if path.is_relative() {
            *path = engine.get_resources_path().join(&*path);
        }
        Ok(())
    }

    fn destroy<H: PillSlotMapKey>(&mut self, _engine: &mut Engine, _self_handle: H) -> Result<()> {
        Ok(())
    }
}
