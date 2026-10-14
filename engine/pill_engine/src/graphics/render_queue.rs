use crate::{
    renderer::resources::{RendererMaterial, RendererMesh},
    resources::{Material, MaterialHandle, Mesh, MeshHandle, ResourceManager},
};

use pill_core::PillSlotMapKey;

use pill_core::Result;

// --- Render queue key type ---

pub type RenderQueueKey = crate::config::RenderQueueKeyType;

// --- Bit-field descriptors ---

pub struct RenderQueueFieldDesc {
    pub mask: RenderQueueKey,
    pub mask_shift: u32,
    pub max: RenderQueueKey,
}

const fn field(mask: u64, shift: u32, max: u64) -> RenderQueueFieldDesc {
    RenderQueueFieldDesc {
        mask,
        mask_shift: shift,
        max,
    }
}

// Layout is documented in config.rs (RenderQueueKeyType)
pub const RENDER_QUEUE_KEY_MATERIAL_INDEX: RenderQueueFieldDesc =
    field(0xFFF0_0000_0000_0000, 52, 4095);
pub const RENDER_QUEUE_KEY_MATERIAL_VERSION: RenderQueueFieldDesc =
    field(0x000F_F000_0000_0000, 44, 255);
pub const RENDER_QUEUE_KEY_MESH_INDEX: RenderQueueFieldDesc =
    field(0x0000_0FFF_F000_0000, 28, 65535);
pub const RENDER_QUEUE_KEY_MESH_VERSION: RenderQueueFieldDesc =
    field(0x0000_0000_0FF0_0000, 20, 255);

// --- Compose ---

#[cfg_attr(feature = "headless", allow(unused_variables))]
pub fn compose_render_queue_key(
    resource_manager: &ResourceManager,
    material_handle: &MaterialHandle,
    mesh_handle: &MeshHandle,
) -> Result<RenderQueueKey> {
    #[cfg(feature = "headless")]
    return Ok(0);

    #[cfg(not(feature = "headless"))]
    {
        let material = resource_manager.get_resource::<Material>(material_handle)?;
        let mesh = resource_manager.get_resource::<Mesh>(mesh_handle)?;

        let renderer_material_handle =
            resource_manager.get_resource_handle::<RendererMaterial>(&material.name)?;
        let renderer_mesh_handle =
            resource_manager.get_resource_handle::<RendererMesh>(&mesh.name)?;

        let material = renderer_material_handle.data();
        let mesh = renderer_mesh_handle.data();
        Ok(
            pack_field(&RENDER_QUEUE_KEY_MATERIAL_INDEX, material.index.into())?
                | pack_field(
                    &RENDER_QUEUE_KEY_MATERIAL_VERSION,
                    material.version.get().into(),
                )?
                | pack_field(&RENDER_QUEUE_KEY_MESH_INDEX, mesh.index.into())?
                | pack_field(&RENDER_QUEUE_KEY_MESH_VERSION, mesh.version.get().into())?,
        )
    }
}

/// Shifts `value` into its field; values above the field's max would corrupt neighbouring fields.
fn pack_field(field: &RenderQueueFieldDesc, value: RenderQueueKey) -> Result<RenderQueueKey> {
    if value > field.max {
        return Err(format!(
            "Render queue key field overflow: {value} > {} (lower MAX_MATERIALS / MAX_MESHES)",
            field.max
        )
        .into());
    }
    Ok(value << field.mask_shift)
}

// --- Decompose ---

pub struct RenderQueueKeyFields {
    pub material_index: u16,
    pub material_version: u8,
    pub mesh_index: u16,
    pub mesh_version: u8,
}

pub fn decompose_render_queue_key(key: RenderQueueKey) -> RenderQueueKeyFields {
    RenderQueueKeyFields {
        material_index: ((key & RENDER_QUEUE_KEY_MATERIAL_INDEX.mask)
            >> RENDER_QUEUE_KEY_MATERIAL_INDEX.mask_shift) as u16,
        material_version: ((key & RENDER_QUEUE_KEY_MATERIAL_VERSION.mask)
            >> RENDER_QUEUE_KEY_MATERIAL_VERSION.mask_shift) as u8,
        mesh_index: ((key & RENDER_QUEUE_KEY_MESH_INDEX.mask)
            >> RENDER_QUEUE_KEY_MESH_INDEX.mask_shift) as u16,
        mesh_version: ((key & RENDER_QUEUE_KEY_MESH_VERSION.mask)
            >> RENDER_QUEUE_KEY_MESH_VERSION.mask_shift) as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIELDS: [RenderQueueFieldDesc; 4] = [
        RENDER_QUEUE_KEY_MATERIAL_INDEX,
        RENDER_QUEUE_KEY_MATERIAL_VERSION,
        RENDER_QUEUE_KEY_MESH_INDEX,
        RENDER_QUEUE_KEY_MESH_VERSION,
    ];

    #[test]
    fn fields_do_not_overlap_and_masks_match_max() {
        for (index, field) in FIELDS.iter().enumerate() {
            assert_eq!(field.mask >> field.mask_shift, field.max, "field {index}");
            for other in &FIELDS[index + 1..] {
                assert_eq!(field.mask & other.mask, 0);
            }
        }
    }

    #[test]
    fn max_material_and_mesh_slots_round_trip() {
        let key = pack_field(&RENDER_QUEUE_KEY_MATERIAL_INDEX, 4095).unwrap()
            | pack_field(&RENDER_QUEUE_KEY_MATERIAL_VERSION, 255).unwrap()
            | pack_field(&RENDER_QUEUE_KEY_MESH_INDEX, 65535).unwrap()
            | pack_field(&RENDER_QUEUE_KEY_MESH_VERSION, 7).unwrap();
        let fields = decompose_render_queue_key(key);
        assert_eq!(
            (
                fields.material_index,
                fields.material_version,
                fields.mesh_index,
                fields.mesh_version
            ),
            (4095, 255, 65535, 7)
        );
        assert!(pack_field(&RENDER_QUEUE_KEY_MESH_INDEX, 65536).is_err());
    }
}
