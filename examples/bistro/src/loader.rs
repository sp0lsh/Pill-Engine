//! Loads a Bistro scene: fetches the glTF, its buffer, the KTX2 textures it uses and the HDR sky,
//! cooks each texture to BC7 + mips (cached in `build/cooked/`, so later runs skip both the
//! download and the cook), parses the scene and bakes image-based lighting.

use pill_assets::ktx2::{ktx2_to_rtex, TextureKind, COOK_VERSION};
use pill_engine::game::{
    gltf_image_kinds, gltf_resource_uris, ibl, parse_gltf_scene, GltfSceneData, TextureType,
};

use crate::fetch;
use crate::ui;

/// HDRI Haven sky shipped with the conversion (CC0).
const HDR_PATH: &str = "san_giuseppe_bridge_4k.hdr";
/// Texture cooks in parallel (CPU bound: transcode + BC7 encode of the mips).
const PARALLEL_COOKS: usize = 12;

const GLTF_PATH: &str = "BistroExterior.gltf";
/// Prefix of the engine resource and scene names.
pub const NAME: &str = "bistro";

pub struct BistroCpu {
    pub scene: GltfSceneData,
    pub equirect: ibl::Equirect,
    pub ibl: (Vec<f32>, Vec<Vec<f32>>), // ibl::bake_all output: diffuse, specular mips
}

pub fn load() -> Result<BistroCpu, String> {
    let start = std::time::Instant::now();
    let gltf_bytes = fetch::get(fetch::BISTRO_RAW, GLTF_PATH)?;
    let uris = gltf_resource_uris(&gltf_bytes).map_err(|e| e.to_string())?;
    let kinds = gltf_image_kinds(&gltf_bytes).map_err(|e| e.to_string())?;

    // Buffers are Git LFS objects; the sky is a plain file
    let buffer_paths: Vec<String> = uris.buffers.iter().flatten().cloned().collect();
    let (fetched, hdr) = std::thread::scope(|scope| {
        let hdr = scope.spawn(|| fetch::get(fetch::BISTRO_RAW, HDR_PATH));
        let fetched = fetch::get_all(fetch::BISTRO_LFS, &buffer_paths);
        (
            fetched,
            hdr.join()
                .unwrap_or_else(|_| Err("HDR fetch panicked".into())),
        )
    });
    let (fetched, hdr) = (fetched?, hdr?);
    let mut fetched = fetched.into_iter();
    let buffers: Vec<Vec<u8>> = uris
        .buffers
        .iter()
        .map(|uri| match uri {
            Some(_) => fetched.next().unwrap_or_default(),
            None => Vec::new(),
        })
        .collect();
    ui::set_status(format!(
        "Bistro: glTF + buffers + sky in {:.1} s",
        start.elapsed().as_secs_f32()
    ));

    // Textures the materials use, cooked (or read back from the cook cache)
    let cook_start = std::time::Instant::now();
    let jobs: Vec<(usize, &String, TextureType)> = uris
        .images
        .iter()
        .zip(&kinds)
        .enumerate()
        .filter_map(|(index, (uri, kind))| Some((index, uri.as_ref()?, (*kind)?)))
        .collect();
    let cooked = fetch::map_parallel(&jobs, PARALLEL_COOKS, |(_, uri, kind)| {
        cook_cached(uri, *kind)
    })?;
    let mut images = vec![Vec::new(); uris.images.len()];
    for ((index, _, _), rtex) in jobs.iter().zip(cooked) {
        images[*index] = rtex;
    }
    ui::set_status(format!(
        "Bistro: {} textures (BC7) in {:.1} s",
        jobs.len(),
        cook_start.elapsed().as_secs_f32()
    ));

    let parse_start = std::time::Instant::now();
    let scene =
        parse_gltf_scene(NAME, &gltf_bytes, &buffers, &mut images).map_err(|e| e.to_string())?;
    drop(buffers);
    let equirect = ibl::equirect_from_hdr(&hdr).map_err(|e| e.to_string())?;
    let ibl = ibl::bake_all(&equirect);
    ui::set_status(format!(
        "Bistro: {} meshes, {} materials, {} textures; parse + IBL in {:.1} s, total {:.1} s",
        scene.meshes.len(),
        scene.materials.len(),
        scene.textures.len(),
        parse_start.elapsed().as_secs_f32(),
        start.elapsed().as_secs_f32()
    ));
    Ok(BistroCpu {
        scene,
        equirect,
        ibl,
    })
}

/// RTEX v5 of a KTX2 image: from `build/cooked/`, else downloaded, cooked and stored there.
fn cook_cached(uri: &str, kind: TextureType) -> Result<Vec<u8>, String> {
    let kind = match kind {
        TextureType::Color | TextureType::Emissive => TextureKind::Color,
        TextureType::Normal => TextureKind::Normal,
        TextureType::MetallicRoughness => TextureKind::Linear,
    };
    let path = fetch::build_dir("cooked").join(format!("{uri}.{kind:?}.v{COOK_VERSION}.rtex"));
    if let Ok(bytes) = std::fs::read(&path) {
        return Ok(bytes);
    }
    let ktx2 = fetch::get(fetch::BISTRO_RAW, uri)?;
    let rtex = ktx2_to_rtex(&ktx2, kind).map_err(|e| format!("{uri}: {e}"))?;
    fetch::store(&path, &rtex);
    Ok(rtex)
}
