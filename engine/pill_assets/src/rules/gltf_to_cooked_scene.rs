use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::gltf_scene::{
    gltf_image_kinds, gltf_resource_uris, parse_gltf_scene, CookedImages, GltfAlphaMode,
    GltfMaterialData,
};
use crate::util::write_atomic;
use crate::{Rule, TextureType};

/// Multi-file glTF (with its buffers and images next to it) → `{stem}.cooked/`, loadable as is:
/// `meshes/{i}.cooked_mesh` (RMSH, world transforms baked in), `textures/{i}.rtex` (RTEX: BC7 +
/// mips from KTX2 with feature `ktx2`, else RGBA8 + mips from PNG/JPEG) and `scene.json`:
///
/// ```json
/// { "textures": [{ "name", "kind": "Color|Normal|MetallicRoughness|Emissive", "file" }],
///   "materials": [{ "name", "base_color", "metallic", "roughness", "emissive_color",
///                   "alpha_mode": "opaque" | "blend" | { "mask": cutoff }, "double_sided",
///                   "transmission", "clearcoat", "clearcoat_roughness", "albedo", "normal", "metallic_roughness", "emissive",
///                   "specular" }],
///   "meshes": [{ "name", "file", "material" }] }
/// ```
///
/// Material texture fields index `textures` (or are null); a mesh's `material` indexes
/// `materials`. Each (image, use) is cooked once, so an image used as two kinds is two textures.
pub struct GltfToCookedScene;

impl Rule for GltfToCookedScene {
    fn name(&self) -> &'static str {
        "gltf_to_cooked_scene"
    }

    fn input_glob(&self) -> &'static str {
        "**/*.gltf"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        cooked_dir(input).join("scene.json")
    }

    fn extra_outputs(&self, input: &Path) -> Vec<PathBuf> {
        // The meshes and textures a previous build wrote (their count is in the glTF)
        ["meshes", "textures"]
            .iter()
            .filter_map(|dir| std::fs::read_dir(cooked_dir(input).join(dir)).ok())
            .flatten()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect()
    }

    fn dependencies(&self, input: &Path, _output: &Path) -> Vec<PathBuf> {
        let Ok(bytes) = std::fs::read(input) else {
            return Vec::new();
        };
        let Ok(uris) = gltf_resource_uris(&bytes) else {
            return Vec::new();
        };
        let dir = input.parent().unwrap_or(Path::new("."));
        uris.buffers
            .iter()
            .chain(&uris.images)
            .flatten()
            .map(|uri| dir.join(uri))
            .collect()
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let gltf = std::fs::read(input).with_context(|| format!("read {input:?}"))?;
        let dir = input.parent().unwrap_or(Path::new("."));
        let read = |uri: &Option<String>| -> Result<Vec<u8>> {
            match uri {
                Some(uri) => std::fs::read(dir.join(uri)).with_context(|| format!("read {uri}")),
                None => Ok(Vec::new()),
            }
        };
        let uris = gltf_resource_uris(&gltf)?;
        let kinds = gltf_image_kinds(&gltf)?;
        let buffers = uris.buffers.iter().map(read).collect::<Result<Vec<_>>>()?;

        // KTX2 images are cooked here per use; PNG/JPEG are decoded by `parse_gltf_scene`
        let is_ktx2 =
            |uri: &Option<String>| uri.as_deref().is_some_and(|uri| uri.ends_with(".ktx2"));
        let mut images: Vec<Vec<u8>> = uris
            .images
            .iter()
            .map(|uri| {
                if is_ktx2(uri) {
                    Ok(Vec::new())
                } else {
                    read(uri)
                }
            })
            .collect::<Result<_>>()?;
        let jobs: Vec<(usize, TextureType)> = uris
            .images
            .iter()
            .zip(&kinds)
            .enumerate()
            .filter(|(_, (uri, _))| is_ktx2(uri))
            .flat_map(|(image, (_, kinds))| kinds.iter().map(move |&kind| (image, kind)))
            .collect();
        let mut cooked: CookedImages = cook_ktx2(&jobs, |image| read(&uris.images[image]))?
            .into_iter()
            .collect();

        let stem = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("scene");
        let scene = parse_gltf_scene(stem, &gltf, &buffers, &mut images, &mut cooked)?;

        let out = cooked_dir(input);
        for dir in ["meshes", "textures"] {
            let _ = std::fs::remove_dir_all(out.join(dir));
            std::fs::create_dir_all(out.join(dir))
                .with_context(|| format!("create {:?}", out.join(dir)))?;
        }
        let mut textures = Vec::with_capacity(scene.textures.len());
        for (index, texture) in scene.textures.iter().enumerate() {
            let file = format!("textures/{index}.rtex");
            write_atomic(&out.join(&file), &texture.rtex)?;
            textures.push(serde_json::json!({
                "name": texture.name,
                "kind": format!("{:?}", texture.kind),
                "file": file,
            }));
        }
        let mut meshes = Vec::with_capacity(scene.meshes.len());
        // Triangles of the meshes and of their trace LODs
        let mut triangles = 0usize;
        #[cfg_attr(not(feature = "mesh_lod"), allow(unused_mut))]
        let mut traced = 0usize;
        for (index, mesh) in scene.meshes.iter().enumerate() {
            let file = format!("meshes/{index}.cooked_mesh");
            triangles += rmsh_index_count(&mesh.rmsh) / 3;
            #[cfg(feature = "mesh_lod")]
            let rmsh = match crate::mesh_lod::trace_lod(&mesh.rmsh, TRACE_CELL) {
                Some((positions, indices)) => {
                    traced += indices.len() / 3;
                    [
                        mesh.rmsh.clone(),
                        crate::convert::trace_lod_bytes(&positions, &indices),
                    ]
                    .concat()
                }
                None => mesh.rmsh.clone(),
            };
            #[cfg(not(feature = "mesh_lod"))]
            let rmsh = mesh.rmsh.clone();
            write_atomic(&out.join(&file), &rmsh)?;
            meshes.push(serde_json::json!({
                "name": mesh.name,
                "file": file,
                "material": mesh.material,
            }));
        }
        if traced > 0 {
            println!("{stem}: trace LODs {traced} of {triangles} triangles");
        }
        let json = serde_json::json!({
            "textures": textures,
            "materials": scene.materials.iter().map(material_json).collect::<Vec<_>>(),
            "meshes": meshes,
        });
        write_atomic(output, serde_json::to_string_pretty(&json)?.as_bytes())
    }
}

/// Error bound (m) of the meshes' trace LODs (`mesh_lod`): an eighth of GI's 2 m fine probe
/// spacing
#[cfg(feature = "mesh_lod")]
const TRACE_CELL: f32 = 0.25;

fn rmsh_index_count(rmsh: &[u8]) -> usize {
    rmsh.get(12..16).map_or(0, |count| {
        u32::from_le_bytes([count[0], count[1], count[2], count[3]]) as usize
    })
}

fn cooked_dir(input: &Path) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("scene");
    input.with_file_name(format!("{stem}.cooked"))
}

fn material_json(material: &GltfMaterialData) -> serde_json::Value {
    serde_json::json!({
        "name": material.name,
        "base_color": material.base_color,
        "metallic": material.metallic,
        "roughness": material.roughness,
        "emissive_color": material.emissive_color,
        "alpha_mode": match material.alpha_mode {
            GltfAlphaMode::Opaque => serde_json::json!("opaque"),
            GltfAlphaMode::Blend => serde_json::json!("blend"),
            GltfAlphaMode::Mask(cutoff) => serde_json::json!({ "mask": cutoff }),
        },
        "double_sided": material.double_sided,
        "transmission": material.transmission,
        "clearcoat": material.clearcoat,
        "clearcoat_roughness": material.clearcoat_roughness,
        "albedo": material.albedo,
        "normal": material.normal,
        "metallic_roughness": material.metallic_roughness,
        "emissive": material.emissive,
        "specular": material.specular,
    })
}

/// RTEX of each (KTX2 image, kind), on all cores.
#[cfg(feature = "ktx2")]
fn cook_ktx2(
    jobs: &[(usize, TextureType)],
    read: impl Fn(usize) -> Result<Vec<u8>> + Sync,
) -> Result<Vec<((usize, TextureType), Vec<u8>)>> {
    use crate::ktx2::{ktx2_to_rtex, TextureKind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let cook = |&(image, kind): &(usize, TextureType)| -> Result<((usize, TextureType), Vec<u8>)> {
        let kind_of_cook = match kind {
            TextureType::Color | TextureType::Emissive => TextureKind::Color,
            TextureType::Normal => TextureKind::Normal,
            TextureType::MetallicRoughness => TextureKind::Linear,
        };
        let rtex =
            ktx2_to_rtex(&read(image)?, kind_of_cook).with_context(|| format!("image {image}"))?;
        Ok(((image, kind), rtex))
    };
    let next = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |count| count.get());
    let mut results: Vec<(usize, Result<((usize, TextureType), Vec<u8>)>)> =
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads.min(jobs.len()).max(1))
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(job) = jobs.get(index) else {
                                return done;
                            };
                            done.push((index, cook(job)));
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap_or_default())
                .collect()
        });
    if results.len() != jobs.len() {
        bail!("a KTX2 cook thread panicked");
    }
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

#[cfg(not(feature = "ktx2"))]
fn cook_ktx2(
    jobs: &[(usize, TextureType)],
    _read: impl Fn(usize) -> Result<Vec<u8>> + Sync,
) -> Result<Vec<((usize, TextureType), Vec<u8>)>> {
    if !jobs.is_empty() {
        bail!("KTX2 images need pill_assets' `ktx2` feature");
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-triangle glTF with an embedded (data URI) buffer and no images.
    fn triangle_gltf() -> String {
        // 3 positions (36 bytes) + 3 u16 indices (6 bytes, padded to 8)
        let mut bytes: Vec<u8> = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend(value.to_le_bytes());
        }
        for index in [0u16, 1, 2, 0] {
            bytes.extend(index.to_le_bytes());
        }
        let base64 = base64_encode(&bytes);
        format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1,"material":0}}]}}],
"materials":[{{"name":"red","pbrMetallicRoughness":{{"baseColorFactor":[1,0,0,1]}}}}],
"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
{{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}}],
"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":6}}],
"buffers":[{{"byteLength":44,"uri":"data:application/octet-stream;base64,{base64}"}}]}}"#
        )
    }

    fn base64_encode(bytes: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk
                .iter()
                .enumerate()
                .fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
            for i in 0..4 {
                out.push(if i <= chunk.len() {
                    ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char
                } else {
                    '='
                });
            }
        }
        out
    }

    #[test]
    fn build_triangle_writes_scene_json_and_mesh() {
        let dir =
            std::env::temp_dir().join(format!("pill_gltf_to_cooked_scene_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("triangle.gltf");
        std::fs::write(&input, triangle_gltf()).unwrap();
        let rule = GltfToCookedScene;
        let output = rule.output_for(&input);
        rule.build(&input, &output).unwrap();

        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
        assert_eq!(json["meshes"].as_array().unwrap().len(), 1);
        assert_eq!(json["materials"][0]["base_color"][0], 1.0);
        assert_eq!(json["materials"][0]["alpha_mode"], "opaque");
        let mesh = dir
            .join("triangle.cooked")
            .join(json["meshes"][0]["file"].as_str().unwrap());
        assert_eq!(&std::fs::read(mesh).unwrap()[0..4], b"RMSH");
        assert_eq!(rule.extra_outputs(&input).len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
