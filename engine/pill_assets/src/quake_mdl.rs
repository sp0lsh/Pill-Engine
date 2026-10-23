//! Quake `.mdl` (IDPO version 6) models → uploadable meshes, one per animation frame (feature
//! `quake_map`). Format per the public MDL specification
//! (http://tfc.duke.free.fr/coding/mdl-specs-en.html): header, palette-indexed skins, texture
//! coordinates (with the seam rule for back-facing triangles), triangles and frames of compressed
//! vertices (`scale · v + translate`). Normals are rebuilt from the triangles, so the 162-entry
//! normal table isn't needed. Positions are converted like the levels: Quake space (Z up, units)
//! → engine space (Y up, meters).

use std::collections::HashMap;

use anyhow::{bail, ensure, Result};

use crate::convert::{self, PrimitiveAttributes};
use crate::gltf_scene::{decode_image, GltfAlphaMode, GltfMaterialData, GltfTextureData};
use crate::quake_map::quake_to_engine;
use crate::TextureType;

const HEADER_SIZE: usize = 84;

/// One animation frame: its name in the model and the mesh of that pose.
pub struct QuakeModelFrame {
    pub name: String,
    pub rmsh: Vec<u8>,
}

/// A model ready to upload: its skin, one material and a mesh per frame (all frames share the
/// vertex layout and texture coordinates, so swapping meshes animates it).
pub struct QuakeModel {
    pub texture: GltfTextureData,
    pub material: GltfMaterialData,
    pub frames: Vec<QuakeModelFrame>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.position + count;
        ensure!(
            end <= self.bytes.len(),
            "MDL truncated at byte {}",
            self.position
        );
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn int(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn float(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn vector(&mut self) -> Result<[f32; 3]> {
        Ok([self.float()?, self.float()?, self.float()?])
    }

    fn count(&mut self, what: &str, limit: i32) -> Result<usize> {
        let value = self.int()?;
        ensure!(
            (0..=limit).contains(&value),
            "MDL {what} {value} out of range"
        );
        Ok(value as usize)
    }
}

struct Frame {
    name: String,
    /// Compressed vertices: x, y, z (normal index dropped).
    vertices: Vec<[u8; 3]>,
}

fn read_simple_frame(reader: &mut Reader, vertex_count: usize) -> Result<Frame> {
    reader.take(8)?; // bounding box min/max (compressed)
    let name_bytes = reader.take(16)?;
    let name_end = name_bytes.iter().position(|&byte| byte == 0).unwrap_or(16);
    let name = String::from_utf8_lossy(&name_bytes[..name_end]).into_owned();
    let raw = reader.take(vertex_count * 4)?;
    let vertices = raw
        .chunks_exact(4)
        .map(|vertex| [vertex[0], vertex[1], vertex[2]])
        .collect();
    Ok(Frame { name, vertices })
}

/// Parses a model and builds its frames. `skin_png` replaces the palette-indexed skin (e.g. the
/// source PNG of a model); without it `palette` (`gfx/palette.lmp`, 768 bytes) decodes skin 0.
pub fn build_quake_model(
    name_prefix: &str,
    bytes: &[u8],
    skin_png: Option<&[u8]>,
    palette: Option<&[u8]>,
) -> Result<QuakeModel> {
    ensure!(bytes.len() >= HEADER_SIZE, "MDL shorter than its header");
    let mut reader = Reader { bytes, position: 0 };
    ensure!(
        reader.take(4)? == b"IDPO",
        "not a Quake model (no IDPO ident)"
    );
    let version = reader.int()?;
    ensure!(version == 6, "MDL version {version}, expected 6");
    let scale = reader.vector()?;
    let translate = reader.vector()?;
    reader.float()?; // bounding radius
    reader.vector()?; // eye position
    let skin_count = reader.count("skin count", 256)?;
    let skin_width = reader.count("skin width", 4096)?;
    let skin_height = reader.count("skin height", 4096)?;
    let vertex_count = reader.count("vertex count", 65_536)?;
    let triangle_count = reader.count("triangle count", 65_536)?;
    let frame_count = reader.count("frame count", 4096)?;
    reader.take(12)?; // sync type, flags, size

    // --- Skins: keep the first image ---
    let skin_texels = skin_width * skin_height;
    let mut first_skin: Option<&[u8]> = None;
    for _ in 0..skin_count {
        if reader.int()? == 0 {
            let texels = reader.take(skin_texels)?;
            first_skin.get_or_insert(texels);
        } else {
            let images = reader.count("skin group size", 256)?;
            reader.take(images * 4)?; // times
            for _ in 0..images {
                let texels = reader.take(skin_texels)?;
                first_skin.get_or_insert(texels);
            }
        }
    }

    // --- Texture coordinates and triangles ---
    let mut st = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let on_seam = reader.int()? != 0;
        st.push((on_seam, reader.int()?, reader.int()?));
    }
    let mut triangles = Vec::with_capacity(triangle_count);
    for _ in 0..triangle_count {
        let faces_front = reader.int()? != 0;
        let corners = [reader.int()?, reader.int()?, reader.int()?];
        ensure!(
            corners
                .iter()
                .all(|&corner| (0..vertex_count as i32).contains(&corner)),
            "MDL triangle references a missing vertex"
        );
        triangles.push((faces_front, corners.map(|corner| corner as usize)));
    }

    // --- Frames (a frame group contributes its first pose) ---
    let mut frames = Vec::with_capacity(frame_count);
    for _ in 0..frame_count {
        if reader.int()? == 0 {
            frames.push(read_simple_frame(&mut reader, vertex_count)?);
        } else {
            let poses = reader.count("frame group size", 1024)?;
            ensure!(poses > 0, "empty MDL frame group");
            reader.take(8)?; // group bounding box
            reader.take(poses * 4)?; // times
            let first = read_simple_frame(&mut reader, vertex_count)?;
            for _ in 1..poses {
                read_simple_frame(&mut reader, vertex_count)?;
            }
            frames.push(first);
        }
    }
    if frames.is_empty() {
        bail!("MDL has no frames");
    }

    // --- Output vertices: one per (vertex, seam side) actually used ---
    let mut output_index: HashMap<(usize, bool), u32> = HashMap::new();
    let mut sources: Vec<usize> = Vec::new();
    let mut texture_coordinates: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::with_capacity(triangle_count * 3);
    for &(faces_front, corners) in &triangles {
        for corner in corners {
            let (on_seam, s, t) = st[corner];
            let back_seam = on_seam && !faces_front;
            let index = *output_index.entry((corner, back_seam)).or_insert_with(|| {
                let s = s as f32
                    + if back_seam {
                        skin_width as f32 * 0.5
                    } else {
                        0.0
                    };
                texture_coordinates.push([
                    (s + 0.5) / skin_width as f32,
                    (t as f32 + 0.5) / skin_height as f32,
                ]);
                sources.push(corner);
                (sources.len() - 1) as u32
            });
            indices.push(index);
        }
    }

    let decode = |frame: &Frame| -> Vec<[f32; 3]> {
        frame
            .vertices
            .iter()
            .map(|vertex| {
                let quake = std::array::from_fn(|axis| {
                    scale[axis] * f32::from(vertex[axis]) + translate[axis]
                });
                quake_to_engine(quake)
            })
            .collect()
    };
    // Counter-clockwise from outside (the engine's front faces): flip the triangles if the first
    // pose's signed volume says they wind the other way
    let first_pose = decode(&frames[0]);
    let signed_volume: f32 = triangles
        .iter()
        .map(|(_, [a, b, c])| {
            let (a, b, c) = (first_pose[*a], first_pose[*b], first_pose[*c]);
            a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
        })
        .sum();
    if signed_volume < 0.0 {
        for triangle in indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }

    let mut frame_meshes = Vec::with_capacity(frames.len());
    for frame in &frames {
        let pose = decode(frame);
        let positions: Vec<[f32; 3]> = sources.iter().map(|&source| pose[source]).collect();
        let (vertices, frame_indices) = convert::build_primitive(
            PrimitiveAttributes {
                positions,
                normals: None, // smooth normals from the faces
                texture_coordinates: Some(texture_coordinates.clone()),
                tangents: Vec::new(),
                indices: Some(indices.clone()),
            },
            convert::IDENTITY,
        )?;
        frame_meshes.push(QuakeModelFrame {
            name: frame.name.clone(),
            rmsh: convert::rmsh_bytes(&vertices, &frame_indices),
        });
    }

    // --- Skin ---
    let (width, height, rgba) = match skin_png {
        Some(png) => {
            let image = decode_image(png)?;
            (image.w, image.h, image.rgba)
        }
        None => {
            let palette = palette.filter(|palette| palette.len() >= 768);
            let (Some(texels), Some(palette)) = (first_skin, palette) else {
                bail!("MDL skin needs a PNG or the Quake palette");
            };
            let rgba = texels
                .iter()
                .flat_map(|&index| {
                    let base = index as usize * 3;
                    [palette[base], palette[base + 1], palette[base + 2], 255]
                })
                .collect();
            (skin_width as u32, skin_height as u32, rgba)
        }
    };
    let texture = GltfTextureData {
        name: format!("{name_prefix}_skin"),
        kind: TextureType::Color,
        rtex: convert::rtex_from_rgba(width, height, rgba),
    };
    let material = GltfMaterialData {
        name: format!("{name_prefix}_material"),
        albedo: Some(0),
        normal: None,
        metallic_roughness: None,
        emissive: None,
        specular: None,
        base_color: [1.0; 4],
        metallic: 0.3,
        roughness: 0.5,
        emissive_color: [0.0; 3],
        alpha_mode: GltfAlphaMode::Opaque,
        double_sided: true,
        transmission: 0.0,
        clearcoat: 0.0,
        clearcoat_roughness: 0.0,
    };
    Ok(QuakeModel {
        texture,
        material,
        frames: frame_meshes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2×2-skin model with 4 vertices, 4 triangles (a tetrahedron) and 2 frames.
    fn tetrahedron(flip: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        let int = |bytes: &mut Vec<u8>, value: i32| bytes.extend_from_slice(&value.to_le_bytes());
        let float = |bytes: &mut Vec<u8>, value: f32| bytes.extend_from_slice(&value.to_le_bytes());
        bytes.extend_from_slice(b"IDPO");
        int(&mut bytes, 6);
        for value in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0] {
            float(&mut bytes, value); // scale, translate, radius, eye
        }
        for value in [1, 2, 2, 4, 4, 2, 0, 0] {
            int(&mut bytes, value); // skins, width, height, verts, tris, frames, sync, flags
        }
        float(&mut bytes, 1.0);
        int(&mut bytes, 0);
        bytes.extend_from_slice(&[0, 1, 2, 3]);
        for (seam, s, t) in [(0, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1)] {
            int(&mut bytes, seam);
            int(&mut bytes, s);
            int(&mut bytes, t);
        }
        // Counter-clockwise from outside (or flipped)
        let mut triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        if flip {
            for triangle in &mut triangles {
                triangle.swap(1, 2);
            }
        }
        for (index, triangle) in triangles.iter().enumerate() {
            int(&mut bytes, (index != 1) as i32); // triangle 1 faces back
            for corner in triangle {
                int(&mut bytes, *corner);
            }
        }
        for (frame, name) in ["idle", "kick"].iter().enumerate() {
            int(&mut bytes, 0);
            bytes.extend_from_slice(&[0; 8]);
            let mut name_bytes = [0u8; 16];
            name_bytes[..name.len()].copy_from_slice(name.as_bytes());
            bytes.extend_from_slice(&name_bytes);
            let lift = frame as u8 * 32;
            for vertex in [
                [0, 0, lift],
                [64, 0, lift],
                [0, 64, lift],
                [0, 0, 64 + lift],
            ] {
                bytes.extend_from_slice(&[vertex[0], vertex[1], vertex[2], 0]);
            }
        }
        bytes
    }

    fn palette() -> Vec<u8> {
        (0..768).map(|value| (value % 256) as u8).collect()
    }

    fn mesh(rmsh: &[u8]) -> (Vec<convert::CookedVertex>, Vec<u32>) {
        let vertex_count = u32::from_le_bytes(rmsh[8..12].try_into().unwrap()) as usize;
        let index_count = u32::from_le_bytes(rmsh[12..16].try_into().unwrap()) as usize;
        let vertex_end = 16 + vertex_count * std::mem::size_of::<convert::CookedVertex>();
        let vertices = bytemuck::cast_slice(&rmsh[16..vertex_end]).to_vec();
        let indices =
            bytemuck::cast_slice(&rmsh[vertex_end..vertex_end + index_count * 4]).to_vec();
        (vertices, indices)
    }

    #[test]
    fn tetrahedron_two_frames_named_and_offset() {
        let model = build_quake_model("t", &tetrahedron(false), None, Some(&palette())).unwrap();
        let names: Vec<&str> = model
            .frames
            .iter()
            .map(|frame| frame.name.as_str())
            .collect();
        assert_eq!(names, ["idle", "kick"]);
        let (idle, indices) = mesh(&model.frames[0].rmsh);
        let (kick, _) = mesh(&model.frames[1].rmsh);
        assert_eq!(indices.len(), 12);
        // Quake z 32 units higher = engine y 1 m higher
        for (a, b) in idle.iter().zip(&kick) {
            assert!((b.position[1] - a.position[1] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn back_facing_seam_vertex_duplicated_with_half_skin_offset() {
        let model = build_quake_model("t", &tetrahedron(false), None, Some(&palette())).unwrap();
        let (vertices, _) = mesh(&model.frames[0].rmsh);
        // Vertex 1 is on the seam and used by back-facing triangle 1: a second copy at s + 1
        assert_eq!(vertices.len(), 5);
        let seam: Vec<[f32; 2]> = vertices
            .iter()
            .filter(|vertex| (vertex.position[0] - 2.0).abs() < 1e-5)
            .map(|vertex| vertex.texture_coordinates)
            .collect();
        assert!(
            seam.contains(&[0.25, 0.75]) && seam.contains(&[0.75, 0.75]),
            "{seam:?}"
        );
    }

    #[test]
    fn clockwise_model_rewound_counter_clockwise() {
        for flip in [false, true] {
            let model = build_quake_model("t", &tetrahedron(flip), None, Some(&palette())).unwrap();
            let (vertices, indices) = mesh(&model.frames[0].rmsh);
            let center = [0.5, 0.5, -0.5]; // roughly inside, engine space
            for triangle in indices.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|i| vertices[triangle[i] as usize].position);
                let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let edge2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let normal = [
                    edge1[1] * edge2[2] - edge1[2] * edge2[1],
                    edge1[2] * edge2[0] - edge1[0] * edge2[2],
                    edge1[0] * edge2[1] - edge1[1] * edge2[0],
                ];
                let outward = [a[0] - center[0], a[1] - center[1], a[2] - center[2]];
                let facing: f32 = (0..3).map(|i| normal[i] * outward[i]).sum();
                assert!(facing > 0.0, "flip {flip}: inward triangle");
            }
        }
    }

    #[test]
    fn palette_skin_decoded_to_rgba() {
        let model = build_quake_model("t", &tetrahedron(false), None, Some(&palette())).unwrap();
        // RTEX header 16 bytes, then texel 1 = palette entry 1 = (3, 4, 5)
        assert_eq!(&model.texture.rtex[16 + 4..16 + 8], &[3, 4, 5, 255]);
    }

    #[test]
    fn wrong_ident_rejected() {
        let mut bytes = tetrahedron(false);
        bytes[0] = b'X';
        assert!(build_quake_model("t", &bytes, None, Some(&palette())).is_err());
    }
}
