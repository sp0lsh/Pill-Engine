use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use bytemuck::{Pod, Zeroable};

use crate::Rule;

/// Must stay bit-for-bit identical to `pill_engine::resources::MeshVertex`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    texture_coordinates: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

/// GLB → cooked_mesh + sidecar textures.
///
/// Primary output: `{stem}.cooked_mesh` (RMSH format, same as ObjToCookedMesh).
/// Side outputs written in the same directory when textures are present:
///   `{stem}_albedo.cooked_tex`  — base color (RTEX RGBA8)
///   `{stem}_normal.cooked_tex`  — normal map (RTEX RGBA8)
pub struct GlbToCookedMesh;

impl Rule for GlbToCookedMesh {
    fn name(&self) -> &'static str {
        "glb_to_cooked_mesh"
    }

    fn input_glob(&self) -> &'static str {
        "**/*.glb"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        input.with_extension("cooked_mesh")
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let bytes = std::fs::read(input).with_context(|| format!("read {input:?}"))?;
        let (doc, buffers, images) =
            gltf::import_slice(&bytes).with_context(|| format!("parse GLB {input:?}"))?;

        // --- Mesh ---

        let mesh = doc
            .meshes()
            .next()
            .with_context(|| format!("{input:?}: no meshes in GLB"))?;
        let prim = mesh
            .primitives()
            .next()
            .with_context(|| format!("{input:?}: mesh has no primitives"))?;
        let reader = prim.reader(|b| Some(&*buffers[b.index()]));

        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .with_context(|| format!("{input:?}: missing positions"))?
            .collect();
        let normals: Vec<[f32; 3]> = reader
            .read_normals()
            .with_context(|| format!("{input:?}: missing normals"))?
            .collect();
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .with_context(|| format!("{input:?}: missing UV set 0"))?
            .into_f32()
            .collect();
        let tangents_glb: Vec<[f32; 4]> = reader
            .read_tangents()
            .map(|t| t.collect())
            .unwrap_or_default();
        let indices: Vec<u32> = reader
            .read_indices()
            .with_context(|| format!("{input:?}: missing indices"))?
            .into_u32()
            .collect();

        let mut vertices: Vec<Vertex> = (0..positions.len())
            .map(|i| {
                let n = normals[i];
                let (tx, ty, tz, sign) = tangents_glb
                    .get(i)
                    .map(|t| (t[0], t[1], t[2], t[3]))
                    .unwrap_or((1.0, 0.0, 0.0, 1.0));
                let bx = (n[1] * tz - n[2] * ty) * sign;
                let by = (n[2] * tx - n[0] * tz) * sign;
                let bz = (n[0] * ty - n[1] * tx) * sign;
                Vertex {
                    position: positions[i],
                    texture_coordinates: uvs[i],
                    normal: n,
                    tangent: [tx, ty, tz],
                    bitangent: [bx, by, bz],
                }
            })
            .collect();

        if tangents_glb.is_empty() {
            compute_tangents(&mut vertices, &indices);
        }

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&indices);
        let mut out = Vec::with_capacity(16 + vertex_bytes.len() + index_bytes.len());
        out.extend_from_slice(b"RMSH");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
        out.extend_from_slice(&(indices.len() as u32).to_le_bytes());
        out.extend_from_slice(vertex_bytes);
        out.extend_from_slice(index_bytes);
        std::fs::write(output, &out).with_context(|| format!("write {output:?}"))?;

        // --- Textures (side outputs) ---

        let stem = input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let dir = output
            .parent()
            .with_context(|| "output has no parent directory")?;

        if let Some(mat) = doc.materials().next() {
            if let Some(info) = mat.pbr_metallic_roughness().base_color_texture() {
                let idx = info.texture().source().index();
                if let Some(img) = images.get(idx) {
                    let rtex = image_to_rtex(img)
                        .with_context(|| format!("{input:?}: base color texture"))?;
                    std::fs::write(dir.join(format!("{stem}_albedo.cooked_tex")), &rtex)?;
                }
            }

            if let Some(info) = mat.normal_texture() {
                let idx = info.texture().source().index();
                if let Some(img) = images.get(idx) {
                    let rtex =
                        image_to_rtex(img).with_context(|| format!("{input:?}: normal texture"))?;
                    std::fs::write(dir.join(format!("{stem}_normal.cooked_tex")), &rtex)?;
                }
            }
        }

        Ok(())
    }
}

fn image_to_rtex(img: &gltf::image::Data) -> Result<Vec<u8>> {
    use gltf::image::Format;

    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => img
            .pixels
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255u8])
            .collect(),
        fmt => bail!("unsupported GLB image format: {fmt:?}"),
    };

    let mut out = Vec::with_capacity(16 + rgba.len());
    out.extend_from_slice(b"RTEX");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&img.width.to_le_bytes());
    out.extend_from_slice(&img.height.to_le_bytes());
    out.extend_from_slice(&rgba);
    Ok(out)
}

fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut triangle_counts = vec![0usize; vertices.len()];

    for tri in indices.chunks(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = vertices[i0].position;
        let p1 = vertices[i1].position;
        let p2 = vertices[i2].position;
        let uv0 = vertices[i0].texture_coordinates;
        let uv1 = vertices[i1].texture_coordinates;
        let uv2 = vertices[i2].texture_coordinates;

        let dp1 = sub3(p1, p0);
        let dp2 = sub3(p2, p0);
        let duv1 = sub2(uv1, uv0);
        let duv2 = sub2(uv2, uv0);

        let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if det.abs() < 1e-8 {
            continue;
        }
        let inv = 1.0 / det;
        let tangent = scale3(sub3(scale3(dp1, duv2[1]), scale3(dp2, duv1[1])), inv);
        let bitangent = scale3(sub3(scale3(dp2, duv1[0]), scale3(dp1, duv2[0])), inv);

        for &i in &[i0, i1, i2] {
            vertices[i].tangent = add3(vertices[i].tangent, tangent);
            vertices[i].bitangent = add3(vertices[i].bitangent, bitangent);
            triangle_counts[i] += 1;
        }
    }

    for (i, &count) in triangle_counts.iter().enumerate() {
        if count > 0 {
            let inv = 1.0 / count as f32;
            vertices[i].tangent = normalize3(scale3(vertices[i].tangent, inv));
            vertices[i].bitangent = normalize3(scale3(vertices[i].bitangent, inv));
        }
    }
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn normalize3(a: [f32; 3]) -> [f32; 3] {
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if len < 1e-10 {
        a
    } else {
        [a[0] / len, a[1] / len, a[2] / len]
    }
}
