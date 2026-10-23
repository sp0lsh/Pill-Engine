//! Skinned glTF input for [`crate::skinning`] (feature `gltf_scene`): node hierarchy with rest
//! TRS, skins, skinned triangle primitives in mesh space, and animations.
//!
//! Textures and materials still come from [`crate::gltf_scene::parse_gltf_scene`]; primitive
//! `material` indices refer to its `materials`.

use anyhow::{bail, Error, Result};
use glam::{Mat4, Quat, Vec3};
use goth_gltf::{ComponentType, NodeTransform, PrimitiveMode};

use crate::convert;
use crate::gltf_scene::{
    accessor_layout, get, read_floats_at, read_primitive_attributes, resolve_buffers, to_vec4, Doc,
};
use crate::skinning::{
    Animation, Channel, ChannelPath, Interpolation, Skin, SkinNode, SkinnedPrimitive, Trs,
};

pub struct GltfSkinnedData {
    pub nodes: Vec<SkinNode>,
    pub skins: Vec<Skin>,
    pub primitives: Vec<SkinnedPrimitive>,
    pub animations: Vec<Animation>,
}

/// Parses the skinning data of a glTF/glb; `external_buffers` as for `parse_gltf_scene`.
pub fn parse_gltf_skinned(
    gltf_bytes: &[u8],
    external_buffers: &[Vec<u8>],
) -> Result<GltfSkinnedData> {
    let (doc, blob) =
        Doc::from_bytes(gltf_bytes).map_err(|e| Error::msg(format!("parse glTF: {e:?}")))?;
    let buffers = resolve_buffers(&doc, blob, external_buffers)?;
    let buffers: Vec<&[u8]> = buffers.iter().map(|b| b.as_ref()).collect();

    let nodes = parse_nodes(&doc)?;

    let mut skins = Vec::with_capacity(doc.skins.len());
    for (skin_idx, skin) in doc.skins.iter().enumerate() {
        if let Some(&joint) = skin.joints.iter().find(|&&joint| joint >= nodes.len()) {
            bail!("glTF skin {skin_idx}: joint node {joint} out of range");
        }
        let inverse_bind = match skin.inverse_bind_matrices {
            Some(acc) => {
                let flat = read_floats_at(&doc, &buffers, acc)?;
                if flat.len() < skin.joints.len() * 16 {
                    bail!("glTF skin {skin_idx}: fewer inverse bind matrices than joints");
                }
                flat.chunks_exact(16)
                    .take(skin.joints.len())
                    .map(Mat4::from_cols_slice)
                    .collect()
            }
            None => vec![Mat4::IDENTITY; skin.joints.len()],
        };
        skins.push(Skin {
            joints: skin.joints.clone(),
            inverse_bind,
        });
    }

    let mut primitives = Vec::new();
    for (node_idx, node) in doc.nodes.iter().enumerate() {
        let (Some(mesh_idx), Some(skin_idx)) = (node.mesh, node.skin) else {
            continue;
        };
        let joint_count = get(&skins, skin_idx, "skin")?.joints.len();
        for (prim_idx, prim) in get(&doc.meshes, mesh_idx, "mesh")?
            .primitives
            .iter()
            .enumerate()
        {
            let attributes = &prim.attributes;
            let (Some(joints_acc), Some(weights_acc)) = (attributes.joints_0, attributes.weights_0)
            else {
                log::warn!(
                    "glTF node {node_idx}: primitive {prim_idx} has no JOINTS_0/WEIGHTS_0, skipped"
                );
                continue;
            };
            if prim.mode != PrimitiveMode::Triangles {
                log::warn!("glTF node {node_idx}: skipping {:?} primitive", prim.mode);
                continue;
            }
            let primitive_attributes = read_primitive_attributes(&doc, &buffers, prim)?;
            let vertex_count = primitive_attributes.positions.len();
            let generated_normals = primitive_attributes.normals.is_none();
            let joints = read_joints(&doc, &buffers, joints_acc)?;
            let weights = to_vec4(read_floats_at(&doc, &buffers, weights_acc)?);
            if joints.len() != vertex_count || weights.len() != vertex_count {
                bail!("glTF mesh {mesh_idx}: JOINTS_0/WEIGHTS_0 count differs from POSITION");
            }
            if let Some(&joint) = joints
                .iter()
                .flatten()
                .find(|&&j| j as usize >= joint_count)
            {
                bail!("glTF mesh {mesh_idx}: joint index {joint} >= {joint_count} skin joints");
            }
            let (rest, indices) = convert::build_primitive(primitive_attributes, convert::IDENTITY)
                .map_err(|e| Error::msg(format!("glTF mesh {mesh_idx}: {e}")))?;
            primitives.push(SkinnedPrimitive {
                name: format!("skinned_{node_idx}_{prim_idx}"),
                skin: skin_idx,
                material: prim.material,
                rest,
                indices,
                joints,
                weights,
                generated_normals,
            });
        }
    }

    let mut animations = Vec::with_capacity(doc.animations.len());
    for (anim_idx, anim) in doc.animations.iter().enumerate() {
        let mut channels = Vec::with_capacity(anim.channels.len());
        let mut duration = 0.0f32;
        for ch in &anim.channels {
            let Some(node) = ch.target.node else {
                continue;
            };
            get(&nodes, node, "node")?;
            let sampler = get(&anim.samplers, ch.sampler, "animation sampler")?;
            let path = match ch.target.path {
                goth_gltf::TargetPath::Translation => ChannelPath::Translation,
                goth_gltf::TargetPath::Rotation => ChannelPath::Rotation,
                goth_gltf::TargetPath::Scale => ChannelPath::Scale,
                goth_gltf::TargetPath::Weights => {
                    log::warn!("glTF animation {anim_idx}: morph target weights channel skipped");
                    continue;
                }
            };
            let interpolation = match sampler.interpolation {
                goth_gltf::Interpolation::Step => Interpolation::Step,
                goth_gltf::Interpolation::Linear => Interpolation::Linear,
                goth_gltf::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };
            let times = read_floats_at(&doc, &buffers, sampler.input)?;
            let values = read_floats_at(&doc, &buffers, sampler.output)?;
            let per_key = path.components()
                * if interpolation == Interpolation::CubicSpline {
                    3
                } else {
                    1
                };
            if times.is_empty() || values.len() != times.len() * per_key {
                bail!(
                    "glTF animation {anim_idx}: sampler {} output/input count mismatch",
                    ch.sampler
                );
            }
            duration = duration.max(*times.last().unwrap());
            channels.push(Channel {
                node,
                path,
                interpolation,
                times,
                values,
            });
        }
        animations.push(Animation {
            name: anim
                .name
                .clone()
                .unwrap_or_else(|| format!("animation_{anim_idx}")),
            channels,
            duration,
        });
    }

    Ok(GltfSkinnedData {
        nodes,
        skins,
        primitives,
        animations,
    })
}

/// Rest TRS and parent of every node; rejects shared children and cycles.
fn parse_nodes(doc: &Doc) -> Result<Vec<SkinNode>> {
    let mut nodes: Vec<SkinNode> = doc
        .nodes
        .iter()
        .map(|node| SkinNode {
            parent: None,
            rest: match node.transform() {
                NodeTransform::Matrix(m) => {
                    let (scale, rotation, translation) =
                        Mat4::from_cols_array(&m).to_scale_rotation_translation();
                    Trs {
                        translation,
                        rotation,
                        scale,
                    }
                }
                NodeTransform::Set {
                    translation,
                    rotation,
                    scale,
                } => Trs {
                    translation: Vec3::from_array(translation),
                    rotation: Quat::from_array(rotation),
                    scale: Vec3::from_array(scale),
                },
            },
        })
        .collect();
    for (idx, node) in doc.nodes.iter().enumerate() {
        for &child in &node.children {
            let child_node = nodes.get_mut(child).ok_or_else(|| {
                Error::msg(format!("glTF node {idx}: child {child} out of range"))
            })?;
            if child_node.parent.replace(idx).is_some() {
                bail!("glTF node {child} has two parents");
            }
        }
    }
    for start in 0..nodes.len() {
        let mut node = nodes[start].parent;
        for _ in 0..nodes.len() {
            match node {
                Some(n) if n == start => bail!("glTF node {start} is its own ancestor"),
                Some(n) => node = nodes[n].parent,
                None => break,
            }
        }
    }
    Ok(nodes)
}

/// `JOINTS_n`: VEC4 of unsigned byte/short, read as integers (never normalized).
fn read_joints(doc: &Doc, buffers: &[&[u8]], accessor_idx: usize) -> Result<Vec<[u16; 4]>> {
    let acc = get(&doc.accessors, accessor_idx, "accessor")?;
    if acc.accessor_type.num_components() != 4 {
        bail!("JOINTS_0 accessor {accessor_idx} is not VEC4");
    }
    let (buf, base, stride, comp_size) = accessor_layout(doc, buffers, accessor_idx)?;
    let mut out = Vec::with_capacity(acc.count);
    for i in 0..acc.count {
        let elem = base + i * stride;
        let mut joint = [0u16; 4];
        for (c, value) in joint.iter_mut().enumerate() {
            let o = elem + c * comp_size;
            *value = match acc.component_type {
                ComponentType::UnsignedByte => buf[o] as u16,
                ComponentType::UnsignedShort => u16::from_le_bytes([buf[o], buf[o + 1]]),
                other => bail!("JOINTS_0 component type {other:?} (must be u8 or u16)"),
            };
        }
        out.push(joint);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skinning::{
        flat_triangles, global_transforms, joint_matrices, pose, skin, skin_bounds, skinned_bounds,
    };

    // One triangle; JOINTS_0 as u8 (accessor 1) and u16 (accessor 2); WEIGHTS_0 as unorm8.
    fn skinned_doc(joints_accessor: usize) -> (String, Vec<u8>) {
        let mut buf: Vec<u8> = [[0f32, 0., 0.], [1., 0., 0.], [0., 1., 0.]]
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(); // 0..36
        buf.extend_from_slice(&[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]); // u8 joints 36..48
        for j in [[0u16, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]] {
            for v in j {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        } // u16 joints 48..72
        buf.extend_from_slice(&[255, 0, 0, 0, 51, 204, 0, 0, 128, 127, 0, 0]); // unorm8 weights 72..84
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0,1]}}],
              "nodes":[{{"mesh":0,"skin":0,"translation":[50,0,0]}},{{"children":[2],"translation":[0,1,0]}},{{"rotation":[0,0,0.7071068,0.7071068]}}],
              "skins":[{{"joints":[1,2]}}],
              "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"JOINTS_0":{joints_accessor},"WEIGHTS_0":3}}}}]}}],
              "buffers":[{{"uri":"b.bin","byteLength":84}}],
              "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36}},
                {{"buffer":0,"byteOffset":36,"byteLength":12}},
                {{"buffer":0,"byteOffset":48,"byteLength":24}},
                {{"buffer":0,"byteOffset":72,"byteLength":12}}],
              "accessors":[
                {{"bufferView":0,"componentType":5126,"type":"VEC3","count":3}},
                {{"bufferView":1,"componentType":5121,"type":"VEC4","count":3}},
                {{"bufferView":2,"componentType":5123,"type":"VEC4","count":3}},
                {{"bufferView":3,"componentType":5121,"type":"VEC4","count":3,"normalized":true}}]}}"#
        );
        (json, buf)
    }

    #[test]
    fn parse_joints_u8_and_u16_are_integer_indices() {
        for accessor in [1, 2] {
            let (json, buf) = skinned_doc(accessor);
            let data = parse_gltf_skinned(json.as_bytes(), &[buf]).expect("parse");
            assert_eq!(
                data.primitives[0].joints,
                [[0, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]]
            );
        }
    }

    #[test]
    fn parse_weights_unorm8_are_divided_by_255() {
        let (json, buf) = skinned_doc(1);
        let data = parse_gltf_skinned(json.as_bytes(), &[buf]).expect("parse");
        let w = &data.primitives[0].weights;
        assert_eq!(w[0], [1.0, 0.0, 0.0, 0.0]);
        assert!((w[1][0] - 51.0 / 255.0).abs() < 1e-6 && (w[1][1] - 204.0 / 255.0).abs() < 1e-6);
        assert!((w[2][0] - 128.0 / 255.0).abs() < 1e-6 && (w[2][1] - 127.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn skin_without_inverse_bind_matrices_uses_identity() {
        let (json, buf) = skinned_doc(1);
        let data = parse_gltf_skinned(json.as_bytes(), &[buf]).expect("parse");
        assert_eq!(data.skins[0].inverse_bind, vec![Mat4::IDENTITY; 2]);
        // vertex 0 is bound to joint 0 (node 1: translate y+1) with weight 1; the skinned
        // node's own translation (50,0,0) is ignored
        let rest: Vec<Trs> = data.nodes.iter().map(|n| n.rest).collect();
        let globals = global_transforms(&data.nodes, &rest);
        let mut out = Vec::new();
        skin(
            &data.primitives[0],
            &joint_matrices(&data.skins[0], &globals),
            &mut out,
        );
        assert!((Vec3::from_array(out[0].position) - Vec3::new(0., 1., 0.)).length() < 1e-6);
        // vertex 1 (1,0,0): 0.2 · joint 1 (node 2 under node 1: rotate 90° about z, then y+1 →
        // (0,2,0)) + 0.8 · joint 0 (→ (1,1,0)) = (0.8, 1.2, 0)
        assert!((Vec3::from_array(out[1].position) - Vec3::new(0.8, 1.2, 0.)).length() < 1e-5);
    }

    fn fox() -> GltfSkinnedData {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/skinning/res/models/Fox.glb"
        );
        let bytes = std::fs::read(path).expect("Fox.glb");
        parse_gltf_skinned(&bytes, &[]).expect("parse Fox.glb")
    }

    #[test]
    fn fox_rest_pose_skinned_equals_rest_positions() {
        let data = fox();
        assert_eq!(data.primitives.len(), 1);
        assert_eq!(data.skins[0].joints.len(), 24);
        let prim = &data.primitives[0];
        let rest: Vec<Trs> = data.nodes.iter().map(|n| n.rest).collect();
        let globals = global_transforms(&data.nodes, &rest);
        let mut out = Vec::new();
        skin(
            prim,
            &joint_matrices(&data.skins[prim.skin], &globals),
            &mut out,
        );
        let extent = prim
            .rest
            .iter()
            .map(|v| Vec3::from_array(v.position).abs().max_element())
            .fold(0.0, f32::max);
        let worst = out
            .iter()
            .zip(&prim.rest)
            .map(|(a, b)| (Vec3::from_array(a.position) - Vec3::from_array(b.position)).length())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4 * extent, "worst {worst} vs extent {extent}");
    }

    #[test]
    fn fox_clips_sampled_every_100ms_are_finite_with_unit_weights() {
        let data = fox();
        let names: Vec<&str> = data.animations.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["Survey", "Walk", "Run"]);
        let prim = &data.primitives[0];
        for w in &prim.weights {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{w:?}");
        }
        let mut out = Vec::new();
        for anim in &data.animations {
            let mut t = 0.0;
            while t <= anim.duration {
                let globals = global_transforms(&data.nodes, &pose(&data.nodes, anim, t));
                skin(
                    prim,
                    &joint_matrices(&data.skins[prim.skin], &globals),
                    &mut out,
                );
                assert!(
                    out.iter()
                        .all(|v| v.position.iter().chain(&v.normal).all(|x| x.is_finite())),
                    "{} t={t}",
                    anim.name
                );
                t += 0.1;
            }
        }
    }

    fn recursive_skeletons() -> GltfSkinnedData {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/skinning/res/models/RecursiveSkeletons.glb"
        );
        let bytes = std::fs::read(path).expect("RecursiveSkeletons.glb");
        parse_gltf_skinned(&bytes, &[]).expect("parse RecursiveSkeletons.glb")
    }

    #[test]
    fn recursive_skeletons_84_skins_one_mesh_track_finite() {
        let data = recursive_skeletons();
        assert_eq!(data.skins.len(), 84);
        assert_eq!(data.primitives.len(), 84);
        let anim = &data.animations[0];
        let mut out = Vec::new();
        for prim in &data.primitives {
            let mut t = 0.0;
            while t <= anim.duration {
                let globals = global_transforms(&data.nodes, &pose(&data.nodes, anim, t));
                skin(
                    prim,
                    &joint_matrices(&data.skins[prim.skin], &globals),
                    &mut out,
                );
                for v in &out {
                    assert!(
                        v.position.iter().chain(&v.normal).all(|x| x.is_finite()),
                        "skin {} t={t}",
                        prim.skin
                    );
                }
                t += 0.1;
            }
        }
    }

    #[test]
    fn skinned_bounds_fox_all_clips_contain_every_cpu_skinned_vertex() {
        let data = fox();
        let prim = &data.primitives[0];
        let skin_data = &data.skins[prim.skin];
        let bounds = skin_bounds(prim, skin_data.joints.len());
        let mut out = Vec::new();
        for anim in &data.animations {
            let mut t = 0.0;
            while t <= anim.duration {
                let joints = joint_matrices(
                    skin_data,
                    &global_transforms(&data.nodes, &pose(&data.nodes, anim, t)),
                );
                skin(prim, &joints, &mut out);
                let (low, high) = skinned_bounds(&bounds, &joints);
                for v in &out {
                    let p = Vec3::from_array(v.position);
                    assert!(
                        p.cmpge(low - 1e-3).all() && p.cmple(high + 1e-3).all(),
                        "{} t={t}: {p} outside {low}..{high}",
                        anim.name
                    );
                }
                t += 0.1;
            }
        }
    }

    #[test]
    fn flat_triangles_fox_true_indexed_quad_false() {
        assert!(flat_triangles(&fox().primitives[0]));
        let mut quad = fox().primitives.remove(0);
        quad.rest.truncate(4);
        quad.indices = vec![0, 1, 2, 2, 1, 3];
        assert!(!flat_triangles(&quad));
    }

    #[test]
    fn skinned_bounds_unweighted_vertex_far_away_is_contained() {
        let mut prim = fox().primitives.remove(0);
        prim.weights[0] = [0.0; 4];
        prim.rest[0].position = [1e4, 0.0, 0.0];
        let joints = vec![Mat4::IDENTITY; 24];
        let (_, high) = skinned_bounds(&skin_bounds(&prim, 24), &joints);
        assert!(high.x >= 1e4, "{high}");
    }
}
