//! CPU skeletal skinning, literal to the glTF 2.0 spec: the correctness baseline for GPU skinning.
//!
//! Per frame: [`pose`] samples an animation over the rest pose (Appendix C interpolation),
//! [`global_transforms`] walks the node hierarchy, [`joint_matrices`] premultiplies the inverse
//! bind matrices and [`skin`] applies `Σ Wᵢ · M[Jᵢ] · B[Jᵢ]` to every vertex ("Skinned Mesh
//! Primitives"). Input comes from [`crate::gltf_skin::parse_gltf_skinned`].

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};

use crate::convert;
pub use crate::convert::CookedVertex;

/// A node's local transform (the rest pose, or a sampled pose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Trs {
    /// `T · R · S`
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

pub struct SkinNode {
    pub parent: Option<usize>,
    pub rest: Trs,
}

pub struct Skin {
    /// Node indices
    pub joints: Vec<usize>,
    /// One per joint; identity when the skin has no `inverseBindMatrices`
    pub inverse_bind: Vec<Mat4>,
}

#[derive(Clone)]
pub struct SkinnedPrimitive {
    pub name: String,
    pub skin: usize,
    /// Index into the document's materials (`GltfSceneData::materials`)
    pub material: Option<usize>,
    /// Rest pose in mesh space (a skinned primitive ignores its node's transform), with the
    /// loader's fallbacks for missing NORMAL/TANGENT/TEXCOORD_0
    pub rest: Vec<CookedVertex>,
    pub indices: Vec<u32>,
    /// `JOINTS_0`: indices into the skin's joints. Only the first 4 influences are read.
    pub joints: Vec<[u16; 4]>,
    /// `WEIGHTS_0`, dequantized
    pub weights: Vec<[f32; 4]>,
    /// No NORMAL in the source: normals are recomputed from the skinned positions
    pub generated_normals: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelPath {
    Translation,
    Rotation,
    Scale,
}

impl ChannelPath {
    /// Floats per key value: xyz, or xyzw for rotation
    pub fn components(self) -> usize {
        if self == ChannelPath::Rotation {
            4
        } else {
            3
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

pub struct Channel {
    pub node: usize,
    pub path: ChannelPath,
    pub interpolation: Interpolation,
    /// Sampler input (seconds)
    pub times: Vec<f32>,
    /// Sampler output, flat; CUBICSPLINE stores (in-tangent, value, out-tangent) per key
    pub values: Vec<f32>,
}

pub struct Animation {
    pub name: String,
    pub channels: Vec<Channel>,
    /// Largest sampler input
    pub duration: f32,
}

/// Samples a translation/scale (xyz) or rotation (xyzw) channel at `time` (spec Appendix C).
/// Outside the input range the output is clamped to the nearest end.
pub fn sample(channel: &Channel, time: f32) -> Vec4 {
    let rotation = channel.path == ChannelPath::Rotation;
    let components = channel.path.components();
    let cubic = channel.interpolation == Interpolation::CubicSpline;
    // part: 0 = in-tangent a, 1 = value v, 2 = out-tangent b (CUBICSPLINE); 0 = value otherwise
    let key = |k: usize, part: usize| -> Vec4 {
        let base = (k * if cubic { 3 } else { 1 } + part) * components;
        let mut v = [0.0; 4];
        v[..components].copy_from_slice(&channel.values[base..base + components]);
        Vec4::from_array(v)
    };
    let value = if cubic { 1 } else { 0 };
    let times = &channel.times;
    let last = times.len() - 1;
    if time <= times[0] {
        return key(0, value);
    }
    if time >= times[last] {
        return key(last, value);
    }
    // t_k <= t_c < t_{k+1}
    let k = times.partition_point(|&t_k| t_k <= time) - 1;
    let t_d = times[k + 1] - times[k];
    let t = (time - times[k]) / t_d;
    match channel.interpolation {
        Interpolation::Step => key(k, 0),
        Interpolation::Linear if rotation => slerp(key(k, 0), key(k + 1, 0), t),
        Interpolation::Linear => (1.0 - t) * key(k, 0) + t * key(k + 1, 0),
        Interpolation::CubicSpline => {
            let (t2, t3) = (t * t, t * t * t);
            let v = (2.0 * t3 - 3.0 * t2 + 1.0) * key(k, 1)
                + t_d * (t3 - 2.0 * t2 + t) * key(k, 2)
                + (-2.0 * t3 + 3.0 * t2) * key(k + 1, 1)
                + t_d * (t3 - t2) * key(k + 1, 0);
            if rotation {
                v.normalize()
            } else {
                v
            }
        }
    }
}

/// Spec slerp: `a = acos(|v_k·v_{k+1}|)`, `s = sign(v_k·v_{k+1})` (short path); as `a` → 0 it
/// turns into linear interpolation, normalized here.
fn slerp(v_k: Vec4, v_k1: Vec4, t: f32) -> Vec4 {
    let dot = v_k.dot(v_k1);
    let s = if dot < 0.0 { -1.0 } else { 1.0 };
    let a = dot.abs().min(1.0).acos();
    let sin_a = a.sin();
    if sin_a < 1e-6 {
        return ((1.0 - t) * v_k + t * s * v_k1).normalize();
    }
    (a * (1.0 - t)).sin() / sin_a * v_k + s * (a * t).sin() / sin_a * v_k1
}

/// Local transforms at `time`: the rest pose with the animated paths overridden.
pub fn pose(nodes: &[SkinNode], animation: &Animation, time: f32) -> Vec<Trs> {
    let mut pose: Vec<Trs> = nodes.iter().map(|node| node.rest).collect();
    for channel in &animation.channels {
        let value = sample(channel, time);
        let trs = &mut pose[channel.node];
        match channel.path {
            ChannelPath::Translation => trs.translation = value.truncate(),
            ChannelPath::Rotation => trs.rotation = Quat::from_vec4(value),
            ChannelPath::Scale => trs.scale = value.truncate(),
        }
    }
    pose
}

/// Global transform of every node: `parent · local`, parents first. The loader rejects cycles.
pub fn global_transforms(nodes: &[SkinNode], pose: &[Trs]) -> Vec<Mat4> {
    let mut world: Vec<Option<Mat4>> = vec![None; nodes.len()];
    let mut chain = Vec::new();
    for start in 0..nodes.len() {
        let mut node = Some(start);
        while let Some(n) = node.filter(|&n| world[n].is_none()) {
            chain.push(n);
            node = nodes[n].parent;
        }
        while let Some(n) = chain.pop() {
            let local = pose[n].matrix();
            world[n] = Some(match nodes[n].parent {
                Some(parent) => world[parent].expect("parent resolved first") * local,
                None => local,
            });
        }
    }
    world
        .into_iter()
        .map(|m| m.expect("every node resolved"))
        .collect()
}

/// `M[j] · B[j]` per joint of the skin.
pub fn joint_matrices(skin: &Skin, globals: &[Mat4]) -> Vec<Mat4> {
    skin.joints
        .iter()
        .zip(&skin.inverse_bind)
        .map(|(&joint, inverse_bind)| globals[joint] * *inverse_bind)
        .collect()
}

/// Skins `primitive` into `out` (resized to its vertex count): per vertex `S = Σ Wᵢ · J[Jᵢ]`,
/// position `S·p`, normal by the inverse-transpose of S's 3×3, tangent and bitangent by the 3×3.
/// A vertex with all-zero weights has no transform in the spec; it keeps its rest position.
pub fn skin(primitive: &SkinnedPrimitive, joint_matrices: &[Mat4], out: &mut Vec<CookedVertex>) {
    out.clear();
    out.extend(primitive.rest.iter().enumerate().map(|(i, rest)| {
        let [w0, w1, w2, w3] = primitive.weights[i];
        if [w0, w1, w2, w3] == [0.0; 4] {
            return *rest;
        }
        let j = primitive.joints[i].map(usize::from);
        let s = joint_matrices[j[0]] * w0
            + joint_matrices[j[1]] * w1
            + joint_matrices[j[2]] * w2
            + joint_matrices[j[3]] * w3;
        let linear = Mat3::from_mat4(s);
        // Generated normals are replaced by face normals below
        let normal = if primitive.generated_normals {
            rest.normal
        } else {
            (linear.inverse().transpose() * Vec3::from_array(rest.normal))
                .normalize_or_zero()
                .to_array()
        };
        CookedVertex {
            position: s
                .transform_point3(Vec3::from_array(rest.position))
                .to_array(),
            texture_coordinates: rest.texture_coordinates,
            normal,
            tangent: (linear * Vec3::from_array(rest.tangent))
                .normalize_or_zero()
                .to_array(),
            bitangent: (linear * Vec3::from_array(rest.bitangent))
                .normalize_or_zero()
                .to_array(),
        }
    }));
    if primitive.generated_normals {
        let positions: Vec<[f32; 3]> = out.iter().map(|v| v.position).collect();
        for (vertex, normal) in out
            .iter_mut()
            .zip(convert::face_normals(&positions, &primitive.indices))
        {
            vertex.normal = normal;
        }
    }
}

/// Rest-pose AABBs `(min, max)` for culling a skinned mesh in any pose (`skinned_bounds`).
#[derive(Clone, Debug, Default)]
pub struct SkinBounds {
    /// Per joint, the vertices it influences (weight > 0); `None` for joints that move no vertex
    pub joints: Vec<Option<(Vec3, Vec3)>>,
    /// Vertices with all-zero weights, which `skin` keeps at rest
    pub unweighted: Option<(Vec3, Vec3)>,
}

pub fn skin_bounds(primitive: &SkinnedPrimitive, joint_count: usize) -> SkinBounds {
    let grow = |bounds: &mut Option<(Vec3, Vec3)>, position: Vec3| {
        let entry = bounds.get_or_insert((position, position));
        *entry = (entry.0.min(position), entry.1.max(position));
    };
    let mut bounds = SkinBounds {
        joints: vec![None; joint_count],
        unweighted: None,
    };
    for ((rest, joints), weights) in primitive
        .rest
        .iter()
        .zip(&primitive.joints)
        .zip(&primitive.weights)
    {
        let position = Vec3::from_array(rest.position);
        if *weights == [0.0; 4] {
            grow(&mut bounds.unweighted, position);
        }
        for (&joint, &weight) in joints.iter().zip(weights) {
            if weight > 0.0 {
                grow(&mut bounds.joints[joint as usize], position);
            }
        }
    }
    bounds
}

/// Conservative AABB of the skinned mesh: the union of every joint's rest box transformed by its
/// joint matrix (centre `M·c`, half extent `|M₃ₓ₃|·e`), and the unweighted vertices' rest box. A
/// skinned vertex `Σ wᵢ·Jᵢ·p` lies in the convex hull of its `Jᵢ·p`, and each `Jᵢ·p` lies in the
/// transformed box of joint i, which holds `p`.
pub fn skinned_bounds(bounds: &SkinBounds, joint_matrices: &[Mat4]) -> (Vec3, Vec3) {
    let (mut min, mut max) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let transformed = bounds
        .joints
        .iter()
        .zip(joint_matrices)
        .map(|(joint, matrix)| (*joint, *matrix))
        .chain(std::iter::once((bounds.unweighted, Mat4::IDENTITY)));
    for (joint, matrix) in transformed {
        let Some((low, high)) = joint else {
            continue;
        };
        let centre = matrix.transform_point3((low + high) * 0.5);
        let linear = Mat3::from_mat4(matrix);
        let absolute = Mat3::from_cols(
            linear.x_axis.abs(),
            linear.y_axis.abs(),
            linear.z_axis.abs(),
        );
        let half_extent = absolute * ((high - low) * 0.5);
        min = min.min(centre - half_extent);
        max = max.max(centre + half_extent);
    }
    if min.x > max.x {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min, max)
    }
}

/// One triangle per vertex (indices `0..n`, n a multiple of 3): each generated normal is the face
/// normal of its own triangle, so GPU skinning can recompute it without a cross-thread sum.
pub fn flat_triangles(primitive: &SkinnedPrimitive) -> bool {
    primitive.indices.len() == primitive.rest.len()
        && primitive.indices.len() % 3 == 0
        && primitive
            .indices
            .iter()
            .enumerate()
            .all(|(i, &index)| index as usize == i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(
        path: ChannelPath,
        interpolation: Interpolation,
        times: &[f32],
        values: &[f32],
    ) -> Channel {
        Channel {
            node: 0,
            path,
            interpolation,
            times: times.to_vec(),
            values: values.to_vec(),
        }
    }

    fn close(a: Vec4, b: Vec4) -> bool {
        (a - b).abs().max_element() < 1e-5
    }

    #[test]
    fn sample_step_between_keys_is_v_k() {
        let ch = channel(
            ChannelPath::Translation,
            Interpolation::Step,
            &[0.0, 1.0],
            &[1., 2., 3., 9., 9., 9.],
        );
        assert_eq!(sample(&ch, 0.7), Vec4::new(1., 2., 3., 0.));
    }

    #[test]
    fn sample_linear_quarter_is_weighted_sum() {
        let ch = channel(
            ChannelPath::Translation,
            Interpolation::Linear,
            &[0.0, 2.0],
            &[0., 0., 0., 4., 8., -4.],
        );
        // t = 0.5 / 2.0 = 0.25 → 0.75·v0 + 0.25·v1
        assert!(close(sample(&ch, 0.5), Vec4::new(1., 2., -1., 0.)));
    }

    #[test]
    fn sample_clamps_outside_input_range() {
        let ch = channel(
            ChannelPath::Translation,
            Interpolation::Linear,
            &[1.0, 2.0],
            &[1., 1., 1., 5., 5., 5.],
        );
        assert_eq!(sample(&ch, 0.0), Vec4::new(1., 1., 1., 0.));
        assert_eq!(sample(&ch, 3.0), Vec4::new(5., 5., 5., 0.));
    }

    #[test]
    fn slerp_negative_dot_takes_short_path() {
        // q0 = identity, q1 = -(90° about z): same rotation as +90°, opposite hemisphere
        let q90 = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let q1 = -Vec4::from(q90);
        let ch = channel(
            ChannelPath::Rotation,
            Interpolation::Linear,
            &[0.0, 1.0],
            &[0., 0., 0., 1., q1.x, q1.y, q1.z, q1.w],
        );
        let half = Quat::from_vec4(sample(&ch, 0.5));
        // short path: 45° about z (not 135°)
        assert!(half.angle_between(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)) < 1e-4);
    }

    #[test]
    fn slerp_identical_quats_has_no_nan() {
        let ch = channel(
            ChannelPath::Rotation,
            Interpolation::Linear,
            &[0.0, 1.0],
            &[0., 0., 0., 1., 0., 0., 0., 1.],
        );
        let q = sample(&ch, 0.3);
        assert!(q.is_finite() && close(q, Vec4::new(0., 0., 0., 1.)));
    }

    #[test]
    fn sample_cubicspline_half_matches_hermite_by_hand() {
        // keys at t=0 and t=2 (t_d = 2), 1-D in x: v0 = 0, b0 = 1; v1 = 1, a1 = 3.
        // t = 0.5: h00 = 0.5, h10 = 0.125, h01 = 0.5, h11 = -0.125
        // v = 0.5·0 + 2·0.125·1 + 0.5·1 + 2·(-0.125)·3 = 0.25 + 0.5 - 0.75 = 0
        let values = [
            9., 0., 0., /* a0 */ 0., 0., 0., /* v0 */ 1., 0., 0., /* b0 */
            3., 0., 0., /* a1 */ 1., 0., 0., /* v1 */ 9., 0., 0., /* b1 */
        ];
        let ch = channel(
            ChannelPath::Translation,
            Interpolation::CubicSpline,
            &[0.0, 2.0],
            &values,
        );
        assert!(close(sample(&ch, 1.0), Vec4::ZERO));
        // ends are the values, not the tangents
        assert!(close(sample(&ch, 2.0), Vec4::new(1., 0., 0., 0.)));
    }

    fn one_vertex(
        position: [f32; 3],
        normal: [f32; 3],
        joints: [u16; 4],
        weights: [f32; 4],
    ) -> SkinnedPrimitive {
        SkinnedPrimitive {
            name: String::new(),
            skin: 0,
            material: None,
            rest: vec![CookedVertex {
                position,
                texture_coordinates: [0.0; 2],
                normal,
                tangent: [1.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, 1.0],
            }],
            indices: Vec::new(),
            joints: vec![joints],
            weights: vec![weights],
            generated_normals: false,
        }
    }

    #[test]
    fn skin_one_joint_bind_pose_leaves_vertex_unchanged() {
        // M = B⁻¹, so M·B = I
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            Quat::from_rotation_y(0.7),
            Vec3::new(1., 2., 3.),
        );
        let skin_ = Skin {
            joints: vec![0],
            inverse_bind: vec![m.inverse()],
        };
        let prim = one_vertex([0.3, -0.2, 0.9], [0., 1., 0.], [0; 4], [1., 0., 0., 0.]);
        let mut out = Vec::new();
        skin(&prim, &joint_matrices(&skin_, &[m]), &mut out);
        let p = Vec3::from_array(out[0].position);
        assert!(
            (p - Vec3::new(0.3, -0.2, 0.9)).abs().max_element() < 1e-5,
            "{p}"
        );
    }

    #[test]
    fn skin_two_joints_half_weights_average_translations() {
        let joints = [
            Mat4::from_translation(Vec3::X),
            Mat4::from_translation(-Vec3::X),
        ];
        let prim = one_vertex([0., 5., 0.], [0., 1., 0.], [0, 1, 0, 0], [0.5, 0.5, 0., 0.]);
        let mut out = Vec::new();
        skin(&prim, &joints, &mut out);
        assert_eq!(out[0].position, [0., 5., 0.]);
    }

    #[test]
    fn skin_non_uniform_scale_transforms_normal_by_inverse_transpose() {
        // scale x by 2: the plane x + y = 0 (normal (1,1,0)/√2) becomes x/2 + y = 0, normal ∝ (1,2,0)
        let joints = [Mat4::from_scale(Vec3::new(2., 1., 1.))];
        let n = Vec3::new(1., 1., 0.).normalize().to_array();
        let prim = one_vertex([0.; 3], n, [0; 4], [1., 0., 0., 0.]);
        let mut out = Vec::new();
        skin(&prim, &joints, &mut out);
        let expected = Vec3::new(1., 2., 0.).normalize();
        assert!(
            (Vec3::from_array(out[0].normal) - expected)
                .abs()
                .max_element()
                < 1e-5
        );
    }

    #[test]
    fn skin_zero_weights_keep_rest_vertex() {
        let joints = [Mat4::from_translation(Vec3::splat(100.0))];
        let prim = one_vertex([1., 2., 3.], [0., 1., 0.], [0; 4], [0.; 4]);
        let mut out = Vec::new();
        skin(&prim, &joints, &mut out);
        assert_eq!(out[0].position, [1., 2., 3.]);
    }

    #[test]
    fn global_transforms_multiply_parent_first() {
        let t = |x: f32| Trs {
            translation: Vec3::new(x, 0., 0.),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        };
        // child listed before its parent
        let nodes = [
            SkinNode {
                parent: Some(1),
                rest: t(1.0),
            },
            SkinNode {
                parent: None,
                rest: t(10.0),
            },
        ];
        let pose: Vec<Trs> = nodes.iter().map(|n| n.rest).collect();
        let world = global_transforms(&nodes, &pose);
        assert_eq!(world[0].w_axis.x, 11.0);
        assert_eq!(world[1].w_axis.x, 10.0);
    }
}
