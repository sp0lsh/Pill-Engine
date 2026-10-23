//! GPU skin cache, after Unreal's GPU Skin Cache: one compute pass at the start of the frame skins
//! every `GpuSkinningComponent` instance into its own `RendererMesh` vertex streams, so the
//! shadow, GI and PBR passes draw the skinned result without knowing about skinning. Per frame
//! the CPU uploads only the joint matrices of all instances, in one write.

use crate::config::SKINNING_COMPUTE_SHADER;
use crate::ecs::{GpuSkinningComponent, SkinSource};
use crate::graphics::shader_bindings::{skinning_compute, structs};
use crate::graphics::{
    ComputePipeline, ComputePipelineDesc, FrameTarget, Pass, PillRenderer, ShaderDesc, WorldQuery,
};
use crate::resources::MeshVertex;

use pill_core::Result;
use wgpu::util::DeviceExt;

const THREADS: u32 = 64;

pub(crate) struct Instance {
    group: wgpu::BindGroup,
    threads: u32,
}

pub(crate) struct JointBuffer {
    pub buffer: wgpu::Buffer,
    rows: usize,
    pub group: wgpu::BindGroup,
}

#[derive(Default)]
pub struct PassSkinning {
    pipeline: Option<ComputePipeline>,
    /// Group 0 per source, uploaded once
    sources: Vec<wgpu::BindGroup>,
    instances: Vec<Instance>,
    joints: Option<JointBuffer>,
}

impl PassSkinning {
    pub fn new() -> Self {
        Self::default()
    }
}

fn pipeline_desc() -> ComputePipelineDesc<'static> {
    ComputePipelineDesc {
        label: Some("skinning"),
        cs: ShaderDesc {
            source: SKINNING_COMPUTE_SHADER,
            entry_func: "cs_main",
        },
        bind_groups: skinning_compute::layout(),
    }
}

/// The GPU inputs of one skinned primitive; `skin` is its skin. Sources without NORMAL must be
/// one triangle per vertex (flat), whose normals the shader recomputes per triangle.
#[cfg(feature = "gltf_loading")]
pub(crate) fn skin_source(
    primitive: &pill_assets::skinning::SkinnedPrimitive,
    skin: &pill_assets::skinning::Skin,
) -> Result<SkinSource> {
    use pill_assets::skinning::{flat_triangles, skin_bounds};
    let flat = primitive.generated_normals && flat_triangles(primitive);
    if primitive.generated_normals && !flat {
        return Err(pill_core::PillError::from(format!(
            "GPU skinning: {:?} has no NORMAL and is indexed; only one triangle per vertex is supported",
            primitive.name
        )));
    }
    let joint_count = skin.joints.len();
    Ok(SkinSource {
        vertex_count: primitive.rest.len() as u32,
        joint_count: joint_count as u32,
        flat,
        rest_positions: primitive.rest.iter().flat_map(|v| v.position).collect(),
        rest_frames: primitive
            .rest
            .iter()
            .flat_map(|v| [v.normal, v.tangent, v.bitangent].map(|[x, y, z]| [x, y, z, 0.0]))
            .collect(),
        skin_vertices: primitive
            .rest
            .iter()
            .zip(&primitive.joints)
            .map(|(v, j)| {
                let [u, w] = bytemuck::cast::<_, MeshVertex>(*v)
                    .packed_attributes()
                    .texture_coordinates;
                [
                    u32::from(j[0]) | u32::from(j[1]) << 16,
                    u32::from(j[2]) | u32::from(j[3]) << 16,
                    u32::from(u) | u32::from(w) << 16,
                    0,
                ]
            })
            .collect(),
        skin_weights: primitive.weights.clone(),
        bounds: skin_bounds(primitive, joint_count),
    })
}

/// Identity joint matrix in the shader's row layout
pub(crate) const IDENTITY_JOINT_ROWS: [[f32; 4]; 3] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

/// Row-major 3x4 rows (the shader's joint layout) of column-major affine `joint_matrices`
pub(crate) fn write_joint_rows(rows: &mut [[f32; 4]], joint_matrices: &[pill_core::Matrix4f]) {
    for (rows, matrix) in rows.chunks_exact_mut(3).zip(joint_matrices) {
        let transposed = matrix.transpose();
        rows.copy_from_slice(&[
            transposed.x_axis.to_array(),
            transposed.y_axis.to_array(),
            transposed.z_axis.to_array(),
        ]);
    }
}

fn storage(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

pub(crate) fn source_group(
    device: &wgpu::Device,
    pipeline: &ComputePipeline,
    source: &SkinSource,
) -> wgpu::BindGroup {
    skinning_compute::Group0 {
        rest_positions: &storage(
            device,
            "skin_rest_positions",
            bytemuck::cast_slice(&source.rest_positions),
        ),
        rest_frames: &storage(
            device,
            "skin_rest_frames",
            bytemuck::cast_slice(&source.rest_frames),
        ),
        skin_vertices: &storage(
            device,
            "skin_vertices",
            bytemuck::cast_slice(&source.skin_vertices),
        ),
        skin_weights: &storage(
            device,
            "skin_weights",
            bytemuck::cast_slice(&source.skin_weights),
        ),
    }
    .create(device, &pipeline.bind_group_layouts[0])
}

/// Group 1 of one instance: its parameters and the mesh streams it writes; returns the thread count
pub(crate) fn instance_group(
    device: &wgpu::Device,
    pipeline: &ComputePipeline,
    source: &SkinSource,
    joint_offset: u32,
    out_positions: &wgpu::Buffer,
    out_attributes: &wgpu::Buffer,
) -> Instance {
    let parameters = structs::SkinInstance {
        joint_offset,
        vertex_count: source.vertex_count,
        flat: source.flat as u32,
        padding: 0,
    };
    let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skin_instance"),
        contents: bytemuck::bytes_of(&parameters),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    Instance {
        group: skinning_compute::Group1 {
            instance: &parameters,
            out_positions,
            out_attributes,
        }
        .create(device, &pipeline.bind_group_layouts[1]),
        threads: if source.flat {
            source.vertex_count / 3
        } else {
            source.vertex_count
        },
    }
}

pub(crate) fn joint_buffer(
    device: &wgpu::Device,
    pipeline: &ComputePipeline,
    rows: usize,
) -> JointBuffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skin_joint_matrices"),
        size: (rows.max(3) * std::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = skinning_compute::Group2 {
        joint_matrices: &buffer,
    }
    .create(device, &pipeline.bind_group_layouts[2]);
    JointBuffer {
        buffer,
        rows,
        group,
    }
}

/// One dispatch per instance: the joint, source and instance groups, `threads` threads
pub(crate) fn encode(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &ComputePipeline,
    sources: &[wgpu::BindGroup],
    instances: &[(usize, &Instance)],
    joints: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("pass_skinning"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(2, joints, &[]);
    let mut bound_source = usize::MAX;
    for &(source, instance) in instances {
        if source != bound_source {
            pass.set_bind_group(0, &sources[source], &[]);
            bound_source = source;
        }
        pass.set_bind_group(1, &instance.group, &[]);
        pass.dispatch_workgroups(instance.threads.div_ceil(THREADS), 1, 1);
    }
}

impl Pass for PassSkinning {
    fn get_label(&self) -> &str {
        "pass_skinning"
    }

    fn init(&mut self, _renderer: &mut dyn PillRenderer) -> Result<()> {
        Ok(())
    }

    fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &mut dyn PillRenderer,
        _frame_target: &FrameTarget,
        world: &WorldQuery<'_>,
    ) -> Result<()> {
        let skinning = world.get_global::<GpuSkinningComponent>()?;
        if skinning.instances.is_empty() {
            return Ok(());
        }
        if self.pipeline.is_none() {
            self.pipeline = Some(renderer.create_compute_pipeline(pipeline_desc())?);
        }
        let pipeline = self.pipeline.as_ref().expect("created above");
        let device = renderer.get_device();

        for source in &skinning.sources[self.sources.len()..] {
            self.sources.push(source_group(device, pipeline, source));
        }
        for instance in &skinning.instances[self.instances.len()..] {
            let mesh = world.renderer_mesh(&instance.mesh)?;
            self.instances.push(instance_group(
                device,
                pipeline,
                &skinning.sources[instance.source.0],
                instance.joint_offset,
                &mesh.position_buffer,
                &mesh.attribute_buffer,
            ));
        }
        let rows = skinning.joint_rows.len();
        if self.joints.as_ref().is_none_or(|joints| joints.rows < rows) {
            self.joints = Some(joint_buffer(device, pipeline, rows));
        }
        let joints = self.joints.as_ref().expect("created above");
        renderer.get_queue().write_buffer(
            &joints.buffer,
            0,
            bytemuck::cast_slice(&skinning.joint_rows),
        );

        let instances: Vec<(usize, &Instance)> = skinning
            .instances
            .iter()
            .map(|instance| instance.source.0)
            .zip(&self.instances)
            .collect();
        encode(encoder, pipeline, &self.sources, &instances, &joints.group);
        Ok(())
    }
}

#[cfg(all(test, feature = "gltf_loading", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use pill_assets::gltf_skin::parse_gltf_skinned;
    use pill_assets::skinning::{global_transforms, joint_matrices, pose, skin};

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u8> {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
        queue.submit([encoder.finish()]);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.expect("map"));
        device.poll(wgpu::PollType::Wait).expect("poll");
        let bytes = readback.slice(..).get_mapped_range().to_vec();
        bytes
    }

    #[test]
    fn gpu_skin_fox_run_3_instances_flat_matches_cpu_skin_and_packing() {
        compare_with_cpu(true);
    }

    /// Fox with its loader-generated rest normals treated as NORMAL: the per-vertex path with the
    /// inverse-transpose normal
    #[test]
    fn gpu_skin_fox_run_3_instances_with_normals_matches_cpu_skin_and_packing() {
        compare_with_cpu(false);
    }

    fn compare_with_cpu(flat: bool) {
        let Some((device, queue)) = device() else {
            eprintln!("no wgpu adapter: skipped");
            return;
        };
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/skinning/res/models/Fox.glb"
        );
        let data = parse_gltf_skinned(&std::fs::read(path).expect("Fox.glb"), &[]).expect("parse");
        let mut primitive = data.primitives[0].clone();
        primitive.generated_normals = flat;
        let primitive = &primitive;
        let skin_data = &data.skins[primitive.skin];
        let source = skin_source(primitive, skin_data).expect("source");
        assert_eq!(source.flat, flat);
        let pipeline = crate::graphics::build_compute_pipeline(&device, &pipeline_desc());
        let sources = [source_group(&device, &pipeline, &source)];

        // Run (clip 2) at three times, one instance each, at joint offsets 0, 24, 48
        let joint_count = skin_data.joints.len();
        let times = [0.5f32, 0.31, 0.77];
        let joint_sets: Vec<Vec<pill_core::Matrix4f>> = times
            .iter()
            .map(|&t| {
                joint_matrices(
                    skin_data,
                    &global_transforms(&data.nodes, &pose(&data.nodes, &data.animations[2], t)),
                )
            })
            .collect();
        let mut rows = vec![[0.0f32; 4]; times.len() * joint_count * 3];
        for (i, joints) in joint_sets.iter().enumerate() {
            write_joint_rows(
                &mut rows[i * joint_count * 3..(i + 1) * joint_count * 3],
                joints,
            );
        }
        let joints = joint_buffer(&device, &pipeline, rows.len());
        queue.write_buffer(&joints.buffer, 0, bytemuck::cast_slice(&rows));

        let stream = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(source.vertex_count) * 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let outputs: Vec<(wgpu::Buffer, wgpu::Buffer)> = times
            .iter()
            .map(|_| (stream("positions"), stream("attributes")))
            .collect();
        let instances: Vec<Instance> = outputs
            .iter()
            .enumerate()
            .map(|(i, (positions, attributes))| {
                instance_group(
                    &device,
                    &pipeline,
                    &source,
                    (i * joint_count) as u32,
                    positions,
                    attributes,
                )
            })
            .collect();
        let mut encoder = device.create_command_encoder(&Default::default());
        encode(
            &mut encoder,
            &pipeline,
            &sources,
            &instances.iter().map(|i| (0, i)).collect::<Vec<_>>(),
            &joints.group,
        );
        queue.submit([encoder.finish()]);

        let extent = primitive
            .rest
            .iter()
            .flat_map(|v| v.position)
            .fold(0.0f32, |a, x| a.max(x.abs()));
        let unorm10 = |packed: u32, shift: u32| ((packed >> shift) & 1023) as i32;
        let mut cpu = Vec::new();
        for (i, (positions, attributes)) in outputs.iter().enumerate() {
            skin(primitive, &joint_sets[i], &mut cpu);
            let gpu_positions: Vec<[f32; 3]> =
                bytemuck::cast_slice(&read(&device, &queue, positions)).to_vec();
            let gpu_attributes: Vec<[u32; 3]> =
                bytemuck::cast_slice(&read(&device, &queue, attributes)).to_vec();
            let (mut worst_position, mut worst_lsb) = (0.0f32, 0);
            for (v, vertex) in cpu.iter().enumerate() {
                let expected: MeshVertex = bytemuck::cast(*vertex);
                let packed = expected.packed_attributes();
                for axis in 0..3 {
                    worst_position =
                        worst_position.max((gpu_positions[v][axis] - vertex.position[axis]).abs());
                }
                for (gpu, cpu) in [
                    (gpu_attributes[v][0], packed.normal),
                    (gpu_attributes[v][1], packed.tangent),
                ] {
                    for shift in [0, 10, 20] {
                        worst_lsb =
                            worst_lsb.max((unorm10(gpu, shift) - unorm10(cpu, shift)).abs());
                    }
                    assert_eq!(
                        gpu >> 30,
                        cpu >> 30,
                        "instance {i} vertex {v}: tangent code"
                    );
                }
                let uv = packed.texture_coordinates;
                assert_eq!(
                    gpu_attributes[v][2],
                    u32::from(uv[0]) | u32::from(uv[1]) << 16,
                    "instance {i} vertex {v}: uv"
                );
            }
            eprintln!("flat {flat} instance {i}: worst position error {worst_position:e} (extent {extent}), worst packed error {worst_lsb} LSB");
            assert!(
                worst_position <= 1e-4 * extent,
                "instance {i}: position error {worst_position}"
            );
            assert!(worst_lsb <= 1, "instance {i}: packed error {worst_lsb} LSB");
        }
    }
}
