//! Plays every `SkeletalAnimationComponent`: gives a new entity its own skinned mesh (on its
//! `MeshRenderingComponent`), then each frame samples the entity's clip at its time, builds the
//! joint matrices and hands them to the GPU skin cache (`graphics::pass_skinning`), refitting the
//! mesh's culling bounds from the joints. Runs after the game systems, before rendering.

use crate::{
    ecs::{
        GpuSkinningComponent, MeshRenderingComponent, SkeletalAnimationComponent, SkinInstance,
        SkinInstanceId,
    },
    engine::Engine,
    graphics::pass_skinning::{write_joint_rows, IDENTITY_JOINT_ROWS},
    resources::{Mesh, MeshData, MeshHandle},
};

use pill_assets::skinning::{global_transforms, joint_matrices, pose, skinned_bounds};
use pill_core::{PillError, Result};
use web_time::Instant;

pub fn skeletal_animation_system(engine: &mut Engine) -> Result<()> {
    let started = Instant::now();
    // Scenes that don't register the component have nothing to animate
    let Ok(components) =
        engine.iterate_two_components_mut::<SkeletalAnimationComponent, MeshRenderingComponent>()
    else {
        return Ok(());
    };
    let new: Vec<(usize, usize)> = components
        .filter(|(_, animation, _)| animation.instance.is_none())
        .map(|(_, animation, _)| (animation.model.0, animation.primitive))
        .collect();
    if !new.is_empty() {
        let mut created = Vec::with_capacity(new.len());
        let mut failure = None;
        for (model, primitive) in new {
            match add_instance(engine, model, primitive) {
                Ok(instance) => created.push(instance),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        // Same iteration order as above: nothing was added or removed in between. Instances
        // created before a failure are still assigned, so the next frame doesn't recreate them.
        let components = engine
            .iterate_two_components_mut::<SkeletalAnimationComponent, MeshRenderingComponent>()?
            .filter(|(_, animation, _)| animation.instance.is_none());
        for ((_, animation, mesh_rendering), (mesh, instance)) in components.zip(created) {
            animation.instance = Some(instance);
            mesh_rendering.set_mesh(&mesh);
        }
        if let Some(error) = failure {
            return Err(error);
        }
    }

    let playing: Vec<(SkinInstanceId, usize, usize, usize, f32, bool)> = engine
        .iterate_one_component::<SkeletalAnimationComponent>()?
        .filter_map(|(_, animation)| {
            animation.instance.map(|instance| {
                (
                    instance,
                    animation.model.0,
                    animation.primitive,
                    animation.animation,
                    animation.time,
                    animation.looping,
                )
            })
        })
        .collect();
    #[cfg(not(feature = "headless"))]
    let mut bounds = Vec::with_capacity(playing.len());
    let skinning = engine.get_global_component_mut::<GpuSkinningComponent>()?;
    for (instance, model, primitive, animation, time, looping) in playing {
        let data = &skinning.models[model].data;
        let clip = data.animations.get(animation).ok_or_else(|| {
            PillError::from(format!(
                "Skeletal animation: model has no animation {animation}"
            ))
        })?;
        let time = if looping && clip.duration > 0.0 {
            time.rem_euclid(clip.duration)
        } else {
            time
        };
        let skin = &data.skins[data.primitives[primitive].skin];
        let joints = joint_matrices(
            skin,
            &global_transforms(&data.nodes, &pose(&data.nodes, clip, time)),
        );
        let entry = &skinning.instances[instance.0];
        let first = entry.joint_offset as usize * 3;
        write_joint_rows(
            &mut skinning.joint_rows[first..first + joints.len() * 3],
            &joints,
        );
        #[cfg(not(feature = "headless"))]
        bounds.push((
            entry.renderer_mesh,
            skinned_bounds(&skinning.sources[entry.source.0].bounds, &joints),
        ));
    }
    #[cfg(not(feature = "headless"))]
    for (renderer_mesh, bounds) in bounds {
        engine
            .resource_manager
            .get_resource_mut::<crate::renderer::resources::RendererMesh>(&renderer_mesh)?
            .local_bounds = bounds;
    }
    engine
        .get_global_component_mut::<GpuSkinningComponent>()?
        .cpu_ms = started.elapsed().as_secs_f32() * 1e3;
    Ok(())
}

/// A mesh resource with the primitive's rest vertices (kept as `Mesh::data` on the CPU, e.g. for
/// GI) and its skin cache instance, at identity joints (the rest pose) until posed
fn add_instance(
    engine: &mut Engine,
    model: usize,
    primitive: usize,
) -> Result<(MeshHandle, SkinInstanceId)> {
    let skinning = engine.get_global_component::<GpuSkinningComponent>()?;
    let skeletal_model = skinning
        .models
        .get(model)
        .ok_or_else(|| PillError::from("Skeletal animation: unknown model"))?;
    let source = *skeletal_model.sources.get(primitive).ok_or_else(|| {
        PillError::from(format!(
            "Skeletal animation: model has no primitive {primitive}"
        ))
    })?;
    let rest = &skeletal_model.data.primitives[primitive];
    let mesh_data = MeshData {
        vertices: bytemuck::cast_slice(&rest.rest).to_vec(),
        indices: rest.indices.clone(),
        trace_lod: None,
    };
    let name = format!("skinned_mesh_{}", skinning.instances.len());
    let mesh = engine.add_resource(Mesh::from_data(&name, mesh_data))?;
    #[cfg(not(feature = "headless"))]
    let renderer_mesh = engine
        .resource_manager
        .get_resource_handle::<crate::renderer::resources::RendererMesh>(&name)?;

    let skinning = engine.get_global_component_mut::<GpuSkinningComponent>()?;
    let joint_count = skinning.sources[source.0].joint_count as usize;
    let joint_offset = (skinning.joint_rows.len() / 3) as u32;
    for _ in 0..joint_count {
        skinning.joint_rows.extend(IDENTITY_JOINT_ROWS);
    }
    skinning.instances.push(SkinInstance {
        mesh,
        #[cfg(not(feature = "headless"))]
        renderer_mesh,
        source,
        joint_offset,
    });
    Ok((mesh, SkinInstanceId(skinning.instances.len() - 1)))
}
