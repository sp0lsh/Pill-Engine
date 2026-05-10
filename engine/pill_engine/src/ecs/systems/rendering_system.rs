use crate::{
    config::RENDERING_SYSTEM,
    ecs::{
        CameraAspectRatio, CameraComponent, EntityHandle,
        MeshRenderingComponent, RenderStateComponent, TransformComponent,
    },
    engine::Engine,
    graphics::RenderQueueItem,
};

use pill_core::{warn, EngineError, LogContext, PillSlotMapKey, PillStyle, RendererError, Timer};

use pill_core::{ErrorContext, Result};
use web_time::Instant;

pub fn rendering_system(engine: &mut Engine) -> Result<()> {
    let mut timer = Timer::new();
    timer.begin_context("rendering_system update");

    // First-frame bootstrap: install default pass chain
    let boot_done = engine
        .get_global_component::<RenderStateComponent>()?
        .boot_done;

    if !boot_done {
        let egui_client = engine
            .get_global_component::<RenderStateComponent>()?
            .egui_client
            .clone();
        engine.renderer.init_default_passes(egui_client)?;
        engine
            .get_global_component_mut::<RenderStateComponent>()?
            .boot_done = true;
        return Ok(());
    }

    timer.record("Get active camera");

    let active_scene_handle = engine.scene_manager.get_active_scene_handle()?;
    let mut active_camera_entity_handle_result: Option<EntityHandle> = None;

    {
        let active_scene = engine.scene_manager.get_active_scene_mut()?;

        for (entity_handle, camera_component) in
            active_scene.get_one_component_iterator_mut::<CameraComponent>()?
        {
            if camera_component.enabled {
                if let CameraAspectRatio::Automatic(_) = camera_component.aspect {
                    let aspect_ratio =
                        engine.window_size.width as f32 / engine.window_size.height as f32;
                    camera_component.aspect = CameraAspectRatio::Automatic(aspect_ratio);
                }
                active_camera_entity_handle_result = Some(entity_handle);
                break;
            }
        }
    }

    let active_camera_entity_handle = active_camera_entity_handle_result
        .ok_or_else(|| -> pill_core::PillError { pill_core::PillError::from(EngineError::NoActiveCamera) })?;

    timer.record("Clear render queue");

    engine.render_queue.clear();
    engine.render_queue.reserve(200000);

    timer.record("Prepare render queue");

    let mut _matrix_calculation_duration: f32 = 0.0;
    let mut add_to_render_queue_duration: f32 = 0.0;

    for (entity_handle, _transform_component, mesh_rendering_component) in engine
        .scene_manager
        .get_two_component_iterator_mut::<TransformComponent, MeshRenderingComponent>(
        active_scene_handle,
    )? {
        let add_to_render_queue_start_time = Instant::now();
        if let Some(render_queue_key) = mesh_rendering_component.render_queue_key {
            let render_queue_item = RenderQueueItem {
                key: render_queue_key,
                entity_index: entity_handle.data().index,
            };
            engine.render_queue.push(render_queue_item);
        } else {
            warn!(LogContext::Rendering => "Invalid render queue key");
            continue;
        }
        add_to_render_queue_duration +=
            add_to_render_queue_start_time.elapsed().as_secs_f32() * 1000.0;
    }

    timer.record(format!(
        "Matrix calculation {} ms",
        _matrix_calculation_duration
    ));
    timer.record(format!(
        "Add to render queue {} ms",
        add_to_render_queue_duration
    ));

    timer.record("Sort render queue");

    engine.render_queue.sort();

    timer.record("Get component storages");

    // Build egui UI and push to egui_client
    #[cfg(feature = "debug_ui")]
    {
        use crate::ecs::EguiManagerComponent;
        let egui_ui = EguiManagerComponent::get_ui(engine);
        let egui_client = engine
            .get_global_component::<RenderStateComponent>()?
            .egui_client
            .clone();
        egui_client.set_ui(egui_ui);
    }

    let active_scene = engine.scene_manager.get_active_scene_mut()?;
    let camera_component_storage = active_scene
        .get_component_storage::<CameraComponent>()
        .context(format!(
            "{}: Cannot get active {}",
            "rendering_system".specific_object_style(),
            "Camera".general_object_style()
        ))?;
    let transform_component_storage = active_scene
        .get_component_storage::<TransformComponent>()
        .context(format!(
            "{}: Cannot get {}",
            "rendering_system".specific_object_style(),
            "TransformComponents".specific_object_style()
        ))
        .unwrap();

    timer.begin_context("Render");

    // Render
    let delta_time = engine.frame_delta_time;

    let render_result = engine.renderer.render(
        active_camera_entity_handle,
        &engine.render_queue,
        camera_component_storage,
        transform_component_storage,
        delta_time,
        &mut timer,
        &engine.resource_manager,
    );
    match render_result {
        Ok(_) => {
            timer.end_context()?;
            engine.system_manager.update_system_timer(
                RENDERING_SYSTEM.name,
                RENDERING_SYSTEM.update_phase,
                timer,
            )?;
            Ok(())
        }
        Err(e) => {
            match e.downcast_ref::<RendererError>() {
                Some(RendererError::SurfaceLost) => {
                    timer.end_context()?;
                    engine.system_manager.update_system_timer(
                        RENDERING_SYSTEM.name,
                        RENDERING_SYSTEM.update_phase,
                        timer,
                    )?;
                    engine.renderer.resize(engine.window_size);
                    Ok(())
                }
                Some(RendererError::SurfaceOutOfMemory) => {
                    panic!("Critical: Renderer error, system out of memory");
                }
                _ => Err(e),
            }
        }
    }
}
