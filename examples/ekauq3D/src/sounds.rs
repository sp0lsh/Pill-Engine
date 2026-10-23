//! One-shot sounds (`res/audio`): from LibreQuake (BSD-3) the shotgun, ricochets, axe swing and
//! hit, jump and land; footsteps are synthesized by the asset pipeline (`ProceduralFootstep`: the
//! `footstep*.procedural_footstep` JSON specs cook to `.cooked.wav`; LibreQuake has none). A
//! source plays one sound at a time, so each sound has a few pooled sources used oldest-first.
//! The engine has no audio on wasm: there `play` does nothing.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sfx {
    Fire,
    Ricochet,
    Footstep,
    Jump,
    Land,
    AxeSwing,
    AxeHit,
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::{play, spawn_sounds, SfxSlot};

#[cfg(target_arch = "wasm32")]
pub fn play(
    _engine: &mut pill_engine::game::Engine,
    _sfx: Sfx,
    _position: Option<pill_engine::game::Vector3f>,
) -> pill_engine::game::Result<()> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use pill_engine::{define_component, game::*};

    use super::Sfx;

    // One pooled source: which sound it plays and when it last started (for oldest-first use).
    define_component!(SfxSlot {
        sfx: Sfx,
        last_used: f32,
        variants: Vec<SoundHandle>,
    });

    /// (sound, files, sources, volume, positional)
    const BANK: [(Sfx, &[&str], usize, f32, bool); 7] = [
        (Sfx::Fire, &["guncock"], 3, 0.9, false),
        (Sfx::Ricochet, &["ric1", "ric2", "ric3"], 6, 0.8, true),
        (
            Sfx::Footstep,
            &[
                "footstep1.cooked",
                "footstep2.cooked",
                "footstep3.cooked",
                "footstep4.cooked",
            ],
            4,
            0.35,
            false,
        ),
        (Sfx::Jump, &["plyrjmp8"], 2, 0.7, false),
        (Sfx::Land, &["land"], 2, 0.9, false),
        (Sfx::AxeSwing, &["ax1"], 2, 0.8, false),
        (Sfx::AxeHit, &["axhit2"], 2, 0.9, false),
    ];

    pub fn spawn_sounds(engine: &mut Engine, scene: SceneHandle) -> Result<()> {
        let mut handles: Vec<(&str, SoundHandle)> = Vec::new();
        for (sfx, files, sources, volume, positional) in BANK {
            let mut variants = Vec::with_capacity(files.len());
            for file in files {
                let handle = match handles.iter().find(|(name, _)| name == file) {
                    Some((_, handle)) => handle.clone(),
                    None => {
                        let handle = engine.add_resource(Sound::new(
                            &format!("ekauq3d_{file}"),
                            format!("audio/{file}.wav").into(),
                        ))?;
                        handles.push((file, handle.clone()));
                        handle
                    }
                };
                variants.push(handle);
            }
            for _ in 0..sources {
                engine
                    .build_entity(scene)
                    .with_component(TransformComponent::builder().build())
                    .with_component(
                        AudioSourceComponent::builder()
                            .sound_type(if positional {
                                SoundType::Sound3D
                            } else {
                                SoundType::Sound2D
                            })
                            .sound(variants[0].clone())
                            .volume(volume)
                            .play_on_awake(false)
                            .build(),
                    )
                    .with_component(SfxSlot {
                        sfx,
                        last_used: f32::MIN,
                        variants: variants.clone(),
                    })
                    .build();
            }
        }
        Ok(())
    }

    /// Plays `sfx` (at `position` for positional sounds) on its least recently used source.
    pub fn play(engine: &mut Engine, sfx: Sfx, position: Option<Vector3f>) -> Result<()> {
        let time = engine.get_global_component::<TimeComponent>()?.time;
        let mut oldest = None;
        let mut oldest_time = f32::MAX;
        for (entity, slot) in engine.iterate_one_component::<SfxSlot>()? {
            if slot.sfx == sfx && slot.last_used < oldest_time {
                oldest_time = slot.last_used;
                oldest = Some(entity);
            }
        }
        let Some(chosen) = oldest else {
            return Ok(());
        };
        for (entity, source, transform, slot) in engine
            .iterate_three_components_mut::<AudioSourceComponent, TransformComponent, SfxSlot>()?
        {
            if entity != chosen {
                continue;
            }
            slot.last_used = time;
            // A variant per play, where the sound has several
            if slot.variants.len() > 1 {
                let pick = (time * 1000.0) as usize % slot.variants.len();
                source.set_sound(slot.variants[pick].clone());
            }
            if let Some(position) = position {
                transform.set_position(position);
            }
            source.play();
        }
        Ok(())
    }
}
