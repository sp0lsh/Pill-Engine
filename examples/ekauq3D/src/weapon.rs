//! What the player's weapons look like, Quake's slots 1 and 2: the axe and the shotgun,
//! LibreQuake's `v_axe.mdl` / `v_shot.mdl` view models (BSD-3) following the view; recoil
//! animation and view punch, muzzle light and flash, two alternating axe swings; impact sparks,
//! dust, blood or explosions with a flash of light, for every fighter's shots. The rules (refire,
//! pellets, reach, damage) are the deathmatch's (`sim::combat`, numbers in `sim::weapons`).
//! Sounds are native only.

use pill_engine::{
    define_component, define_global_component,
    game::{
        quake_map::{quake_direction_to_engine, quake_to_engine},
        quake_mdl::QuakeModel,
        *,
    },
};

use crate::{
    player::{euler_from_columns, Mode, PlayerRig},
    sim::{add, combat::ImpactKind, scale, Rng},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeaponKind {
    Axe,
    Shotgun,
}

impl WeaponKind {
    /// The weapon's slot in `sim::weapons::WEAPONS`.
    pub fn slot(self) -> usize {
        match self {
            WeaponKind::Axe => 0,
            WeaponKind::Shotgun => 1,
        }
    }

    pub fn from_slot(slot: usize) -> Self {
        if slot == 0 {
            WeaponKind::Axe
        } else {
            WeaponKind::Shotgun
        }
    }
}

/// View punch per shot, degrees up.
const PUNCH: f32 = 2.0;
/// Recoil animation: model frames 1..=6 at 10 frames per second, then frame 0 (rest).
const FIRE_FRAMES: usize = 6;
const FRAME_TIME: f32 = 0.1;
/// Swing animations: frames 1..=4 and 5..=8, alternating, at 10 frames per second.
const AXE_SWING_FRAMES: usize = 4;
const AXE_PUNCH: f32 = 1.0;
/// Lowering the weapon away, then raising the new one: each half the deathmatch's switch.
const SWITCH_TIME: f32 = crate::sim::combat::SWITCH_TIME * 0.5;
/// How far the weapon drops while switching (units).
const SWITCH_DROP: f32 = 24.0;
/// View models are drawn this much smaller, scaled about the eye: the same on screen (perspective
/// doesn't change), but inside the player's box, so walls in front can't hide them (the engine has
/// no separate view-model pass, as Quake's depth range trick).
const VIEW_MODEL_SCALE: f32 = 0.3;
/// Muzzle, relative to the eye in Quake units (along forward, right, up): the barrel's end.
const MUZZLE: [f32; 3] = [26.0, 0.0, -7.0];
const MUZZLE_FLASH_TIME: f32 = 0.06;
/// The muzzle light sits this far ahead of the eye (units): it lights the room in front, not the
/// view model a hand's width away.
const MUZZLE_LIGHT_AHEAD: f32 = 48.0;
const PARTICLES: usize = 400;
/// Tracer streak and impact mark lifetimes (seconds); marks shrink away over the last second.
const TRACER_TIME: f32 = 0.05;
const DECAL_TIME: f32 = 12.0;
/// View-model recoil: a spring (stiffness, damping) kicked back by each shot (units/s).
const RECOIL_STIFFNESS: f32 = 260.0;
const RECOIL_DAMPING: f32 = 16.0;
const RECOIL_KICK: f32 = 90.0;
const IMPACT_LIGHTS: usize = 6;
const PARTICLE_GRAVITY: f32 = 9.81;

// The view model entity: the weapon and frame it shows.
define_component!(ViewModel {
    kind: WeaponKind,
    frame: usize,
});

/// An uploaded model: a mesh per frame and its material.
pub struct WeaponModel {
    pub frames: Vec<MeshHandle>,
    pub material: MaterialHandle,
}

// One pooled particle (hidden while `life` ≤ 0); engine space, meters.
define_component!(Particle {
    // Motion (sparks, dust), or the direction a streak runs or a mark faces
    velocity: Vector3f,
    life: f32,
    max_life: f32,
    // Sparks and dust: size growing or shrinking to `end_size`; a streak's width and length, a
    // mark's size
    size: f32,
    end_size: f32,
    gravity: f32,
    kind: ParticleKind,
});

/// What a pooled particle is: flying sparks or dust (dust slows faster), a tracer streak, a mark
/// on a wall.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParticleKind {
    Spark,
    Dust,
    Streak,
    Mark,
}

// A light that fades out over `time` seconds after being set off.
define_component!(Flash {
    life: f32,
    time: f32,
    peak: f32,
    muzzle: bool,
});

// The emissive flash quad at the muzzle.
define_component!(MuzzleFlash { life: f32 });

define_global_component!(Weapon {
    // By slot (`WeaponKind::slot`); `None` if its model failed to load
    models: [Option<WeaponModel>; 2],
    current: WeaponKind,
    // Switching to this weapon: `switch_time` runs 0..2·SWITCH_TIME (down, then up)
    switching_to: Option<WeaponKind>,
    switch_time: f32,
    spark_material: Option<MaterialHandle>,
    dust_material: Option<MaterialHandle>,
    blood_material: Option<MaterialHandle>,
    brass_material: Option<MaterialHandle>,
    gore_material: Option<MaterialHandle>,
    tracer_material: Option<MaterialHandle>,
    mark_material: Option<MaterialHandle>,
    // View-model recoil (units back along the view) and its velocity
    recoil: f32,
    recoil_velocity: f32,
    // Seconds since the last attack, while its animation runs
    animation: Option<f32>,
    // Axe: the second swing animation next
    alternate_swing: bool,
    // From the deathmatch this frame: the player attacked, switched, the axe connected
    fired: Option<WeaponKind>,
    switched: Option<WeaponKind>,
    axe_hit: bool,
    // Seconds until the muzzle light and all effects are over: their per-frame work stops then
    muzzle_left: f32,
    effects_left: f32,
    // Requested by a script (capture runs)
    scripted_fire: bool,
    scripted_select: Option<WeaponKind>,
    random: Rng,
});

impl Weapon {
    pub fn new() -> Self {
        Self {
            models: [None, None],
            current: WeaponKind::Shotgun,
            switching_to: None,
            switch_time: 0.0,
            spark_material: None,
            dust_material: None,
            blood_material: None,
            brass_material: None,
            gore_material: None,
            tracer_material: None,
            mark_material: None,
            recoil: 0.0,
            recoil_velocity: 0.0,
            animation: None,
            alternate_swing: false,
            fired: None,
            switched: None,
            axe_hit: false,
            muzzle_left: 0.0,
            effects_left: 0.0,
            scripted_fire: false,
            scripted_select: None,
            random: Rng::new(1),
        }
    }

    fn model(&self, kind: WeaponKind) -> Option<&WeaponModel> {
        self.models[kind.slot()].as_ref()
    }

    /// The player's attack, switch and axe hit of this frame (from the deathmatch's events).
    pub fn player_fired(&mut self, slot: usize) {
        self.fired = Some(WeaponKind::from_slot(slot));
    }

    pub fn player_switched(&mut self, slot: usize) {
        self.switched = Some(WeaponKind::from_slot(slot));
    }

    pub fn player_axe_hit(&mut self) {
        self.axe_hit = true;
    }

    /// Fire and weapon choice requested by a capture script, taken once.
    pub fn take_script(&mut self) -> (bool, Option<usize>) {
        (
            std::mem::take(&mut self.scripted_fire),
            self.scripted_select.take().map(WeaponKind::slot),
        )
    }
}

/// Hidden (scale 0, far below the level) pooled entities stay out of view and of light lists.
const PARKED: Vector3f = Vector3f::new(0.0, -10_000.0, 0.0);

fn park(transform: &mut TransformComponent) {
    transform.set_position(PARKED);
    transform.set_scale(Vector3f::ZERO);
}

pub fn upload_model(engine: &mut Engine, prefix: &str, model: QuakeModel) -> Result<WeaponModel> {
    let QuakeModel {
        texture,
        material,
        frames,
    } = model;
    let skin = engine.add_resource(Texture::from_bytes_owned(
        &texture.name,
        texture.kind,
        texture.rtex,
    ))?;
    let material = engine.add_resource(
        Material::new(&material.name)
            .albedo_texture(skin)
            .metallic(material.metallic)
            .roughness(material.roughness)
            .double_sided(material.double_sided),
    )?;
    let mut frame_meshes = Vec::with_capacity(frames.len());
    for frame in &frames {
        frame_meshes.push(engine.add_resource(Mesh::from_cooked_mesh_bytes(
            &format!("{prefix}_{}", frame.name),
            &frame.rmsh,
        )?)?);
    }
    Ok(WeaponModel {
        frames: frame_meshes,
        material,
    })
}

/// Uploads the weapons (either may be missing) and creates the effect pools.
pub fn spawn_weapons(
    engine: &mut Engine,
    scene: SceneHandle,
    axe: Option<QuakeModel>,
    shotgun: Option<QuakeModel>,
) -> Result<()> {
    let axe = axe
        .map(|model| upload_model(engine, "v_axe", model))
        .transpose()?;
    let shotgun = shotgun
        .map(|model| upload_model(engine, "v_shot", model))
        .transpose()?;
    let current = if shotgun.is_some() {
        WeaponKind::Shotgun
    } else {
        WeaponKind::Axe
    };
    let shown = match current {
        WeaponKind::Axe => axe.as_ref(),
        WeaponKind::Shotgun => shotgun.as_ref(),
    };
    let Some(shown) = shown else {
        return Ok(());
    };
    engine
        .build_entity(scene)
        .with_component(TransformComponent::builder().position(PARKED).build())
        .with_component(
            MeshRenderingComponent::builder()
                .mesh(&shown.frames[0])
                .material(&shown.material)
                .cast_shadows(false)
                .build(),
        )
        .with_component(ViewModel {
            kind: current,
            frame: 0,
        })
        .build();

    let cube = engine.add_resource(Mesh::cube("ekauq3d_particle", 1.0))?;
    let spark_material = engine.add_resource(
        Material::new("ekauq3d_spark")
            .albedo(Color::new(0.0, 0.0, 0.0))
            .emissive(Color::new(4.0, 2.2, 0.7))
            .blend_mode(BlendMode::Additive),
    )?;
    let dust_material = engine.add_resource(
        Material::new("ekauq3d_dust")
            .albedo(Color::new(0.35, 0.32, 0.28))
            .roughness(1.0)
            .opacity(0.4),
    )?;
    let blood_material = engine.add_resource(
        Material::new("ekauq3d_blood")
            .albedo(Color::new(0.45, 0.02, 0.02))
            .roughness(0.6)
            .opacity(0.8),
    )?;
    let brass_material = engine.add_resource(
        Material::new("ekauq3d_brass")
            .albedo(Color::new(0.75, 0.55, 0.2))
            .metallic(0.9)
            .roughness(0.35),
    )?;
    let gore_material = engine.add_resource(
        Material::new("ekauq3d_gore")
            .albedo(Color::new(0.3, 0.02, 0.02))
            .roughness(0.5),
    )?;
    let tracer_material = engine.add_resource(
        Material::new("ekauq3d_tracer")
            .albedo(Color::new(0.0, 0.0, 0.0))
            .emissive(Color::new(6.0, 4.0, 1.6))
            .blend_mode(BlendMode::Additive),
    )?;
    let decal_material = engine.add_resource(
        // Black: no diffuse, no reflection (opaque; the translucent pass loses thin quads)
        Material::new("ekauq3d_impact_mark")
            .albedo(Color::new(0.0, 0.0, 0.0))
            .metallic(1.0)
            .roughness(1.0),
    )?;
    let flash_material = engine.add_resource(
        Material::new("ekauq3d_muzzle_flash")
            .albedo(Color::new(0.0, 0.0, 0.0))
            .emissive(Color::new(20.0, 12.0, 4.0))
            .blend_mode(BlendMode::Additive),
    )?;
    for _ in 0..PARTICLES {
        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(PARKED)
                    .scale(Vector3f::ZERO)
                    .build(),
            )
            .with_component(
                MeshRenderingComponent::builder()
                    .mesh(&cube)
                    .material(&spark_material)
                    .cast_shadows(false)
                    .build(),
            )
            .with_component(Particle {
                velocity: Vector3f::ZERO,
                life: 0.0,
                max_life: 1.0,
                size: 0.0,
                end_size: 0.0,
                gravity: 0.0,
                kind: ParticleKind::Spark,
            })
            .build();
    }
    engine
        .build_entity(scene)
        .with_component(
            TransformComponent::builder()
                .position(PARKED)
                .scale(Vector3f::ZERO)
                .build(),
        )
        .with_component(
            MeshRenderingComponent::builder()
                .mesh(&cube)
                .material(&flash_material)
                .cast_shadows(false)
                .build(),
        )
        .with_component(MuzzleFlash { life: 0.0 })
        .build();
    for index in 0..=IMPACT_LIGHTS {
        let muzzle = index == 0;
        engine
            .build_entity(scene)
            .with_component(TransformComponent::builder().position(PARKED).build())
            .with_component(
                LightComponent::builder()
                    .color(if muzzle {
                        [1.0, 0.75, 0.45]
                    } else {
                        [1.0, 0.6, 0.3]
                    })
                    .intensity(0.0)
                    .range(if muzzle { 6.0 } else { 2.5 })
                    .build(),
            )
            .with_component(Flash {
                life: 0.0,
                time: if muzzle { MUZZLE_FLASH_TIME } else { 0.08 },
                peak: if muzzle { 5.0 } else { 1.5 },
                muzzle,
            })
            .build();
    }

    let weapon = engine.get_global_component_mut::<Weapon>()?;
    weapon.models = [axe, shotgun];
    weapon.current = current;
    weapon.spark_material = Some(spark_material);
    weapon.dust_material = Some(dust_material);
    weapon.blood_material = Some(blood_material);
    weapon.brass_material = Some(brass_material);
    weapon.gore_material = Some(gore_material);
    weapon.tracer_material = Some(tracer_material);
    weapon.mark_material = Some(decal_material);
    Ok(())
}

/// View model placement, attack and switch animations (player mode), from the deathmatch's
/// events of this frame.
pub fn weapon_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    let (mode, view) = {
        let rig = engine.get_global_component::<PlayerRig>()?;
        (rig.mode, rig.view)
    };
    if engine
        .get_global_component::<Weapon>()?
        .models
        .iter()
        .all(Option::is_none)
    {
        return Ok(());
    }
    if mode != Mode::Player {
        for (_, transform, _) in
            engine.iterate_two_components_mut::<TransformComponent, ViewModel>()?
        {
            park(transform);
        }
        return Ok(());
    }

    // --- Switching: lower the weapon, swap, raise the new one (the deathmatch's switch time) ---
    let (drop, attacked, axe_hit) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        if let Some(kind) = weapon.switched.take() {
            if kind != weapon.current {
                weapon.switching_to = Some(kind);
                weapon.switch_time = 0.0;
                weapon.animation = None;
            }
        }
        let drop = match weapon.switching_to {
            Some(kind) => {
                weapon.switch_time += delta_time;
                if weapon.switch_time >= SWITCH_TIME && weapon.current != kind {
                    weapon.current = kind;
                }
                if weapon.switch_time >= 2.0 * SWITCH_TIME {
                    weapon.switching_to = None;
                    0.0
                } else {
                    let linear = 1.0 - (weapon.switch_time / SWITCH_TIME - 1.0).abs();
                    linear * linear * (3.0 - 2.0 * linear)
                }
            }
            None => 0.0,
        };
        let attacked = weapon.fired.take();
        if let Some(kind) = attacked {
            weapon.current = kind;
            weapon.switching_to = None;
            weapon.animation = Some(0.0);
            if kind == WeaponKind::Axe {
                weapon.alternate_swing = !weapon.alternate_swing;
            }
        }
        (drop, attacked, std::mem::take(&mut weapon.axe_hit))
    };
    match attacked {
        Some(WeaponKind::Shotgun) => {
            engine.get_global_component_mut::<PlayerRig>()?.punch(PUNCH);
            engine.get_global_component_mut::<Weapon>()?.recoil_velocity -= RECOIL_KICK;
            eject_shell(engine, &view)?;
            set_off_muzzle(engine)?;
            crate::sounds::play(engine, crate::sounds::Sfx::Fire, None)?;
        }
        Some(WeaponKind::Axe) => {
            crate::sounds::play(engine, crate::sounds::Sfx::AxeSwing, None)?;
        }
        None => {}
    }
    if axe_hit {
        engine
            .get_global_component_mut::<PlayerRig>()?
            .punch(AXE_PUNCH);
        crate::sounds::play(engine, crate::sounds::Sfx::AxeHit, None)?;
    }

    // --- Animation frame ---
    let (kind, frame, frame_mesh, material) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let kind = weapon.current;
        let frame = match weapon.animation.as_mut() {
            Some(time) => {
                *time += delta_time;
                let elapsed = *time;
                let (first, count) = match kind {
                    WeaponKind::Shotgun => (1, FIRE_FRAMES),
                    WeaponKind::Axe if weapon.alternate_swing => (1, AXE_SWING_FRAMES),
                    WeaponKind::Axe => (1 + AXE_SWING_FRAMES, AXE_SWING_FRAMES),
                };
                let step = (elapsed / FRAME_TIME) as usize;
                if step < count {
                    first + step
                } else {
                    weapon.animation = None;
                    0
                }
            }
            None => 0,
        };
        let model = weapon.model(kind);
        (
            kind,
            frame,
            model.and_then(|model| model.frames.get(frame).cloned()),
            model.map(|model| model.material.clone()),
        )
    };

    // --- Placement: at the eye along the view, bobbing forward with the head, dropped while
    // switching; offsets scaled with the model ---
    let recoil = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        weapon.recoil_velocity += (-weapon.recoil * RECOIL_STIFFNESS
            - weapon.recoil_velocity * RECOIL_DAMPING)
            * delta_time;
        weapon.recoil += weapon.recoil_velocity * delta_time;
        weapon.recoil
    };
    // Recoil: back along the view and a little up (`recoil` is negative while kicked)
    let position = add(
        view.eye,
        scale(
            add(
                scale(view.forward, view.bob * 0.4 + recoil),
                scale(view.up, -drop * SWITCH_DROP - recoil * 0.4),
            ),
            VIEW_MODEL_SCALE,
        ),
    );
    let position = Vector3f::from_array(quake_to_engine(position));
    // Mesh axes: +X along the model's forward, +Y up, +Z its right (engine-converted Quake)
    let rotation = euler_from_columns([
        quake_direction_to_engine(view.forward),
        quake_direction_to_engine(view.up),
        quake_direction_to_engine(view.right),
    ]);
    for (_, transform, rendering, model) in engine
        .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, ViewModel>()?
    {
        transform.set_position(position);
        transform.set_rotation(rotation);
        transform.set_scale(Vector3f::splat(VIEW_MODEL_SCALE));
        if model.kind != kind {
            if let Some(material) = material.as_ref() {
                rendering.set_material(material);
            }
            model.kind = kind;
            model.frame = usize::MAX;
        }
        if let Some(mesh) = frame_mesh.as_ref().filter(|_| model.frame != frame) {
            rendering.set_mesh(mesh);
            model.frame = frame;
        }
    }

    // --- Muzzle flash quad, facing the view ---
    let muzzle = Vector3f::from_array(quake_to_engine(muzzle_point(&view)));
    let flash_rotation = euler_from_columns([
        quake_direction_to_engine(view.right),
        quake_direction_to_engine(view.up),
        quake_direction_to_engine(view.forward.map(|value| -value)),
    ]);
    for (_, transform, flash) in
        engine.iterate_two_components_mut::<TransformComponent, MuzzleFlash>()?
    {
        flash.life -= delta_time;
        if flash.life > 0.0 {
            let size = 0.22 * VIEW_MODEL_SCALE * (0.6 + flash.life / MUZZLE_FLASH_TIME * 0.4);
            transform.set_position(muzzle);
            transform.set_rotation(flash_rotation);
            transform.set_scale(Vector3f::new(size, size, 0.01));
        } else {
            park(transform);
        }
    }
    let muzzle_active = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        weapon.muzzle_left -= delta_time;
        weapon.muzzle_left > 0.0
    };
    if !muzzle_active {
        return Ok(());
    }
    // Ahead of the eye, but kept 16 units off a wall in front
    let ahead = {
        let state = engine.get_global_component::<crate::game::LevelState>()?;
        let reach = state.collision.as_ref().map_or(1.0, |collision| {
            collision
                .trace_ray(
                    view.eye,
                    add(view.eye, scale(view.forward, MUZZLE_LIGHT_AHEAD)),
                )
                .fraction
        });
        (reach * MUZZLE_LIGHT_AHEAD - 16.0).max(0.0)
    };
    let muzzle_light =
        Vector3f::from_array(quake_to_engine(add(view.eye, scale(view.forward, ahead))));
    for (_, transform, flash) in engine.iterate_two_components_mut::<TransformComponent, Flash>()? {
        if flash.muzzle && flash.life > 0.0 {
            transform.set_position(muzzle_light);
        }
    }
    Ok(())
}

fn set_off_muzzle(engine: &mut Engine) -> Result<()> {
    {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        weapon.muzzle_left = MUZZLE_FLASH_TIME;
        weapon.effects_left = weapon.effects_left.max(MUZZLE_FLASH_TIME + 0.1);
    }
    for (_, flash) in engine.iterate_one_component_mut::<MuzzleFlash>()? {
        flash.life = MUZZLE_FLASH_TIME;
    }
    for (_, flash) in engine.iterate_one_component_mut::<Flash>()? {
        if flash.muzzle {
            flash.life = flash.time;
        }
    }
    Ok(())
}

/// Where a shot hit, for its effect.
pub struct Impact {
    pub point: [f32; 3],
    pub normal: [f32; 3],
    pub metal: bool,
    pub kind: ImpactKind,
}

pub fn is_metal(texture: &str) -> bool {
    ["met", "tech", "tek", "steel", "iron"]
        .iter()
        .any(|word| texture.contains(word))
}

/// Sparks (metal) or dust, blood on a fighter, a burst for an explosion; a short flash of light
/// and maybe a ricochet sound.
pub fn spawn_impact(engine: &mut Engine, impact: &Impact) -> Result<()> {
    let point = Vector3f::from_array(quake_to_engine(impact.point));
    let normal = Vector3f::from_array(quake_direction_to_engine(impact.normal));
    let melee = impact.kind == ImpactKind::Melee;
    let flesh = impact.kind == ImpactKind::Flesh;
    let (sparks, dust) = match (impact.kind, impact.metal) {
        (ImpactKind::Melee, true) => (4, 2),
        (ImpactKind::Melee, false) => (0, 5),
        (ImpactKind::Flesh, _) => (0, 6),
        (ImpactKind::Explosion, _) => (24, 14),
        (ImpactKind::Wall, true) => (7, 3),
        (ImpactKind::Wall, false) => (3, 6),
    };

    // Particle parameters first (the random state lives in Weapon)
    let mut emitted = Vec::with_capacity(sparks + dust);
    let (spark_material, dust_material, ricochet) = {
        let explosion = impact.kind == ImpactKind::Explosion;
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        for index in 0..sparks + dust {
            let is_dust = index >= sparks;
            let random = Vector3f::new(
                weapon.random.signed(),
                weapon.random.signed(),
                weapon.random.signed(),
            );
            let burst = if explosion { 3.0 } else { 1.0 };
            let (speed, spread, life, size, end_size, gravity) = if is_dust {
                (
                    weapon.random.range(0.3, 0.9),
                    0.5,
                    weapon.random.range(0.4, 0.7),
                    0.02,
                    weapon.random.range(0.05, 0.08),
                    1.0,
                )
            } else {
                (
                    weapon.random.range(1.5, 4.0),
                    2.0,
                    weapon.random.range(0.2, 0.45),
                    weapon.random.range(0.008, 0.016),
                    0.0,
                    PARTICLE_GRAVITY,
                )
            };
            emitted.push(Particle {
                velocity: (normal * speed + random * spread) * burst,
                life,
                max_life: life,
                size,
                end_size,
                gravity,
                kind: if is_dust {
                    ParticleKind::Dust
                } else {
                    ParticleKind::Spark
                },
            });
        }
        let ricochet = impact.kind == ImpactKind::Wall && weapon.random.signed() > 0.33;
        let dust_material = if flesh {
            weapon.blood_material.clone()
        } else {
            weapon.dust_material.clone()
        };
        (weapon.spark_material.clone(), dust_material, ricochet)
    };

    let emitted = emitted
        .into_iter()
        .map(|particle| {
            let material = if particle.kind == ParticleKind::Dust {
                dust_material.clone()
            } else {
                spark_material.clone()
            };
            (particle, material)
        })
        .collect();
    emit(engine, point, emitted)?;

    if melee || flesh {
        return Ok(());
    }
    // The oldest impact light
    let mut chosen = None;
    let mut lowest = f32::MAX;
    for (entity, flash) in engine.iterate_one_component::<Flash>()? {
        if !flash.muzzle && flash.life < lowest {
            lowest = flash.life;
            chosen = Some(entity);
        }
    }
    for (entity, transform, flash) in
        engine.iterate_two_components_mut::<TransformComponent, Flash>()?
    {
        if Some(entity) == chosen {
            flash.life = flash.time;
            // Half a meter off the surface: a soft glow, not an inverse-square hot spot
            transform.set_position(point + normal * 0.5);
        }
    }

    if ricochet {
        crate::sounds::play(engine, crate::sounds::Sfx::Ricochet, Some(point))?;
    }
    Ok(())
}

/// Starts pooled particles at `point` (engine space), each with its material.
fn emit(
    engine: &mut Engine,
    point: Vector3f,
    particles: Vec<(Particle, Option<MaterialHandle>)>,
) -> Result<()> {
    {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let longest = particles
            .iter()
            .fold(0.0f32, |longest, (particle, _)| longest.max(particle.life));
        weapon.effects_left = weapon.effects_left.max(longest + 0.1);
    }
    let mut queue = particles.into_iter();
    for (_, transform, rendering, particle) in engine
        .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, Particle>()?
    {
        if particle.life > 0.0 {
            continue;
        }
        let Some((next, material)) = queue.next() else {
            break;
        };
        if let Some(material) = material.filter(|m| rendering.material_handle.as_ref() != Some(m)) {
            rendering.set_material(&material);
        }
        *particle = next;
        transform.set_position(point);
        match particle.kind {
            ParticleKind::Spark | ParticleKind::Dust => {
                transform.set_scale(Vector3f::splat(particle.size));
            }
            ParticleKind::Streak => {
                transform.set_rotation(along(particle.velocity));
                transform.set_scale(Vector3f::new(
                    particle.end_size,
                    particle.size,
                    particle.size,
                ));
            }
            ParticleKind::Mark => {
                transform.set_rotation(along(particle.velocity));
                transform.set_scale(Vector3f::new(0.002, particle.size, particle.size));
            }
        }
    }
    Ok(())
}

/// The muzzle in Quake space: the barrel's end, scaled with the view model.
fn muzzle_point(view: &crate::player::ViewPose) -> [f32; 3] {
    add(
        view.eye,
        scale(
            add(
                scale(view.forward, MUZZLE[0]),
                add(scale(view.right, MUZZLE[1]), scale(view.up, MUZZLE[2])),
            ),
            VIEW_MODEL_SCALE,
        ),
    )
}

/// A spent shell flung to the right of the player's shotgun.
fn eject_shell(engine: &mut Engine, view: &crate::player::ViewPose) -> Result<()> {
    let (particle, material) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let right = Vector3f::from_array(quake_direction_to_engine(view.right));
        let up = Vector3f::from_array(quake_direction_to_engine(view.up));
        let life = 0.8;
        let particle = Particle {
            velocity: right * weapon.random.range(1.5, 2.2) + up * weapon.random.range(1.0, 1.6),
            life,
            max_life: life,
            size: 0.012,
            end_size: 0.012,
            gravity: PARTICLE_GRAVITY,
            kind: ParticleKind::Spark,
        };
        (particle, weapon.brass_material.clone())
    };
    let start = add(
        muzzle_point(view),
        scale(view.forward, -12.0 * VIEW_MODEL_SCALE),
    );
    emit(
        engine,
        Vector3f::from_array(quake_to_engine(start)),
        vec![(particle, material)],
    )
}

/// A killed fighter bursts: a blood cloud and dark chunks thrown up and falling (Quake space).
pub fn spawn_kill_burst(engine: &mut Engine, point: [f32; 3]) -> Result<()> {
    let particles = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let mut particles = Vec::with_capacity(22);
        for index in 0..22 {
            let chunk = index >= 14;
            let random = Vector3f::new(
                weapon.random.signed(),
                weapon.random.signed().abs() + 0.3,
                weapon.random.signed(),
            );
            let life = if chunk {
                weapon.random.range(0.8, 1.3)
            } else {
                weapon.random.range(0.5, 0.9)
            };
            let particle = Particle {
                velocity: random * if chunk { 3.0 } else { 1.2 },
                life,
                max_life: life,
                size: if chunk {
                    weapon.random.range(0.025, 0.045)
                } else {
                    0.04
                },
                end_size: if chunk { 0.02 } else { 0.12 },
                gravity: if chunk { PARTICLE_GRAVITY } else { 1.0 },
                kind: if chunk {
                    ParticleKind::Spark
                } else {
                    ParticleKind::Dust
                },
            };
            let material = if chunk {
                weapon.gore_material.clone()
            } else {
                weapon.blood_material.clone()
            };
            particles.push((particle, material));
        }
        particles
    };
    emit(
        engine,
        Vector3f::from_array(quake_to_engine(point)),
        particles,
    )
}

/// Rotation (degrees) turning the unit cube's X axis along `direction` (engine space).
fn along(direction: Vector3f) -> Vector3f {
    let x = direction.normalize_or_zero();
    let helper = if x.y.abs() > 0.9 {
        Vector3f::X
    } else {
        Vector3f::Y
    };
    let z = x.cross(helper).normalize_or_zero();
    let y = z.cross(x);
    euler_from_columns([x.to_array(), y.to_array(), z.to_array()])
}

/// Streaks from the player's muzzle to where the pellets hit (Quake space).
pub fn spawn_tracers(
    engine: &mut Engine,
    view: &crate::player::ViewPose,
    hits: &[[f32; 3]],
) -> Result<()> {
    let start = Vector3f::from_array(quake_to_engine(muzzle_point(view)));
    let material = engine
        .get_global_component::<Weapon>()?
        .tracer_material
        .clone();
    for hit in hits {
        let end = Vector3f::from_array(quake_to_engine(*hit));
        let span = end - start;
        // The far part of the shot: a streak, not a beam out of the barrel
        let streak = Particle {
            velocity: span,
            life: TRACER_TIME,
            max_life: TRACER_TIME,
            size: 0.006,
            end_size: span.length() * 0.6,
            gravity: 0.0,
            kind: ParticleKind::Streak,
        };
        emit(engine, end - span * 0.3, vec![(streak, material.clone())])?;
    }
    Ok(())
}

/// A dark mark on the wall where a shot hit (Quake space).
pub fn spawn_decal(engine: &mut Engine, point: [f32; 3], normal: [f32; 3]) -> Result<()> {
    let (size, material) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        (
            weapon.random.range(0.08, 0.12),
            weapon.mark_material.clone(),
        )
    };
    let normal = Vector3f::from_array(quake_direction_to_engine(normal));
    let mark = Particle {
        velocity: normal,
        life: DECAL_TIME,
        max_life: DECAL_TIME,
        size,
        end_size: size,
        gravity: 0.0,
        kind: ParticleKind::Mark,
    };
    let point = Vector3f::from_array(quake_to_engine(point)) + normal * 0.004;
    emit(engine, point, vec![(mark, material)])
}

/// Moves, shrinks or grows, and retires particles; fades the flash lights (both modes).
pub fn effects_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        if weapon.effects_left <= 0.0 {
            return Ok(());
        }
        weapon.effects_left -= delta_time;
    }
    for (_, transform, particle) in
        engine.iterate_two_components_mut::<TransformComponent, Particle>()?
    {
        if particle.life <= 0.0 {
            continue;
        }
        particle.life -= delta_time;
        if particle.life <= 0.0 {
            park(transform);
            continue;
        }
        match particle.kind {
            ParticleKind::Streak => {
                let width = particle.size * particle.life / particle.max_life;
                transform.set_scale(Vector3f::new(particle.end_size, width, width));
                continue;
            }
            ParticleKind::Mark => {
                if particle.life < 1.0 {
                    let size = particle.size * particle.life;
                    transform.set_scale(Vector3f::new(0.002, size, size));
                }
                continue;
            }
            ParticleKind::Spark | ParticleKind::Dust => {}
        }
        particle.velocity.y -= particle.gravity * delta_time;
        let drag = if particle.kind == ParticleKind::Dust {
            2.5
        } else {
            0.8
        };
        particle.velocity *= 1.0 - drag * delta_time;
        transform.set_position(transform.position + particle.velocity * delta_time);
        let age = 1.0 - particle.life / particle.max_life;
        let size = particle.size + (particle.end_size - particle.size) * age;
        transform.set_scale(Vector3f::splat(size.max(0.0)));
    }
    for (_, transform, light, flash) in
        engine.iterate_three_components_mut::<TransformComponent, LightComponent, Flash>()?
    {
        if flash.life > 0.0 {
            flash.life -= delta_time;
            light.intensity = flash.peak * (flash.life / flash.time).max(0.0);
        } else if light.intensity != 0.0 {
            light.intensity = 0.0;
            transform.set_position(PARKED);
        }
    }
    Ok(())
}
