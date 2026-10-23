//! The player's weapons, Quake's slots 1 and 2: the axe (melee) and the shotgun, LibreQuake's
//! `v_axe.mdl` / `v_shot.mdl` view models (BSD-3) following the view. Shotgun: Quake's numbers
//! (6 pellets, spread 0.04 × 0.04, 0.5 s refire, 2048 units range; quakewiki), hitscan against the
//! level's brushes, recoil animation and view punch, muzzle light and flash, impact sparks or dust
//! with a flash of light. Axe: 64 units reach, 0.5 s refire, two alternating swings, a dust puff
//! where it hits a wall. Sounds are native only.

use pill_engine::{
    define_component, define_global_component,
    game::{
        quake_map::{quake_direction_to_engine, quake_to_engine, QuakeCollision},
        quake_mdl::QuakeModel,
        *,
    },
};

use crate::player::{euler_from_columns, Mode, PlayerRig};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeaponKind {
    Axe,
    Shotgun,
}

impl WeaponKind {
    fn slot(self) -> usize {
        match self {
            WeaponKind::Axe => 0,
            WeaponKind::Shotgun => 1,
        }
    }
}

const PELLETS: usize = 6;
const SPREAD: [f32; 2] = [0.04, 0.04];
const RANGE: f32 = 2048.0;
const REFIRE: f32 = 0.5;
/// View punch per shot, degrees up.
const PUNCH: f32 = 2.0;
/// Recoil animation: model frames 1..=6 at 10 frames per second, then frame 0 (rest).
const FIRE_FRAMES: usize = 6;
const FRAME_TIME: f32 = 0.1;
const AXE_RANGE: f32 = 64.0;
const AXE_REFIRE: f32 = 0.5;
/// The axe connects this far into its swing.
const AXE_HIT_TIME: f32 = 0.2;
/// Swing animations: frames 1..=4 and 5..=8, alternating, at 10 frames per second.
const AXE_SWING_FRAMES: usize = 4;
const AXE_PUNCH: f32 = 1.0;
/// Lowering the weapon away, then raising the new one, each this long (seconds).
const SWITCH_TIME: f32 = 0.15;
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
const PARTICLES: usize = 192;
const IMPACT_LIGHTS: usize = 6;
const PARTICLE_GRAVITY: f32 = 9.81;

// The view model entity: the weapon and frame it shows.
define_component!(ViewModel {
    kind: WeaponKind,
    frame: usize,
});

/// A weapon's uploaded view model: a mesh per frame and its material.
pub struct WeaponModel {
    frames: Vec<MeshHandle>,
    material: MaterialHandle,
}

// One pooled particle (hidden while `life` ≤ 0); engine space, meters.
define_component!(Particle {
    velocity: Vector3f,
    life: f32,
    max_life: f32,
    size: f32,
    end_size: f32,
    gravity: f32,
    dust: bool,
});

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
    refire: f32,
    // Seconds since the last attack, while its animation runs
    animation: Option<f32>,
    // Axe: the second swing animation next; the hit of the current swing is still to check
    alternate_swing: bool,
    axe_hit_pending: bool,
    // Seconds until the muzzle light and all effects are over: their per-frame work stops then
    muzzle_left: f32,
    effects_left: f32,
    // Requested by a script (capture runs)
    scripted_fire: bool,
    scripted_select: Option<WeaponKind>,
    random: u32,
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
            refire: 0.0,
            animation: None,
            alternate_swing: false,
            axe_hit_pending: false,
            muzzle_left: 0.0,
            effects_left: 0.0,
            scripted_fire: false,
            scripted_select: None,
            random: 0x9E37_79B9,
        }
    }

    fn model(&self, kind: WeaponKind) -> Option<&WeaponModel> {
        self.models[kind.slot()].as_ref()
    }

    /// Uniform in −1..1 (xorshift).
    fn random_signed(&mut self) -> f32 {
        self.random ^= self.random << 13;
        self.random ^= self.random >> 17;
        self.random ^= self.random << 5;
        (self.random as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn random_range(&mut self, low: f32, high: f32) -> f32 {
        low + (self.random_signed() * 0.5 + 0.5) * (high - low)
    }
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], factor: f32) -> [f32; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let length = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if length > 0.0 {
        scale(a, 1.0 / length)
    } else {
        a
    }
}

/// Hidden (scale 0, far below the level) pooled entities stay out of view and of light lists.
const PARKED: Vector3f = Vector3f::new(0.0, -10_000.0, 0.0);

fn park(transform: &mut TransformComponent) {
    transform.set_position(PARKED);
    transform.set_scale(Vector3f::ZERO);
}

fn upload_model(engine: &mut Engine, prefix: &str, model: QuakeModel) -> Result<WeaponModel> {
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
                dust: false,
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
    Ok(())
}

/// View model placement, firing, recoil animation (player mode).
pub fn weapon_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    let (mode, view, locked_fire, select) = {
        let rig = engine.get_global_component::<PlayerRig>()?;
        let input = engine.get_global_component::<InputComponent>()?;
        let fire = input.cursor_locked() && input.get_mouse_button(MouseButton::Left);
        let select = if input.get_key_pressed(KeyboardKey::Digit1) {
            Some(WeaponKind::Axe)
        } else if input.get_key_pressed(KeyboardKey::Digit2) {
            Some(WeaponKind::Shotgun)
        } else {
            None
        };
        (rig.mode, rig.view, fire, select)
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

    // --- Switching: 1 = axe, 2 = shotgun; lower the weapon, swap, raise the new one ---
    let drop = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let select = select.or(weapon.scripted_select.take());
        if let Some(kind) = select {
            let available = weapon.model(kind).is_some();
            if available && weapon.switching_to.is_none() && kind != weapon.current {
                weapon.switching_to = Some(kind);
                weapon.switch_time = 0.0;
                weapon.animation = None;
            }
        }
        match weapon.switching_to {
            Some(kind) => {
                weapon.switch_time += delta_time;
                if weapon.switch_time >= SWITCH_TIME && weapon.current != kind {
                    weapon.current = kind;
                }
                if weapon.switch_time >= 2.0 * SWITCH_TIME {
                    weapon.switching_to = None;
                    0.0
                } else {
                    1.0 - (weapon.switch_time / SWITCH_TIME - 1.0).abs()
                }
            }
            None => 0.0,
        }
    };

    let mut attacked = None;
    {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        weapon.refire = (weapon.refire - delta_time).max(0.0);
        let trigger = locked_fire || std::mem::take(&mut weapon.scripted_fire);
        if trigger && weapon.refire <= 0.0 && weapon.switching_to.is_none() {
            weapon.refire = match weapon.current {
                WeaponKind::Axe => AXE_REFIRE,
                WeaponKind::Shotgun => REFIRE,
            };
            weapon.animation = Some(0.0);
            if weapon.current == WeaponKind::Axe {
                weapon.alternate_swing = !weapon.alternate_swing;
                weapon.axe_hit_pending = true;
            }
            attacked = Some(weapon.current);
        }
    }
    match attacked {
        Some(WeaponKind::Shotgun) => {
            engine.get_global_component_mut::<PlayerRig>()?.punch(PUNCH);
            fire_pellets(engine, &view)?;
            set_off_muzzle(engine)?;
            crate::sounds::play(engine, crate::sounds::Sfx::Fire, None)?;
        }
        Some(WeaponKind::Axe) => {
            crate::sounds::play(engine, crate::sounds::Sfx::AxeSwing, None)?;
        }
        None => {}
    }

    // --- Animation frame; the axe connects partway through its swing ---
    let (kind, frame, frame_mesh, material, axe_hit) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        let kind = weapon.current;
        let mut axe_hit = false;
        let frame = match weapon.animation.as_mut() {
            Some(time) => {
                *time += delta_time;
                let elapsed = *time;
                let (first, count) = match kind {
                    WeaponKind::Shotgun => (1, FIRE_FRAMES),
                    WeaponKind::Axe if weapon.alternate_swing => (1, AXE_SWING_FRAMES),
                    WeaponKind::Axe => (1 + AXE_SWING_FRAMES, AXE_SWING_FRAMES),
                };
                if kind == WeaponKind::Axe && weapon.axe_hit_pending && elapsed >= AXE_HIT_TIME {
                    weapon.axe_hit_pending = false;
                    axe_hit = true;
                }
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
            axe_hit,
        )
    };
    if axe_hit {
        swing_axe(engine, &view)?;
    }

    // --- Placement: at the eye along the view, bobbing forward with the head, dropped while
    // switching; offsets scaled with the model ---
    let position = add(
        view.eye,
        scale(
            add(
                scale(view.forward, view.bob * 0.4),
                scale(view.up, -drop * SWITCH_DROP),
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
    let muzzle = add(
        view.eye,
        scale(
            add(
                scale(view.forward, MUZZLE[0]),
                add(scale(view.right, MUZZLE[1]), scale(view.up, MUZZLE[2])),
            ),
            VIEW_MODEL_SCALE,
        ),
    );
    let muzzle = Vector3f::from_array(quake_to_engine(muzzle));
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

struct Impact {
    point: [f32; 3],
    normal: [f32; 3],
    metal: bool,
    // Axe: a small puff, no light and no ricochet
    melee: bool,
}

fn is_metal(texture: &str) -> bool {
    ["met", "tech", "tek", "steel", "iron"]
        .iter()
        .any(|word| texture.contains(word))
}

/// What a trace from `from` to `to` hits, if anything.
fn trace_impact(
    collision: &QuakeCollision,
    from: [f32; 3],
    to: [f32; 3],
    melee: bool,
) -> Option<Impact> {
    let trace = collision.trace_ray(from, to);
    if trace.fraction >= 1.0 || trace.start_solid {
        return None;
    }
    let texture = trace
        .texture
        .map(|id| collision.texture_name(id))
        .unwrap_or("");
    Some(Impact {
        point: trace.end,
        normal: trace.normal,
        metal: is_metal(texture),
        melee,
    })
}

/// The axe's hit: a short trace ahead of the eye; a puff, a thud and a small punch on a wall.
fn swing_axe(engine: &mut Engine, view: &crate::player::ViewPose) -> Result<()> {
    let impact = {
        let state = engine.get_global_component::<crate::game::LevelState>()?;
        let Some(collision) = state.collision.as_ref() else {
            return Ok(());
        };
        let reach = add(view.eye, scale(view.forward, AXE_RANGE));
        let Some(impact) = trace_impact(collision, view.eye, reach, true) else {
            return Ok(());
        };
        impact
    };
    spawn_impact(engine, &impact)?;
    engine
        .get_global_component_mut::<PlayerRig>()?
        .punch(AXE_PUNCH);
    crate::sounds::play(engine, crate::sounds::Sfx::AxeHit, None)
}

/// Six pellets from the eye, spread over the view's right and up axes; effects where they hit.
fn fire_pellets(engine: &mut Engine, view: &crate::player::ViewPose) -> Result<()> {
    let mut impacts = Vec::with_capacity(PELLETS);
    {
        let directions: Vec<[f32; 3]> = {
            let weapon = engine.get_global_component_mut::<Weapon>()?;
            (0..PELLETS)
                .map(|_| {
                    let spread_right = weapon.random_signed() * SPREAD[0];
                    let spread_up = weapon.random_signed() * SPREAD[1];
                    normalize(add(
                        view.forward,
                        add(scale(view.right, spread_right), scale(view.up, spread_up)),
                    ))
                })
                .collect()
        };
        let state = engine.get_global_component::<crate::game::LevelState>()?;
        let Some(collision) = state.collision.as_ref() else {
            return Ok(());
        };
        impacts.extend(directions.into_iter().filter_map(|direction| {
            trace_impact(
                collision,
                view.eye,
                add(view.eye, scale(direction, RANGE)),
                false,
            )
        }));
    }
    for impact in &impacts {
        spawn_impact(engine, impact)?;
    }
    Ok(())
}

/// Sparks (metal) or dust, a short flash of light and maybe a ricochet sound.
fn spawn_impact(engine: &mut Engine, impact: &Impact) -> Result<()> {
    let point = Vector3f::from_array(quake_to_engine(impact.point));
    let normal = Vector3f::from_array(quake_direction_to_engine(impact.normal));
    let (sparks, dust) = match (impact.melee, impact.metal) {
        (true, true) => (4, 2),
        (true, false) => (0, 5),
        (false, true) => (7, 3),
        (false, false) => (3, 6),
    };

    // Particle parameters first (the random state lives in Weapon)
    let mut emitted = Vec::with_capacity(sparks + dust);
    let (spark_material, dust_material, ricochet) = {
        let weapon = engine.get_global_component_mut::<Weapon>()?;
        for index in 0..sparks + dust {
            let is_dust = index >= sparks;
            let random = Vector3f::new(
                weapon.random_signed(),
                weapon.random_signed(),
                weapon.random_signed(),
            );
            let (speed, spread, life, size, end_size, gravity) = if is_dust {
                (
                    weapon.random_range(0.3, 0.9),
                    0.5,
                    weapon.random_range(0.4, 0.7),
                    0.02,
                    weapon.random_range(0.05, 0.08),
                    1.0,
                )
            } else {
                (
                    weapon.random_range(1.5, 4.0),
                    2.0,
                    weapon.random_range(0.2, 0.45),
                    weapon.random_range(0.008, 0.016),
                    0.0,
                    PARTICLE_GRAVITY,
                )
            };
            emitted.push(Particle {
                velocity: normal * speed + random * spread,
                life,
                max_life: life,
                size,
                end_size,
                gravity,
                dust: is_dust,
            });
        }
        let ricochet = !impact.melee && weapon.random_signed() > 0.33;
        let longest = emitted
            .iter()
            .fold(0.0f32, |longest, particle| longest.max(particle.life));
        weapon.effects_left = weapon.effects_left.max(longest + 0.1);
        (
            weapon.spark_material.clone(),
            weapon.dust_material.clone(),
            ricochet,
        )
    };

    let mut queue = emitted.into_iter();
    for (_, transform, rendering, particle) in engine
        .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, Particle>()?
    {
        if particle.life > 0.0 {
            continue;
        }
        let Some(next) = queue.next() else {
            break;
        };
        let material = if next.dust {
            dust_material.as_ref()
        } else {
            spark_material.as_ref()
        };
        if let Some(material) = material.filter(|m| rendering.material_handle.as_ref() != Some(*m))
        {
            rendering.set_material(material);
        }
        *particle = next;
        transform.set_position(point);
        transform.set_scale(Vector3f::splat(particle.size));
    }

    if impact.melee {
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
        particle.velocity.y -= particle.gravity * delta_time;
        particle.velocity *= 1.0 - (if particle.dust { 2.5 } else { 0.8 }) * delta_time;
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
