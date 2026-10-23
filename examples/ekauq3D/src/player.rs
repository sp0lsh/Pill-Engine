//! FPS player: Quake movement (engine `quake_player`) against the level's brush collision, mouse
//! look with a locked cursor, and the view feel: head bob, strafe roll, landing dip and weapon
//! punch. N toggles between this and the free-fly spectator camera.
//!
//! Bob shape after Quake's documented view variables (cl_bob 0.02, cl_bobcycle 0.6, cl_bobup
//! 0.5) and roll (cl_rollangle 2, cl_rollspeed 200); written fresh, no GPL code.

use pill_engine::{
    define_global_component,
    game::{
        quake_map::{engine_to_quake, quake_direction_to_engine, quake_to_engine, QuakeCollision},
        quake_player::{
            player_move, trace_player, MoveSettings, PlayerInput, PlayerState, EYE_HEIGHT,
        },
        *,
    },
};

use crate::game::{FlyCamera, LevelState};

/// Degrees of view rotation per pixel of mouse movement.
const MOUSE_SENSITIVITY: f32 = 0.12;
/// Quake's look limits.
const MAX_PITCH: f32 = 80.0;
/// Longest movement step (stable stepping and landing at low frame rates).
const MAX_MOVE_STEP: f32 = 1.0 / 72.0;
const BOB_AMOUNT: f32 = 0.02;
const BOB_CYCLE: f32 = 0.6;
const BOB_UP: f32 = 0.5;
const ROLL_ANGLE: f32 = 2.0;
const ROLL_SPEED: f32 = 200.0;
/// Weapon punch decay, degrees per second.
const PUNCH_DECAY: f32 = 10.0;
/// Landing dip spring (stiffness, damping) and impulse per unit/s of landing speed.
const DIP_STIFFNESS: f32 = 80.0;
const DIP_DAMPING: f32 = 14.0;
const DIP_IMPULSE: f32 = 0.35;
/// Landings faster than this (units/s; a hop lands at about 270) dip the view and sound heavy.
pub const HARD_LANDING_SPEED: f32 = 300.0;
/// Footsteps only above this ground speed (units/s).
const FOOTSTEP_MIN_SPEED: f32 = 100.0;

/// Shows the crosshair (the egui overlay has no engine access): set in player mode.
pub static CROSSHAIR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Spectator,
    Player,
}

/// The view of this frame, in Quake space (Z up, units): where the weapon and shots come from.
#[derive(Clone, Copy, Default)]
pub struct ViewPose {
    pub eye: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    /// Current head bob (units), for the weapon's own bob.
    pub bob: f32,
}

/// What happened this frame, for sounds.
#[derive(Clone, Copy, Default)]
pub struct PlayerEvents {
    pub footstep: bool,
    pub jumped: bool,
    pub landed_speed: f32,
}

define_global_component!(PlayerRig {
    mode: Mode,
    state: PlayerState,
    // Quake view angles in degrees: yaw counter-clockwise from +X, pitch up positive
    yaw: f32,
    pitch: f32,
    bob_time: f32,
    dip: f32,
    dip_velocity: f32,
    punch: f32,
    view: ViewPose,
    events: PlayerEvents,
    // Scripted input instead of the devices (capture runs)
    scripted: Option<PlayerInput>,
});

impl PlayerRig {
    pub fn new() -> Self {
        Self {
            mode: Mode::Spectator,
            state: PlayerState::default(),
            yaw: 0.0,
            pitch: 0.0,
            bob_time: 0.0,
            dip: 0.0,
            dip_velocity: 0.0,
            punch: 0.0,
            view: ViewPose::default(),
            events: PlayerEvents::default(),
            scripted: None,
        }
    }

    /// Kicks the view up by `degrees` (weapon recoil).
    pub fn punch(&mut self, degrees: f32) {
        self.punch = degrees;
    }
}

/// `FlyCamera` yaw of a Quake yaw: FlyCamera yaw 0 looks down −Z, Quake yaw 0 down +X (engine +X).
pub fn fly_yaw_from_quake(yaw: f32) -> f32 {
    yaw - 90.0
}

/// Quake view basis from yaw and pitch (degrees, pitch up positive): forward, right, up.
fn view_basis(yaw: f32, pitch: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    let forward = [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ];
    let right = [yaw.sin(), -yaw.cos(), 0.0];
    let up = [
        -pitch.sin() * yaw.cos(),
        -pitch.sin() * yaw.sin(),
        pitch.cos(),
    ];
    (forward, right, up)
}

/// `TransformComponent::rotation` (degrees; Rx·Ry·Rz) of a rotation matrix given by its columns.
pub fn euler_from_columns(columns: [[f32; 3]; 3]) -> Vector3f {
    // m[row][column]; Rx(a)·Ry(b)·Rz(c) has m02 = sin b, m12 = −sin a·cos b, m22 = cos a·cos b,
    // m01 = −cos b·sin c, m00 = cos b·cos c
    let m = |row: usize, column: usize| columns[column][row];
    let b = m(0, 2).clamp(-1.0, 1.0).asin();
    let a = (-m(1, 2)).atan2(m(2, 2));
    let c = (-m(0, 1)).atan2(m(0, 0));
    Vector3f::new(a.to_degrees(), b.to_degrees(), c.to_degrees())
}

fn rotate_about(vector: [f32; 3], axis: [f32; 3], degrees: f32) -> [f32; 3] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    let cross = [
        axis[1] * vector[2] - axis[2] * vector[1],
        axis[2] * vector[0] - axis[0] * vector[2],
        axis[0] * vector[1] - axis[1] * vector[0],
    ];
    let dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
    std::array::from_fn(|i| vector[i] * cos + cross[i] * sin + axis[i] * dot * (1.0 - cos))
}

/// Switches between spectator and player (N). The player starts where the camera is, dropped
/// to the floor; the spectator starts at the player's eye.
pub fn toggle_mode(engine: &mut Engine) -> Result<()> {
    let mut camera = None;
    for (_, transform, fly) in engine.iterate_two_components::<TransformComponent, FlyCamera>()? {
        camera = Some((transform.position, fly.yaw, fly.pitch));
    }
    let Some((camera_position, fly_yaw, fly_pitch)) = camera else {
        return Ok(());
    };
    let mode = engine.get_global_component::<PlayerRig>()?.mode;
    match mode {
        Mode::Player => {
            let (yaw, pitch) = {
                CROSSHAIR.store(false, std::sync::atomic::Ordering::Relaxed);
                let rig = engine.get_global_component_mut::<PlayerRig>()?;
                rig.mode = Mode::Spectator;
                (rig.yaw, rig.pitch)
            };
            engine
                .get_global_component_mut::<InputComponent>()?
                .set_cursor_locked(false);
            for (_, fly) in engine.iterate_one_component_mut::<FlyCamera>()? {
                fly.yaw = fly_yaw_from_quake(yaw);
                fly.pitch = pitch;
            }
        }
        Mode::Spectator => {
            let state = engine.get_global_component::<LevelState>()?;
            let Some(collision) = state.collision.as_ref() else {
                return Ok(());
            };
            let eye = engine_to_quake(camera_position.to_array());
            let spawns: Vec<[f32; 3]> = state
                .spawns
                .iter()
                .map(|spawn| engine_to_quake(spawn.position.to_array()))
                .collect();
            let origin =
                drop_to_floor(collision, [eye[0], eye[1], eye[2] - EYE_HEIGHT]).or_else(|| {
                    spawns.iter().find_map(|spawn| {
                        drop_to_floor(collision, [spawn[0], spawn[1], spawn[2] - EYE_HEIGHT])
                    })
                });
            let Some(origin) = origin else {
                log::warn!("ekauq3D: no free spot for the player");
                return Ok(());
            };
            CROSSHAIR.store(true, std::sync::atomic::Ordering::Relaxed);
            let rig = engine.get_global_component_mut::<PlayerRig>()?;
            rig.mode = Mode::Player;
            rig.state = PlayerState {
                origin,
                ..PlayerState::default()
            };
            rig.yaw = fly_yaw + 90.0; // inverse of fly_yaw_from_quake
            rig.pitch = fly_pitch.clamp(-MAX_PITCH, MAX_PITCH);
            rig.dip = 0.0;
            rig.dip_velocity = 0.0;
            rig.punch = 0.0;
            engine
                .get_global_component_mut::<InputComponent>()?
                .set_cursor_locked(true);
        }
    }
    engine
        .get_global_component_mut::<RenderStateComponent>()?
        .reset_motion_history();
    Ok(())
}

/// The box at `origin` moved down onto the floor below it, if it isn't stuck in a wall.
fn drop_to_floor(collision: &QuakeCollision, origin: [f32; 3]) -> Option<[f32; 3]> {
    let down = trace_player(
        collision,
        origin,
        [origin[0], origin[1], origin[2] - 4096.0],
    );
    (!down.start_solid && down.fraction < 1.0).then_some(down.end)
}

/// Mouse look, movement and the view of the player (player mode only).
pub fn player_system(engine: &mut Engine) -> Result<()> {
    if engine.get_global_component::<PlayerRig>()?.mode != Mode::Player {
        return Ok(());
    }
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);

    // --- Input: Esc frees the cursor, a click takes it back ---
    let (mouse_delta, keys, jump, locked) = {
        let input = engine.get_global_component::<InputComponent>()?;
        let axis = |positive: KeyboardKey, negative: KeyboardKey| {
            input.get_key(positive) as i32 as f32 - input.get_key(negative) as i32 as f32
        };
        let keys = (
            axis(KeyboardKey::KeyW, KeyboardKey::KeyS),
            axis(KeyboardKey::KeyD, KeyboardKey::KeyA),
        );
        let escape = input.get_key_pressed(KeyboardKey::Escape);
        let click = input.get_mouse_button_pressed(MouseButton::Left);
        let mut locked = input.cursor_locked();
        if escape {
            locked = false;
        } else if click {
            locked = true;
        }
        (
            input.get_mouse_delta(),
            keys,
            input.get_key(KeyboardKey::Space),
            locked,
        )
    };
    engine
        .get_global_component_mut::<InputComponent>()?
        .set_cursor_locked(locked);

    let state = engine.get_global_component::<LevelState>()?;
    let Some(collision) = state.collision.as_ref() else {
        return Ok(());
    };
    let rig = engine.get_global_component::<PlayerRig>()?;
    let mut rig_state = rig.state;
    let (mut yaw, mut pitch) = (rig.yaw, rig.pitch);
    let input = match rig.scripted {
        Some(scripted) => {
            yaw = scripted.yaw;
            scripted
        }
        None => {
            if locked {
                yaw -= mouse_delta.x * MOUSE_SENSITIVITY;
                pitch = (pitch - mouse_delta.y * MOUSE_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
            }
            PlayerInput {
                forward: keys.0,
                side: keys.1,
                jump,
                yaw,
            }
        }
    };

    let settings = MoveSettings::default();
    let steps = (delta_time / MAX_MOVE_STEP).ceil().max(1.0);
    let step = delta_time / steps;
    let mut events = PlayerEvents::default();
    for _ in 0..steps as usize {
        let moved = player_move(&mut rig_state, &input, step, collision, &settings);
        events.jumped |= moved.jumped;
        events.landed_speed = events.landed_speed.max(moved.landed_speed);
    }

    // --- View feel ---
    let rig = engine.get_global_component_mut::<PlayerRig>()?;
    rig.state = rig_state;
    rig.yaw = yaw.rem_euclid(360.0);
    rig.pitch = pitch;
    let velocity = rig_state.velocity;
    let ground_speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1]).sqrt();

    // Bob: a cycle of BOB_CYCLE seconds, rising over its first BOB_UP part; scaled by speed
    let previous_half = (rig.bob_time / (BOB_CYCLE * 0.5)).floor();
    rig.bob_time += delta_time;
    let mut cycle = (rig.bob_time / BOB_CYCLE).fract();
    cycle = if cycle < BOB_UP {
        std::f32::consts::PI * cycle / BOB_UP
    } else {
        std::f32::consts::PI + std::f32::consts::PI * (cycle - BOB_UP) / (1.0 - BOB_UP)
    };
    let amplitude = ground_speed * BOB_AMOUNT;
    let bob = if rig_state.on_ground {
        (amplitude * 0.3 + amplitude * 0.7 * cycle.sin()).clamp(-7.0, 4.0)
    } else {
        0.0
    };
    // A footstep at each half cycle while running on the ground
    let half = (rig.bob_time / (BOB_CYCLE * 0.5)).floor();
    events.footstep =
        half != previous_half && rig_state.on_ground && ground_speed > FOOTSTEP_MIN_SPEED;

    // Landing dip: a damped spring kicked by the landing speed; not for a normal hop's landing
    // (about 270 units/s) or one jumped straight out of (buffered press), so hopping stays smooth
    if events.landed_speed > HARD_LANDING_SPEED && !events.jumped {
        rig.dip_velocity -= events.landed_speed * DIP_IMPULSE;
    }
    rig.dip_velocity += (-rig.dip * DIP_STIFFNESS - rig.dip_velocity * DIP_DAMPING) * delta_time;
    rig.dip += rig.dip_velocity * delta_time;

    rig.punch = if rig.punch > 0.0 {
        (rig.punch - PUNCH_DECAY * delta_time).max(0.0)
    } else {
        (rig.punch + PUNCH_DECAY * delta_time).min(0.0)
    };

    // Roll: lean into strafing
    let (_, right, _) = view_basis(rig.yaw, 0.0);
    let side = velocity[0] * right[0] + velocity[1] * right[1];
    let roll = side.signum() * side.abs().min(ROLL_SPEED) / ROLL_SPEED * ROLL_ANGLE;

    let origin = rig_state.origin;
    let eye = [origin[0], origin[1], origin[2] + EYE_HEIGHT + bob + rig.dip];
    let (forward, right, up) = view_basis(rig.yaw, rig.pitch + rig.punch);
    let (right, up) = (
        rotate_about(right, forward, roll),
        rotate_about(up, forward, roll),
    );
    rig.view = ViewPose {
        eye,
        forward,
        right,
        up,
        bob,
    };
    rig.events = events;

    let position = Vector3f::from_array(quake_to_engine(eye));
    let columns = [
        quake_direction_to_engine(right),
        quake_direction_to_engine(up),
        quake_direction_to_engine(forward.map(|value| -value)),
    ];
    let rotation = euler_from_columns(columns);
    for (_, transform, _) in engine.iterate_two_components_mut::<TransformComponent, FlyCamera>()? {
        transform.set_position(position);
        transform.set_rotation(rotation);
    }
    Ok(())
}

/// Crosshair in player mode: a small white plus in the middle of the screen.
pub fn crosshair(ctx: &egui::Context) {
    let center = ctx.screen_rect().center();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("ekauq3d.crosshair"),
    ));
    let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
    for (from, to) in [
        (egui::vec2(-7.0, 0.0), egui::vec2(-2.5, 0.0)),
        (egui::vec2(2.5, 0.0), egui::vec2(7.0, 0.0)),
        (egui::vec2(0.0, -7.0), egui::vec2(0.0, -2.5)),
        (egui::vec2(0.0, 2.5), egui::vec2(0.0, 7.0)),
    ] {
        painter.line_segment([center + from, center + to], stroke);
    }
}
