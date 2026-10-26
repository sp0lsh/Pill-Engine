//! FPS player: mouse look with a locked cursor and movement keys (the deathmatch moves the player,
//! Quake movement against the level's brushes, at its fixed tick), and the view feel: head bob,
//! strafe roll, landing dip and weapon punch. N toggles between this and the free-fly spectator
//! camera.
//!
//! Bob shape after Quake's documented view variables (cl_bob 0.02, cl_bobcycle 0.6, cl_bobup
//! 0.5) and roll (cl_rollangle 2, cl_rollspeed 200); written fresh, no GPL code.

use pill_engine::{
    define_global_component,
    game::{
        quake_map::{engine_to_quake, quake_direction_to_engine, quake_to_engine},
        quake_player::{MoveEvents, PlayerInput, PlayerState, EYE_HEIGHT},
        *,
    },
};

use crate::{
    game::{FlyCamera, LevelState},
    sim::{arena::drop_to_floor, combat::MAX_PITCH, view_basis},
};

/// Degrees of view rotation per pixel of mouse movement.
const MOUSE_SENSITIVITY: f32 = 0.12;
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
/// Hit kick: a spring (stiffness, damping) on pitch and roll, kicked by damage and blasts.
const KICK_STIFFNESS: f32 = 160.0;
const KICK_DAMPING: f32 = 14.0;
/// Landings faster than this (units/s; a hop lands at about 270) dip the view and sound heavy.
pub const HARD_LANDING_SPEED: f32 = 300.0;
/// Footsteps only above this ground speed (units/s).
const FOOTSTEP_MIN_SPEED: f32 = 100.0;
/// The view when dead: the head knocked up by the shot (units, seconds), falling to rest this
/// far below the origin (about 8 above the floor) at `DEATH_LAND_TIME`, rebounding off the
/// floor; turning toward where the killer goes (at most `DEATH_MAX_TURN` degrees from the view
/// at death, rate per second), pitched and rolled over (Quake's dead view rolls 80°). The hit
/// tumbles the head (degrees per axis, a damped spring: rad/s, damping ratio); landing shakes
/// it along the view's own up axis (units, degrees of nod, Hz, decay seconds), so the shake is
/// vertical on screen however the head lies.
const DEATH_KNOCK: f32 = 4.0;
const DEATH_KNOCK_TIME: f32 = 0.1;
const DEATH_REST_HEIGHT: f32 = -16.0;
const DEATH_LAND_TIME: f32 = 0.7;
const DEATH_REBOUND: f32 = 3.0;
const DEATH_REBOUND_TIME: f32 = 0.25;
const DEATH_TUMBLE: [f32; 3] = [25.0, 18.0, 22.0];
const DEATH_TUMBLE_RATE: f32 = 11.0;
const DEATH_TUMBLE_DAMPING: f32 = 0.45;
const DEATH_SHAKE_LIFT: f32 = 2.5;
const DEATH_SHAKE_NOD: f32 = 4.0;
const DEATH_SHAKE_RATE: f32 = 14.0;
const DEATH_SHAKE_DECAY: f32 = 0.25;
const DEATH_MAX_TURN: f32 = 90.0;
const DEATH_TURN_RATE: f32 = 4.0;
const DEATH_PITCH: f32 = -10.0;
const DEATH_ROLL: f32 = 80.0;

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
    /// The dead player's head hit the floor.
    pub head_landed: bool,
}

/// The dead player's view: seconds since the death, the view then, the killer and the yaw
/// the head turns toward, the yaw turned so far.
#[derive(Clone, Copy)]
pub struct DeathView {
    time: f32,
    yaw: f32,
    pitch: f32,
    pub killer: Option<usize>,
    target_yaw: f32,
    turned: f32,
    /// Tumble directions.
    seed: f32,
}

impl DeathView {
    pub fn new(yaw: f32, pitch: f32, killer: Option<usize>, seed: f32) -> Self {
        Self {
            time: 0.0,
            yaw,
            pitch,
            killer,
            target_yaw: yaw,
            turned: yaw,
            seed,
        }
    }

    /// Turns the head toward `yaw` (degrees), at most `DEATH_MAX_TURN` from the view at death.
    pub fn look_toward(&mut self, yaw: f32) {
        let turn = wrap_degrees(yaw - self.yaw).clamp(-DEATH_MAX_TURN, DEATH_MAX_TURN);
        self.target_yaw = self.yaw + turn;
    }

    /// Eye height over the origin, yaw, pitch and roll at `time`.
    fn pose(&self) -> (f32, f32, f32, f32) {
        let smooth = |x: f32| {
            let x = x.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let time = self.time;
        let height = if time < DEATH_KNOCK_TIME {
            EYE_HEIGHT + DEATH_KNOCK * (std::f32::consts::FRAC_PI_2 * time / DEATH_KNOCK_TIME).sin()
        } else if time < DEATH_LAND_TIME {
            // Falling: gravity-like, from the knock's top to the rest height
            let fall = (time - DEATH_KNOCK_TIME) / (DEATH_LAND_TIME - DEATH_KNOCK_TIME);
            let top = EYE_HEIGHT + DEATH_KNOCK;
            top + (DEATH_REST_HEIGHT - top) * fall * fall
        } else {
            let rebound = (time - DEATH_LAND_TIME) / DEATH_REBOUND_TIME;
            DEATH_REST_HEIGHT
                + if rebound < 1.0 {
                    DEATH_REBOUND * (std::f32::consts::PI * rebound).sin()
                } else {
                    0.0
                }
        };
        // The hit's tumble: each axis swings to its offset and overshoots a little (step
        // response of a damped spring), directions from the seed
        let damped = DEATH_TUMBLE_RATE * (1.0 - DEATH_TUMBLE_DAMPING.powi(2)).sqrt();
        let response = 1.0
            - (-DEATH_TUMBLE_DAMPING * DEATH_TUMBLE_RATE * time).exp()
                * ((damped * time).cos()
                    + DEATH_TUMBLE_DAMPING * DEATH_TUMBLE_RATE / damped * (damped * time).sin());
        let tumble: [f32; 3] = std::array::from_fn(|axis| {
            let sign = (self.seed * (axis as f32 + 1.7)).sin().signum();
            sign * DEATH_TUMBLE[axis] * response
        });
        let over = smooth(time / DEATH_LAND_TIME);
        (
            height,
            self.turned + tumble[0],
            self.pitch + (DEATH_PITCH - self.pitch) * over + tumble[1],
            DEATH_ROLL * over + tumble[2],
        )
    }

    /// After the head hits the floor: lift along the view's up axis (units) and nod about its
    /// right axis (degrees), decaying.
    fn shake(&self) -> (f32, f32) {
        let since = self.time - DEATH_LAND_TIME;
        if since < 0.0 {
            return (0.0, 0.0);
        }
        let decay = (-since / DEATH_SHAKE_DECAY).exp();
        let phase = std::f32::consts::TAU * DEATH_SHAKE_RATE * since;
        (
            DEATH_SHAKE_LIFT * decay * phase.sin(),
            DEATH_SHAKE_NOD * decay * (phase + 0.6).sin(),
        )
    }
}

/// `degrees` in −180..180.
fn wrap_degrees(degrees: f32) -> f32 {
    (degrees + 180.0).rem_euclid(360.0) - 180.0
}

impl PlayerEvents {
    /// Adds a movement step's jump and landing.
    pub fn merge(&mut self, moved: &MoveEvents) {
        self.jumped |= moved.jumped;
        self.landed_speed = self.landed_speed.max(moved.landed_speed);
    }
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
    // Hit kick, degrees of pitch and roll, and their velocity
    kick: [f32; 2],
    kick_velocity: [f32; 2],
    view: ViewPose,
    events: PlayerEvents,
    // Scripted input instead of the devices (capture runs)
    scripted: Option<PlayerInput>,
    // Killed in the deathmatch, until it respawns the player
    dead: bool,
    // The view while dead (from the death until the respawn)
    death: Option<DeathView>,
    // This frame's movement keys and view yaw, for the deathmatch
    input: PlayerInput,
    // Where the player is drawn: between the deathmatch's last two ticks
    origin: [f32; 3],
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
            kick: [0.0; 2],
            kick_velocity: [0.0; 2],
            view: ViewPose::default(),
            events: PlayerEvents::default(),
            scripted: None,
            dead: false,
            death: None,
            input: PlayerInput::default(),
            origin: [0.0; 3],
        }
    }

    /// Kicks the view up by `degrees` (weapon recoil).
    pub fn punch(&mut self, degrees: f32) {
        self.punch = degrees;
    }

    /// Jolts the view (degrees per second of pitch and roll): a hit, a blast nearby.
    pub fn kick(&mut self, pitch: f32, roll: f32) {
        self.kick_velocity[0] += pitch;
        self.kick_velocity[1] += roll;
    }
}

/// `FlyCamera` yaw of a Quake yaw: FlyCamera yaw 0 looks down −Z, Quake yaw 0 down +X (engine +X).
pub fn fly_yaw_from_quake(yaw: f32) -> f32 {
    yaw - 90.0
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
            rig.origin = origin;
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

/// Mouse look and movement keys of the player (player mode only), for the deathmatch.
pub fn player_input_system(engine: &mut Engine) -> Result<()> {
    if engine.get_global_component::<PlayerRig>()?.mode != Mode::Player {
        return Ok(());
    }

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

    let rig = engine.get_global_component_mut::<PlayerRig>()?;
    let (mut yaw, mut pitch) = (rig.yaw, rig.pitch);
    let input = match rig.scripted {
        _ if rig.dead => PlayerInput {
            yaw,
            ..PlayerInput::default()
        },
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
    rig.yaw = yaw.rem_euclid(360.0);
    rig.pitch = pitch;
    rig.input = PlayerInput {
        yaw: rig.yaw,
        ..input
    };
    Ok(())
}

/// The player's view (player mode only) after the deathmatch moved them: bob, roll, landing
/// dip, punch, the camera at the eye.
pub fn player_view_system(engine: &mut Engine) -> Result<()> {
    if engine.get_global_component::<PlayerRig>()?.mode != Mode::Player {
        return Ok(());
    }
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    let rig = engine.get_global_component_mut::<PlayerRig>()?;
    let rig_state = rig.state;
    let mut events = rig.events;

    // --- View feel ---
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

    for axis in 0..2 {
        rig.kick_velocity[axis] += (-rig.kick[axis] * KICK_STIFFNESS
            - rig.kick_velocity[axis] * KICK_DAMPING)
            * delta_time;
        rig.kick[axis] += rig.kick_velocity[axis] * delta_time;
    }

    rig.punch = if rig.punch > 0.0 {
        (rig.punch - PUNCH_DECAY * delta_time).max(0.0)
    } else {
        (rig.punch + PUNCH_DECAY * delta_time).min(0.0)
    };

    // Roll: lean into strafing
    let (_, right, _) = view_basis(rig.yaw, 0.0);
    let side = velocity[0] * right[0] + velocity[1] * right[1];
    let roll = side.signum() * side.abs().min(ROLL_SPEED) / ROLL_SPEED * ROLL_ANGLE;

    let origin = rig.origin;
    let mut eye = [origin[0], origin[1], origin[2] + EYE_HEIGHT + bob + rig.dip];
    let (mut yaw, mut pitch, mut roll) = (
        rig.yaw,
        rig.pitch + rig.punch + rig.kick[0],
        roll + rig.kick[1],
    );
    // Dead: the head knocked, falling to the floor, turning after the killer
    events.head_landed = false;
    if let Some(death) = rig.death.as_mut() {
        let before = death.time;
        death.time += delta_time;
        events.head_landed = before < DEATH_LAND_TIME && death.time >= DEATH_LAND_TIME;
        let turn = wrap_degrees(death.target_yaw - death.turned);
        death.turned += turn * (1.0 - (-DEATH_TURN_RATE * delta_time).exp());
        let (height, death_yaw, death_pitch, death_roll) = death.pose();
        eye = [origin[0], origin[1], origin[2] + height];
        (yaw, pitch, roll) = (death_yaw, death_pitch, death_roll);
    }
    let (forward, right, up) = view_basis(yaw, pitch);
    let (right, mut up) = (
        rotate_about(right, forward, roll),
        rotate_about(up, forward, roll),
    );
    let mut forward = forward;
    // The dead head's landing shake, in the view's own frame: vertical on screen
    if let Some(death) = rig.death.as_ref() {
        let (lift, nod) = death.shake();
        eye = std::array::from_fn(|axis| eye[axis] + up[axis] * lift);
        forward = rotate_about(forward, right, nod);
        up = rotate_about(up, right, nod);
    }
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
