//! Quake-style player movement against [`QuakeCollision`] (feature `quake_map`), in Quake space
//! (Z up, units, seconds): friction with a stop speed, ground and air acceleration (air
//! acceleration only steers: wish speed capped at 30), gravity, jumping, sliding along walls and
//! stepping up stairs. Written from the publicly documented behavior and server defaults
//! (sv_maxspeed 320, sv_accelerate 10, sv_friction 4, sv_stopspeed 100, sv_gravity 800) and the
//! structure of MIT-licensed reimplementations (pmove-godot, bevy_fps_controller); no GPL code.

use crate::quake_map::{QuakeCollision, QuakeTrace};
use crate::util::vec_math::{add3 as add, cross3 as cross, dot3 as dot, scale3 as scale};

/// Player box around `PlayerState::origin`, Quake's player hull.
pub const PLAYER_MINS: [f32; 3] = [-16.0, -16.0, -24.0];
pub const PLAYER_MAXS: [f32; 3] = [16.0, 16.0, 32.0];
/// Eye height above `PlayerState::origin`.
pub const EYE_HEIGHT: f32 = 22.0;

/// Steepest ground the player stands on: the up component of its normal.
const MIN_GROUND_NORMAL: f32 = 0.7;
/// How far below the box the ground is looked for.
const GROUND_PROBE: f32 = 0.25;
/// Velocity left after clipping against a plane, slightly pushing away from it.
const OVERBOUNCE: f32 = 1.001;
const MAX_CLIP_PLANES: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct MoveSettings {
    pub max_speed: f32,
    pub accelerate: f32,
    pub air_accelerate: f32,
    /// Wish speed air acceleration works up to (what makes air strafing turn, not speed up).
    pub air_wish_speed: f32,
    pub friction: f32,
    pub stop_speed: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub step_height: f32,
    /// A jump pressed this long (seconds) before landing still happens, on the landing move
    /// itself (no friction tick, so speed carries into the next hop). Every jump needs a fresh
    /// press.
    pub jump_buffer: f32,
}

impl Default for MoveSettings {
    fn default() -> Self {
        Self {
            max_speed: 320.0,
            accelerate: 10.0,
            air_accelerate: 10.0,
            air_wish_speed: 30.0,
            friction: 4.0,
            stop_speed: 100.0,
            gravity: 800.0,
            jump_speed: 270.0,
            step_height: 18.0,
            jump_buffer: 0.15,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerState {
    /// Box center reference (the hull spans `PLAYER_MINS`..`PLAYER_MAXS` around it).
    pub origin: [f32; 3],
    pub velocity: [f32; 3],
    pub on_ground: bool,
    /// Jump held on the previous move: a new jump needs a fresh press.
    pub jump_held: bool,
    /// Seconds left of a press made in the air (see `MoveSettings::jump_buffer`).
    pub jump_buffered: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerInput {
    /// −1..1, scaled to `MoveSettings::max_speed`.
    pub forward: f32,
    pub side: f32,
    pub jump: bool,
    /// Degrees, counter-clockwise from +X seen from above.
    pub yaw: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveEvents {
    pub jumped: bool,
    /// Downward speed when landing this move, 0 otherwise.
    pub landed_speed: f32,
    /// Height climbed by stepping up.
    pub stepped: f32,
}

type Vec3 = [f32; 3];

fn length(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

/// Half size and center offset of the player box.
fn hull() -> (Vec3, Vec3) {
    let half = std::array::from_fn(|axis| (PLAYER_MAXS[axis] - PLAYER_MINS[axis]) * 0.5);
    let offset = std::array::from_fn(|axis| (PLAYER_MAXS[axis] + PLAYER_MINS[axis]) * 0.5);
    (half, offset)
}

/// Sweeps the player box from `origin` to `target` (origins).
pub fn trace_player(world: &QuakeCollision, origin: Vec3, target: Vec3) -> QuakeTrace {
    let (half, offset) = hull();
    let mut trace = world.trace_box(add(origin, offset), add(target, offset), half);
    trace.end = add(trace.end, scale(offset, -1.0));
    trace
}

/// Removes the part of `velocity` going into a plane with `normal`.
fn clip_velocity(velocity: Vec3, normal: Vec3) -> Vec3 {
    let backoff = dot(velocity, normal) * OVERBOUNCE;
    let mut clipped = add(velocity, scale(normal, -backoff));
    for value in &mut clipped {
        if value.abs() < 0.001 {
            *value = 0.0;
        }
    }
    clipped
}

/// Moves `origin` by `velocity·time`, sliding along what it hits; `velocity` loses the parts into
/// the surfaces.
fn slide_move(world: &QuakeCollision, origin: &mut Vec3, velocity: &mut Vec3, time: f32) {
    let original = *velocity;
    let mut planes: Vec<Vec3> = Vec::with_capacity(MAX_CLIP_PLANES);
    let mut time_left = time;
    for _ in 0..4 {
        if length(*velocity) == 0.0 {
            break;
        }
        let target = add(*origin, scale(*velocity, time_left));
        let trace = trace_player(world, *origin, target);
        if trace.all_solid {
            *velocity = [0.0; 3];
            return;
        }
        if trace.fraction > 0.0 {
            *origin = trace.end;
        }
        if trace.fraction >= 1.0 {
            break;
        }
        time_left -= time_left * trace.fraction;
        if planes.len() >= MAX_CLIP_PLANES {
            *velocity = [0.0; 3];
            break;
        }
        planes.push(trace.normal);

        // A velocity along every plane hit so far, else along the crease of two, else stop
        let mut found = None;
        for (index, plane) in planes.iter().enumerate() {
            let candidate = clip_velocity(original, *plane);
            let clear = planes
                .iter()
                .enumerate()
                .all(|(other, normal)| other == index || dot(candidate, *normal) >= 0.0);
            if clear {
                found = Some(candidate);
                break;
            }
        }
        match found {
            Some(candidate) => *velocity = candidate,
            None if planes.len() == 2 => {
                let crease = cross(planes[0], planes[1]);
                *velocity = scale(crease, dot(crease, *velocity));
            }
            None => {
                *velocity = [0.0; 3];
                break;
            }
        }
        // Never move back against the original direction (no jitter in corners)
        if dot(*velocity, original) <= 0.0 {
            *velocity = [0.0; 3];
            break;
        }
    }
}

/// Slide move that also tries the move `step_height` higher and keeps it when it gets further
/// horizontally (stairs); returns the height stepped up.
fn step_slide_move(
    world: &QuakeCollision,
    settings: &MoveSettings,
    origin: &mut Vec3,
    velocity: &mut Vec3,
    time: f32,
) -> f32 {
    let start = *origin;
    let start_velocity = *velocity;
    let mut low_origin = start;
    let mut low_velocity = start_velocity;
    slide_move(world, &mut low_origin, &mut low_velocity, time);

    let up = trace_player(world, start, add(start, [0.0, 0.0, settings.step_height]));
    let mut high_origin = up.end;
    let mut high_velocity = start_velocity;
    slide_move(world, &mut high_origin, &mut high_velocity, time);
    let climbed = up.end[2] - start[2];
    let down = trace_player(world, high_origin, add(high_origin, [0.0, 0.0, -climbed]));
    let landed_on_ground = down.fraction < 1.0 && down.normal[2] >= MIN_GROUND_NORMAL;
    high_origin = down.end;

    let horizontal = |point: Vec3| {
        let dx = point[0] - start[0];
        let dy = point[1] - start[1];
        dx * dx + dy * dy
    };
    if landed_on_ground && horizontal(high_origin) > horizontal(low_origin) + 0.01 {
        *origin = high_origin;
        *velocity = [high_velocity[0], high_velocity[1], low_velocity[2]];
        (high_origin[2] - start[2]).max(0.0)
    } else {
        *origin = low_origin;
        *velocity = low_velocity;
        0.0
    }
}

fn ground_below(world: &QuakeCollision, origin: Vec3) -> Option<QuakeTrace> {
    let trace = trace_player(world, origin, add(origin, [0.0, 0.0, -GROUND_PROBE]));
    (trace.fraction < 1.0 && trace.normal[2] >= MIN_GROUND_NORMAL).then_some(trace)
}

/// Adds speed along `direction` until it reaches `wish_speed`, at `accel · accel_scale_speed` per
/// second (in the air `wish_speed` is capped but the rate keeps the full wish speed: that's what
/// lets air strafing turn without speeding up).
fn accelerate(
    velocity: &mut Vec3,
    direction: Vec3,
    wish_speed: f32,
    accel: f32,
    accel_scale_speed: f32,
    time: f32,
) {
    let current = dot(*velocity, direction);
    let add_speed = wish_speed - current;
    if add_speed <= 0.0 {
        return;
    }
    let accel_speed = (accel * time * accel_scale_speed).min(add_speed);
    *velocity = add(*velocity, scale(direction, accel_speed));
}

/// Advances the player by `time` seconds (use steps of about 1/72 s or less).
pub fn player_move(
    state: &mut PlayerState,
    input: &PlayerInput,
    time: f32,
    world: &QuakeCollision,
    settings: &MoveSettings,
) -> MoveEvents {
    let mut events = MoveEvents::default();
    let was_on_ground = state.on_ground;
    let mut velocity = state.velocity;

    state.on_ground = velocity[2] <= 180.0 && ground_below(world, state.origin).is_some();
    // A fresh press arms the jump; it fires on the ground, now or on landing within the buffer
    if input.jump && !state.jump_held {
        state.jump_buffered = settings.jump_buffer.max(time);
    }
    state.jump_held = input.jump;
    if state.on_ground && state.jump_buffered > 0.0 {
        velocity[2] = settings.jump_speed;
        state.on_ground = false;
        state.jump_buffered = 0.0;
        events.jumped = true;
    }
    state.jump_buffered = (state.jump_buffered - time).max(0.0);

    // Friction on the ground
    if state.on_ground {
        let speed = length([velocity[0], velocity[1], 0.0]);
        if speed > 0.0 {
            let control = speed.max(settings.stop_speed);
            let new_speed = (speed - control * settings.friction * time).max(0.0);
            let factor = new_speed / speed;
            velocity[0] *= factor;
            velocity[1] *= factor;
        }
        velocity[2] = 0.0;
    }

    // Wish direction from the view yaw
    let yaw = input.yaw.to_radians();
    let forward = [yaw.cos(), yaw.sin(), 0.0];
    let right = [yaw.sin(), -yaw.cos(), 0.0];
    let wish = add(
        scale(forward, input.forward.clamp(-1.0, 1.0)),
        scale(right, input.side.clamp(-1.0, 1.0)),
    );
    let wish_length = length(wish);
    let (direction, wish_speed) = if wish_length > 0.0 {
        (
            scale(wish, 1.0 / wish_length),
            settings.max_speed * wish_length.min(1.0),
        )
    } else {
        ([0.0; 3], 0.0)
    };

    if state.on_ground {
        accelerate(
            &mut velocity,
            direction,
            wish_speed,
            settings.accelerate,
            wish_speed,
            time,
        );
        events.stepped = step_slide_move(world, settings, &mut state.origin, &mut velocity, time);
    } else {
        accelerate(
            &mut velocity,
            direction,
            wish_speed.min(settings.air_wish_speed),
            settings.air_accelerate,
            wish_speed,
            time,
        );
        velocity[2] -= settings.gravity * time;
        let falling_speed = -velocity[2];
        slide_move(world, &mut state.origin, &mut velocity, time);
        if !was_on_ground || events.jumped {
            if let Some(ground) = ground_below(world, state.origin).filter(|_| velocity[2] <= 0.0) {
                state.origin = ground.end;
                state.on_ground = true;
                if !events.jumped {
                    events.landed_speed = falling_speed.max(0.0);
                }
                velocity[2] = 0.0;
            }
        }
    }
    // Walking off a ledge leaves the ground; standing on it again keeps the player there
    if state.on_ground && !events.jumped {
        state.on_ground = ground_below(world, state.origin).is_some();
    }
    state.velocity = velocity;
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quake_map::parse_map;

    /// Axis-aligned box brushes (Quake units) in a worldspawn, as a collision world. Per axis: the
    /// min face's three points (outward normal) and the same points reversed on the max side.
    fn world(boxes: &[([i32; 3], [i32; 3])]) -> QuakeCollision {
        let mut text = String::from("{\n\"classname\" \"worldspawn\"\n");
        for &([x0, y0, z0], [x1, y1, z1]) in boxes {
            text.push_str(&format!(
                "{{\n\
                 ( {x0} {y0} {z0} ) ( {x0} {y1} {z0} ) ( {x0} {y0} {z1} ) stone 0 0 0 1 1\n\
                 ( {x0} {y0} {z0} ) ( {x0} {y0} {z1} ) ( {x1} {y0} {z0} ) stone 0 0 0 1 1\n\
                 ( {x0} {y0} {z0} ) ( {x1} {y0} {z0} ) ( {x0} {y1} {z0} ) stone 0 0 0 1 1\n\
                 ( {x1} {y0} {z1} ) ( {x1} {y1} {z0} ) ( {x1} {y0} {z0} ) stone 0 0 0 1 1\n\
                 ( {x1} {y1} {z0} ) ( {x0} {y1} {z1} ) ( {x0} {y1} {z0} ) stone 0 0 0 1 1\n\
                 ( {x0} {y1} {z1} ) ( {x1} {y0} {z1} ) ( {x0} {y0} {z1} ) stone 0 0 0 1 1\n\
                 }}\n"
            ));
        }
        text.push_str("}\n");
        QuakeCollision::from_map(&parse_map(&text).unwrap())
    }

    const STEP: f32 = 1.0 / 72.0;

    /// A floor 0..2048 × −512..512 with its top at z 0.
    fn floor() -> ([i32; 3], [i32; 3]) {
        ([-64, -512, -64], [2048, 512, 0])
    }

    fn standing_at(x: f32) -> PlayerState {
        PlayerState {
            origin: [x, 0.0, 24.0 + 1.0 / 32.0],
            on_ground: true,
            ..PlayerState::default()
        }
    }

    fn run(
        state: &mut PlayerState,
        input: PlayerInput,
        seconds: f32,
        world: &QuakeCollision,
    ) -> Vec<MoveEvents> {
        let settings = MoveSettings::default();
        (0..(seconds / STEP).round() as usize)
            .map(|_| player_move(state, &input, STEP, world, &settings))
            .collect()
    }

    fn horizontal_speed(state: &PlayerState) -> f32 {
        length([state.velocity[0], state.velocity[1], 0.0])
    }

    #[test]
    fn running_forward_on_flat_ground_reaches_320_and_no_more() {
        let world = world(&[floor()]);
        let mut state = standing_at(0.0);
        let forward = PlayerInput {
            forward: 1.0,
            ..PlayerInput::default()
        };
        run(&mut state, forward, 1.0, &world);
        let speed = horizontal_speed(&state);
        assert!((speed - 320.0).abs() < 1.0, "speed {speed}");
        assert!(state.on_ground);
        assert!(
            (state.origin[2] - 24.0).abs() < 0.1,
            "z {}",
            state.origin[2]
        );
    }

    #[test]
    fn released_input_friction_stops_player() {
        let world = world(&[floor()]);
        let mut state = standing_at(0.0);
        state.velocity = [320.0, 0.0, 0.0];
        run(&mut state, PlayerInput::default(), 1.0, &world);
        assert_eq!(horizontal_speed(&state), 0.0);
    }

    fn step_world(height: i32) -> QuakeCollision {
        world(&[floor(), ([256, -512, 0], [2048, 512, height])])
    }

    #[test]
    fn step_18_units_climbed_step_19_blocks() {
        let forward = PlayerInput {
            forward: 1.0,
            ..PlayerInput::default()
        };
        let low = step_world(18);
        let mut state = standing_at(100.0);
        run(&mut state, forward, 1.0, &low);
        assert!(state.origin[0] > 300.0, "x {}", state.origin[0]);
        assert!(
            (state.origin[2] - (24.0 + 18.0)).abs() < 0.5,
            "z {}",
            state.origin[2]
        );

        let high = step_world(19);
        let mut state = standing_at(100.0);
        run(&mut state, forward, 1.0, &high);
        assert!(
            state.origin[0] < 256.0 - 16.0 + 0.1,
            "x {}",
            state.origin[0]
        );
    }

    #[test]
    fn running_diagonally_into_wall_slides_along_it() {
        let world = world(&[floor(), ([300, -512, 0], [364, 512, 128])]);
        let mut state = standing_at(200.0);
        // 45° between +X (into the wall) and +Y (along it)
        let input = PlayerInput {
            forward: 1.0,
            yaw: 45.0,
            ..PlayerInput::default()
        };
        run(&mut state, input, 1.0, &world);
        assert!(
            state.origin[0] <= 300.0 - 16.0 + 0.1,
            "x {}",
            state.origin[0]
        );
        assert!(state.origin[1] > 100.0, "y {}", state.origin[1]);
    }

    #[test]
    fn jump_apex_about_45_units_then_lands_with_landed_speed() {
        let world = world(&[floor()]);
        let mut state = standing_at(0.0);
        let jump = PlayerInput {
            jump: true,
            ..PlayerInput::default()
        };
        let events = run(&mut state, jump, 0.0 + STEP, &world);
        assert!(events[0].jumped);
        let mut apex = state.origin[2];
        let mut landed = 0.0;
        let settings = MoveSettings::default();
        for _ in 0..200 {
            let events = player_move(&mut state, &PlayerInput::default(), STEP, &world, &settings);
            apex = apex.max(state.origin[2]);
            if events.landed_speed > 0.0 {
                landed = events.landed_speed;
                break;
            }
        }
        let height = apex - 24.0;
        assert!((height - 45.6).abs() < 3.0, "apex {height}");
        assert!((landed - 270.0).abs() < 20.0, "landed {landed}");
        assert!(state.on_ground);
    }

    #[test]
    fn held_jump_does_not_rejump_after_landing() {
        let world = world(&[floor()]);
        let mut state = standing_at(0.0);
        let jump = PlayerInput {
            jump: true,
            ..PlayerInput::default()
        };
        let jumps = run(&mut state, jump, 2.0, &world)
            .iter()
            .filter(|events| events.jumped)
            .count();
        assert_eq!(jumps, 1);
    }

    /// Jumps once, then presses jump again `before_landing` seconds before it would land; returns
    /// whether the second press jumped and the horizontal speed then.
    fn press_before_landing(before_landing: f32) -> (bool, f32) {
        let world = world(&[([-64, -512, -64], [8192, 512, 0])]);
        let settings = MoveSettings::default();
        let mut state = standing_at(0.0);
        state.velocity = [320.0, 0.0, 0.0];
        let forward = PlayerInput {
            forward: 1.0,
            ..PlayerInput::default()
        };
        let jump = PlayerInput {
            jump: true,
            ..forward
        };
        assert!(player_move(&mut state, &jump, STEP, &world, &settings).jumped);
        player_move(&mut state, &forward, STEP, &world, &settings);
        // Airtime 2·270/800 = 0.675 s; press at airtime − before_landing, then hold
        let press_at = 0.675 - before_landing;
        let mut time = 2.0 * STEP;
        while time < press_at {
            player_move(&mut state, &forward, STEP, &world, &settings);
            time += STEP;
        }
        for _ in 0..(0.6 / STEP) as usize {
            if player_move(&mut state, &jump, STEP, &world, &settings).jumped {
                return (true, horizontal_speed(&state));
            }
        }
        (false, horizontal_speed(&state))
    }

    #[test]
    fn jump_pressed_just_before_landing_hops_at_full_speed() {
        let (jumped, speed) = press_before_landing(0.1);
        assert!(jumped);
        assert!(speed >= 319.0, "speed {speed}");
    }

    #[test]
    fn jump_pressed_too_early_in_air_is_dropped() {
        let (jumped, _) = press_before_landing(0.3);
        assert!(!jumped);
    }

    #[test]
    fn walking_off_ledge_falls() {
        let world = world(&[
            ([-64, -512, -64], [128, 512, 0]),
            ([-64, -512, -512], [2048, 512, -448]),
        ]);
        let mut state = standing_at(0.0);
        let forward = PlayerInput {
            forward: 1.0,
            ..PlayerInput::default()
        };
        run(&mut state, forward, 2.0, &world);
        assert!(state.origin[2] < -400.0, "z {}", state.origin[2]);
    }
}
