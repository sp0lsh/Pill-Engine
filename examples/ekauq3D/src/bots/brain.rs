//! The bot: thinks at 10 Hz (enemy, state, goal, weapon) and steers every step (route following
//! with the navigation graph's walk, drop and jump edges, fight movement, aim). States after the
//! Quake III bot's AI network (van Waveren 2001, §15): roam for items, fight, chase a lost enemy,
//! retreat to health. Limited like a human (§7.1): reaction time, turn rate, aim wander.

use pill_engine::game::{
    quake_map::QuakeCollision,
    quake_player::{trace_player, PlayerInput},
};

use super::{
    genome::{self, Calibration, Genome, Skill},
    nav::{self, EdgeKind, NavGraph, RouteCache},
};
use crate::sim::{
    add, angle_delta, angles_of,
    arena::Arena,
    combat::{Command, Fighter, Match, MAX_PITCH},
    distance, dot, horizontal_distance, length, normalize, scale, sub, view_basis,
    weapons::Kind,
    Rng, Vec3,
};

const THINK_TIME: f32 = 0.1;
/// Field of view for noticing enemies (degrees, full width).
const FIELD_OF_VIEW: f32 = 110.0;
/// A node counts as reached this close (units, horizontally).
const REACHED: f32 = 16.0;
/// Route lookahead: walk straight to the farthest of the next nodes in clear line.
const LOOKAHEAD: usize = 6;
/// Roaming goals are dropped after this long (seconds).
const GOAL_TIME: f32 = 12.0;
/// Aim wander: Ornstein–Uhlenbeck mean reversion (1/s).
const AIM_REVERSION: f32 = 2.0;
/// Box half width: a target's angular radius uses it.
const TARGET_RADIUS: f32 = 16.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum State {
    Roam,
    Fight,
    Chase,
    Retreat,
}

/// A jump edge under way: the phase it is in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum JumpPhase {
    /// Walking to the run-up start (running jumps) or stopping (standing jumps).
    LineUp,
    /// Running along the jump, jumping at the take-off node.
    Run,
    Airborne,
}

#[derive(Clone, Copy, Debug)]
struct Jump {
    from: u32,
    kind: EdgeKind,
    phase: JumpPhase,
    left: f32,
}

/// What the bot needs to see of the world.
pub struct World<'a> {
    pub collision: &'a QuakeCollision,
    pub arena: &'a Arena,
    pub nav: &'a NavGraph,
    pub fight: &'a Match,
    pub calibration: &'a Calibration,
    pub routes: &'a mut RouteCache,
}

pub struct Bot {
    pub genome: Genome,
    pub skill: Skill,
    pub state: State,
    rng: Rng,
    think_left: f32,
    enemy: Option<usize>,
    /// Seconds the current enemy has been in sight.
    seen: f32,
    last_seen: Option<Vec3>,
    chase_left: f32,
    goal: Option<u32>,
    goal_left: f32,
    /// Per arena spot (health, then other items, then spawns): seconds since last visited.
    unvisited: Vec<f32>,
    strafe: f32,
    jump: Option<Jump>,
    aim_error: [f32; 2],
    stuck: f32,
    stuck_origin: Vec3,
}

impl Bot {
    pub fn new(genome: Genome, skill: Skill, seed: u32) -> Self {
        Self {
            genome,
            skill,
            state: State::Roam,
            rng: Rng::new(seed),
            think_left: 0.0,
            enemy: None,
            seen: 0.0,
            last_seen: None,
            chase_left: 0.0,
            goal: None,
            goal_left: 0.0,
            unvisited: Vec::new(),
            strafe: 1.0,
            jump: None,
            aim_error: [0.0; 2],
            stuck: 0.0,
            stuck_origin: [0.0; 3],
        }
    }

    /// The command of fighter `me` for the next `dt` seconds.
    pub fn command(&mut self, me: usize, world: &mut World, dt: f32) -> Command {
        let fighter = world.fight.fighters[me].clone();
        if !fighter.alive {
            self.state = State::Roam;
            self.enemy = None;
            self.goal = None;
            self.jump = None;
            self.seen = 0.0;
            return Command {
                input: PlayerInput {
                    yaw: fighter.yaw,
                    ..PlayerInput::default()
                },
                pitch: fighter.pitch,
                ..Command::default()
            };
        }
        self.think_left -= dt;
        let mut select = None;
        if self.think_left <= 0.0 {
            self.think_left += THINK_TIME;
            select = self.think(me, &fighter, world);
        }
        let (move_direction, jump, look) = self.steer(&fighter, world, dt);
        let (yaw, pitch, fire) = self.aim(&fighter, world, look, dt);
        // Movement relative to the view: wish direction = forward·cos + right·sin
        let (forward, side) = match move_direction {
            Some(direction) => {
                let (wanted, _) = angles_of(direction);
                let relative = angle_delta(yaw, wanted).to_radians();
                (relative.cos(), -relative.sin())
            }
            None => (0.0, 0.0),
        };
        Command {
            input: PlayerInput {
                forward,
                side,
                jump,
                yaw,
            },
            pitch,
            fire,
            select,
        }
    }

    /// Enemy, state, goal and weapon; the weapon to switch to, if any.
    fn think(&mut self, me: usize, fighter: &Fighter, world: &mut World) -> Option<usize> {
        let fight = world.fight;
        let arena = world.arena;
        let spots = arena.health.len() + arena.spots.len() + arena.spawns.len();
        if self.unvisited.len() != spots {
            self.unvisited = (0..spots).map(|_| self.rng.range(0.0, 30.0)).collect();
        }
        for (index, time) in self.unvisited.iter_mut().enumerate() {
            *time += THINK_TIME;
            if distance(spot(arena, index), fighter.state.origin) < 64.0 {
                *time = 0.0;
            }
        }

        // --- Enemy: the nearest one in view (or whoever hit me last), seen for the reaction time
        let eye = fighter.eye();
        let (forward, _, _) = view_basis(fighter.yaw, 0.0);
        let mut best: Option<(usize, f32)> = None;
        for (index, other) in fight.fighters.iter().enumerate() {
            if index == me || !other.targetable() {
                continue;
            }
            let to = sub(other.center(), eye);
            let flat = normalize([to[0], to[1], 0.0]);
            let in_view = dot(flat, forward) >= (FIELD_OF_VIEW * 0.5).to_radians().cos();
            let hurt_me = fighter.last_attacker == Some(index);
            if !(in_view || hurt_me || Some(index) == self.enemy) {
                continue;
            }
            if !Match::visible(world.collision, fighter, other) {
                continue;
            }
            let mut score = length(to);
            if hurt_me {
                score *= 0.6;
            }
            if best.map_or(true, |(_, known)| score < known) {
                best = Some((index, score));
            }
        }
        match best {
            Some((index, _)) => {
                if self.enemy == Some(index) {
                    self.seen += THINK_TIME;
                } else {
                    self.enemy = Some(index);
                    self.seen = 0.0;
                }
                self.last_seen = Some(fight.fighters[index].state.origin);
            }
            None => self.seen = 0.0,
        }
        let visible = best.is_some();
        let reacted = visible && self.seen >= self.skill.reaction;

        // --- State
        let health = fighter.health;
        let low = health < self.genome.get(genome::RETREAT_HEALTH) && !arena.health.is_empty();
        self.state = match self.state {
            _ if reacted && low => State::Retreat,
            State::Retreat if health < self.genome.get(genome::RETURN_HEALTH) => State::Retreat,
            _ if reacted => State::Fight,
            State::Fight | State::Retreat if !visible && self.last_seen.is_some() => {
                self.chase_left = self.genome.get(genome::CHASE_TIME);
                self.goal = None;
                State::Chase
            }
            State::Chase => {
                self.chase_left -= THINK_TIME;
                let arrived = self
                    .last_seen
                    .is_some_and(|seen| horizontal_distance(seen, fighter.state.origin) < 48.0);
                if self.chase_left <= 0.0 || arrived {
                    self.enemy = None;
                    self.last_seen = None;
                    self.goal = None;
                    State::Roam
                } else {
                    State::Chase
                }
            }
            State::Fight | State::Retreat => State::Roam,
            State::Roam => State::Roam,
        };

        // --- Goal
        let nav = world.nav;
        let here = nav.nearest(fighter.state.origin);
        self.goal_left -= THINK_TIME;
        match self.state {
            State::Chase => {
                self.goal = self.last_seen.and_then(|seen| nav.nearest(seen));
            }
            State::Retreat => {
                // The health item I reach first compared with the enemy
                let enemy = self
                    .enemy
                    .and_then(|enemy| nav.nearest(fight.fighters[enemy].state.origin));
                let mut chosen = None;
                let mut best = f32::MIN;
                for (item, &spot) in arena.health.iter().enumerate() {
                    if fight.health_items.get(item).is_some_and(|&left| left > 3.0) {
                        continue;
                    }
                    let (Some(here), Some(node)) = (here, nav.nearest(spot)) else {
                        continue;
                    };
                    let Some((mine, _)) = world.routes.route(nav, here, node) else {
                        continue;
                    };
                    let theirs = enemy
                        .and_then(|enemy| world.routes.route(nav, enemy, node))
                        .map_or(10.0, |(time, _)| time);
                    let score = theirs - mine;
                    if score > best {
                        best = score;
                        chosen = Some(node);
                    }
                }
                self.goal = chosen;
            }
            State::Roam if self.goal.is_none() || self.goal_left <= 0.0 => {
                self.goal = self.roam_goal(fighter, world, here);
                self.goal_left = GOAL_TIME;
            }
            _ => {}
        }
        if let (Some(goal), Some(here)) = (self.goal, here) {
            if goal == here && self.state == State::Roam {
                self.goal = None;
            }
        }

        // --- Weapon: the most damage at the enemy's distance (measured), with a margin
        let enemy_distance = self
            .enemy
            .map(|enemy| distance(fight.fighters[enemy].state.origin, fighter.state.origin))
            .unwrap_or(256.0);
        let current = fighter
            .switching
            .map_or(fighter.weapon, |(weapon, _)| weapon);
        let mut best_weapon = current;
        let mut best_dps =
            world.calibration.dps(current, enemy_distance) * self.genome.get(genome::SWITCH_MARGIN);
        for weapon in 0..fight.weapons.len() {
            let mut dps = world.calibration.dps(weapon, enemy_distance);
            let definition = &fight.weapons[weapon];
            if definition.splash > 0.0
                && enemy_distance < definition.splash * self.genome.get(genome::SELF_SPLASH_MARGIN)
            {
                dps = 0.0;
            }
            if dps > best_dps {
                best_dps = dps;
                best_weapon = weapon;
            }
        }
        (best_weapon != current).then_some(best_weapon)
    }

    /// A roaming goal: health when hurt, else the spot not visited for longest, nearer first.
    fn roam_goal(
        &mut self,
        fighter: &Fighter,
        world: &mut World,
        here: Option<u32>,
    ) -> Option<u32> {
        let here = here?;
        let arena = world.arena;
        let nav = world.nav;
        let greed = self.genome.get(genome::ITEM_GREED);
        let mut chosen = None;
        let mut best = f32::MIN;
        for index in 0..self.unvisited.len() {
            let Some(node) = nav.nearest(spot(arena, index)) else {
                continue;
            };
            if node == here {
                continue;
            }
            let Some((time, _)) = world.routes.route(nav, here, node) else {
                continue;
            };
            let mut value = self.unvisited[index].min(60.0) / 60.0;
            if index < arena.health.len() {
                let available = world
                    .fight
                    .health_items
                    .get(index)
                    .is_some_and(|&left| left <= time);
                if available {
                    value += greed * (1.0 - fighter.health / 100.0);
                }
            }
            let score = value / (1.0 + time * 0.3) + self.rng.unit() * 0.05;
            if score > best {
                best = score;
                chosen = Some(node);
            }
        }
        chosen
    }

    /// Where to move (direction), whether to jump, and where to look when not aiming.
    fn steer(
        &mut self,
        fighter: &Fighter,
        world: &mut World,
        dt: f32,
    ) -> (Option<Vec3>, bool, Option<Vec3>) {
        let origin = fighter.state.origin;
        let nav = world.nav;

        // Stuck: barely moved for a while although trying to
        if distance(origin, self.stuck_origin) > 24.0 {
            self.stuck = 0.0;
            self.stuck_origin = origin;
        } else {
            self.stuck += dt;
        }
        let unstick = self.stuck > 1.5;
        if self.stuck > 3.0 {
            self.goal = None;
            self.jump = None;
            self.stuck = 0.0;
            self.stuck_origin = origin;
        }

        // A jump under way comes first
        if let Some(jump) = self.jump {
            if let Some(result) = self.continue_jump(fighter, world, jump, dt) {
                return result;
            }
        }

        if self.state == State::Fight {
            if let Some(enemy) = self.enemy {
                let (direction, jump) = self.fight_move(fighter, world, enemy, dt);
                return (direction, jump || unstick, None);
            }
        }

        let Some(goal) = self.goal else {
            return (None, unstick, None);
        };
        let Some(here) = nav.nearest(origin) else {
            return (None, unstick, None);
        };
        let Some((_, next)) = world.routes.route(nav, here, goal) else {
            self.goal = None;
            return (None, unstick, None);
        };
        if next == here {
            let target = nav.nodes[goal as usize];
            let direction = normalize([target[0] - origin[0], target[1] - origin[1], 0.0]);
            return (Some(direction), unstick, Some(direction));
        }
        // A jump edge starts here: take it
        if let Some(edge) = nav.edge(here, next) {
            if matches!(
                edge.kind,
                EdgeKind::Jump { .. } | EdgeKind::StandingJump { .. }
            ) {
                let jump = Jump {
                    from: here,
                    kind: edge.kind,
                    phase: JumpPhase::LineUp,
                    left: 4.0,
                };
                self.jump = Some(jump);
                if let Some(result) = self.continue_jump(fighter, world, jump, dt) {
                    return result;
                }
            }
        }
        // Walk to the farthest of the next walk/drop nodes in a clear line
        let mut target = next;
        let mut node = next;
        for _ in 0..LOOKAHEAD {
            let Some((_, after)) = world.routes.route(nav, node, goal) else {
                break;
            };
            if after == node {
                break;
            }
            let Some(edge) = nav.edge(node, after) else {
                break;
            };
            if edge.kind != EdgeKind::Walk {
                break;
            }
            let point = nav.nodes[after as usize];
            let line = trace_player(
                world.collision,
                add(origin, [0.0, 0.0, 18.0]),
                add(point, [0.0, 0.0, 18.0]),
            );
            if line.fraction < 1.0 || (point[2] - origin[2]).abs() > 40.0 {
                break;
            }
            target = after;
            node = after;
        }
        let point = nav.nodes[target as usize];
        let direction = normalize([point[0] - origin[0], point[1] - origin[1], 0.0]);
        (Some(direction), unstick, Some(direction))
    }

    /// Runs a jump edge: line up, run and jump at the take-off node, fly along the edge's yaw.
    fn continue_jump(
        &mut self,
        fighter: &Fighter,
        world: &mut World,
        mut jump: Jump,
        dt: f32,
    ) -> Option<(Option<Vec3>, bool, Option<Vec3>)> {
        let nav = world.nav;
        let origin = fighter.state.origin;
        let from = nav.nodes[jump.from as usize];
        let yaw = match jump.kind {
            EdgeKind::Jump { yaw } | EdgeKind::StandingJump { yaw } => yaw,
            _ => {
                self.jump = None;
                return None;
            }
        };
        let (along, _, _) = view_basis(yaw, 0.0);
        jump.left -= dt;
        if jump.left <= 0.0 {
            self.jump = None;
            return None;
        }
        let aligned = angle_delta(fighter.yaw, yaw).abs() < 5.0;
        let result = match jump.phase {
            JumpPhase::LineUp => match jump.kind {
                EdgeKind::Jump { .. } => {
                    let Some(start) = nav::run_up_start(world.collision, from, jump.kind) else {
                        self.jump = None;
                        return None;
                    };
                    if horizontal_distance(origin, start) < 10.0 && aligned {
                        jump.phase = JumpPhase::Run;
                        (Some(along), false, Some(along))
                    } else if horizontal_distance(origin, start) < 10.0 {
                        (None, false, Some(along))
                    } else {
                        let direction =
                            normalize([start[0] - origin[0], start[1] - origin[1], 0.0]);
                        (Some(direction), false, Some(along))
                    }
                }
                _ => {
                    let speed = length([fighter.state.velocity[0], fighter.state.velocity[1], 0.0]);
                    if horizontal_distance(origin, from) > REACHED {
                        let direction = normalize([from[0] - origin[0], from[1] - origin[1], 0.0]);
                        (Some(direction), false, Some(along))
                    } else if speed > 30.0 || !aligned {
                        (None, false, Some(along))
                    } else {
                        jump.phase = JumpPhase::Airborne;
                        (Some(along), true, Some(along))
                    }
                }
            },
            JumpPhase::Run => {
                let ahead = dot(sub(origin, from), along);
                if ahead >= -2.0 {
                    jump.phase = JumpPhase::Airborne;
                    (Some(along), true, Some(along))
                } else {
                    (Some(along), false, Some(along))
                }
            }
            JumpPhase::Airborne => {
                if fighter.state.on_ground && fighter.state.velocity[2] <= 0.0 && jump.left < 3.8 {
                    self.jump = None;
                    return None;
                }
                (Some(along), false, Some(along))
            }
        };
        if self.jump.is_some() {
            self.jump = Some(jump);
        }
        Some(result)
    }

    /// Fight movement: hold the distance the current weapon does best at, strafe, dodge-jump;
    /// never off a ledge.
    fn fight_move(
        &mut self,
        fighter: &Fighter,
        world: &mut World,
        enemy: usize,
        dt: f32,
    ) -> (Option<Vec3>, bool) {
        let target = &world.fight.fighters[enemy];
        let origin = fighter.state.origin;
        let to = sub(target.state.origin, origin);
        let flat = normalize([to[0], to[1], 0.0]);
        let side = [flat[1], -flat[0], 0.0];
        let gap = horizontal_distance(origin, target.state.origin);
        let preferred = world.calibration.best_distance(fighter.weapon)
            * self.genome.get(genome::PREFERRED_RANGE);
        let tolerance = self.genome.get(genome::RANGE_TOLERANCE);
        let approach = if gap > preferred + tolerance {
            0.4 + 0.6 * self.genome.get(genome::AGGRESSION)
        } else if gap < preferred - tolerance {
            -1.0
        } else {
            0.0
        };
        // Strafe, harder while the enemy looks at me
        let (enemy_forward, _, _) = view_basis(target.yaw, target.pitch);
        let aimed_at = dot(
            enemy_forward,
            normalize(sub(fighter.center(), target.eye())),
        ) > 0.97;
        let mut switch_rate = self.genome.get(genome::STRAFE_SWITCH);
        if aimed_at {
            switch_rate *= 1.0 + 2.0 * self.genome.get(genome::DODGE_WHEN_AIMED);
        }
        if self.rng.unit() < switch_rate * dt {
            self.strafe = -self.strafe;
        }
        let strafing = self.genome.get(genome::STRAFE) > 0.5
            || (aimed_at && self.genome.get(genome::DODGE_WHEN_AIMED) > 0.5);
        let mut direction = scale(flat, approach);
        if strafing {
            direction = add(direction, scale(side, self.strafe));
        }
        if length(direction) < 0.1 {
            return (None, false);
        }
        let direction = normalize(direction);
        let direction = if safe_ground(world.collision, origin, direction) {
            direction
        } else {
            self.strafe = -self.strafe;
            let other = normalize(add(scale(flat, approach), scale(side, self.strafe)));
            if safe_ground(world.collision, origin, other) {
                other
            } else {
                return (None, false);
            }
        };
        let jump =
            fighter.state.on_ground && self.rng.unit() < self.genome.get(genome::JUMP_DODGE) * dt;
        (Some(direction), jump)
    }

    /// View and fire: at the enemy (projectiles led, splash at the feet), with the aim's wander
    /// and the turn rate; else along `look`.
    fn aim(
        &mut self,
        fighter: &Fighter,
        world: &World,
        look: Option<Vec3>,
        dt: f32,
    ) -> (f32, f32, bool) {
        let fight = world.fight;
        let eye = fighter.eye();
        let target = self
            .enemy
            .filter(|_| matches!(self.state, State::Fight | State::Retreat | State::Chase))
            .map(|enemy| &fight.fighters[enemy])
            .filter(|enemy| enemy.targetable());
        let visible = target.is_some_and(|enemy| {
            self.seen >= self.skill.reaction
                && self.state != State::Chase
                && Match::visible(world.collision, fighter, enemy)
        });
        let (wanted_yaw, wanted_pitch, aim_point) = match (target, visible) {
            (Some(enemy), true) => {
                let point = self.aim_point(fighter, enemy, world);
                let (yaw, pitch) = angles_of(sub(point, eye));
                (yaw, pitch, Some(point))
            }
            (Some(_), false) if self.state == State::Chase => {
                let seen = self.last_seen.unwrap_or(eye);
                let (yaw, _) = angles_of(sub(seen, eye));
                (yaw, 0.0, None)
            }
            _ => match look {
                Some(direction) => (angles_of(direction).0, 0.0, None),
                None => (fighter.yaw, 0.0, None),
            },
        };
        // Aim wander (stationary deviation = skill.aim_error), bigger on a fast target
        let mut deviation = self.skill.aim_error;
        if let Some(enemy) = target {
            let speed = length(enemy.state.velocity);
            let gap = distance(enemy.eye(), eye).max(32.0);
            deviation += (speed / gap).to_degrees() * 0.05;
        }
        let sigma = deviation * (2.0 * AIM_REVERSION).sqrt();
        for error in &mut self.aim_error {
            *error += -*error * AIM_REVERSION * dt + sigma * dt.sqrt() * self.rng.normal();
        }
        let (wanted_yaw, wanted_pitch) = if aim_point.is_some() {
            (
                wanted_yaw + self.aim_error[0],
                wanted_pitch + self.aim_error[1],
            )
        } else {
            (wanted_yaw, wanted_pitch)
        };
        let turn = self.skill.turn_rate * dt;
        let yaw = fighter.yaw + angle_delta(fighter.yaw, wanted_yaw).clamp(-turn, turn);
        let pitch = fighter.pitch + (wanted_pitch - fighter.pitch).clamp(-turn, turn);

        // Fire when the view is within the cone around the target
        let fire = aim_point.is_some_and(|point| {
            let Some(enemy) = target else {
                return false;
            };
            let gap = distance(point, eye).max(1.0);
            let dps = world
                .calibration
                .dps(fighter.weapon, distance(enemy.center(), eye));
            if dps <= 0.0 {
                return false;
            }
            let (forward, _, _) = view_basis(yaw, pitch);
            let off = dot(forward, normalize(sub(point, eye)))
                .clamp(-1.0, 1.0)
                .acos();
            let radius = (TARGET_RADIUS / gap).atan();
            off <= radius * self.genome.get(genome::FIRE_CONE)
        });
        (
            yaw.rem_euclid(360.0),
            pitch.clamp(-MAX_PITCH, MAX_PITCH),
            fire,
        )
    }

    /// Where to aim at `enemy` with the current weapon: the center, or for projectiles where the
    /// enemy will be (linear lead, × the genome's lead), splash weapons maybe at the feet.
    fn aim_point(&mut self, fighter: &Fighter, enemy: &Fighter, world: &World) -> Vec3 {
        let definition = &world.fight.weapons[fighter.weapon];
        let eye = fighter.eye();
        let center = enemy.center();
        if definition.kind != Kind::Projectile {
            return center;
        }
        let mut point = center;
        if definition.splash > 0.0
            && enemy.state.on_ground
            && self.rng.unit() < self.genome.get(genome::FEET_AIM)
        {
            point = add(enemy.state.origin, [0.0, 0.0, -20.0]);
        }
        let speed = definition.speed.max(1.0);
        let mut time = distance(point, eye) / speed;
        for _ in 0..2 {
            let ahead = add(
                point,
                scale(enemy.state.velocity, time * self.genome.get(genome::LEAD)),
            );
            time = distance(ahead, eye) / speed;
        }
        add(
            point,
            scale(enemy.state.velocity, time * self.genome.get(genome::LEAD)),
        )
    }
}

/// The arena spot `index`: health items, then other items, then spawns.
fn spot(arena: &Arena, index: usize) -> Vec3 {
    let health = arena.health.len();
    let spots = arena.spots.len();
    if index < health {
        arena.health[index]
    } else if index < health + spots {
        arena.spots[index - health]
    } else {
        arena.spawns[index - health - spots].origin
    }
}

/// Whether moving 72 units (the box clear of its spot) along `direction` keeps ground under the
/// box within a step and a little (no walking off ledges while fighting).
fn safe_ground(collision: &QuakeCollision, origin: Vec3, direction: Vec3) -> bool {
    let ahead = add(origin, scale(direction, 72.0));
    let line = trace_player(collision, origin, ahead);
    let end = line.end;
    let down = trace_player(collision, end, [end[0], end[1], end[2] - 64.0]);
    down.fraction < 1.0
}

#[cfg(test)]
mod tests {
    use super::super::genome::Calibration;
    use super::*;
    use crate::sim::{
        arena::{box_world, Spawn},
        combat::Match,
        weapons::WEAPONS,
        STEP,
    };

    fn calibration() -> Calibration {
        // Axe: only up close; shotgun: falling with distance
        Calibration {
            dps: vec![
                [30.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    45.0, 45.0, 40.0, 35.0, 28.0, 22.0, 15.0, 10.0, 5.0, 3.0, 1.0, 0.0,
                ],
            ],
        }
    }

    fn arena_of(spawns: &[[f32; 3]], health: Vec<Vec3>) -> Arena {
        let spawns = spawns
            .iter()
            .map(|&origin| Spawn { origin, yaw: 0.0 })
            .collect();
        Arena::from_points(spawns, health, Vec::new())
    }

    /// Runs bot 0 (and bot 1, idle unless `both`) for `seconds`; returns the match.
    fn run(
        collision: &QuakeCollision,
        arena: &Arena,
        bot: &mut Bot,
        fight: &mut Match,
        seconds: f32,
        mut each: impl FnMut(&Match, &Command),
    ) {
        let calibration = calibration();
        let nav = NavGraph::of(collision, arena);
        let mut routes = RouteCache::new(8);
        for _ in 0..(seconds / STEP) as usize {
            let command = {
                let mut world = World {
                    collision,
                    arena,
                    nav: &nav,
                    fight,
                    calibration: &calibration,
                    routes: &mut routes,
                };
                bot.command(0, &mut world, STEP)
            };
            let idle = Command {
                input: PlayerInput {
                    yaw: fight.fighters.get(1).map_or(0.0, |f| f.yaw),
                    ..PlayerInput::default()
                },
                ..Command::default()
            };
            fight.step(collision, arena, &[command, idle], STEP);
            each(fight, &command);
        }
    }

    const Z: f32 = 24.0 + 1.0 / 32.0;

    #[test]
    fn bot_on_flat_floor_reaches_far_goal_by_route() {
        let collision = box_world(&[
            ([-1024, -256, -64], [1024, 256, 0]),
            // A wall across the middle with a gap at one end
            ([-16, -256, 0], [16, 160, 128]),
        ]);
        let goal = [512.0, 0.0, Z];
        let arena = arena_of(&[[-512.0, 0.0, Z]], vec![goal]);
        let mut fight = Match::new(&WEAPONS, &arena, 1, 1);
        fight.fighters[0].health = 50.0;
        let mut bot = Bot::new(Genome::default(), Skill::at(1.0), 2);
        let mut reached = false;
        run(&collision, &arena, &mut bot, &mut fight, 8.0, |fight, _| {
            reached |= horizontal_distance(fight.fighters[0].state.origin, goal) < 32.0;
        });
        assert!(reached, "ended at {:?}", fight.fighters[0].state.origin);
    }

    #[test]
    fn bot_crosses_gap_by_jump_edge() {
        let collision = box_world(&[
            ([-512, -128, -64], [0, 128, 0]),
            ([128, -128, -64], [640, 128, 0]),
            ([-512, -128, -1024], [640, 128, -960]),
        ]);
        let goal = [448.0, 0.0, Z];
        let arena = arena_of(&[[-256.0, 0.0, Z]], vec![goal]);
        let mut fight = Match::new(&WEAPONS, &arena, 1, 1);
        fight.fighters[0].health = 50.0;
        let mut bot = Bot::new(Genome::default(), Skill::at(1.0), 3);
        let mut reached = false;
        run(
            &collision,
            &arena,
            &mut bot,
            &mut fight,
            10.0,
            |fight, _| {
                reached |= horizontal_distance(fight.fighters[0].state.origin, goal) < 48.0
                    && fight.fighters[0].state.origin[2] > 0.0;
            },
        );
        assert!(reached, "ended at {:?}", fight.fighters[0].state.origin);
    }

    /// Bot 0 at the origin facing +x; bot 1 (idle target) behind it at −x 256.
    fn behind() -> (QuakeCollision, Arena, Match) {
        let collision = box_world(&[([-1024, -1024, -64], [1024, 1024, 0])]);
        let arena = arena_of(&[[0.0, 0.0, Z], [-256.0, 0.0, Z]], Vec::new());
        let mut fight = Match::new(&WEAPONS, &arena, 2, 5);
        fight.fighters[0].state.origin = [0.0, 0.0, Z];
        fight.fighters[0].yaw = 0.0;
        fight.fighters[1].state.origin = [-256.0, 0.0, Z];
        fight.fighters[1].yaw = 0.0;
        // Bot 1 hit bot 0: bot 0 notices it although it's behind
        fight.fighters[0].last_attacker = Some(1);
        (collision, arena, fight)
    }

    #[test]
    fn bot_turns_no_faster_than_turn_rate() {
        let (collision, arena, mut fight) = behind();
        let skill = Skill::at(0.0);
        let mut bot = Bot::new(Genome::default(), skill, 4);
        let mut previous = fight.fighters[0].yaw;
        let mut fastest = 0.0f32;
        run(&collision, &arena, &mut bot, &mut fight, 1.0, |fight, _| {
            let yaw = fight.fighters[0].yaw;
            fastest = fastest.max(angle_delta(previous, yaw).abs() / STEP);
            previous = yaw;
        });
        assert!(fastest <= skill.turn_rate * 1.01, "turned {fastest}°/s");
    }

    #[test]
    fn bot_waits_reaction_time_before_firing() {
        let (collision, arena, mut fight) = behind();
        let skill = Skill::at(1.0);
        let mut bot = Bot::new(Genome::default(), skill, 6);
        let mut first_shot = None;
        let mut time = 0.0;
        run(
            &collision,
            &arena,
            &mut bot,
            &mut fight,
            3.0,
            |_, command| {
                time += STEP;
                if command.fire && first_shot.is_none() {
                    first_shot = Some(time);
                }
            },
        );
        let first_shot = first_shot.expect("never fired");
        assert!(first_shot >= skill.reaction, "fired after {first_shot}");
    }

    #[test]
    fn projectile_lead_hits_target_running_sideways() {
        let rocket = crate::sim::weapons::WeaponDef {
            name: "rocket",
            kind: Kind::Projectile,
            damage: 100.0,
            pellets: 1,
            spread: 0.0,
            range: 4096.0,
            refire: 0.8,
            delay: 0.0,
            speed: 1000.0,
            splash: 0.0,
            self_damage: 0.0,
        };
        let collision = box_world(&[([-2048, -2048, -64], [2048, 2048, 0])]);
        let arena = arena_of(&[[0.0, 0.0, Z], [600.0, 0.0, Z]], Vec::new());
        let weapons = [WEAPONS[0], rocket];
        let mut fight = Match::new(&weapons, &arena, 2, 9);
        fight.fighters[1].state.velocity = [0.0, 320.0, 0.0];
        let shooter = fight.fighters[0].clone();
        let mut bot = Bot::new(Genome::default(), Skill::at(1.0), 10);
        let world = World {
            collision: &collision,
            arena: &arena,
            nav: &NavGraph::of(&collision, &arena),
            fight: &fight,
            calibration: &calibration(),
            routes: &mut RouteCache::new(1),
        };
        let mut shooter = shooter;
        shooter.weapon = 1;
        let point = bot.aim_point(&shooter, &fight.fighters[1], &world);
        // Rocket flight ≈ 0.6 s: about 190 units ahead of the target's y
        assert!((point[1] - 0.6 * 320.0).abs() < 30.0, "aimed at {point:?}");
    }

    #[test]
    fn bot_refuses_strafe_off_ledge() {
        // A narrow ledge (y −48..48) high above a pit
        let collision = box_world(&[
            ([-512, -48, -64], [512, 48, 0]),
            ([-512, -512, -1024], [512, 512, -960]),
        ]);
        assert!(!safe_ground(&collision, [0.0, 0.0, Z], [0.0, 1.0, 0.0]));
        assert!(safe_ground(&collision, [0.0, 0.0, Z], [1.0, 0.0, 0.0]));
    }
}
