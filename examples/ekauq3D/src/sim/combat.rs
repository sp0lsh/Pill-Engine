//! The deathmatch rules, for the player and the bots alike: weapons fire (melee, hitscan,
//! projectiles with splash) against the level and the fighters' boxes, damage, death, respawn at
//! the spawn farthest from enemies, health items and frags. Moves the bots (`player_move`); the
//! players are fighters like the bots, moved by their commands.

use pill_engine::game::{
    quake_map::QuakeCollision,
    quake_player::{
        player_move, MoveEvents, MoveSettings, PlayerInput, PlayerState, EYE_HEIGHT, PLAYER_MAXS,
        PLAYER_MINS,
    },
};

use serde::{Deserialize, Serialize};

use super::{
    add,
    arena::Arena,
    distance, horizontal_distance, normalize, scale, sub, view_basis,
    weapons::{Kind, WeaponDef},
    Rng, Vec3, STEP,
};

pub const MAX_HEALTH: f32 = 100.0;
/// Quake's look limit up and down (degrees).
pub const MAX_PITCH: f32 = 80.0;
const RESPAWN_TIME: f32 = 2.0;
/// Lowering one weapon and raising the next (seconds).
pub const SWITCH_TIME: f32 = 0.3;
const HEALTH_PICKUP: f32 = 25.0;
const HEALTH_RESPAWN: f32 = 20.0;
/// How close (horizontally, vertically) a fighter takes an item.
const PICKUP_REACH: [f32; 2] = [32.0, 48.0];
/// Projectiles leave this far ahead of the eye.
const MUZZLE_AHEAD: f32 = 16.0;

/// What a fighter does this step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Command {
    pub input: PlayerInput,
    pub pitch: f32,
    pub fire: bool,
    pub select: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct Fighter {
    pub state: PlayerState,
    pub yaw: f32,
    pub pitch: f32,
    pub health: f32,
    pub weapon: usize,
    /// Seconds until the next attack.
    pub refire: f32,
    /// Melee swing under way: seconds to the hit, weapon.
    pub pending: Option<(f32, usize)>,
    /// Switching to this weapon, seconds left.
    pub switching: Option<(usize, f32)>,
    pub alive: bool,
    pub respawn: f32,
    pub frags: i32,
    /// What the last move did (jumped, landed): the player's sounds.
    pub moved: MoveEvents,
    /// Takes part (a player slot only while someone plays it).
    pub active: bool,
    pub last_attacker: Option<usize>,
    pub damage_dealt: f32,
}

impl Fighter {
    pub fn eye(&self) -> Vec3 {
        let origin = self.state.origin;
        [origin[0], origin[1], origin[2] + EYE_HEIGHT]
    }

    pub fn center(&self) -> Vec3 {
        let origin = self.state.origin;
        [
            origin[0],
            origin[1],
            origin[2] + (PLAYER_MINS[2] + PLAYER_MAXS[2]) * 0.5,
        ]
    }

    pub fn targetable(&self) -> bool {
        self.alive && self.active
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ImpactKind {
    Wall,
    Melee,
    Flesh,
    Explosion,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Event {
    Fired {
        fighter: usize,
        weapon: usize,
    },
    Switched {
        fighter: usize,
        weapon: usize,
    },
    Impact {
        /// The fighter whose shot it is.
        by: usize,
        point: Vec3,
        normal: Vec3,
        texture: Option<u32>,
        kind: ImpactKind,
    },
    Hit {
        attacker: usize,
        target: usize,
        damage: f32,
    },
    Killed {
        attacker: Option<usize>,
        target: usize,
    },
    Respawned {
        fighter: usize,
    },
    Picked {
        fighter: usize,
        item: usize,
    },
}

#[derive(Clone, Debug)]
pub struct Projectile {
    pub owner: usize,
    pub weapon: usize,
    pub origin: Vec3,
    pub velocity: Vec3,
    pub travelled: f32,
}

pub struct Match {
    pub weapons: Vec<WeaponDef>,
    pub fighters: Vec<Fighter>,
    pub projectiles: Vec<Projectile>,
    /// Per health item, seconds until it is back (≤ 0: there).
    pub health_items: Vec<f32>,
    pub events: Vec<Event>,
    settings: MoveSettings,
    rng: Rng,
}

/// Where the ray `from + t·direction` (t in 0..`length`) enters the box, if it does.
fn ray_box(from: Vec3, direction: Vec3, length: f32, mins: Vec3, maxs: Vec3) -> Option<f32> {
    let mut enter = 0.0f32;
    let mut exit = length;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-8 {
            if from[axis] < mins[axis] || from[axis] > maxs[axis] {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / direction[axis];
        let (mut near, mut far) = (
            (mins[axis] - from[axis]) * inverse,
            (maxs[axis] - from[axis]) * inverse,
        );
        if near > far {
            std::mem::swap(&mut near, &mut far);
        }
        enter = enter.max(near);
        exit = exit.min(far);
        if enter > exit {
            return None;
        }
    }
    Some(enter)
}

impl Match {
    /// `fighters` fighters spawned at the arena's spawns.
    pub fn new(weapons: &[WeaponDef], arena: &Arena, fighters: usize, seed: u32) -> Self {
        let mut created = Self {
            weapons: weapons.to_vec(),
            fighters: Vec::with_capacity(fighters),
            projectiles: Vec::new(),
            health_items: vec![0.0; arena.health.len()],
            events: Vec::new(),
            settings: MoveSettings::default(),
            rng: Rng::new(seed),
        };
        for _ in 0..fighters {
            created.fighters.push(Fighter {
                state: PlayerState::default(),
                yaw: 0.0,
                pitch: 0.0,
                health: 0.0,
                weapon: weapons.len() - 1,
                refire: 0.0,
                pending: None,
                switching: None,
                alive: false,
                respawn: 0.0,
                frags: 0,
                moved: MoveEvents::default(),
                active: true,
                last_attacker: None,
                damage_dealt: 0.0,
            });
        }
        for index in 0..fighters {
            created.spawn(arena, index);
        }
        created.events.clear();
        created
    }

    /// Advances `dt` seconds in steps of at most `STEP`; `commands` per fighter. What happened
    /// is added to `events` (the caller takes them).
    pub fn step(
        &mut self,
        collision: &QuakeCollision,
        arena: &Arena,
        commands: &[Command],
        dt: f32,
    ) {
        let steps = (dt / STEP).ceil().max(1.0);
        let step = dt / steps;
        for _ in 0..steps as usize {
            self.tick(collision, arena, commands, step);
        }
    }

    fn tick(&mut self, collision: &QuakeCollision, arena: &Arena, commands: &[Command], dt: f32) {
        for index in 0..self.fighters.len() {
            let command = commands.get(index).copied().unwrap_or_default();
            let fighter = &mut self.fighters[index];
            if !fighter.active {
                continue;
            }
            if !fighter.alive {
                fighter.respawn -= dt;
                if fighter.respawn <= 0.0 {
                    self.spawn(arena, index);
                }
                continue;
            }
            fighter.yaw = command.input.yaw;
            fighter.pitch = command.pitch.clamp(-MAX_PITCH, MAX_PITCH);
            fighter.moved = player_move(
                &mut fighter.state,
                &command.input,
                dt,
                collision,
                &self.settings,
            );
            self.weapon(collision, index, &command, dt);
        }
        self.fly_projectiles(collision, dt);
        self.pick_up(arena, dt);
    }

    /// Switching, refire, attacks and delayed melee hits of one fighter.
    fn weapon(&mut self, collision: &QuakeCollision, index: usize, command: &Command, dt: f32) {
        let count = self.weapons.len();
        let fighter = &mut self.fighters[index];
        fighter.refire = (fighter.refire - dt).max(0.0);
        if let Some(select) = command.select.filter(|&select| select < count) {
            if select != fighter.weapon && fighter.switching.is_none() {
                fighter.switching = Some((select, SWITCH_TIME));
                fighter.pending = None;
                self.events.push(Event::Switched {
                    fighter: index,
                    weapon: select,
                });
            }
        }
        let fighter = &mut self.fighters[index];
        if let Some((weapon, left)) = fighter.switching {
            let left = left - dt;
            if left <= SWITCH_TIME * 0.5 {
                fighter.weapon = weapon;
            }
            fighter.switching = (left > 0.0).then_some((weapon, left));
        }
        if let Some((left, weapon)) = fighter.pending {
            let left = left - dt;
            if left <= 0.0 {
                fighter.pending = None;
                self.attack(collision, index, weapon);
            } else {
                self.fighters[index].pending = Some((left, weapon));
            }
        }
        let fighter = &mut self.fighters[index];
        if command.fire && fighter.refire <= 0.0 && fighter.switching.is_none() {
            let weapon = fighter.weapon;
            let definition = self.weapons[weapon];
            fighter.refire = definition.refire;
            self.events.push(Event::Fired {
                fighter: index,
                weapon,
            });
            if definition.delay > 0.0 {
                self.fighters[index].pending = Some((definition.delay, weapon));
            } else {
                self.attack(collision, index, weapon);
            }
        }
    }

    /// The nearest active fighter (other than `skip`) the segment enters before `length`.
    fn fighter_on_ray(
        &self,
        from: Vec3,
        direction: Vec3,
        length: f32,
        skip: usize,
        grow: f32,
    ) -> Option<(usize, f32)> {
        let mut nearest = None;
        let mut nearest_distance = length;
        for (index, fighter) in self.fighters.iter().enumerate() {
            if index == skip || !fighter.targetable() {
                continue;
            }
            let origin = fighter.state.origin;
            let mins = std::array::from_fn(|axis| origin[axis] + PLAYER_MINS[axis] - grow);
            let maxs = std::array::from_fn(|axis| origin[axis] + PLAYER_MAXS[axis] + grow);
            if let Some(distance) = ray_box(from, direction, nearest_distance, mins, maxs) {
                nearest_distance = distance;
                nearest = Some((index, distance));
            }
        }
        nearest
    }

    fn attack(&mut self, collision: &QuakeCollision, index: usize, weapon: usize) {
        let definition = self.weapons[weapon];
        let fighter = &self.fighters[index];
        if !fighter.alive {
            return;
        }
        let eye = fighter.eye();
        let (forward, right, up) = view_basis(fighter.yaw, fighter.pitch);
        match definition.kind {
            Kind::Melee | Kind::Hitscan => {
                for _ in 0..definition.pellets {
                    let direction = if definition.spread > 0.0 {
                        let (side, rise) = (self.rng.signed(), self.rng.signed());
                        normalize(add(
                            forward,
                            add(
                                scale(right, side * definition.spread),
                                scale(up, rise * definition.spread),
                            ),
                        ))
                    } else {
                        forward
                    };
                    let end = add(eye, scale(direction, definition.range));
                    let trace = collision.trace_ray(eye, end);
                    let wall = trace.fraction * definition.range;
                    if let Some((target, along)) =
                        self.fighter_on_ray(eye, direction, wall, index, 0.0)
                    {
                        self.events.push(Event::Impact {
                            by: index,
                            point: add(eye, scale(direction, along)),
                            normal: scale(direction, -1.0),
                            texture: None,
                            kind: ImpactKind::Flesh,
                        });
                        self.damage(Some(index), target, definition.damage);
                    } else if trace.fraction < 1.0 && !trace.start_solid {
                        self.events.push(Event::Impact {
                            by: index,
                            point: trace.end,
                            normal: trace.normal,
                            texture: trace.texture,
                            kind: if definition.kind == Kind::Melee {
                                ImpactKind::Melee
                            } else {
                                ImpactKind::Wall
                            },
                        });
                    }
                }
            }
            Kind::Projectile => {
                let start = add(eye, scale(forward, MUZZLE_AHEAD));
                let clear = collision.trace_ray(eye, start);
                let origin = if clear.fraction < 1.0 { eye } else { start };
                self.projectiles.push(Projectile {
                    owner: index,
                    weapon,
                    origin,
                    velocity: scale(forward, definition.speed),
                    travelled: 0.0,
                });
            }
        }
    }

    fn fly_projectiles(&mut self, collision: &QuakeCollision, dt: f32) {
        let mut index = 0;
        while index < self.projectiles.len() {
            let projectile = self.projectiles[index].clone();
            let definition = self.weapons[projectile.weapon];
            let step = scale(projectile.velocity, dt);
            let length = super::length(step);
            let direction = normalize(step);
            let end = add(projectile.origin, step);
            let trace = collision.trace_ray(projectile.origin, end);
            let wall = trace.fraction * length;
            let hit =
                self.fighter_on_ray(projectile.origin, direction, wall, projectile.owner, 4.0);
            let exploded = match hit {
                Some((target, along)) => {
                    self.damage(Some(projectile.owner), target, definition.damage);
                    Some((
                        add(projectile.origin, scale(direction, along)),
                        Some(target),
                        scale(direction, -1.0),
                    ))
                }
                None if trace.fraction < 1.0 => Some((trace.end, None, trace.normal)),
                None => None,
            };
            if let Some((point, direct, normal)) = exploded {
                self.events.push(Event::Impact {
                    by: projectile.owner,
                    point,
                    normal,
                    texture: None,
                    kind: ImpactKind::Explosion,
                });
                self.splash(collision, &projectile, point, direct);
                self.projectiles.swap_remove(index);
                continue;
            }
            let moved = &mut self.projectiles[index];
            moved.origin = end;
            moved.travelled += length;
            if moved.travelled >= definition.range {
                self.projectiles.swap_remove(index);
                continue;
            }
            index += 1;
        }
    }

    /// Splash damage around an explosion, falling linearly to its radius; not through walls.
    fn splash(
        &mut self,
        collision: &QuakeCollision,
        projectile: &Projectile,
        point: Vec3,
        direct: Option<usize>,
    ) {
        let definition = self.weapons[projectile.weapon];
        if definition.splash <= 0.0 {
            return;
        }
        for target in 0..self.fighters.len() {
            if Some(target) == direct || !self.fighters[target].targetable() {
                continue;
            }
            let center = self.fighters[target].center();
            let distance = (distance(point, center) - 16.0).max(0.0);
            if distance >= definition.splash {
                continue;
            }
            if collision.trace_ray(point, center).fraction < 1.0 {
                continue;
            }
            let mut damage = definition.damage * (1.0 - distance / definition.splash);
            if target == projectile.owner {
                damage *= definition.self_damage;
            }
            if damage > 0.0 {
                self.damage(Some(projectile.owner), target, damage);
            }
        }
    }

    fn damage(&mut self, attacker: Option<usize>, target: usize, damage: f32) {
        let fighter = &mut self.fighters[target];
        if !fighter.targetable() {
            return;
        }
        fighter.health -= damage;
        if attacker != Some(target) {
            fighter.last_attacker = attacker;
        }
        if let Some(attacker) = attacker.filter(|&attacker| attacker != target) {
            self.fighters[attacker].damage_dealt += damage;
            self.events.push(Event::Hit {
                attacker,
                target,
                damage,
            });
        }
        let fighter = &mut self.fighters[target];
        if fighter.health > 0.0 {
            return;
        }
        fighter.alive = false;
        fighter.respawn = RESPAWN_TIME;
        fighter.pending = None;
        fighter.switching = None;
        match attacker {
            Some(attacker) if attacker != target => self.fighters[attacker].frags += 1,
            _ => self.fighters[target].frags -= 1,
        }
        self.events.push(Event::Killed { attacker, target });
    }

    /// Brings fighter `index` back at the spawn farthest from the living enemies.
    pub fn spawn(&mut self, arena: &Arena, index: usize) {
        if arena.spawns.is_empty() {
            return;
        }
        let enemies: Vec<Vec3> = self
            .fighters
            .iter()
            .enumerate()
            .filter(|(other, fighter)| *other != index && fighter.targetable())
            .map(|(_, fighter)| fighter.state.origin)
            .collect();
        let spawn = arena
            .spawns
            .iter()
            .max_by(|a, b| {
                let nearest = |origin: Vec3| {
                    enemies
                        .iter()
                        .map(|&enemy| distance(enemy, origin))
                        .fold(f32::MAX, f32::min)
                };
                // A little noise so equal spawns alternate
                let jitter = self.rng.unit() - self.rng.unit();
                (nearest(a.origin) + jitter).total_cmp(&nearest(b.origin))
            })
            .copied()
            .expect("spawns");
        let weapon = self.weapons.len() - 1;
        let fighter = &mut self.fighters[index];
        fighter.state = PlayerState {
            origin: spawn.origin,
            ..PlayerState::default()
        };
        fighter.yaw = spawn.yaw;
        fighter.pitch = 0.0;
        fighter.health = MAX_HEALTH;
        fighter.weapon = weapon;
        fighter.refire = 0.0;
        fighter.alive = true;
        fighter.last_attacker = None;
        self.events.push(Event::Respawned { fighter: index });
    }

    fn pick_up(&mut self, arena: &Arena, dt: f32) {
        for (item, spot) in arena.health.iter().enumerate() {
            let left = &mut self.health_items[item];
            if *left > 0.0 {
                *left -= dt;
                continue;
            }
            for (index, fighter) in self.fighters.iter_mut().enumerate() {
                let origin = fighter.state.origin;
                if fighter.targetable()
                    && fighter.health < MAX_HEALTH
                    && horizontal_distance(origin, *spot) <= PICKUP_REACH[0]
                    && (origin[2] - spot[2]).abs() <= PICKUP_REACH[1]
                {
                    fighter.health = (fighter.health + HEALTH_PICKUP).min(MAX_HEALTH);
                    *left = HEALTH_RESPAWN;
                    self.events.push(Event::Picked {
                        fighter: index,
                        item,
                    });
                    break;
                }
            }
        }
    }

    /// Whether `from` sees `to` (eye to eye or eye to center, through no wall).
    pub fn visible(collision: &QuakeCollision, from: &Fighter, to: &Fighter) -> bool {
        let eye = from.eye();
        collision.trace_ray(eye, to.eye()).fraction >= 1.0
            || collision.trace_ray(eye, to.center()).fraction >= 1.0
    }
}

pub fn direction_to(from: Vec3, to: Vec3) -> Vec3 {
    normalize(sub(to, from))
}

#[cfg(test)]
mod tests {
    use super::super::{
        arena::{box_world, Spawn},
        weapons::WEAPONS,
    };
    use super::*;

    const SHOTGUN: usize = 1;
    const AXE: usize = 0;

    /// A flat floor (top z 0) with two spawns `distance` apart on x, facing each other.
    fn duel(distance: f32, boxes: &[([i32; 3], [i32; 3])]) -> (QuakeCollision, Arena) {
        let mut all = vec![([-1024, -1024, -64], [2048, 1024, 0])];
        all.extend_from_slice(boxes);
        let collision = box_world(&all);
        let z = 24.0 + 1.0 / 32.0;
        let spawns = vec![
            Spawn {
                origin: [0.0, 0.0, z],
                yaw: 0.0,
            },
            Spawn {
                origin: [distance, 0.0, z],
                yaw: 180.0,
            },
        ];
        let arena = Arena::from_points(spawns, Vec::new(), Vec::new());
        (collision, arena)
    }

    fn placed(arena: &Arena, distance: f32) -> Match {
        let mut fight = Match::new(&WEAPONS, arena, 2, 7);
        let z = 24.0 + 1.0 / 32.0;
        fight.fighters[0].state.origin = [0.0, 0.0, z];
        fight.fighters[0].yaw = 0.0;
        fight.fighters[1].state.origin = [distance, 0.0, z];
        fight.fighters[1].yaw = 180.0;
        fight
    }

    /// Fighter 0 fires straight ahead, fighter 1 stands.
    fn fire() -> [Command; 2] {
        let mut first = Command {
            fire: true,
            ..Command::default()
        };
        first.input.yaw = 0.0;
        let mut idle = Command::default();
        idle.input.yaw = 180.0;
        [first, idle]
    }

    #[test]
    fn shotgun_point_blank_hits_with_all_six_pellets() {
        let (collision, arena) = duel(64.0, &[]);
        let mut fight = placed(&arena, 64.0);
        fight.fighters[0].weapon = SHOTGUN;
        fight.step(&collision, &arena, &fire(), STEP);
        assert_eq!(fight.fighters[1].health, MAX_HEALTH - 24.0);
    }

    #[test]
    fn hitscan_stops_at_wall_in_front_of_target() {
        let (collision, arena) = duel(256.0, &[([120, -256, 0], [136, 256, 256])]);
        let mut fight = placed(&arena, 256.0);
        fight.fighters[0].weapon = SHOTGUN;
        fight.step(&collision, &arena, &fire(), STEP);
        assert_eq!(fight.fighters[1].health, MAX_HEALTH);
        assert!(fight.events.iter().any(|event| matches!(
            event,
            Event::Impact {
                kind: ImpactKind::Wall,
                ..
            }
        )));
    }

    #[test]
    fn axe_hits_only_within_range_after_delay() {
        let (collision, arena) = duel(48.0, &[]);
        let mut fight = placed(&arena, 48.0);
        fight.fighters[0].weapon = AXE;
        fight.step(&collision, &arena, &fire(), STEP);
        assert_eq!(
            fight.fighters[1].health, MAX_HEALTH,
            "hit before the swing lands"
        );
        let idle = [Command::default(), Command::default()];
        fight.step(&collision, &arena, &idle, 0.25);
        assert_eq!(fight.fighters[1].health, MAX_HEALTH - 20.0);

        let (collision, arena) = duel(128.0, &[]);
        let mut fight = placed(&arena, 128.0);
        fight.fighters[0].weapon = AXE;
        fight.step(&collision, &arena, &fire(), STEP);
        fight.step(&collision, &arena, &idle, 0.25);
        assert_eq!(fight.fighters[1].health, MAX_HEALTH);
    }

    #[test]
    fn rocket_splash_hurts_shooter_scaled_by_self_damage() {
        let rocket = WeaponDef {
            name: "rocket",
            kind: Kind::Projectile,
            damage: 100.0,
            pellets: 1,
            spread: 0.0,
            range: 4096.0,
            refire: 0.8,
            delay: 0.0,
            speed: 1000.0,
            splash: 120.0,
            self_damage: 0.5,
        };
        // A wall right in front of the shooter's eye
        let (collision, arena) = duel(512.0, &[([40, -256, 0], [56, 256, 256])]);
        let mut fight = Match::new(&[WEAPONS[0], rocket], &arena, 2, 3);
        let z = 24.0 + 1.0 / 32.0;
        fight.fighters[0].state.origin = [0.0, 0.0, z];
        fight.fighters[1].state.origin = [512.0, 0.0, z];
        fight.fighters[0].weapon = 1;
        let mut commands = fire();
        commands[0].pitch = -30.0;
        fight.step(&collision, &arena, &commands, STEP);
        let idle = [Command::default(), Command::default()];
        fight.step(&collision, &arena, &idle, 0.2);
        let lost = MAX_HEALTH - fight.fighters[0].health;
        assert!(lost > 5.0 && lost <= 50.0, "lost {lost}");
        assert_eq!(fight.fighters[1].health, MAX_HEALTH);
    }

    #[test]
    fn killed_fighter_respawns_at_spawn_farthest_from_enemies() {
        let (collision, arena) = duel(64.0, &[]);
        let mut fight = placed(&arena, 64.0);
        fight.fighters[0].weapon = SHOTGUN;
        fight.fighters[1].health = 10.0;
        fight.step(&collision, &arena, &fire(), STEP);
        assert!(!fight.fighters[1].alive);
        assert_eq!(fight.fighters[0].frags, 1);
        // Shooter stands at the first spawn: the victim comes back at the second
        fight.fighters[0].state.origin = arena.spawns[0].origin;
        let idle = [Command::default(), Command::default()];
        fight.step(&collision, &arena, &idle, RESPAWN_TIME + 0.1);
        assert!(fight.fighters[1].alive);
        assert!(distance(fight.fighters[1].state.origin, arena.spawns[1].origin) < 1.0);
    }

    #[test]
    fn health_pickup_heals_and_respawns_after_twenty_seconds() {
        let (collision, mut arena) = duel(512.0, &[]);
        let z = 24.0 + 1.0 / 32.0;
        arena.health = vec![[0.0, 0.0, z]];
        let mut fight = placed(&arena, 512.0);
        fight.health_items = vec![0.0];
        fight.fighters[0].health = 50.0;
        let idle = [Command::default(), Command::default()];
        fight.step(&collision, &arena, &idle, STEP);
        assert_eq!(fight.fighters[0].health, 75.0);
        fight.step(&collision, &arena, &idle, 10.0);
        assert_eq!(fight.fighters[0].health, 75.0, "item back too early");
        fight.step(&collision, &arena, &idle, 10.5);
        assert_eq!(fight.fighters[0].health, 100.0);
    }
}
