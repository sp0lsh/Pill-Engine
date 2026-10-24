//! What each weapon does, measured: a bot (reference skill) standing on a flat range shoots a
//! target strafing at each distance of `BINS`; damage per second per weapon and distance. Stands
//! in for the Quake III bot's hand-written weapon weights (van Waveren 2001, §9).

use pill_engine::game::quake_player::PlayerInput;

use super::{
    brain::{Bot, World},
    genome::{Calibration, Genome, Skill, BINS},
    nav::{NavGraph, RouteCache},
};
use crate::sim::{
    angles_of,
    arena::{box_world, Arena, Spawn},
    combat::{Command, Event, Match, MAX_HEALTH},
    horizontal_distance, sub,
    weapons::WeaponDef,
    Rng, STEP,
};

/// Seconds measured per weapon and distance.
const SECONDS: f32 = 20.0;
/// Difficulty level the measurement is made at.
pub const REFERENCE_LEVEL: f32 = 0.75;
const Z: f32 = 24.0 + 1.0 / 32.0;

/// Damage 1 wherever a weapon reaches: enough for the measuring bot to pull the trigger.
fn nominal(weapons: &[WeaponDef]) -> Calibration {
    Calibration {
        dps: weapons
            .iter()
            .map(|weapon| {
                std::array::from_fn(|bin| (BINS[bin] <= weapon.range + 16.0) as u8 as f32)
            })
            .collect(),
    }
}

pub fn calibrate(weapons: &[WeaponDef], seed: u32) -> Calibration {
    let collision = box_world(&[([-4096, -4096, -64], [4096, 4096, 0])]);
    let reach = nominal(weapons);
    // Shooter and target at each distance (the same for every weapon)
    let ranges: Vec<(Arena, NavGraph)> = BINS
        .iter()
        .map(|&range| {
            let spawns = vec![
                Spawn {
                    origin: [0.0, 0.0, Z],
                    yaw: 0.0,
                },
                Spawn {
                    origin: [range, 0.0, Z],
                    yaw: 180.0,
                },
            ];
            let arena = Arena::from_points(spawns, Vec::new(), Vec::new());
            let nav = NavGraph::of(&collision, &arena);
            (arena, nav)
        })
        .collect();
    let mut dps = Vec::with_capacity(weapons.len());
    for weapon in 0..weapons.len() {
        let mut curve = [0.0; BINS.len()];
        for (bin, &range) in BINS.iter().enumerate() {
            let (arena, nav) = &ranges[bin];
            curve[bin] = measure(
                &collision,
                arena,
                nav,
                weapons,
                weapon,
                range,
                &reach,
                seed + bin as u32,
            );
        }
        dps.push(curve);
    }
    Calibration { dps }
}

fn measure(
    collision: &pill_engine::game::quake_map::QuakeCollision,
    arena: &Arena,
    nav: &NavGraph,
    weapons: &[WeaponDef],
    weapon: usize,
    range: f32,
    reach: &Calibration,
    seed: u32,
) -> f32 {
    let mut fight = Match::new(weapons, arena, 2, seed);
    fight.fighters[0].state.origin = [0.0, 0.0, Z];
    fight.fighters[0].yaw = 0.0;
    fight.fighters[0].weapon = weapon;
    fight.fighters[1].state.origin = [range, 0.0, Z];
    fight.fighters[0].last_attacker = Some(1);
    let mut bot = Bot::new(Genome::default(), Skill::at(REFERENCE_LEVEL), seed);
    let mut routes = RouteCache::new(1);
    let mut rng = Rng::new(seed ^ 0x5eed);
    let mut strafe = 1.0;
    let mut strafe_left = 0.0;
    let mut damage = 0.0;
    for _ in 0..(SECONDS / STEP) as usize {
        let mut shooter = {
            let mut world = World {
                collision,
                arena,
                nav,
                fight: &fight,
                calibration: reach,
                routes: &mut routes,
            };
            bot.command(0, &mut world, STEP)
        };
        // The shooter stands and keeps its weapon: only aim and trigger are measured
        shooter.input.forward = 0.0;
        shooter.input.side = 0.0;
        shooter.input.jump = false;
        shooter.select = None;
        // The target strafes at the distance, facing the shooter
        strafe_left -= STEP;
        if strafe_left <= 0.0 {
            strafe = -strafe;
            strafe_left = rng.range(0.4, 1.0);
        }
        let target = fight.fighters[1].state.origin;
        let (yaw, _) = angles_of(sub(fight.fighters[0].state.origin, target));
        let off = horizontal_distance(target, fight.fighters[0].state.origin) - range;
        let target = Command {
            input: PlayerInput {
                forward: (off / 64.0).clamp(-1.0, 1.0),
                side: strafe,
                jump: false,
                yaw,
            },
            ..Command::default()
        };
        fight.step(collision, arena, &[shooter, target], STEP);
        for event in fight.events.drain(..) {
            if let Event::Hit {
                attacker: 0,
                damage: dealt,
                ..
            } = event
            {
                damage += dealt;
            }
        }
        fight.fighters[1].health = MAX_HEALTH;
        fight.fighters[0].health = MAX_HEALTH;
    }
    damage / SECONDS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::weapons::WEAPONS;

    #[test]
    fn axe_dps_is_zero_beyond_its_range_and_shotgun_falls_with_distance() {
        let calibration = calibrate(&WEAPONS, 1);
        let axe = &calibration.dps[0];
        let shotgun = &calibration.dps[1];
        assert!(axe[0] > 10.0, "axe at 32: {}", axe[0]);
        assert!(axe[3..].iter().all(|&dps| dps == 0.0), "axe {axe:?}");
        assert!(shotgun[1] > shotgun[9] * 2.0, "shotgun {shotgun:?}");
        assert!(shotgun[1] > 20.0, "shotgun {shotgun:?}");
    }

    #[test]
    fn shorter_shotgun_range_measures_no_damage_beyond_it() {
        let mut weapons = WEAPONS;
        weapons[1].range = 512.0;
        let calibration = calibrate(&weapons, 1);
        let shotgun = &calibration.dps[1];
        assert!(shotgun[5] > 0.0, "shotgun {shotgun:?}");
        assert!(
            shotgun[8..].iter().all(|&dps| dps == 0.0),
            "shotgun {shotgun:?}"
        );
    }
}
