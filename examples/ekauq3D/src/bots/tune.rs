//! Self-play tuning of the genome (native): bot duels on real maps, a simple evolution strategy
//! (diagonal, rank-weighted recombination of the best) in the genome's 0..1 space; every candidate
//! of a generation meets the same opponents, maps and seeds (common random numbers). After the
//! Quake III bot's genetic selection of fuzzy weights (van Waveren 2001, §9.4) and population-based
//! training (Jaderberg et al. 2019).

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

use pill_engine::game::quake_map::QuakeCollision;

use super::{
    brain::{Bot, World},
    genome::{Calibration, Genome, Skill, GENES},
    nav::{NavGraph, RouteCache},
};
use crate::sim::{arena::Arena, combat::Match, weapons::WeaponDef, Rng, STEP};

/// Duel outcome for the first bot: frag difference, damage difference.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    pub frags: i32,
    pub damage: f32,
    /// Per bot: frags, damage dealt.
    pub kills: [i32; 2],
    pub dealt: [f32; 2],
}

impl Outcome {
    pub fn score(&self) -> f32 {
        self.frags as f32 + self.damage * 0.01
    }
}

/// A map as the tuner plays it: collision, arena, navigation (`loader::load_arena`).
pub type Map = (QuakeCollision, Arena, NavGraph);

/// Two bots fight for `seconds` on a map.
pub fn duel(
    (collision, arena, nav): &Map,
    weapons: &[WeaponDef],
    calibration: &Calibration,
    genomes: [Genome; 2],
    skill: Skill,
    seconds: f32,
    seed: u32,
) -> Outcome {
    let mut fight = Match::new(weapons, arena, 2, seed);
    let mut bots = [
        Bot::new(genomes[0], skill, seed.wrapping_mul(3) + 1),
        Bot::new(genomes[1], skill, seed.wrapping_mul(5) + 2),
    ];
    let mut routes = RouteCache::for_arena(arena);
    let mut commands = [Default::default(); 2];
    for _ in 0..(seconds / STEP) as usize {
        for (index, bot) in bots.iter_mut().enumerate() {
            let mut world = World {
                collision,
                arena,
                nav,
                fight: &fight,
                calibration,
                routes: &mut routes,
            };
            commands[index] = bot.command(index, &mut world, STEP);
        }
        fight.step(collision, arena, &commands, STEP);
        fight.events.clear();
    }
    let [first, second] = [&fight.fighters[0], &fight.fighters[1]];
    Outcome {
        frags: first.frags - second.frags,
        damage: first.damage_dealt - second.damage_dealt,
        kills: [first.frags, second.frags],
        dealt: [first.damage_dealt, second.damage_dealt],
    }
}

pub struct Settings {
    pub generations: usize,
    pub population: usize,
    pub parents: usize,
    /// Seconds per duel.
    pub seconds: f32,
    /// Duels per map and opponent.
    pub seeds: u32,
    pub threads: usize,
    pub skill: Skill,
}

/// Runs `jobs` jobs on `threads` threads; results in job order.
fn parallel<T: Send>(jobs: usize, threads: usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<T>>> = Mutex::new((0..jobs).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= jobs {
                    break;
                }
                let result = job(index);
                results.lock().expect("results")[index] = Some(result);
            });
        }
    });
    results
        .into_inner()
        .expect("results")
        .into_iter()
        .map(|result| result.expect("every job ran"))
        .collect()
}

/// Evolves the genome from `start`; `report(generation, best score, mean genome)` after each
/// generation. Returns the final mean (the recombined best).
pub fn tune(
    maps: &[Map],
    weapons: &[WeaponDef],
    calibration: &Calibration,
    start: Genome,
    settings: &Settings,
    seed: u32,
    mut report: impl FnMut(usize, f32, &Genome),
) -> Genome {
    let mut rng = Rng::new(seed);
    let mut mean: [f32; GENES] = std::array::from_fn(|gene| start.normalized(gene));
    let mut sigma = [0.15f32; GENES];
    let mut champions: Vec<Genome> = Vec::new();
    // Rank weights: log-decreasing over the parents
    let weights: Vec<f32> = (0..settings.parents)
        .map(|rank| ((settings.parents as f32 + 0.5).ln() - (rank as f32 + 1.0).ln()).max(0.0))
        .collect();
    let total: f32 = weights.iter().sum();
    for generation in 0..settings.generations {
        let candidates: Vec<[f32; GENES]> = (0..settings.population)
            .map(|_| {
                std::array::from_fn(|gene| {
                    (mean[gene] + sigma[gene] * rng.normal()).clamp(0.0, 1.0)
                })
            })
            .collect();
        // Opponents: the default genome, the current mean, the last champions
        let mut opponents = vec![Genome::default(), Genome::from_normalized(mean)];
        opponents.extend(champions.iter().rev().take(2).copied());
        let seeds: Vec<u32> = (0..maps.len() as u32 * settings.seeds)
            .map(|_| rng.next_u32())
            .collect();
        let per_candidate = maps.len() * settings.seeds as usize * opponents.len();
        let outcomes = parallel(candidates.len() * per_candidate, settings.threads, |job| {
            let candidate = Genome::from_normalized(candidates[job / per_candidate]);
            let within = job % per_candidate;
            let opponent = opponents[within % opponents.len()];
            let duel_index = within / opponents.len();
            duel(
                &maps[duel_index % maps.len()],
                weapons,
                calibration,
                [candidate, opponent],
                settings.skill,
                settings.seconds,
                seeds[duel_index],
            )
            .score()
        });
        let mut scored: Vec<(f32, usize)> = (0..candidates.len())
            .map(|candidate| {
                let scores = &outcomes[candidate * per_candidate..(candidate + 1) * per_candidate];
                (scores.iter().sum::<f32>() / per_candidate as f32, candidate)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let old = mean;
        for gene in 0..GENES {
            let mut value = 0.0;
            let mut spread = 0.0;
            for (rank, &(_, candidate)) in scored.iter().take(settings.parents).enumerate() {
                let x = candidates[candidate][gene];
                value += weights[rank] / total * x;
                spread += weights[rank] / total * (x - old[gene]).powi(2);
            }
            mean[gene] = value;
            sigma[gene] = (0.7 * sigma[gene] + 0.3 * spread.sqrt()).clamp(0.02, 0.3);
        }
        champions.push(Genome::from_normalized(candidates[scored[0].1]));
        report(generation, scored[0].0, &Genome::from_normalized(mean));
    }
    Genome::from_normalized(mean)
}

#[cfg(test)]
mod tests {
    use super::super::{calibrate::calibrate, genome};
    use super::*;
    use crate::sim::{
        arena::{box_world, Spawn},
        weapons::WEAPONS,
    };

    #[test]
    fn evolution_beats_a_poor_genome_on_flat_arena() {
        let collision = box_world(&[
            ([-1024, -1024, -64], [1024, 1024, 0]),
            ([-64, -64, 0], [64, 64, 128]),
        ]);
        let z = 24.0 + 1.0 / 32.0;
        let spawns = vec![
            Spawn {
                origin: [-512.0, -512.0, z],
                yaw: 45.0,
            },
            Spawn {
                origin: [512.0, 512.0, z],
                yaw: 225.0,
            },
        ];
        let arena = Arena::from_points(spawns, Vec::new(), Vec::new());
        let calibration = calibrate(&WEAPONS, 1);
        // A poor start: never fires unless dead on target, no strafing
        let mut poor = Genome::default();
        poor.0[genome::FIRE_CONE] = 0.5;
        poor.0[genome::STRAFE] = 0.0;
        poor.0[genome::PREFERRED_RANGE] = 1.5;
        let nav = NavGraph::of(&collision, &arena);
        let maps = vec![(collision, arena, nav)];
        let settings = Settings {
            generations: 4,
            population: 8,
            parents: 3,
            seconds: 30.0,
            seeds: 2,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            skill: Skill::at(1.0),
        };
        let tuned = tune(
            &maps,
            &WEAPONS,
            &calibration,
            poor,
            &settings,
            11,
            |_, _, _| {},
        );
        let total: f32 = (0..6)
            .map(|seed| {
                duel(
                    &maps[0],
                    &WEAPONS,
                    &calibration,
                    [tuned, poor],
                    Skill::at(1.0),
                    30.0,
                    100 + seed,
                )
                .score()
            })
            .sum();
        assert!(total > 0.0, "tuned scored {total} against the start");
    }
}
