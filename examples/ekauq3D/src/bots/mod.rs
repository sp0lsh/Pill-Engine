//! Bots playing the deathmatch (`crate::sim`) like players: their own commands every tick.
//!
//! - `nav`: navigation built from the level at load (floor flood fill, jumps simulated with the
//!   player's own movement), routing
//! - `brain`: the bot (think at 10 Hz, steer every step), skill limited like a human
//! - `genome`: tactics parameters, skill, the knowledge file the tuner writes
//! - `calibrate`: per weapon damage per second over distance, measured in the simulation
//! - `tune`: evolution of the genome by self-play (native)
//!
//! Approach after van Waveren, "The Quake III Arena Bot" (2001); see `bots_quake_I-III.md`.

pub mod brain;
pub mod calibrate;
pub mod genome;
pub mod nav;
#[cfg(not(target_arch = "wasm32"))]
pub mod tune;

use pill_engine::game::quake_map::QuakeCollision;

use crate::sim::{arena::Arena, combat::Command, session::Session, STEP};
use brain::{Bot, World};
use genome::{Knowledge, Skill};
use nav::{NavGraph, RouteCache};

/// The bots of a session: a brain per bot fighter (none for player slots), the map's navigation.
pub struct Bots {
    pub brains: Vec<Option<Bot>>,
    pub nav: NavGraph,
    pub knowledge: Knowledge,
    routes: RouteCache,
}

impl Bots {
    /// Brains for the session's bot slots (after its player slots), at its difficulty.
    pub fn new(session: &Session, arena: &Arena, nav: NavGraph, knowledge: Knowledge) -> Self {
        let skill = Skill::at(session.level);
        let seed = session.seed;
        let brains = (0..session.fight.fighters.len())
            .map(|index| {
                (index >= session.players).then(|| {
                    Bot::new(
                        knowledge.genome,
                        skill,
                        seed.wrapping_add(100 + index as u32),
                    )
                })
            })
            .collect();
        Self {
            brains,
            nav,
            knowledge,
            routes: RouteCache::for_arena(arena),
        }
    }

    /// Everyone's commands for the next tick: the bots think, players' come from `players`
    /// (slot, command); a slot without one stands still.
    pub fn commands(
        &mut self,
        session: &Session,
        collision: &QuakeCollision,
        arena: &Arena,
        players: &[(usize, Command)],
    ) -> Vec<Command> {
        let mut commands = Vec::with_capacity(self.brains.len());
        for index in 0..self.brains.len() {
            let fighter = &session.fight.fighters[index];
            let mut command = Command::default();
            command.input.yaw = fighter.yaw;
            command.pitch = fighter.pitch;
            if let Some(bot) = self.brains[index].as_mut() {
                let mut world = World {
                    collision,
                    arena,
                    nav: &self.nav,
                    fight: &session.fight,
                    calibration: &self.knowledge.calibration,
                    routes: &mut self.routes,
                };
                command = bot.command(index, &mut world, STEP);
            } else if let Some((_, given)) = players.iter().find(|(slot, _)| *slot == index) {
                command = *given;
            }
            commands.push(command);
        }
        commands
    }

    /// The difficulty changed (after a match).
    pub fn set_level(&mut self, level: f32) {
        let skill = Skill::at(level);
        for bot in self.brains.iter_mut().flatten() {
            bot.skill = skill;
        }
    }
}
