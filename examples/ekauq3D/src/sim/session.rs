//! The authoritative deathmatch at a fixed tick (`STEP`): player slots and bot slots, matches to
//! the frag or time limit, the difficulty adapting to the players. Runs the same in the game
//! (single player, listen server), the dedicated server and replays: `advance` applies
//! everyone's commands (live, from the bots, or recorded) and the match flow.

use pill_engine::game::quake_map::QuakeCollision;

use super::{
    arena::Arena,
    combat::{Command, Event, Match},
    weapons::WEAPONS,
    Vec3, STEP,
};

pub const FRAG_LIMIT: i32 = 10;
pub const TIME_LIMIT: f32 = 300.0;
/// A match counts for the difficulty only if players played this long (seconds).
const COUNTED_PLAY: f32 = 60.0;

/// A player joining or leaving a session slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlotChange {
    Join(usize),
    Leave(usize),
}

/// A finished match.
#[derive(Clone, Copy, Debug)]
pub struct Ended {
    pub best_player: i32,
    pub best_bot: i32,
    /// Difficulty before and after (the same when the match didn't count).
    pub level_before: f32,
    pub level: f32,
}

pub struct Session {
    pub fight: Match,
    /// The first `players` fighters are player slots, the rest bots.
    pub players: usize,
    pub level: f32,
    /// Seconds into the match.
    pub clock: f32,
    pub tick: u64,
    pub seed: u32,
    /// Per fighter, seconds since its last attack (animation).
    pub since_attack: Vec<f32>,
    played: f32,
}

impl Session {
    /// `players` player slots (inactive until joined), then `bots` bots at difficulty `level`.
    pub fn new(arena: &Arena, players: usize, bots: usize, level: f32, seed: u32) -> Self {
        let count = players + bots;
        let mut fight = Match::new(&WEAPONS, arena, count, seed);
        for fighter in &mut fight.fighters[..players] {
            fighter.active = false;
        }
        Self {
            fight,
            players,
            level,
            clock: 0.0,
            tick: 0,
            seed,
            since_attack: vec![f32::MAX; count],
            played: 0.0,
        }
    }

    /// A player takes slot `slot`: at `place` (origin, yaw) if given, else at a spawn.
    pub fn join(&mut self, arena: &Arena, slot: usize, place: Option<(Vec3, f32)>) {
        let fighter = &mut self.fight.fighters[slot];
        fighter.active = true;
        fighter.frags = 0;
        self.fight.spawn(arena, slot);
        if let Some((origin, yaw)) = place {
            let fighter = &mut self.fight.fighters[slot];
            fighter.state.origin = origin;
            fighter.yaw = yaw;
        }
    }

    pub fn leave(&mut self, slot: usize) {
        self.fight.fighters[slot].active = false;
    }

    /// One tick with these commands (one per fighter): the rules, then the match flow. What
    /// happened, and the match result when it ended (all fighters then respawn with no frags).
    pub fn advance(
        &mut self,
        collision: &QuakeCollision,
        arena: &Arena,
        commands: &[Command],
    ) -> (Vec<Event>, Option<Ended>) {
        self.fight.step(collision, arena, commands, STEP);
        self.tick += 1;
        let mut events = std::mem::take(&mut self.fight.events);
        self.age_attacks(&events, STEP);
        self.clock += STEP;
        let playing = self.fight.fighters[..self.players]
            .iter()
            .any(|fighter| fighter.active);
        if playing {
            self.played += STEP;
        }
        let best = |range: std::ops::Range<usize>| {
            self.fight.fighters[range]
                .iter()
                .filter(|fighter| fighter.active)
                .map(|fighter| fighter.frags)
                .max()
                .unwrap_or(0)
        };
        let best_player = best(0..self.players);
        let best_bot = best(self.players..self.fight.fighters.len());
        if best_player.max(best_bot) < FRAG_LIMIT && self.clock < TIME_LIMIT {
            return (events, None);
        }
        let level_before = self.level;
        if self.played >= COUNTED_PLAY {
            self.level = level_after_match(self.level, best_player, best_bot);
        }
        self.clock = 0.0;
        self.played = 0.0;
        for index in 0..self.fight.fighters.len() {
            let fighter = &mut self.fight.fighters[index];
            fighter.frags = 0;
            fighter.damage_dealt = 0.0;
            if fighter.active {
                self.fight.spawn(arena, index);
            }
        }
        events.append(&mut self.fight.events);
        let ended = Ended {
            best_player,
            best_bot,
            level_before,
            level: self.level,
        };
        (events, Some(ended))
    }

    /// Seconds since each fighter's attack, `dt` later, these events happened in between.
    pub fn age_attacks(&mut self, events: &[Event], dt: f32) {
        for (index, since) in self.since_attack.iter_mut().enumerate() {
            let attacked = events
                .iter()
                .any(|event| matches!(event, Event::Fired { fighter, .. } if *fighter == index));
            *since = if attacked { 0.0 } else { *since + dt };
        }
    }

    /// FNV-1a over every fighter's simulated state: replays compare it.
    pub fn checksum(&self) -> u64 {
        let mut hash = super::Fnv::new();
        for fighter in &self.fight.fighters {
            for value in fighter.state.origin.iter().chain(&fighter.state.velocity) {
                hash.u32(value.to_bits());
            }
            hash.u32(fighter.health.to_bits());
            hash.u32(fighter.frags as u32);
            hash.u32(fighter.alive as u32);
        }
        hash.finish()
    }
}

/// Difficulty after a match: up when the players won, down when a bot did, by the frag margin.
pub fn level_after_match(level: f32, player_frags: i32, best_bot_frags: i32) -> f32 {
    let margin = (player_frags - best_bot_frags) as f32 / FRAG_LIMIT as f32;
    (level + 0.1 * margin.clamp(-1.0, 1.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_rises_after_human_wins_and_falls_after_loss() {
        assert!(level_after_match(0.5, 10, 4) > 0.5);
        assert!(level_after_match(0.5, 3, 10) < 0.5);
        assert_eq!(level_after_match(0.98, 10, 0), 1.0);
        assert_eq!(level_after_match(0.02, 0, 10), 0.0);
    }
}
