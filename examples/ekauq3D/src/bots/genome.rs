//! What a bot knows: tactics (the genome, evolved by self-play), skill (human-like limits, the
//! only thing difficulty adaptation changes) and the weapons' measured damage over distance; the
//! knowledge file (`knowledge.txt`) holds the evolved genome and the calibration with the hash
//! of the weapons it was measured with.

use crate::sim::{weapons::WeaponDef, Rng};

/// Tactics parameters; none is about a particular weapon (those are measured, `Calibration`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Genome(pub [f32; GENES]);

pub const GENES: usize = 16;

/// Distance held in a fight, × the current weapon's best distance.
pub const PREFERRED_RANGE: usize = 0;
/// Slack around that distance (units).
pub const RANGE_TOLERANCE: usize = 1;
/// Switch weapons only when the other does this many times the damage.
pub const SWITCH_MARGIN: usize = 2;
/// Fire when the aim is within this many target radii.
pub const FIRE_CONE: usize = 3;
/// Chance to strafe in a fight; strafe direction changes per second.
pub const STRAFE: usize = 4;
pub const STRAFE_SWITCH: usize = 5;
/// Dodging jumps per second in a fight.
pub const JUMP_DODGE: usize = 6;
/// Retreat to health below this health; fight again above `RETURN_HEALTH`.
pub const RETREAT_HEALTH: usize = 7;
pub const RETURN_HEALTH: usize = 8;
/// Seconds chasing a lost enemy.
pub const CHASE_TIME: usize = 9;
/// Closing in (1) or holding back (0) when the distance is wrong.
pub const AGGRESSION: usize = 10;
/// Weight of health items as goals.
pub const ITEM_GREED: usize = 11;
/// Projectile lead, × the computed lead.
pub const LEAD: usize = 12;
/// Chance to aim splash weapons at the feet.
pub const FEET_AIM: usize = 13;
/// Keep splash explosions this many radii away from yourself.
pub const SELF_SPLASH_MARGIN: usize = 14;
/// Strafing harder while the enemy aims at you.
pub const DODGE_WHEN_AIMED: usize = 15;

// prettier-ignore
/// Per gene: low, high, default.
pub const BOUNDS: [(f32, f32, f32); GENES] = [
    (0.3, 1.5, 0.8),     // PREFERRED_RANGE
    (16.0, 256.0, 96.0), // RANGE_TOLERANCE
    (1.0, 2.0, 1.2),     // SWITCH_MARGIN
    (0.5, 4.0, 1.5),     // FIRE_CONE
    (0.0, 1.0, 0.7),     // STRAFE
    (0.1, 3.0, 0.8),     // STRAFE_SWITCH
    (0.0, 2.0, 0.3),     // JUMP_DODGE
    (0.0, 70.0, 30.0),   // RETREAT_HEALTH
    (30.0, 100.0, 70.0), // RETURN_HEALTH
    (0.0, 10.0, 4.0),    // CHASE_TIME
    (0.0, 1.0, 0.6),     // AGGRESSION
    (0.0, 3.0, 1.0),     // ITEM_GREED
    (0.0, 1.3, 1.0),     // LEAD
    (0.0, 1.0, 0.5),     // FEET_AIM
    (1.0, 2.5, 1.3),     // SELF_SPLASH_MARGIN
    (0.0, 1.0, 0.5),     // DODGE_WHEN_AIMED
];

impl Default for Genome {
    fn default() -> Self {
        Self(std::array::from_fn(|gene| BOUNDS[gene].2))
    }
}

impl Genome {
    pub fn get(&self, gene: usize) -> f32 {
        self.0[gene]
    }

    pub fn clamped(mut self) -> Self {
        for (gene, value) in self.0.iter_mut().enumerate() {
            *value = value.clamp(BOUNDS[gene].0, BOUNDS[gene].1);
        }
        self
    }

    /// Gene in 0..1 of its range (the tuner's space) and back.
    pub fn normalized(&self, gene: usize) -> f32 {
        let (low, high, _) = BOUNDS[gene];
        (self.0[gene] - low) / (high - low)
    }

    pub fn from_normalized(values: [f32; GENES]) -> Self {
        Self(std::array::from_fn(|gene| {
            let (low, high, _) = BOUNDS[gene];
            low + values[gene].clamp(0.0, 1.0) * (high - low)
        }))
    }

    pub fn random(rng: &mut Rng) -> Self {
        Self::from_normalized(std::array::from_fn(|_| rng.unit()))
    }
}

/// Human-like limits, from one difficulty level in 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Skill {
    /// Standard deviation of the aim's wander (degrees).
    pub aim_error: f32,
    /// Seconds from seeing an enemy to reacting.
    pub reaction: f32,
    /// Fastest turn (degrees per second).
    pub turn_rate: f32,
}

// prettier-ignore
const EASY: Skill = Skill {
    aim_error: 9.0,
    reaction: 0.5,
    turn_rate: 200.0,
};
// prettier-ignore
const HARD: Skill = Skill {
    aim_error: 1.2,
    reaction: 0.12,
    turn_rate: 900.0,
};

impl Skill {
    pub fn at(level: f32) -> Self {
        let level = level.clamp(0.0, 1.0);
        let mix = |easy: f32, hard: f32| easy + (hard - easy) * level;
        Self {
            aim_error: mix(EASY.aim_error, HARD.aim_error),
            reaction: mix(EASY.reaction, HARD.reaction),
            turn_rate: mix(EASY.turn_rate, HARD.turn_rate),
        }
    }
}

/// Distances (units) at which weapons are measured.
// prettier-ignore
pub const BINS: [f32; 12] = [
    32.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0, 1536.0, 2048.0,
];

/// Per weapon, damage per second against a strafing target at each of `BINS`.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    pub dps: Vec<[f32; BINS.len()]>,
}

impl Calibration {
    /// Damage per second of `weapon` at `distance` (linear between bins, 0 past the last).
    pub fn dps(&self, weapon: usize, distance: f32) -> f32 {
        let Some(curve) = self.dps.get(weapon) else {
            return 0.0;
        };
        if distance <= BINS[0] {
            return curve[0];
        }
        for bin in 1..BINS.len() {
            if distance <= BINS[bin] {
                let t = (distance - BINS[bin - 1]) / (BINS[bin] - BINS[bin - 1]);
                return curve[bin - 1] + (curve[bin] - curve[bin - 1]) * t;
            }
        }
        0.0
    }

    /// The distance `weapon` does the most damage at.
    pub fn best_distance(&self, weapon: usize) -> f32 {
        let Some(curve) = self.dps.get(weapon) else {
            return BINS[0];
        };
        let mut best = 0;
        for bin in 1..BINS.len() {
            // Farther wins near-ties: the same damage from farther is safer
            if curve[bin] >= curve[best] * 0.9 && curve[bin] > 0.0 {
                best = bin;
            }
        }
        BINS[best]
    }
}

/// The knowledge file: genome, and the calibration with the weapons hash it was measured with.
#[derive(Clone, Debug, PartialEq)]
pub struct Knowledge {
    pub weapons: u64,
    pub calibration: Calibration,
    pub genome: Genome,
}

/// The knowledge compiled into the game (written by the tuner).
pub const KNOWLEDGE: &str = include_str!("knowledge.txt");

impl Knowledge {
    pub fn parse(text: &str) -> Self {
        let mut weapons = 0;
        let mut dps = Vec::new();
        let mut genome = Genome::default();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let mut words = line.split_whitespace();
            match words.next() {
                Some("weapons") => {
                    weapons = words
                        .next()
                        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                        .unwrap_or(0);
                }
                Some("calibration") => {
                    let _name = words.next();
                    let values: Vec<f32> = words.filter_map(|word| word.parse().ok()).collect();
                    if values.len() == BINS.len() {
                        dps.push(std::array::from_fn(|bin| values[bin]));
                    }
                }
                Some("genome") => {
                    let values: Vec<f32> = words.filter_map(|word| word.parse().ok()).collect();
                    if values.len() == GENES {
                        genome = Genome(std::array::from_fn(|gene| values[gene])).clamped();
                    }
                }
                _ => {}
            }
        }
        Self {
            weapons,
            calibration: Calibration { dps },
            genome,
        }
    }

    pub fn write(&self, weapons: &[WeaponDef], provenance: &str) -> String {
        let mut text = String::from(
            "# Written by ekauq_tune (examples/ekauq_tune.rs); measured with the weapons of this hash\n",
        );
        text.push_str(&format!("weapons {:016x}\n", self.weapons));
        for (curve, weapon) in self.calibration.dps.iter().zip(weapons) {
            let values: Vec<String> = curve.iter().map(|value| format!("{value:.1}")).collect();
            text.push_str(&format!(
                "calibration {} {}\n",
                weapon.name,
                values.join(" ")
            ));
        }
        let genes: Vec<String> = self
            .genome
            .0
            .iter()
            .map(|value| format!("{value:.3}"))
            .collect();
        text.push_str(&format!("genome {}\n", genes.join(" ")));
        text.push_str(&format!("# {provenance}\n"));
        text
    }

    /// Measures these weapons again (`calibrate`).
    pub fn recalibrate(&mut self, weapons: &[WeaponDef]) {
        self.calibration = super::calibrate::calibrate(weapons, 1);
        self.weapons = crate::sim::weapons::hash(weapons);
    }

    /// Whether the calibration was measured with these weapons.
    pub fn matches(&self, weapons: &[WeaponDef]) -> bool {
        self.weapons == crate::sim::weapons::hash(weapons)
            && self.calibration.dps.len() == weapons.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::weapons::WEAPONS;

    #[test]
    fn knowledge_file_round_trips() {
        let knowledge = Knowledge {
            weapons: crate::sim::weapons::hash(&WEAPONS),
            calibration: Calibration {
                dps: vec![[1.5; BINS.len()], [20.0; BINS.len()]],
            },
            genome: Genome::default(),
        };
        let parsed = Knowledge::parse(&knowledge.write(&WEAPONS, "test"));
        assert_eq!(parsed, knowledge);
        assert!(parsed.matches(&WEAPONS));
    }
}
