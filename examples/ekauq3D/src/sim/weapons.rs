//! The weapons, defined once (player and bots alike), Quake's numbers. Bots know nothing else
//! about them: they measure what each does (`calibrate`), again whenever these values change
//! (`hash`).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Melee,
    Hitscan,
    Projectile,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeaponDef {
    pub name: &'static str,
    pub kind: Kind,
    /// Per pellet, hit or direct projectile hit (health is 100).
    pub damage: f32,
    pub pellets: u32,
    /// Per pellet, along the view's right and up.
    pub spread: f32,
    /// Units: melee reach, hitscan length, projectile flight.
    pub range: f32,
    /// Seconds between attacks.
    pub refire: f32,
    /// Seconds from the attack to the hit (melee swing).
    pub delay: f32,
    /// Projectile units per second.
    pub speed: f32,
    /// Projectile splash radius (units), damage falls linearly to its edge.
    pub splash: f32,
    /// Splash damage factor on the shooter.
    pub self_damage: f32,
}

/// Slot order (keys 1, 2): the axe, the shotgun (6 pellets, spread 0.04, 2048 units; quakewiki).
// prettier-ignore
pub const WEAPONS: [WeaponDef; 2] = [
    WeaponDef {
        name: "axe",
        kind: Kind::Melee,
        damage: 20.0,
        pellets: 1,
        spread: 0.0,
        range: 64.0,
        refire: 0.5,
        delay: 0.2,
        speed: 0.0,
        splash: 0.0,
        self_damage: 0.0,
    },
    WeaponDef {
        name: "shotgun",
        kind: Kind::Hitscan,
        damage: 4.0,
        pellets: 6,
        spread: 0.04,
        range: 2048.0,
        refire: 0.5,
        delay: 0.0,
        speed: 0.0,
        splash: 0.0,
        self_damage: 0.0,
    },
];

/// FNV-1a over the weapons' values: what the stored bot knowledge was measured with.
pub fn hash(weapons: &[WeaponDef]) -> u64 {
    let mut hash = super::Fnv::new();
    for weapon in weapons {
        hash.bytes(weapon.name.as_bytes());
        hash.bytes(&[weapon.kind as u8]);
        for value in [
            weapon.damage,
            weapon.pellets as f32,
            weapon.spread,
            weapon.range,
            weapon.refire,
            weapon.delay,
            weapon.speed,
            weapon.splash,
            weapon.self_damage,
        ] {
            hash.u32(value.to_bits());
        }
    }
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_changes_when_shotgun_range_changes() {
        let mut changed = WEAPONS;
        changed[1].range = 512.0;
        assert_ne!(hash(&WEAPONS), hash(&changed));
        assert_eq!(hash(&WEAPONS), hash(&WEAPONS));
    }
}
