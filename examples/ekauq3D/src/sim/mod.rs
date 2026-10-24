//! The game's deathmatch, independent of the engine (only the Quake collision and movement of
//! `pill_assets`): the same rules in the game, the dedicated server, tests, replays and the bots'
//! self-play.
//!
//! - `weapons`: the weapons, defined once (hashed so bots notice changes)
//! - `arena`: a map's spawns and items
//! - `combat`: fighters, weapons fire, damage, death, respawn, frags
//! - `session`: the authoritative fixed tick, player slots, matches, difficulty
//! - `record`: session recordings and replay

pub mod arena;
pub mod combat;
pub mod record;
pub mod session;
pub mod weapons;

pub type Vec3 = [f32; 3];

/// Simulation step (seconds), the player's longest movement step.
pub const STEP: f32 = 1.0 / 72.0;

/// FNV-1a: weapon hashes and state checksums (stable across builds and platforms).
pub struct Fnv(u64);

impl Fnv {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    pub fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= *byte as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }

    pub fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// Small deterministic generator (xorshift32).
#[derive(Clone, Debug)]
pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1).wrapping_mul(0x9E37_79B9) | 1)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in 0..1.
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in −1..1.
    pub fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + self.unit() * (high - low)
    }

    /// Standard normal (Box–Muller).
    pub fn normal(&mut self) -> f32 {
        let a = self.unit().max(1e-7);
        let b = self.unit();
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }

    pub fn index(&mut self, count: usize) -> usize {
        (self.next_u32() as usize) % count.max(1)
    }
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: Vec3, factor: f32) -> Vec3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

pub fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn length(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

pub fn distance(a: Vec3, b: Vec3) -> f32 {
    length(sub(a, b))
}

pub fn horizontal_distance(a: Vec3, b: Vec3) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub fn normalize(a: Vec3) -> Vec3 {
    let length = length(a);
    if length > 0.0 {
        scale(a, 1.0 / length)
    } else {
        a
    }
}

/// Quake view basis from yaw and pitch (degrees, pitch up positive): forward, right, up.
pub fn view_basis(yaw: f32, pitch: f32) -> (Vec3, Vec3, Vec3) {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    let forward = [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ];
    let right = [yaw.sin(), -yaw.cos(), 0.0];
    let up = [
        -pitch.sin() * yaw.cos(),
        -pitch.sin() * yaw.sin(),
        pitch.cos(),
    ];
    (forward, right, up)
}

/// Yaw and pitch (degrees) looking along `direction`.
pub fn angles_of(direction: Vec3) -> (f32, f32) {
    let horizontal = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt();
    (
        direction[1].atan2(direction[0]).to_degrees(),
        direction[2].atan2(horizontal).to_degrees(),
    )
}

/// Signed smallest difference `to − from` of two angles (degrees), in −180..180.
pub fn angle_delta(from: f32, to: f32) -> f32 {
    (to - from + 540.0).rem_euclid(360.0) - 180.0
}
