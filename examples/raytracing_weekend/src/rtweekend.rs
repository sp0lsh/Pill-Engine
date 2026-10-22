//! The book's `rtweekend.h`: constants and `random_double`. The book's `std::rand()` is a
//! per-thread PCG32 here; each thread gets its own stream, seeded in creation order.

use std::{
    cell::Cell,
    sync::atomic::{AtomicU64, Ordering},
};

pub const PI: f64 = std::f64::consts::PI;

pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

static NEXT_STREAM: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new(seed(NEXT_STREAM.fetch_add(1, Ordering::Relaxed)));
}

fn seed(stream: u64) -> u64 {
    stream
        .wrapping_add(0x853C_49E6_748F_EA9B)
        .wrapping_mul(6_364_136_223_846_793_005)
}

fn next_u32() -> u32 {
    RNG_STATE.with(|state| {
        let old = state.get();
        state.set(
            old.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407),
        );
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    })
}

/// A random real in [0, 1).
pub fn random_double() -> f64 {
    next_u32() as f64 / (u32::MAX as f64 + 1.0)
}

/// A random real in [min, max).
pub fn random_double_range(min: f64, max: f64) -> f64 {
    min + (max - min) * random_double()
}
