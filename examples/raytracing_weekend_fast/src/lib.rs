//! aras_p's ToyPathTracer workload (github.com/aras-p/ToyPathTracer, Cpp/), optimized in Rust:
//! `scene.rs` is his scene and settings, `fast.rs` the tracer, `game.rs` the CPU branch of his
//! Apple/Renderer.mm (frame timing and his Mrays/s line). NEON only.

#[cfg(not(target_arch = "aarch64"))]
compile_error!("raytracing_weekend_fast uses NEON intrinsics");

mod fast;
mod game;
mod scene;
use pill_engine::game::create_game;

create_game!(crate::game::Game {}, pill_engine::game::PillGame);
