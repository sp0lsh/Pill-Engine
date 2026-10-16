mod game;
mod gpu_net;
mod gpu_tracker;
mod hands;
mod onnx;
#[cfg(target_arch = "wasm32")]
mod camera;
#[cfg(target_arch = "wasm32")]
mod camera_pass;
// CPU (RTen) reference of the tracker, for the tests
#[cfg(test)]
mod tracker;

use pill_engine::game::create_game;
create_game!(crate::game::Game {}, pill_engine::game::PillGame);
