pub mod bots;
pub mod deathmatch;
mod fetch;
mod game;
pub mod loader;
#[cfg(not(target_arch = "wasm32"))]
pub mod net;
mod player;
pub mod sim;
mod sounds;
mod tour;
mod weapon;

use pill_engine::game::create_game;
create_game!(crate::game::Game {}, pill_engine::game::PillGame);
