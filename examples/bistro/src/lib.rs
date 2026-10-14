#[cfg(target_arch = "wasm32")]
compile_error!("examples/bistro is native only (~600 MB of assets, BC7 cooked on the CPU)");

mod daylight;
mod fetch;
mod game;
mod loader;
mod ui;

use pill_engine::game::create_game;
create_game!(crate::game::Game {}, pill_engine::game::PillGame);
