mod fetch;
mod game;
mod loader;
mod player;
mod sounds;
mod weapon;

use pill_engine::game::create_game;
create_game!(crate::game::Game {}, pill_engine::game::PillGame);
