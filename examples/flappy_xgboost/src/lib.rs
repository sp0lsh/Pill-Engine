mod director;
mod game;
mod gbt;

use pill_engine::game::create_game;
create_game!(crate::game::WebGame {}, pill_engine::game::PillGame);
