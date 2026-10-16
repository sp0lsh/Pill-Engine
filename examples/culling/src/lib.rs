mod editor;
mod game;
mod pass_editor;
mod pass_wireframe;

use pill_engine::game::create_game;
create_game!(crate::game::Game {}, pill_engine::game::PillGame);
