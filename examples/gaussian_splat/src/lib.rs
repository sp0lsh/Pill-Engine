mod free_fly;
mod game;
mod gaussian_cloud;
mod pass_splat;

pub use game::GaussianGame;
pub use game::GaussianGame as WebGame;

use pill_engine::game::create_game;
create_game!(crate::game::GaussianGame {}, pill_engine::game::PillGame);
