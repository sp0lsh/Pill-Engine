mod camera;
mod color;
mod game;
mod hittable;
mod hittable_list;
mod interval;
mod material;
mod ray;
mod rtweekend;
mod scene;
mod sphere;
mod vec3;
use pill_engine::game::create_game;

create_game!(crate::game::Game {}, pill_engine::game::PillGame);
