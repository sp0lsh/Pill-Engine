//! Flappy Bird from unit cubes, with pipes placed by a booster that learns the player
//! (`director`): Space or click flaps, pipes scroll in from the right, a 7-segment score counts the
//! pipes passed. Everything is spawned once and moved; a crash freezes the game until the next flap
//! restarts it.

use pill_engine::{define_component, define_global_component, game::*};

use crate::director::Director;

// World units. The camera sees y ±7.9 (60 · tan 7.5°) and, at the 3:2 window, x ±11.9.
const CAMERA_DISTANCE: f32 = 60.0;
const CAMERA_FOV: f32 = 15.0;
const VIEW_HALF_WIDTH: f32 = 11.9;
const VIEW_HALF_HEIGHT: f32 = 7.9;
const CEILING: f32 = 7.0;
const GROUND: f32 = -6.0;
const GRAVITY: f32 = -30.0;
const FLAP_SPEED: f32 = 9.0;
/// Frame time is capped so a stalled frame doesn't carry the bird through a pipe.
const MAX_DELTA_TIME: f32 = 1.0 / 30.0;
const BIRD_X: f32 = -4.0;
const BIRD_SIZE: f32 = 0.8;
const PIPE_WIDTH: f32 = 1.6;
const PIPE_SPEED: f32 = 4.0;
const PIPE_SPACING: f32 = 7.5;
/// Enough pipes that one leaving on the left reappears off-screen on the right.
const PIPES: usize = 4;
const PIPE_EXIT_X: f32 = -VIEW_HALF_WIDTH - PIPE_WIDTH;
const FIRST_PIPE_X: f32 = VIEW_HALF_WIDTH + PIPE_WIDTH;
/// A pipe scores when its right edge passes the bird's left edge.
const SCORE_X: f32 = BIRD_X - (BIRD_SIZE + PIPE_WIDTH) * 0.5;
/// Score digits: count, top center, size and segment thickness; drawn in front of the pipes.
const DIGITS: usize = 3;
const DIGIT_Y: f32 = 6.0;
const DIGIT_Z: f32 = 0.5;
const DIGIT_WIDTH: f32 = 0.8;
const DIGIT_HEIGHT: f32 = 1.4;
const DIGIT_STEP: f32 = 1.2;
const SEGMENT: f32 = 0.18;
/// Lit segments a..g (bits 0..6) of 0..9.
const DIGIT_SEGMENTS: [u8; 10] = [
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F,
];
/// Unlit segments, and the pipes before the first flap, wait here, off-screen.
const PARKED: Vector3f = Vector3f::new(0.0, 1000.0, 0.0);

// Parts, by `Part::id`: the bird, a top and a bottom cube per pipe, the ground, the segments
const BIRD: usize = 0;
const FIRST_PIPE: usize = 1;
const GROUND_PART: usize = FIRST_PIPE + 2 * PIPES;
const FIRST_SEGMENT: usize = GROUND_PART + 1;
const PARTS: usize = FIRST_SEGMENT + 7 * DIGITS;

define_component!(Part { id: usize });

define_global_component!(Flappy {
    alive: bool,
    bird_y: f32,
    bird_speed: f32,
    pipe_x: [f32; PIPES],
    gap_y: [f32; PIPES],
    gap_size: [f32; PIPES],
    score: u32,
    runs: u32,
    director: Director,
});

pub struct WebGame {}

impl Flappy {
    fn place(&mut self, pipe: usize) {
        (self.gap_y[pipe], self.gap_size[pipe]) = self.director.place(pipe);
    }

    fn restart(&mut self) {
        self.alive = true;
        self.bird_y = 0.0;
        self.bird_speed = FLAP_SPEED;
        self.score = 0;
        self.runs += 1;
        self.director.start_run();
        for pipe in 0..PIPES {
            self.pipe_x[pipe] = FIRST_PIPE_X + pipe as f32 * PIPE_SPACING;
            self.place(pipe);
        }
    }

    /// The bird died before clearing the nearest pipe still ahead of it.
    fn die(&mut self) {
        self.alive = false;
        let ahead = (0..PIPES)
            .filter(|&pipe| self.pipe_x[pipe] >= SCORE_X)
            .min_by(|&a, &b| self.pipe_x[a].total_cmp(&self.pipe_x[b]));
        if let Some(pipe) = ahead {
            self.director.died(pipe, self.score);
        }
    }

    fn step(&mut self, flap: bool, delta_time: f32, time: f32) {
        if !self.alive {
            if flap {
                // The moment of the first flap varies the pipe sequence from launch to launch
                if self.runs == 0 {
                    self.director.mix_seed(time.to_bits());
                }
                self.restart();
            }
            return;
        }
        if flap {
            self.bird_speed = FLAP_SPEED;
        }
        self.bird_speed += GRAVITY * delta_time;
        self.bird_y += self.bird_speed * delta_time;
        let half = BIRD_SIZE * 0.5;
        if self.bird_y - half < GROUND || self.bird_y + half > CEILING {
            self.die();
            return;
        }
        for pipe in 0..PIPES {
            let before = self.pipe_x[pipe];
            self.pipe_x[pipe] -= PIPE_SPEED * delta_time;
            if before >= SCORE_X && self.pipe_x[pipe] < SCORE_X {
                self.score += 1;
                self.director.scored(pipe);
            }
            if self.pipe_x[pipe] < PIPE_EXIT_X {
                self.pipe_x[pipe] += PIPES as f32 * PIPE_SPACING;
                self.place(pipe);
            }
            let overlaps_x = (self.pipe_x[pipe] - BIRD_X).abs() < (PIPE_WIDTH + BIRD_SIZE) * 0.5;
            let inside_gap =
                (self.bird_y - self.gap_y[pipe]).abs() < (self.gap_size[pipe] - BIRD_SIZE) * 0.5;
            if overlaps_x && !inside_gap {
                self.die();
                return;
            }
        }
    }

    /// Center and size of every part.
    fn layout(&self) -> [(Vector3f, Vector3f); PARTS] {
        let mut parts = [(PARKED, Vector3f::ONE); PARTS];
        parts[BIRD] = (
            Vector3f::new(BIRD_X, self.bird_y, 0.0),
            Vector3f::splat(BIRD_SIZE),
        );
        for pipe in 0..PIPES {
            let x = self.pipe_x[pipe];
            let gap_top = self.gap_y[pipe] + self.gap_size[pipe] * 0.5;
            let gap_bottom = self.gap_y[pipe] - self.gap_size[pipe] * 0.5;
            // Each pipe reaches from its gap edge past the top of the view or down to the ground
            let top_height = VIEW_HALF_HEIGHT + 0.5 - gap_top;
            let bottom_height = gap_bottom - GROUND;
            parts[FIRST_PIPE + 2 * pipe] = (
                Vector3f::new(x, gap_top + top_height * 0.5, 0.0),
                Vector3f::new(PIPE_WIDTH, top_height, PIPE_WIDTH),
            );
            parts[FIRST_PIPE + 2 * pipe + 1] = (
                Vector3f::new(x, GROUND + bottom_height * 0.5, 0.0),
                Vector3f::new(PIPE_WIDTH, bottom_height, PIPE_WIDTH),
            );
        }
        // Ground: from the ground line down past the bottom of the view, wider than the view
        let ground_depth = VIEW_HALF_HEIGHT + GROUND + 1.0;
        parts[GROUND_PART] = (
            Vector3f::new(0.0, GROUND - ground_depth * 0.5, 0.0),
            Vector3f::new(4.0 * VIEW_HALF_WIDTH, ground_depth, 1.0),
        );
        // Score centered, without leading zeros
        let score = self.score.min(10u32.pow(DIGITS as u32) - 1);
        let count = (1..DIGITS).filter(|&place| score >= 10u32.pow(place as u32)).count() + 1;
        let first_left = -(count as f32 * DIGIT_STEP - (DIGIT_STEP - DIGIT_WIDTH)) * 0.5;
        let horizontal = Vector3f::new(DIGIT_WIDTH, SEGMENT, SEGMENT);
        let vertical = Vector3f::new(SEGMENT, DIGIT_HEIGHT * 0.5, SEGMENT);
        for place in 0..count {
            let digit = score / 10u32.pow((count - 1 - place) as u32) % 10;
            let left = first_left + place as f32 * DIGIT_STEP;
            let (right, center) = (left + DIGIT_WIDTH, left + DIGIT_WIDTH * 0.5);
            let (upper, lower) = (DIGIT_Y - DIGIT_HEIGHT * 0.25, DIGIT_Y - DIGIT_HEIGHT * 0.75);
            // a top, b upper right, c lower right, d bottom, e lower left, f upper left, g middle
            let segments = [
                (center, DIGIT_Y, horizontal),
                (right, upper, vertical),
                (right, lower, vertical),
                (center, DIGIT_Y - DIGIT_HEIGHT, horizontal),
                (left, lower, vertical),
                (left, upper, vertical),
                (center, DIGIT_Y - DIGIT_HEIGHT * 0.5, horizontal),
            ];
            let mask = DIGIT_SEGMENTS[digit as usize];
            for (segment, (x, y, size)) in segments.into_iter().enumerate() {
                if mask & (1 << segment) != 0 {
                    parts[FIRST_SEGMENT + place * 7 + segment] = (Vector3f::new(x, y, DIGIT_Z), size);
                }
            }
        }
        parts
    }
}

fn flappy_system(engine: &mut Engine) -> Result<()> {
    let input = engine.get_global_component::<InputComponent>()?;
    let flap = input.get_key_pressed(KeyboardKey::Space)
        || input.get_mouse_button_pressed(MouseButton::Left);
    let time = engine.get_global_component::<TimeComponent>()?;
    let (delta_time, time) = (time.delta_time.min(MAX_DELTA_TIME), time.time);
    let flappy = engine.get_global_component_mut::<Flappy>()?;
    flappy.step(flap, delta_time, time);
    let parts = flappy.layout();
    for (_, transform, part) in engine.iterate_two_components_mut::<TransformComponent, Part>()? {
        let (position, scale) = parts[part.id];
        transform.set_position(position);
        transform.set_scale(scale);
    }
    Ok(())
}

impl PillGame for WebGame {
    fn start(&self, engine: &mut Engine) -> Result<()> {
        let scene = engine.create_scene("default")?;
        engine.set_active_scene(scene)?;
        {
            let render_state = engine.get_global_component_mut::<RenderStateComponent>()?;
            render_state.background_color = [0.45, 0.75, 0.95];
            render_state.ibl_intensity = 0.0;
            render_state.motion_blur = 0.0;
        }
        engine.register_component::<TransformComponent>(scene)?;
        engine.register_component::<CameraComponent>(scene)?;
        engine.register_component::<MeshRenderingComponent>(scene)?;
        engine.register_component::<Part>(scene)?;

        let cube = engine.add_resource(Mesh::cube("cube", 1.0))?;
        // Black albedo + emissive: flat, unlit colors
        let flat = |name: &str, color: Color| {
            Material::new(name)
                .albedo(Color::new(0.0, 0.0, 0.0))
                .emissive(color)
        };
        let bird = engine.add_resource(flat("bird", Color::new(1.0, 0.85, 0.1)))?;
        let pipe = engine.add_resource(flat("pipe", Color::new(0.2, 0.7, 0.2)))?;
        let ground = engine.add_resource(flat("ground", Color::new(0.85, 0.75, 0.5)))?;
        let digit = engine.add_resource(flat("digit", Color::new(1.0, 1.0, 1.0)))?;

        engine
            .build_entity(scene)
            .with_component(
                TransformComponent::builder()
                    .position(Vector3f::new(0.0, 0.0, CAMERA_DISTANCE))
                    .build(),
            )
            .with_component(
                CameraComponent::builder()
                    .enabled(true)
                    .fov(CAMERA_FOV)
                    .build(),
            )
            .build();

        for id in 0..PARTS {
            let material = match id {
                BIRD => &bird,
                FIRST_PIPE..GROUND_PART => &pipe,
                GROUND_PART => &ground,
                _ => &digit,
            };
            engine
                .build_entity(scene)
                .with_component(TransformComponent::builder().position(PARKED).build())
                .with_component(
                    MeshRenderingComponent::builder()
                        .mesh(&cube)
                        .material(material)
                        .build(),
                )
                .with_component(Part { id })
                .build();
        }

        // Pipes wait off-screen until the first flap starts the game
        let flappy = Flappy {
            alive: false,
            bird_y: 0.0,
            bird_speed: 0.0,
            pipe_x: [PARKED.y; PIPES],
            gap_y: [0.0; PIPES],
            gap_size: [0.0; PIPES],
            score: 0,
            runs: 0,
            director: Director::new(0x9E37_79B9, PIPES),
        };
        engine.add_global_component(flappy)?;
        engine.add_system("flappy", flappy_system)?;
        Ok(())
    }
}
