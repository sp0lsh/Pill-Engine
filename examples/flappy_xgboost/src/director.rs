//! Pipe director: a booster (`gbt`) learns from the player's own runs whether they clear a pipe,
//! retrains after every death, and each new gap is the candidate whose predicted pass rate is
//! closest to a pressure-and-release wave: easy stretches, then a ramp to a spike that deepens
//! wave by wave. Native builds keep the rows in `build/runs.csv` across launches and write every
//! trained model to `res/model.txt`; builds embed that file, so wasm starts trained.

use crate::gbt::{self, Booster, Params};

pub const FEATURES: usize = 6;
pub type Features = [f32; FEATURES];
/// Column names of the runs file and the model; skill is the mean score of the recent runs.
const FEATURE_NAMES: [&str; FEATURES] = [
    "gap_jump",
    "gap_jump_abs",
    "gap_y",
    "gap_size",
    "pipe_index",
    "skill",
];
/// The model of the last native training, as of this build.
const MODEL: &str = include_str!("../res/model.txt");

const GAP_LOW: f32 = -3.0;
const GAP_HIGH: f32 = 4.0;
const GAP_SIZE_MIN: f32 = 3.2;
const GAP_SIZE_MAX: f32 = 4.4;
const MAX_JUMP: f32 = 5.0;
const CANDIDATES: usize = 16;
/// Share of pipes placed at random, so the rows keep covering every kind of gap.
const EXPLORE: f32 = 0.1;
const MAX_ROWS: usize = 5000;
const MIN_ROWS: usize = 150;
/// Runs averaged into the skill feature.
const SKILL_RUNS: usize = 5;
// Wave: `RELEASE_PIPES` easy pipes, then a ramp to the pressure pass rate at the end of the period
const PERIOD: u32 = 10;
const RELEASE_PIPES: u32 = 7;
const RELEASE: f32 = 0.97;
const PRESSURE_START: f32 = 0.85;
const PRESSURE_STEP: f32 = 0.02;
const PRESSURE_MIN: f32 = 0.65;

/// Target pass rate of the run's `pipe`-th pipe.
fn target(pipe: u32) -> f32 {
    let pressure = (PRESSURE_START - PRESSURE_STEP * (pipe / PERIOD) as f32).max(PRESSURE_MIN);
    let phase = pipe % PERIOD;
    if phase < RELEASE_PIPES {
        return RELEASE;
    }
    let ramp = (phase - RELEASE_PIPES + 1) as f32 / (PERIOD - RELEASE_PIPES) as f32;
    RELEASE + (pressure - RELEASE) * ramp
}

/// A placed pipe waiting for its label.
#[derive(Clone, Copy)]
struct Placed {
    features: Features,
    predicted: Option<f32>,
}

pub struct Director {
    seed: u32,
    rows: Vec<Features>,
    labels: Vec<f32>,
    /// Rows not yet appended to the runs file.
    unsaved: usize,
    booster: Option<Booster<FEATURES>>,
    placed: Vec<Placed>,
    recent_scores: Vec<u32>,
    pipe_index: u32,
    last_gap_y: f32,
    /// This run: predicted pass rates and outcomes of the labeled pipes.
    run_predicted: Vec<f32>,
    run_passed: Vec<f32>,
}

impl Director {
    pub fn new(seed: u32, pipes: usize) -> Self {
        let (rows, labels) = load();
        let mut director = Self {
            seed,
            rows,
            labels,
            unsaved: 0,
            booster: None,
            placed: vec![
                Placed {
                    features: [0.0; FEATURES],
                    predicted: None,
                };
                pipes
            ],
            recent_scores: Vec::new(),
            pipe_index: 0,
            last_gap_y: 0.0,
            run_predicted: Vec::new(),
            run_passed: Vec::new(),
        };
        director.train();
        if director.booster.is_none() {
            director.booster = Booster::from_text(MODEL, &FEATURE_NAMES);
        }
        director
    }

    /// Mixes entropy (e.g. the time of the first flap) into the generator.
    pub fn mix_seed(&mut self, entropy: u32) {
        self.seed ^= entropy;
        if self.seed == 0 {
            self.seed = 0x9E37_79B9;
        }
    }

    fn random(&mut self) -> f32 {
        // xorshift32
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed >> 8) as f32 / (1 << 24) as f32
    }

    fn skill(&self) -> f32 {
        let count = self.recent_scores.len().max(1) as f32;
        self.recent_scores.iter().sum::<u32>() as f32 / count
    }

    pub fn start_run(&mut self) {
        self.pipe_index = 0;
        self.last_gap_y = 0.0;
        self.run_predicted.clear();
        self.run_passed.clear();
    }

    /// Gap center and height of the pipe in `slot`.
    pub fn place(&mut self, slot: usize) -> (f32, f32) {
        let low = GAP_LOW.max(self.last_gap_y - MAX_JUMP);
        let high = GAP_HIGH.min(self.last_gap_y + MAX_JUMP);
        let target = target(self.pipe_index);
        let explore = self.booster.is_none() || self.random() < EXPLORE;
        let candidates = if explore { 1 } else { CANDIDATES };
        // (error, gap center, gap height, placed)
        let mut best: Option<(f32, f32, f32, Placed)> = None;
        for _ in 0..candidates {
            let gap_y = low + self.random() * (high - low);
            let gap_size = GAP_SIZE_MIN + self.random() * (GAP_SIZE_MAX - GAP_SIZE_MIN);
            let jump = gap_y - self.last_gap_y;
            let features = [
                jump,
                jump.abs(),
                gap_y,
                gap_size,
                self.pipe_index as f32,
                self.skill(),
            ];
            let predicted = self.booster.as_ref().map(|booster| booster.predict(&features));
            let error = predicted.map_or(0.0, |predicted| (predicted - target).abs());
            if best.map_or(true, |(best_error, ..)| error < best_error) {
                best = Some((error, gap_y, gap_size, Placed { features, predicted }));
            }
        }
        let (_, gap_y, gap_size, placed) = best.expect("at least one candidate");
        self.placed[slot] = placed;
        self.pipe_index += 1;
        self.last_gap_y = gap_y;
        (gap_y, gap_size)
    }

    fn label(&mut self, slot: usize, passed: bool) {
        let placed = self.placed[slot];
        let label = if passed { 1.0 } else { 0.0 };
        self.rows.push(placed.features);
        self.labels.push(label);
        self.unsaved += 1;
        if let Some(predicted) = placed.predicted {
            self.run_predicted.push(predicted);
            self.run_passed.push(label);
        }
    }

    pub fn scored(&mut self, slot: usize) {
        self.label(slot, true);
    }

    /// The bird died before clearing the pipe in `slot`; retrains on everything so far.
    pub fn died(&mut self, slot: usize, score: u32) {
        self.label(slot, false);
        let count = self.run_predicted.len().max(1) as f32;
        log::info!(
            "flappy_xgboost: score {score}, labeled pipes {}, predicted pass {:.2}, actual {:.2}",
            self.run_predicted.len(),
            self.run_predicted.iter().sum::<f32>() / count,
            self.run_passed.iter().sum::<f32>() / count,
        );
        self.recent_scores.push(score);
        if self.recent_scores.len() > SKILL_RUNS {
            self.recent_scores.remove(0);
        }
        save(&self.rows, &self.labels, self.unsaved);
        self.unsaved = 0;
        if self.rows.len() > MAX_ROWS {
            let excess = self.rows.len() - MAX_ROWS;
            self.rows.drain(..excess);
            self.labels.drain(..excess);
        }
        self.train();
    }

    fn train(&mut self) {
        let passes = self.labels.iter().filter(|&&label| label > 0.5).count();
        if self.rows.len() < MIN_ROWS || passes == 0 || passes == self.labels.len() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        let booster = gbt::fit(&self.rows, &self.labels, &Params::default());
        let log_loss = self
            .rows
            .iter()
            .zip(&self.labels)
            .map(|(row, &label)| {
                let p = booster.predict(row).clamp(1e-6, 1.0 - 1e-6);
                -(label * p.ln() + (1.0 - label) * (1.0 - p).ln())
            })
            .sum::<f32>()
            / self.rows.len() as f32;
        #[cfg(not(target_arch = "wasm32"))]
        let milliseconds = started.elapsed().as_secs_f32() * 1000.0;
        #[cfg(target_arch = "wasm32")]
        let milliseconds = f32::NAN;
        log::info!(
            "flappy_xgboost: trained rows {}, trees {}, {milliseconds:.1} ms, log-loss {log_loss:.3}",
            self.rows.len(),
            booster.tree_count(),
        );
        save_model(&booster.to_text(&FEATURE_NAMES));
        self.booster = Some(booster);
    }
}

#[cfg(not(target_arch = "wasm32"))]
const RUNS_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/build/runs.csv");
/// Under `res/`, which the native hot reload treats as assets: writing it doesn't rebuild the game.
#[cfg(not(target_arch = "wasm32"))]
const MODEL_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/res/model.txt");

/// Rows of the runs file: a header, then per line the features and the label.
#[cfg(not(target_arch = "wasm32"))]
fn load() -> (Vec<Features>, Vec<f32>) {
    let (mut rows, mut labels) = (Vec::new(), Vec::new());
    let Ok(text) = std::fs::read_to_string(RUNS_FILE) else {
        return (rows, labels);
    };
    for line in text.lines() {
        let values: Vec<f32> = line.split(',').filter_map(|value| value.parse().ok()).collect();
        if values.len() != FEATURES + 1 {
            continue;
        }
        let mut row = [0.0; FEATURES];
        row.copy_from_slice(&values[..FEATURES]);
        rows.push(row);
        labels.push(values[FEATURES]);
    }
    let excess = rows.len().saturating_sub(MAX_ROWS);
    rows.drain(..excess);
    labels.drain(..excess);
    log::info!("flappy_xgboost: loaded {} rows from {RUNS_FILE}", rows.len());
    (rows, labels)
}

/// Appends the last `unsaved` rows to the runs file.
#[cfg(not(target_arch = "wasm32"))]
fn save(rows: &[Features], labels: &[f32], unsaved: usize) {
    use std::io::Write;
    let start = rows.len() - unsaved;
    let mut text = String::new();
    if std::fs::metadata(RUNS_FILE).map_or(true, |metadata| metadata.len() == 0) {
        text.push_str(&format!("{},passed\n", FEATURE_NAMES.join(",")));
    }
    for (row, label) in rows[start..].iter().zip(&labels[start..]) {
        for value in row {
            text.push_str(&format!("{value},"));
        }
        text.push_str(&format!("{label}\n"));
    }
    let written = std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/build"))
        .and_then(|()| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(RUNS_FILE)
        })
        .and_then(|mut file| file.write_all(text.as_bytes()));
    if let Err(error) = written {
        log::warn!("flappy_xgboost: saving {RUNS_FILE} failed: {error}");
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn save_model(text: &str) {
    if let Err(error) = std::fs::write(MODEL_FILE, text) {
        log::warn!("flappy_xgboost: saving {MODEL_FILE} failed: {error}");
    }
}

#[cfg(target_arch = "wasm32")]
fn save_model(_text: &str) {}

#[cfg(target_arch = "wasm32")]
fn load() -> (Vec<Features>, Vec<f32>) {
    (Vec::new(), Vec::new())
}

#[cfg(target_arch = "wasm32")]
fn save(_rows: &[Features], _labels: &[f32], _unsaved: usize) {}
