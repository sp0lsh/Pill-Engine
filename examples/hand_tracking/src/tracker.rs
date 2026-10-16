//! Runs the two ONNX models with RTen (pure Rust) in MediaPipe's loop: tracked hands get their
//! next crop from the previous landmarks; the palm detector runs only while fewer than
//! `MAX_HANDS` hands are tracked, and its palms that overlap a tracked hand are dropped.

use crate::hands::{self, HandRoi, Image, Palm, LANDMARK_COUNT};
use pill_engine::game::Vector2f;
use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, Tensor};

pub const MAX_HANDS: usize = 2;

pub struct Hand {
    pub landmarks: [Vector2f; LANDMARK_COUNT],
}

/// Milliseconds spent in each model in the last `step`.
#[derive(Clone, Copy, Default)]
pub struct Timings {
    pub palm_ms: f32,
    pub hand_ms: f32,
}

pub struct Tracker {
    palm_model: Model,
    hand_model: Model,
    hands: Vec<Hand>,
}

fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|window| window.performance())
            .map_or(0.0, |performance| performance.now())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

/// Runs `model` on an NHWC `size`² RGB input; returns its outputs flattened.
fn run(model: &Model, size: usize, data: Vec<f32>) -> Result<Vec<Vec<f32>>, String> {
    let input = NdTensor::from_data([1, size, size, 3], data);
    let outputs = model
        .run(
            vec![(model.input_ids()[0], input.view().into())],
            model.output_ids(),
            None,
        )
        .map_err(|error| format!("inference: {error}"))?;
    outputs
        .into_iter()
        .map(|output| {
            let tensor: Tensor<f32> = output
                .try_into()
                .map_err(|_| "inference: output is not f32".to_string())?;
            Ok(tensor.to_vec())
        })
        .collect()
}

impl Tracker {
    pub fn new(palm_onnx: Vec<u8>, hand_onnx: Vec<u8>) -> Result<Self, String> {
        Ok(Self {
            palm_model: Model::load(palm_onnx).map_err(|error| format!("palm model: {error}"))?,
            hand_model: Model::load(hand_onnx).map_err(|error| format!("hand model: {error}"))?,
            hands: Vec::new(),
        })
    }

    pub fn hands(&self) -> &[Hand] {
        &self.hands
    }

    fn detect_palms(&self, image: &Image) -> Result<Vec<Palm>, String> {
        let [regressors, logits]: [Vec<f32>; 2] =
            run(&self.palm_model, hands::PALM_INPUT_SIZE, hands::palm_input(image))?
                .try_into()
                .map_err(|_| "palm model: expected 2 outputs".to_string())?;
        Ok(hands::decode_palms(&regressors, &logits, image.width, image.height))
    }

    fn landmarks(&self, image: &Image, roi: &HandRoi) -> Result<Option<Hand>, String> {
        let outputs = run(&self.hand_model, hands::HAND_INPUT_SIZE, roi.input(image))?;
        let (Some(landmarks), Some(confidence)) = (outputs.first(), outputs.get(1)) else {
            return Err("hand model: expected landmark and confidence outputs".to_string());
        };
        let confidence = confidence.first().copied().unwrap_or(0.0);
        Ok((confidence >= hands::HAND_CONFIDENCE_THRESHOLD).then(|| Hand {
            landmarks: roi.landmarks(landmarks),
        }))
    }

    /// Tracks hands in `image` (one camera frame).
    pub fn step(&mut self, image: &Image) -> Result<Timings, String> {
        let mut timings = Timings::default();
        let mut rois: Vec<HandRoi> = self
            .hands
            .iter()
            .map(|hand| HandRoi::from_landmarks(&hand.landmarks))
            .collect();
        if rois.len() < MAX_HANDS {
            let start = now_ms();
            let tracked: Vec<(Vector2f, Vector2f)> = self.hands.iter().map(bounds).collect();
            for palm in self.detect_palms(image)? {
                let overlaps = tracked
                    .iter()
                    .any(|&(min, max)| palm.max.cmpge(min).all() && palm.min.cmple(max).all());
                if rois.len() < MAX_HANDS && !overlaps {
                    rois.push(HandRoi::from_palm(palm.min, palm.max, &palm.keypoints));
                }
            }
            timings.palm_ms = (now_ms() - start) as f32;
        }
        let start = now_ms();
        let mut hands = Vec::with_capacity(rois.len());
        for roi in &rois {
            hands.extend(self.landmarks(image, roi)?);
        }
        timings.hand_ms = (now_ms() - start) as f32;
        self.hands = hands;
        Ok(timings)
    }
}

fn bounds(hand: &Hand) -> (Vector2f, Vector2f) {
    let min = hand.landmarks.iter().copied().reduce(Vector2f::min).unwrap_or_default();
    let max = hand.landmarks.iter().copied().reduce(Vector2f::max).unwrap_or_default();
    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PALM_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/palm_detection_mediapipe_2023feb.onnx"
    );
    const HAND_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/handpose_estimation_mediapipe_2023feb.onnx"
    );
    /// MediaPipe's test image (storage.googleapis.com/mediapipe-assets/victory.jpg, Apache 2.0)
    const VICTORY_PNG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/victory.png");

    fn victory() -> Image {
        let decoder = png::Decoder::new(std::io::BufReader::new(
            std::fs::File::open(VICTORY_PNG).expect("test image"),
        ));
        let mut reader = decoder.read_info().expect("png header");
        let mut rgba = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut rgba).expect("png frame");
        assert_eq!(info.color_type, png::ColorType::Rgba);
        Image {
            width: info.width as usize,
            height: info.height as usize,
            rgba,
        }
    }

    fn tracker() -> Tracker {
        Tracker::new(
            std::fs::read(PALM_ONNX).expect("palm model"),
            std::fs::read(HAND_ONNX).expect("hand model"),
        )
        .expect("models load")
    }

    #[test]
    fn step_victory_image_one_hand_landmarks_inside_image() {
        let image = victory();
        let mut tracker = tracker();
        tracker.step(&image).expect("step");
        assert_eq!(tracker.hands().len(), 1);
        let (min, max) = bounds(&tracker.hands()[0]);
        assert!(min.cmpge(Vector2f::ZERO).all(), "{min}");
        assert!(max.cmple(Vector2f::new(342.0, 362.0)).all(), "{max}");
        // The second step crops from the first step's landmarks and finds the same hand
        tracker.step(&image).expect("step");
        assert_eq!(tracker.hands().len(), 1);
    }

    #[test]
    fn step_black_image_no_hands() {
        let image = Image {
            width: 64,
            height: 48,
            rgba: vec![0; 64 * 48 * 4],
        };
        let mut tracker = tracker();
        tracker.step(&image).expect("step");
        assert!(tracker.hands().is_empty());
    }
}
