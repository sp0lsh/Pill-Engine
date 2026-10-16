//! Pre- and post-processing of OpenCV Zoo's MediaPipe hand models (opencv_zoo
//! `palm_detection_mediapipe/mp_palmdet.py` and `handpose_estimation_mediapipe/mp_handpose.py`),
//! with the crops written as one affine map instead of OpenCV's crop / pad / rotate / resize.
//! Coordinates are image pixels, y down.

// Only the overlay constants are used natively; the tracker runs in the web build and tests
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use pill_engine::game::{Vector2f, Vector3f};

/// Palm detector input, pixels (square).
pub const PALM_INPUT_SIZE: usize = 192;
/// Hand landmark model input, pixels (square).
pub const HAND_INPUT_SIZE: usize = 224;
pub const ANCHOR_COUNT: usize = 2016;
pub const LANDMARK_COUNT: usize = 21;
/// Landmark indices (MediaPipe hand topology)
pub const WRIST: usize = 0;
pub const THUMB_TIP: usize = 4;
pub const INDEX_MCP: usize = 5;
pub const INDEX_TIP: usize = 8;
pub const MIDDLE_MCP: usize = 9;
pub const RING_MCP: usize = 13;
pub const PINKY_MCP: usize = 17;
const PALM_SCORE_THRESHOLD: f32 = 0.5;
const PALM_NMS_THRESHOLD: f32 = 0.3;
pub const HAND_CONFIDENCE_THRESHOLD: f32 = 0.8;
/// Hand landmarks that stand in for the palm detector's 7 keypoints when tracking.
const PALM_LANDMARK_IDS: [usize; 7] = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, 1, 2];
const PALM_BASE: usize = 0;
const MIDDLE_FINGER_BASE: usize = 2;
const PALM_BOX_SHIFT: Vector2f = Vector2f::new(0.0, -0.4);
const PALM_BOX_ENLARGE: f32 = 3.0;

/// Bone pairs of the 21 landmarks, for drawing.
pub const SKELETON: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

/// RGBA8 image, rows top to bottom.
#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Image {
    /// Bilinear RGB at pixel position `p` (pixel centres at +0.5), 0..1; black outside.
    fn sample(&self, p: Vector2f) -> [f32; 3] {
        let x = p.x - 0.5;
        let y = p.y - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let texel = |xi: f32, yi: f32| -> [f32; 3] {
            if xi < 0.0 || yi < 0.0 || xi >= self.width as f32 || yi >= self.height as f32 {
                return [0.0; 3];
            }
            let index = (yi as usize * self.width + xi as usize) * 4;
            [
                self.rgba[index] as f32,
                self.rgba[index + 1] as f32,
                self.rgba[index + 2] as f32,
            ]
        };
        let (a, b, c, d) = (
            texel(x0, y0),
            texel(x0 + 1.0, y0),
            texel(x0, y0 + 1.0),
            texel(x0 + 1.0, y0 + 1.0),
        );
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bottom = c[i] + (d[i] - c[i]) * fx;
            (top + (bottom - top) * fy) / 255.0
        })
    }

    /// `size`² NHWC RGB 0..1 tensor data, where pixel (u, v) samples `map(u + 0.5, v + 0.5)`.
    fn warp(&self, size: usize, map: impl Fn(Vector2f) -> Vector2f) -> Vec<f32> {
        let mut data = Vec::with_capacity(size * size * 3);
        for v in 0..size {
            for u in 0..size {
                let p = map(Vector2f::new(u as f32 + 0.5, v as f32 + 0.5));
                data.extend_from_slice(&self.sample(p));
            }
        }
        data
    }
}

/// The palm detector's SSD anchor centres (normalized): a 24×24 grid with 2 anchors per cell,
/// then a 12×12 grid with 6, cell centres (the table in `mp_palmdet.py`).
pub fn palm_anchors() -> Vec<Vector2f> {
    let mut anchors = Vec::with_capacity(ANCHOR_COUNT);
    for (grid, per_cell) in [(24, 2), (12, 6)] {
        for y in 0..grid {
            for x in 0..grid {
                let centre = Vector2f::new(
                    (x as f32 + 0.5) / grid as f32,
                    (y as f32 + 0.5) / grid as f32,
                );
                anchors.extend(std::iter::repeat_n(centre, per_cell));
            }
        }
    }
    anchors
}

/// The image letterboxed into the detector's square input: the square of side max(w, h)
/// centred on the image.
struct Letterbox {
    side: f32,
    offset: Vector2f,
}

impl Letterbox {
    fn new(width: usize, height: usize) -> Self {
        let side = width.max(height) as f32;
        Self {
            side,
            offset: (Vector2f::splat(side) - Vector2f::new(width as f32, height as f32)) / 2.0,
        }
    }

    /// Normalized detector coordinates → image pixels.
    fn to_image(&self, normalized: Vector2f) -> Vector2f {
        normalized * self.side - self.offset
    }
}

/// Palm detector input for `image`.
pub fn palm_input(image: &Image) -> Vec<f32> {
    let letterbox = Letterbox::new(image.width, image.height);
    image.warp(PALM_INPUT_SIZE, |uv| {
        letterbox.to_image(uv / PALM_INPUT_SIZE as f32)
    })
}

/// A detected palm: box corners and 7 keypoints (wrist, finger bases, palm base) in image pixels.
#[derive(Clone, Debug)]
pub struct Palm {
    pub min: Vector2f,
    pub max: Vector2f,
    pub keypoints: [Vector2f; 7],
    pub score: f32,
}

fn iou(a: (Vector2f, Vector2f), b: (Vector2f, Vector2f)) -> f32 {
    let size = (a.1.min(b.1) - a.0.max(b.0)).max(Vector2f::ZERO);
    let intersection = size.x * size.y;
    let area = |(min, max): (Vector2f, Vector2f)| (max - min).x * (max - min).y;
    intersection / (area(a) + area(b) - intersection).max(1e-6)
}

/// Palms from the detector outputs (`regressors` 2016×18: box cx, cy, w, h then 7 keypoint
/// x, y, in input pixels relative to the anchor; `logits` 2016) for a `width` × `height` image:
/// sigmoid, threshold, decode, greedy NMS by score.
pub fn decode_palms(regressors: &[f32], logits: &[f32], width: usize, height: usize) -> Vec<Palm> {
    let letterbox = Letterbox::new(width, height);
    let anchors = palm_anchors();
    let input = PALM_INPUT_SIZE as f32;
    let mut candidates: Vec<Palm> = (0..ANCHOR_COUNT)
        .filter_map(|i| {
            let score = 1.0 / (1.0 + (-logits[i]).exp());
            if score < PALM_SCORE_THRESHOLD {
                return None;
            }
            let r = &regressors[i * 18..i * 18 + 18];
            let centre = anchors[i] + Vector2f::new(r[0], r[1]) / input;
            let half = Vector2f::new(r[2], r[3]) / input / 2.0;
            Some(Palm {
                min: letterbox.to_image(centre - half),
                max: letterbox.to_image(centre + half),
                keypoints: std::array::from_fn(|k| {
                    letterbox.to_image(anchors[i] + Vector2f::new(r[4 + k * 2], r[5 + k * 2]) / input)
                }),
                score,
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Palm> = Vec::new();
    for palm in candidates {
        if kept
            .iter()
            .all(|other| iou((palm.min, palm.max), (other.min, other.max)) <= PALM_NMS_THRESHOLD)
        {
            kept.push(palm);
        }
    }
    kept
}

/// Rotated square crop fed to the landmark model: `centre` and `side` in the frame rotated by
/// `angle` (radians) around `pivot`.
#[derive(Clone, Copy, Debug)]
pub struct HandRoi {
    pivot: Vector2f,
    angle: f32,
    centre: Vector2f,
    side: f32,
}

/// `p` rotated by `angle` around `pivot` (`cv.getRotationMatrix2D(pivot, degrees(angle), 1)`).
fn rotate(p: Vector2f, pivot: Vector2f, angle: f32) -> Vector2f {
    let (sin, cos) = angle.sin_cos();
    let d = p - pivot;
    pivot + Vector2f::new(cos * d.x + sin * d.y, -sin * d.x + cos * d.y)
}

impl HandRoi {
    /// Crop of the hand around a palm box and its 7 keypoints: rotated so wrist → middle finger
    /// base points up, keypoint bounds shifted by (0, −0.4) and enlarged 3×, made square.
    pub fn from_palm(min: Vector2f, max: Vector2f, keypoints: &[Vector2f; 7]) -> Self {
        let (wrist, middle) = (keypoints[PALM_BASE], keypoints[MIDDLE_FINGER_BASE]);
        let angle = std::f32::consts::FRAC_PI_2 - (-(middle.y - wrist.y)).atan2(middle.x - wrist.x);
        let angle = angle - std::f32::consts::TAU * ((angle + std::f32::consts::PI) / std::f32::consts::TAU).floor();
        let pivot = (min + max) / 2.0;
        let rotated = keypoints.map(|p| rotate(p, pivot, angle));
        let low = rotated.iter().copied().reduce(Vector2f::min).unwrap_or(pivot);
        let high = rotated.iter().copied().reduce(Vector2f::max).unwrap_or(pivot);
        let size = high - low;
        let centre = (low + high) / 2.0 + PALM_BOX_SHIFT * size;
        let size = size * PALM_BOX_ENLARGE;
        Self {
            pivot,
            angle,
            centre,
            side: size.x.max(size.y),
        }
    }

    /// The next frame's crop from tracked landmarks (their palm subset stands in for the palm
    /// detection).
    pub fn from_landmarks(landmarks: &[Vector2f; LANDMARK_COUNT]) -> Self {
        let keypoints = PALM_LANDMARK_IDS.map(|i| landmarks[i]);
        let min = keypoints.iter().copied().reduce(Vector2f::min).unwrap_or_default();
        let max = keypoints.iter().copied().reduce(Vector2f::max).unwrap_or_default();
        Self::from_palm(min, max, &keypoints)
    }

    /// Model input pixel (0..HAND_INPUT_SIZE) → image pixel.
    fn to_image(&self, input: Vector2f) -> Vector2f {
        let rotated = self.centre + (input / HAND_INPUT_SIZE as f32 - 0.5) * self.side;
        rotate(rotated, self.pivot, -self.angle)
    }

    /// Landmark model input for `image`.
    pub fn input(&self, image: &Image) -> Vec<f32> {
        image.warp(HAND_INPUT_SIZE, |uv| self.to_image(uv))
    }

    /// Image-pixel landmarks from the model's 63 outputs (x, y, z in input pixels).
    pub fn landmarks(&self, output: &[f32]) -> [Vector2f; LANDMARK_COUNT] {
        std::array::from_fn(|i| self.to_image(Vector2f::new(output[i * 3], output[i * 3 + 1])))
    }

    /// Metric 3D landmarks (metres around the hand's centre; x right, y down, z away from the
    /// camera, image axes) from the model's 63 world outputs, which are in the crop's rotated
    /// frame (`mp_handpose.py` `rotated_landmarks_world`).
    pub fn world_landmarks(&self, output: &[f32]) -> [Vector3f; LANDMARK_COUNT] {
        std::array::from_fn(|i| {
            let xy = rotate(Vector2f::new(output[i * 3], output[i * 3 + 1]), Vector2f::ZERO, -self.angle);
            xy.extend(output[i * 3 + 2])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: usize, height: usize) -> Image {
        Image {
            width,
            height,
            rgba: vec![0; width * height * 4],
        }
    }

    fn close(a: Vector2f, b: Vector2f) -> bool {
        (a - b).length() < 1e-3
    }

    #[test]
    fn palm_anchors_count_2016_first_last_cell_centres() {
        let anchors = palm_anchors();
        assert_eq!(anchors.len(), ANCHOR_COUNT);
        // mp_palmdet.py: [0.02083333, 0.02083333] first, [0.9583333, 0.9583333] last
        assert!(close(anchors[0], Vector2f::splat(0.020_833_33)));
        assert!(close(anchors[1152], Vector2f::splat(0.041_666_67)));
        assert!(close(anchors[ANCHOR_COUNT - 1], Vector2f::splat(0.958_333_3)));
    }

    #[test]
    fn letterbox_wide_image_centre_maps_to_image_centre() {
        let letterbox = Letterbox::new(640, 480);
        assert!(close(letterbox.to_image(Vector2f::splat(0.5)), Vector2f::new(320.0, 240.0)));
        assert!(close(letterbox.to_image(Vector2f::ZERO), Vector2f::new(0.0, -80.0)));
    }

    #[test]
    fn decode_palms_one_anchor_above_threshold_box_around_anchor() {
        let mut regressors = vec![0.0; ANCHOR_COUNT * 18];
        let mut logits = vec![-10.0; ANCHOR_COUNT];
        // Anchor 0 at (0.0208, 0.0208): a 19.2 × 19.2 input-pixel box (0.1 normalized)
        regressors[2] = 19.2;
        regressors[3] = 19.2;
        logits[0] = 10.0;
        let palms = decode_palms(&regressors, &logits, 192, 192);
        assert_eq!(palms.len(), 1);
        let centre = (palms[0].min + palms[0].max) / 2.0;
        assert!(close(centre, Vector2f::splat(4.0)));
        assert!(close(palms[0].max - palms[0].min, Vector2f::splat(19.2)));
    }

    #[test]
    fn decode_palms_two_overlapping_boxes_nms_keeps_higher_score() {
        let mut regressors = vec![0.0; ANCHOR_COUNT * 18];
        let mut logits = vec![-10.0; ANCHOR_COUNT];
        // Anchors 0 and 1 share a cell: identical boxes
        for i in [0, 1] {
            regressors[i * 18 + 2] = 40.0;
            regressors[i * 18 + 3] = 40.0;
        }
        logits[0] = 1.0;
        logits[1] = 3.0;
        let palms = decode_palms(&regressors, &logits, 192, 192);
        assert_eq!(palms.len(), 1);
        assert!((palms[0].score - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn hand_roi_upright_hand_angle_0_crop_above_palm() {
        // Wrist below the middle finger base: no rotation
        let mut keypoints = [Vector2f::new(100.0, 100.0); 7];
        keypoints[PALM_BASE] = Vector2f::new(100.0, 120.0);
        keypoints[MIDDLE_FINGER_BASE] = Vector2f::new(100.0, 80.0);
        keypoints[1] = Vector2f::new(80.0, 100.0);
        keypoints[3] = Vector2f::new(120.0, 100.0);
        let roi = HandRoi::from_palm(Vector2f::new(80.0, 80.0), Vector2f::new(120.0, 120.0), &keypoints);
        assert!(roi.angle.abs() < 1e-6);
        // Bounds 40 × 40, shifted up 16, enlarged to 120
        assert!(close(roi.centre, Vector2f::new(100.0, 84.0)));
        assert!((roi.side - 120.0).abs() < 1e-3);
        assert!(close(roi.to_image(Vector2f::splat(112.0)), Vector2f::new(100.0, 84.0)));
    }

    #[test]
    fn hand_roi_hand_pointing_right_angle_half_pi_input_up_is_image_right() {
        let mut keypoints = [Vector2f::new(100.0, 100.0); 7];
        keypoints[PALM_BASE] = Vector2f::new(80.0, 100.0);
        keypoints[MIDDLE_FINGER_BASE] = Vector2f::new(120.0, 100.0);
        let roi = HandRoi::from_palm(Vector2f::new(80.0, 80.0), Vector2f::new(120.0, 120.0), &keypoints);
        assert!((roi.angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        // The top of the crop (fingers) lies towards +x in the image
        let top = roi.to_image(Vector2f::new(112.0, 0.0));
        let bottom = roi.to_image(Vector2f::new(112.0, 224.0));
        assert!(top.x > bottom.x + 1.0);
        assert!((top.y - bottom.y).abs() < 1e-3);
    }

    #[test]
    fn hand_roi_landmarks_input_centre_maps_to_crop_centre() {
        let roi = HandRoi {
            pivot: Vector2f::new(10.0, 20.0),
            angle: 0.7,
            centre: Vector2f::new(30.0, 40.0),
            side: 100.0,
        };
        let output: Vec<f32> = (0..LANDMARK_COUNT).flat_map(|_| [112.0, 112.0, 0.0]).collect();
        let landmarks = roi.landmarks(&output);
        assert!(close(landmarks[0], rotate(roi.centre, roi.pivot, -roi.angle)));
    }

    #[test]
    fn image_sample_pixel_centre_returns_texel() {
        let mut img = image(2, 1);
        img.rgba[4..8].copy_from_slice(&[255, 0, 0, 255]);
        let rgb = img.sample(Vector2f::new(1.5, 0.5));
        assert_eq!(rgb, [1.0, 0.0, 0.0]);
    }
}
