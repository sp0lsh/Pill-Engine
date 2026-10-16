//! AR interaction gestures from the 21 hand landmarks, by rules on the hand's shape: which fingers
//! are extended and whether thumb and index tips touch. Distances are relative to the palm size
//! (wrist → middle finger base), so they hold at any distance from the camera.
//!
//! Priority: Pinch (with Start / Hold / Release phases), Point (index only), Open palm (all four
//! fingers), Grab (no finger), else None. Pinch switches at once, with hysteresis; the others are
//! reported once seen in `DEBOUNCE_RESULTS` consecutive tracker results.

// Used by the web build and the tests
#![cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]

use crate::hands::{INDEX_MCP, INDEX_TIP, LANDMARK_COUNT, MIDDLE_MCP, PINKY_MCP, RING_MCP, THUMB_TIP, WRIST};
use pill_engine::game::Vector2f;

/// (MCP, PIP, tip) of index, middle, ring and pinky.
const FINGERS: [(usize, usize, usize); 4] = [
    (INDEX_MCP, INDEX_MCP + 1, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_MCP + 1, MIDDLE_MCP + 3),
    (RING_MCP, RING_MCP + 1, RING_MCP + 3),
    (PINKY_MCP, PINKY_MCP + 1, PINKY_MCP + 3),
];

/// A finger is extended when its tip is this much farther from the wrist than its PIP joint…
const EXTENDED_REACH: f32 = 1.1;
/// …and MCP → PIP → tip is this straight (cosine of the bend).
const EXTENDED_STRAIGHTNESS: f32 = 0.7;
/// The thumb is extended when its tip is this far from the index finger base, in palm sizes.
const THUMB_EXTENDED: f32 = 0.9;
/// Thumb–index tip distance (palm sizes) to start and to end a pinch.
const PINCH_ENTER: f32 = 0.25;
const PINCH_EXIT: f32 = 0.40;
/// Consecutive results a non-pinch gesture needs before it is reported.
const DEBOUNCE_RESULTS: u32 = 3;
/// A hand keeps its identity when its wrist is within this many palm sizes of the previous one.
const SAME_HAND: f32 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PinchPhase {
    Start,
    Hold { seconds: f32 },
    Release,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gesture {
    None,
    Point,
    OpenPalm,
    Grab,
    Pinch(PinchPhase),
}

/// The finger state of one hand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub thumb: bool,
    /// Index, middle, ring, pinky
    pub fingers: [bool; 4],
    /// Thumb–index tip distance, palm sizes
    pub pinch_distance: f32,
}

/// Midpoint of the thumb and index tips, where a pinch grabs.
pub fn pinch_point(landmarks: &[Vector2f; LANDMARK_COUNT]) -> Vector2f {
    (landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) / 2.0
}

pub fn palm_size(landmarks: &[Vector2f; LANDMARK_COUNT]) -> f32 {
    landmarks[WRIST].distance(landmarks[MIDDLE_MCP]).max(1e-3)
}

pub fn shape(landmarks: &[Vector2f; LANDMARK_COUNT]) -> Shape {
    let palm = palm_size(landmarks);
    let wrist = landmarks[WRIST];
    let fingers = FINGERS.map(|(mcp, pip, tip)| {
        let (mcp, pip, tip) = (landmarks[mcp], landmarks[pip], landmarks[tip]);
        let reach = tip.distance(wrist) > EXTENDED_REACH * pip.distance(wrist);
        let straight = (pip - mcp)
            .normalize_or_zero()
            .dot((tip - pip).normalize_or_zero())
            > EXTENDED_STRAIGHTNESS;
        reach && straight
    });
    Shape {
        thumb: landmarks[THUMB_TIP].distance(landmarks[INDEX_MCP]) > THUMB_EXTENDED * palm,
        fingers,
        pinch_distance: landmarks[THUMB_TIP].distance(landmarks[INDEX_TIP]) / palm,
    }
}

/// The gesture of `shape`, without pinch (decided with hysteresis by the caller).
fn pose(shape: &Shape) -> Gesture {
    match shape.fingers {
        [true, false, false, false] => Gesture::Point,
        [true, true, true, true] => Gesture::OpenPalm,
        [false, false, false, false] => Gesture::Grab,
        _ => Gesture::None,
    }
}

struct TrackedHand {
    id: u64,
    wrist: Vector2f,
    pinching: bool,
    pinch_start: f64,
    candidate: Gesture,
    candidate_results: u32,
    reported: Gesture,
}

/// Gesture state of the tracked hands, updated once per tracker result.
#[derive(Default)]
pub struct Gestures {
    hands: Vec<TrackedHand>,
    next_id: u64,
}

impl Gestures {
    /// The gesture of each of `hands` (same order) at `time` (seconds).
    pub fn update(&mut self, hands: &[[Vector2f; LANDMARK_COUNT]], time: f64) -> Vec<Gesture> {
        let mut previous = std::mem::take(&mut self.hands);
        hands
            .iter()
            .map(|landmarks| {
                let wrist = landmarks[WRIST];
                let reach = SAME_HAND * palm_size(landmarks);
                let nearest = previous
                    .iter()
                    .enumerate()
                    .map(|(i, hand)| (i, hand.wrist.distance(wrist)))
                    .filter(|&(_, distance)| distance < reach)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i);
                let mut hand = match nearest {
                    Some(i) => previous.swap_remove(i),
                    None => TrackedHand {
                        id: {
                            self.next_id += 1;
                            self.next_id
                        },
                        wrist,
                        pinching: false,
                        pinch_start: time,
                        candidate: Gesture::None,
                        candidate_results: 0,
                        reported: Gesture::None,
                    },
                };
                hand.wrist = wrist;
                let gesture = hand.step(&shape(landmarks), time);
                self.hands.push(hand);
                gesture
            })
            .collect()
    }
}

impl Gestures {
    /// Ids of the hands of the last `update`, same order; a hand keeps its id while tracked.
    pub fn ids(&self) -> Vec<u64> {
        self.hands.iter().map(|hand| hand.id).collect()
    }
}

impl TrackedHand {
    fn step(&mut self, shape: &Shape, time: f64) -> Gesture {
        let was_pinching = self.pinching;
        self.pinching = shape.pinch_distance
            < if was_pinching {
                PINCH_EXIT
            } else {
                PINCH_ENTER
            };
        match (was_pinching, self.pinching) {
            (false, true) => {
                self.pinch_start = time;
                self.reported = Gesture::Pinch(PinchPhase::Start);
                return self.reported;
            }
            (true, true) => {
                self.reported = Gesture::Pinch(PinchPhase::Hold {
                    seconds: (time - self.pinch_start) as f32,
                });
                return self.reported;
            }
            (true, false) => {
                self.candidate = pose(shape);
                self.candidate_results = 1;
                self.reported = Gesture::Pinch(PinchPhase::Release);
                return self.reported;
            }
            (false, false) => {}
        }
        let gesture = pose(shape);
        if gesture == self.candidate {
            self.candidate_results += 1;
        } else {
            self.candidate = gesture;
            self.candidate_results = 1;
        }
        if self.candidate_results >= DEBOUNCE_RESULTS {
            self.reported = self.candidate;
        } else if matches!(self.reported, Gesture::Pinch(_)) {
            // A released pinch shows Release once, then None until the next gesture settles
            self.reported = Gesture::None;
        }
        self.reported
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upright hand (fingers up, y down) with the wrist at the origin and a palm size of 100:
    /// each finger in `extended` straight up from its base, the others curled into the palm; the
    /// thumb out to the side, or its tip at `thumb_tip`.
    fn hand(extended: [bool; 4], thumb_tip: Option<Vector2f>) -> [Vector2f; LANDMARK_COUNT] {
        let mut points = [Vector2f::ZERO; LANDMARK_COUNT];
        let bases = [-30.0, -10.0, 10.0, 30.0];
        for (finger, &(mcp, pip, tip)) in FINGERS.iter().enumerate() {
            let base = Vector2f::new(bases[finger], -100.0);
            points[mcp] = base;
            if extended[finger] {
                points[pip] = base + Vector2f::new(0.0, -40.0);
                points[pip + 1] = base + Vector2f::new(0.0, -65.0);
                points[tip] = base + Vector2f::new(0.0, -85.0);
            } else {
                points[pip] = base + Vector2f::new(0.0, -30.0);
                points[pip + 1] = base + Vector2f::new(0.0, -10.0);
                points[tip] = base + Vector2f::new(0.0, 10.0);
            }
        }
        points[1] = Vector2f::new(-30.0, -20.0);
        points[2] = Vector2f::new(-55.0, -40.0);
        points[3] = Vector2f::new(-75.0, -55.0);
        points[THUMB_TIP] = thumb_tip.unwrap_or(Vector2f::new(-120.0, -50.0));
        points
    }

    fn settled(landmarks: [Vector2f; LANDMARK_COUNT]) -> Gesture {
        let mut gestures = Gestures::default();
        (0..DEBOUNCE_RESULTS)
            .map(|i| gestures.update(&[landmarks], i as f64)[0])
            .last()
            .unwrap_or(Gesture::None)
    }

    #[test]
    fn shape_upright_open_hand_all_fingers_and_thumb_extended() {
        let shape = shape(&hand([true; 4], None));
        assert_eq!(shape.fingers, [true; 4]);
        assert!(shape.thumb);
    }

    #[test]
    fn index_only_extended_settles_to_point() {
        assert_eq!(settled(hand([true, false, false, false], None)), Gesture::Point);
    }

    #[test]
    fn all_fingers_extended_settles_to_open_palm() {
        assert_eq!(settled(hand([true; 4], None)), Gesture::OpenPalm);
    }

    #[test]
    fn all_fingers_curled_settles_to_grab() {
        assert_eq!(settled(hand([false; 4], None)), Gesture::Grab);
    }

    #[test]
    fn index_and_middle_extended_settles_to_none() {
        assert_eq!(settled(hand([true, true, false, false], None)), Gesture::None);
    }

    #[test]
    fn point_before_debounce_reports_previous_gesture() {
        let mut gestures = Gestures::default();
        let point = hand([true, false, false, false], None);
        assert_eq!(gestures.update(&[point], 0.0)[0], Gesture::None);
        assert_eq!(gestures.update(&[point], 0.1)[0], Gesture::None);
        assert_eq!(gestures.update(&[point], 0.2)[0], Gesture::Point);
    }

    #[test]
    fn pinch_touch_hold_open_start_hold_release() {
        let mut gestures = Gestures::default();
        let index_tip = hand([true; 4], None)[INDEX_TIP];
        let touching = hand([true; 4], Some(index_tip + Vector2f::new(5.0, 0.0)));
        let apart = hand([true; 4], None);
        assert_eq!(gestures.update(&[touching], 1.0)[0], Gesture::Pinch(PinchPhase::Start));
        assert_eq!(
            gestures.update(&[touching], 1.5)[0],
            Gesture::Pinch(PinchPhase::Hold { seconds: 0.5 })
        );
        assert_eq!(gestures.update(&[apart], 2.0)[0], Gesture::Pinch(PinchPhase::Release));
    }

    #[test]
    fn pinch_distance_between_enter_and_exit_keeps_pinching() {
        let mut gestures = Gestures::default();
        let index_tip = hand([true; 4], None)[INDEX_TIP];
        // Palm size 100: 20 enters (< 0.25), 30 stays (< 0.40) but wouldn't enter
        let enter = hand([true; 4], Some(index_tip + Vector2f::new(20.0, 0.0)));
        let between = hand([true; 4], Some(index_tip + Vector2f::new(30.0, 0.0)));
        assert_eq!(gestures.update(&[between], 0.0)[0], Gesture::None);
        assert_eq!(gestures.update(&[enter], 0.1)[0], Gesture::Pinch(PinchPhase::Start));
        assert!(matches!(
            gestures.update(&[between], 0.2)[0],
            Gesture::Pinch(PinchPhase::Hold { .. })
        ));
    }

    #[test]
    fn hand_moved_far_away_starts_new_debounce() {
        let mut gestures = Gestures::default();
        let point = hand([true, false, false, false], None);
        for i in 0..DEBOUNCE_RESULTS {
            gestures.update(&[point], i as f64);
        }
        let moved = point.map(|p| p + Vector2f::new(1000.0, 0.0));
        assert_eq!(gestures.update(&[moved], 9.0)[0], Gesture::None);
    }

    /// Settled gesture of the hand the GPU tracker finds in `tests/<name>.png`.
    fn image_gesture(name: &str) -> Gesture {
        use crate::test_support::{test_device, test_image, test_tracker};
        use pill_engine::game::wgpu;
        let image = test_image(name);
        let (device, queue) = test_device();
        let mut tracker = test_tracker(&device);
        let mut gestures = Gestures::default();
        let mut gesture = Gesture::None;
        // Job 1 finds the palm, then each job tracks the landmarks; enough results to settle
        for result in 0..2 + DEBOUNCE_RESULTS {
            tracker.step(&device, &queue, || Some(image.clone()));
            let _ = device.poll(wgpu::PollType::Wait);
            tracker.step(&device, &queue, || None);
            if result >= 1 {
                assert_eq!(tracker.hands().len(), 1, "{name}: hands");
                gesture = gestures.update(tracker.hands(), result as f64)[0];
            }
        }
        gesture
    }

    #[test]
    fn pointing_up_image_point() {
        assert_eq!(image_gesture("pointing_up"), Gesture::Point);
    }

    #[test]
    fn thumb_up_image_grab() {
        assert_eq!(image_gesture("thumb_up"), Gesture::Grab);
    }

    #[test]
    fn victory_image_none() {
        assert_eq!(image_gesture("victory"), Gesture::None);
    }

    #[test]
    fn ids_two_hands_swapped_order_keep_ids() {
        let mut gestures = Gestures::default();
        let left = hand([true; 4], None);
        let right = left.map(|p| p + Vector2f::new(500.0, 0.0));
        gestures.update(&[left, right], 0.0);
        let ids = gestures.ids();
        gestures.update(&[right, left], 0.1);
        assert_eq!(gestures.ids(), [ids[1], ids[0]]);
    }
}
