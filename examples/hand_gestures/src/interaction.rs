//! Pinch to hold a box in the editor prototype: the pinch point in the camera frame becomes a
//! screen position (the video's cover + mirror mapping), then a ray from the game camera that
//! picks the nearest box. While the pinch holds, the box follows the hand in 3D: across the
//! screen with the pinch ray, in depth with the hand's scale on screen, and in rotation with the
//! hand's 3D frame (from the model's metric landmarks). All three are smoothed with a One Euro
//! filter, so landmark jitter doesn't shake the box.

use crate::game::cover_uv_scale;
use crate::gestures::pinch_point;
use crate::hands::{INDEX_MCP, LANDMARK_COUNT, MIDDLE_MCP, PINKY_MCP, RING_MCP, WRIST};
use pill_engine::game::{
    camera_forward, camera_view_projection, CameraComponent, TransformComponent, Vector2f, Vector3f,
};

/// Half extent of the cube mesh (res/models/cube.obj is 1 × 1 × 1).
const BOX_HALF_EXTENT: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vector3f,
    pub direction: Vector3f,
}

/// Screen uv (0..1, y down) of `point` in a camera frame of `frame_size`, shown cover-fitted and
/// mirrored on a screen of aspect `screen_aspect` (width / height).
pub fn screen_uv(point: Vector2f, frame_size: Vector2f, screen_aspect: f32) -> Vector2f {
    let uv_scale = cover_uv_scale(frame_size, Vector2f::new(screen_aspect, 1.0));
    (point / frame_size - 0.5) / uv_scale + 0.5
}

/// The camera ray through screen `uv`, from the inverse of its view-projection.
pub fn ray(inverse_view_projection: glam::Mat4, uv: Vector2f) -> Ray {
    let ndc = Vector2f::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let near = inverse_view_projection.project_point3(ndc.extend(0.0));
    let far = inverse_view_projection.project_point3(ndc.extend(1.0));
    Ray {
        origin: near,
        direction: (far - near).normalize_or_zero(),
    }
}

/// Distance along `ray` to the box of `transform` (cube of `BOX_HALF_EXTENT`, rotated and
/// scaled by it), if the ray hits it.
pub fn hit_box(ray: &Ray, transform: &TransformComponent) -> Option<f32> {
    // Into the box's local frame: undo translation, rotation, then scale
    let rotation = transform.rotation_matrix();
    let to_local = |v: Vector3f| rotation.transpose() * v / transform.scale;
    let origin = to_local(ray.origin - transform.position);
    let direction = to_local(ray.direction);
    let (mut near, mut far) = (f32::NEG_INFINITY, f32::INFINITY);
    for axis in 0..3 {
        let (o, d) = (origin[axis], direction[axis]);
        if d.abs() < 1e-8 {
            if o.abs() > BOX_HALF_EXTENT {
                return None;
            }
            continue;
        }
        let (t0, t1) = ((-BOX_HALF_EXTENT - o) / d, (BOX_HALF_EXTENT - o) / d);
        near = near.max(t0.min(t1));
        far = far.min(t0.max(t1));
    }
    // Local and world distances agree only for uniform scale (the boxes' case)
    (near <= far && far >= 0.0).then_some(near.max(0.0) * transform.scale.x)
}

/// A value the One Euro filter can smooth.
pub trait Filtered:
    Copy + std::ops::Sub<Output = Self> + std::ops::Div<f32, Output = Self>
{
    const ZERO: Self;
    fn lerp_to(self, other: Self, t: f32) -> Self;
    fn magnitude(self) -> f32;
}

impl Filtered for f32 {
    const ZERO: Self = 0.0;
    fn lerp_to(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }
    fn magnitude(self) -> f32 {
        self.abs()
    }
}

impl Filtered for Vector2f {
    const ZERO: Self = Vector2f::ZERO;
    fn lerp_to(self, other: Self, t: f32) -> Self {
        self.lerp(other, t)
    }
    fn magnitude(self) -> f32 {
        self.length()
    }
}

impl Filtered for Vector3f {
    const ZERO: Self = Vector3f::ZERO;
    fn lerp_to(self, other: Self, t: f32) -> Self {
        self.lerp(other, t)
    }
    fn magnitude(self) -> f32 {
        self.length()
    }
}

/// One Euro filter (Casiez et al., CHI 2012): a low-pass whose cutoff rises with speed, so it
/// removes jitter at rest and adds little lag to fast moves.
#[derive(Clone, Copy, Debug)]
pub struct OneEuroFilter<T> {
    value: T,
    derivative: T,
    /// Cutoff increase (Hz) per unit of speed of the value
    beta: f32,
}

/// Cutoff at rest (Hz) and the derivative's cutoff.
const MIN_CUTOFF: f32 = 1.0;
const DERIVATIVE_CUTOFF: f32 = 1.0;
/// Speed coefficients: pinch position (screen widths/s), hand axes (unit vectors/s), hand scale
/// (start-relative/s).
const POSITION_BETA: f32 = 4.0;
const AXIS_BETA: f32 = 1.0;
const SCALE_BETA: f32 = 2.0;

/// Exponential smoothing factor for `cutoff` Hz over `dt` seconds.
fn smoothing(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (std::f32::consts::TAU * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl<T: Filtered> OneEuroFilter<T> {
    pub fn new(value: T, beta: f32) -> Self {
        Self {
            value,
            derivative: T::ZERO,
            beta,
        }
    }

    /// The filtered value after `value` arrives `dt` seconds after the previous one.
    pub fn filter(&mut self, value: T, dt: f32) -> T {
        if dt <= 0.0 {
            return self.value;
        }
        let derivative = (value - self.value) / dt;
        self.derivative = self
            .derivative
            .lerp_to(derivative, smoothing(DERIVATIVE_CUTOFF, dt));
        let cutoff = MIN_CUTOFF + self.beta * self.derivative.magnitude();
        self.value = self.value.lerp_to(value, smoothing(cutoff, dt));
        self.value
    }
}

/// Palm bones measured for the hand's scale on screen.
const PALM_BONES: [(usize, usize); 6] = [
    (WRIST, INDEX_MCP),
    (WRIST, MIDDLE_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
];

/// Centre of the palm: the mean of the wrist and the index, middle and pinky bases.
fn palm_centre(landmarks: &[Vector2f; LANDMARK_COUNT]) -> Vector2f {
    [WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP]
        .iter()
        .map(|&i| landmarks[i])
        .sum::<Vector2f>()
        / 4.0
}

/// Orthonormal frame of a hand from two of its directions: `up` (wrist → middle finger base) and
/// `across` (pinky base → index base). Columns: across, up, palm normal.
fn frame(up: Vector3f, across: Vector3f) -> glam::Mat3 {
    let y = up.normalize_or_zero();
    let z = across.cross(y).normalize_or_zero();
    glam::Mat3::from_cols(y.cross(z), y, z)
}

/// Horizontal field of view of the webcam, degrees (browsers don't report it). Fitted on this
/// laptop: 89° minimises the yaw spread of a hand moved across the image at a fixed orientation.
const WEBCAM_HFOV: f32 = 89.0;

/// The hand's metric landmarks turned from the model's frame into the webcam's: the landmark
/// model sees the hand along the line of sight to it (its crop), so a hand off the image centre
/// comes out turned by the angle between that line and the optical axis. `pixel` is the hand's
/// position in a `frame_size` image.
fn perspective_corrected(
    world: &[Vector3f; LANDMARK_COUNT],
    pixel: Vector2f,
    frame_size: Vector2f,
) -> [Vector3f; LANDMARK_COUNT] {
    let focal = frame_size.x / 2.0 / (WEBCAM_HFOV.to_radians() / 2.0).tan();
    // Image axes: x right, y down, z from the webcam into the scene
    let sight = ((pixel - frame_size / 2.0) / focal).extend(1.0).normalize();
    let turn = glam::Quat::from_rotation_arc(Vector3f::Z, sight);
    world.map(|point| turn * point)
}

/// The hand's (up, across) directions from its metric 3D landmarks.
pub fn hand_axes(world: &[Vector3f; LANDMARK_COUNT]) -> (Vector3f, Vector3f) {
    (
        world[MIDDLE_MCP] - world[WRIST],
        world[INDEX_MCP] - world[PINKY_MCP],
    )
}

/// The hand's size on screen as image pixels per metre of palm, from the palm bones' image
/// lengths over their metric lengths across the image plane (so tilting the hand doesn't change
/// it, moving it toward the camera does).
fn pixels_per_metre(landmarks: &[Vector2f; LANDMARK_COUNT], world: &[Vector3f; LANDMARK_COUNT]) -> f32 {
    let (image, metric) = PALM_BONES.iter().fold((0.0, 0.0), |(image, metric), &(a, b)| {
        (
            image + landmarks[a].distance(landmarks[b]),
            metric + (world[a] - world[b]).truncate().length(),
        )
    });
    image / metric.max(1e-6)
}

/// The game camera, as the holding math needs it.
pub struct CameraFrame {
    inverse_view_projection: glam::Mat4,
    position: Vector3f,
    forward: Vector3f,
    rotation: glam::Mat3,
    /// Screen width / height
    aspect: f32,
}

impl CameraFrame {
    pub fn new(camera: &CameraComponent, transform: &TransformComponent) -> Self {
        Self {
            inverse_view_projection: camera_view_projection(camera, transform).inverse(),
            position: transform.position,
            forward: camera_forward(transform),
            rotation: transform.rotation_matrix(),
            aspect: camera.aspect.get_value(),
        }
    }

    /// The camera ray through screen `uv`.
    pub fn ray(&self, uv: Vector2f) -> Ray {
        ray(self.inverse_view_projection, uv)
    }
}

/// What a held box follows of a hand: its pinch on screen, its 3D frame and its scale.
pub struct HandPose {
    pub pinch_uv: Vector2f,
    up: Vector3f,
    across: Vector3f,
    pixels_per_metre: f32,
}

/// The pose of a hand from its image landmarks and metric 3D landmarks in a `frame_size` camera
/// frame, shown on the screen of `camera`.
pub fn hand_pose(
    landmarks: &[Vector2f; LANDMARK_COUNT],
    world: &[Vector3f; LANDMARK_COUNT],
    frame_size: Vector2f,
    camera: &CameraFrame,
) -> HandPose {
    let world = perspective_corrected(world, palm_centre(landmarks), frame_size);
    let (up, across) = hand_axes(&world);
    HandPose {
        pinch_uv: screen_uv(pinch_point(landmarks), frame_size, camera.aspect),
        up,
        across,
        pixels_per_metre: pixels_per_metre(landmarks, &world),
    }
}

/// A box held by a pinch. Position: the hit point's depth along the camera's forward axis and
/// its offset from the box centre, then the box follows the pinch ray at that depth scaled by the
/// hand's scale change. Rotation: the hand's 3D rotation since the pinch started, applied to the
/// box's rotation then; the hand's landmark axes (image x right, y down, z away from the webcam)
/// map to the mirrored view's camera axes by −I, which leaves rotations unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Hold {
    depth: f32,
    offset: Vector3f,
    pinch_uv: OneEuroFilter<Vector2f>,
    start_box: glam::Quat,
    start_hand_inverse: glam::Mat3,
    up: OneEuroFilter<Vector3f>,
    across: OneEuroFilter<Vector3f>,
    start_scale: f32,
    scale: OneEuroFilter<f32>,
}

impl Hold {
    /// Picks up the box of `transform` at `hit` with a hand at `pose`.
    pub fn new(hit: Vector3f, transform: &TransformComponent, camera: &CameraFrame, pose: &HandPose) -> Self {
        Self {
            depth: (hit - camera.position).dot(camera.forward),
            offset: hit - transform.position,
            pinch_uv: OneEuroFilter::new(pose.pinch_uv, POSITION_BETA),
            start_box: glam::Quat::from_mat3(&transform.rotation_matrix()),
            start_hand_inverse: frame(pose.up, pose.across).transpose(),
            up: OneEuroFilter::new(pose.up.normalize_or_zero(), AXIS_BETA),
            across: OneEuroFilter::new(pose.across.normalize_or_zero(), AXIS_BETA),
            start_scale: pose.pixels_per_metre,
            scale: OneEuroFilter::new(1.0, SCALE_BETA),
        }
    }

    /// Box position (if the pinch ray reaches the held depth) and rotation (Euler XYZ degrees)
    /// for the hand at `pose`, smoothed over `dt` seconds.
    pub fn update(&mut self, pose: &HandPose, dt: f32, camera: &CameraFrame) -> (Option<Vector3f>, Vector3f) {
        let ray = camera.ray(self.pinch_uv.filter(pose.pinch_uv, dt));
        // The hand twice as large on screen (twice as close) halves the depth
        let scale = self.scale.filter(pose.pixels_per_metre / self.start_scale.max(1e-6), dt);
        let depth = self.depth / scale.max(1e-3);
        let along = ray.direction.dot(camera.forward);
        let position = (along > 1e-6).then(|| {
            let t = (depth - (ray.origin - camera.position).dot(camera.forward)) / along;
            ray.origin + ray.direction * t - self.offset
        });

        let up = self.up.filter(pose.up.normalize_or_zero(), dt);
        let across = self.across.filter(pose.across.normalize_or_zero(), dt);
        let change = camera.rotation * frame(up, across) * self.start_hand_inverse * camera.rotation.transpose();
        let rotation = glam::Quat::from_mat3(&change).normalize() * self.start_box;
        let (x, y, z) = rotation.to_euler(glam::EulerRot::XYZ);
        (position, Vector3f::new(x.to_degrees(), y.to_degrees(), z.to_degrees()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pill_engine::game::{camera_view_projection, CameraComponent};

    fn camera() -> (CameraComponent, TransformComponent) {
        // At the origin looking down −Z, 16:9
        (
            CameraComponent::builder()
                .enabled(true)
                .fov(60.0)
                .aspect(pill_engine::game::CameraAspectRatio::Manual(16.0 / 9.0))
                .range(0.1..100.0)
                .build(),
            TransformComponent::builder().build(),
        )
    }

    fn centre_ray() -> Ray {
        let (camera, transform) = camera();
        ray(camera_view_projection(&camera, &transform).inverse(), Vector2f::splat(0.5))
    }

    fn box_at(position: Vector3f) -> TransformComponent {
        TransformComponent::builder()
            .position(position)
            .rotation(Vector3f::new(10.0, 30.0, 0.0))
            .build()
    }

    #[test]
    fn ray_screen_centre_points_down_minus_z() {
        let ray = centre_ray();
        assert!((ray.direction - Vector3f::NEG_Z).length() < 1e-4, "{:?}", ray.direction);
    }

    #[test]
    fn hit_box_straight_ahead_hits_box_aside_misses() {
        let ray = centre_ray();
        let distance = hit_box(&ray, &box_at(Vector3f::new(0.0, 0.0, -3.0))).expect("hit");
        assert!(distance > 2.3 && distance < 2.7, "{distance}");
        assert!(hit_box(&ray, &box_at(Vector3f::new(2.0, 0.0, -3.0))).is_none());
    }

    #[test]
    fn hit_box_two_boxes_on_ray_nearer_is_closer() {
        let ray = centre_ray();
        let near = hit_box(&ray, &box_at(Vector3f::new(0.0, 0.0, -3.0))).expect("near");
        let far = hit_box(&ray, &box_at(Vector3f::new(0.0, 0.0, -6.0))).expect("far");
        assert!(near < far);
    }

    fn camera_frame() -> CameraFrame {
        let (camera, transform) = camera();
        CameraFrame::new(&camera, &transform)
    }

    /// A hand of `world` landmarks pinching at screen `pinch_uv`, at 1000 pixels per metre.
    fn pose(world: &[Vector3f; LANDMARK_COUNT], pinch_uv: Vector2f, pixels_per_metre: f32) -> HandPose {
        let (up, across) = hand_axes(world);
        HandPose {
            pinch_uv,
            up,
            across,
            pixels_per_metre,
        }
    }

    /// Holds a box 3 m ahead by its centre-screen hit, then feeds `pose` for 4 s at 60 Hz.
    fn hold_then(start_rotation: Vector3f, later: HandPose) -> (Option<Vector3f>, Vector3f) {
        let camera = camera_frame();
        let transform = TransformComponent::builder()
            .position(Vector3f::new(0.0, 0.0, -3.0))
            .rotation(start_rotation)
            .build();
        let start = camera.ray(Vector2f::splat(0.5));
        let hit = start.origin + start.direction * hit_box(&start, &transform).expect("hit");
        let mut hold = Hold::new(hit, &transform, &camera, &pose(&flat_hand(), Vector2f::splat(0.5), 1000.0));
        let mut result = (None, Vector3f::ZERO);
        for _ in 0..240 {
            result = hold.update(&later, 1.0 / 60.0, &camera);
        }
        result
    }

    #[test]
    fn hold_unmoved_hand_keeps_box() {
        let start = Vector3f::new(20.0, 30.0, 0.0);
        let (position, rotation) = hold_then(start, pose(&flat_hand(), Vector2f::splat(0.5), 1000.0));
        assert!((position.expect("position") - Vector3f::new(0.0, 0.0, -3.0)).length() < 1e-3);
        assert!((rotation - start).length() < 1e-2, "{rotation}");
    }

    #[test]
    fn hold_pinch_moved_right_moves_box_right_at_same_depth() {
        let (position, _) = hold_then(Vector3f::ZERO, pose(&flat_hand(), Vector2f::new(0.7, 0.5), 1000.0));
        let position = position.expect("position");
        assert!(position.x > 0.5);
        assert!((position.z + 3.0).abs() < 1e-3);
    }

    #[test]
    fn hold_hand_twice_as_large_halves_grab_depth() {
        // Grabbed on the front face 2.5 m away: the face comes to 1.25 m, the centre 0.5 m behind
        let (position, _) = hold_then(Vector3f::ZERO, pose(&flat_hand(), Vector2f::splat(0.5), 2000.0));
        assert!((position.expect("position").z + 1.75).abs() < 0.03);
    }

    #[test]
    fn hold_hand_turned_about_vertical_turns_box_the_same() {
        let turn = glam::Quat::from_rotation_y(0.6);
        let (_, rotation) = hold_then(
            Vector3f::ZERO,
            pose(&turned(&flat_hand(), turn), Vector2f::splat(0.5), 1000.0),
        );
        let got = glam::Quat::from_mat3(
            &TransformComponent::builder().rotation(rotation).build().rotation_matrix(),
        );
        assert!(got.angle_between(turn) < 0.01, "{rotation}");
    }

    #[test]
    fn screen_uv_left_of_frame_lands_right_of_screen() {
        // Mirrored selfie view
        let uv = screen_uv(Vector2f::new(100.0, 240.0), Vector2f::new(640.0, 480.0), 16.0 / 9.0);
        assert!(uv.x > 0.5);
        assert!((uv.y - 0.5).abs() < 1e-6);
    }

    #[test]
    fn one_euro_jitter_at_rest_reduced_tenfold() {
        // ±0.01 alternating noise around 0.5 at 60 Hz
        let mut filter = OneEuroFilter::new(Vector2f::splat(0.5), POSITION_BETA);
        let mut worst = 0.0f32;
        for i in 0..120 {
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
            let out = filter.filter(Vector2f::splat(0.5 + noise), 1.0 / 60.0);
            if i > 60 {
                worst = worst.max((out.x - 0.5).abs());
            }
        }
        assert!(worst < 0.001, "{worst}");
    }

    #[test]
    fn one_euro_steady_move_lags_under_3_percent_of_screen() {
        // 0.5 screen widths per second for 1 s at 60 Hz; cutoff 1 + 4 · 0.5 = 3 Hz lags by
        // 0.5 / (2π · 3) ≈ 0.027
        let mut filter = OneEuroFilter::new(Vector2f::ZERO, POSITION_BETA);
        let mut out = Vector2f::ZERO;
        for i in 1..=60 {
            out = filter.filter(Vector2f::new(i as f32 / 120.0, 0.0), 1.0 / 60.0);
        }
        assert!((out.x - 0.5).abs() < 0.03, "{}", out.x);
    }

    /// Metric landmarks of a flat, upright hand (palm facing the camera, image axes): wrist at
    /// the origin, finger bases 9 cm up, spread 2 cm apart; every finger point 2 cm above its base
    /// (only the palm points matter here).
    fn flat_hand() -> [Vector3f; LANDMARK_COUNT] {
        let mut world = [Vector3f::ZERO; LANDMARK_COUNT];
        for (i, base) in [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP].into_iter().enumerate() {
            let point = Vector3f::new(0.03 - i as f32 * 0.02, -0.09, 0.0);
            for k in 0..4 {
                world[base + k] = point + Vector3f::new(0.0, -0.02 * k as f32, 0.0);
            }
        }
        world
    }

    fn turned(world: &[Vector3f; LANDMARK_COUNT], rotation: glam::Quat) -> [Vector3f; LANDMARK_COUNT] {
        world.map(|point| rotation * point)
    }

    #[test]
    fn pixels_per_metre_tilted_hand_keeps_scale() {
        let world = flat_hand();
        // The image of the flat hand at 1000 px per metre
        let image = world.map(|p| p.truncate() * 1000.0);
        assert!((pixels_per_metre(&image, &world) - 1000.0).abs() < 1e-2);
        // Tilted 60° back: foreshortened in the image and across the image plane alike
        let tilted = turned(&world, glam::Quat::from_rotation_x(1.0));
        let tilted_image = tilted.map(|p| p.truncate() * 1000.0);
        assert!((pixels_per_metre(&tilted_image, &tilted) - 1000.0).abs() < 1e-2);
    }

    #[test]
    fn perspective_corrected_hand_at_image_centre_unchanged() {
        let world = flat_hand();
        let corrected = perspective_corrected(&world, Vector2f::new(320.0, 240.0), Vector2f::new(640.0, 480.0));
        assert!(world.iter().zip(&corrected).all(|(a, b)| (*a - *b).length() < 1e-6));
    }

    #[test]
    fn perspective_corrected_off_centre_hand_undoes_line_of_sight_turn() {
        // A flat hand at the right edge, as the model reports it: seen along the line of sight,
        // so turned against it
        let (pixel, frame_size) = (Vector2f::new(600.0, 240.0), Vector2f::new(640.0, 480.0));
        let focal = 320.0 / (WEBCAM_HFOV.to_radians() / 2.0).tan();
        let sight = Vector3f::new(280.0 / focal, 0.0, 1.0).normalize();
        let seen = turned(&flat_hand(), glam::Quat::from_rotation_arc(Vector3f::Z, sight).inverse());
        let corrected = perspective_corrected(&seen, pixel, frame_size);
        assert!(flat_hand().iter().zip(&corrected).all(|(a, b)| (*a - *b).length() < 1e-5));
    }
}
