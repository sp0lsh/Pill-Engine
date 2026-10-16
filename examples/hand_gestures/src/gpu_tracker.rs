//! MediaPipe's tracking loop with both models on the GPU (`GpuNet`), never waiting for it: each
//! `step` collects the previous job's read-back if the GPU has finished, then submits the next
//! job. A job runs the landmark model on every tracked hand (crop from its last landmarks) and
//! on the palms found by the last palm detection, plus the palm detector while fewer than
//! `MAX_HANDS` hands are tracked.

// Used by the web build and the tests
#![cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]

use crate::gpu_net::GpuNet;
use crate::hands::{self, HandRoi, Image, ANCHOR_COUNT, HAND_INPUT_SIZE, LANDMARK_COUNT, PALM_INPUT_SIZE};
use pill_engine::game::{bytemuck, wgpu, Vector2f, Vector3f};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const MAX_HANDS: usize = 2;
/// Read-back floats per palm detection (box regressors, logits) and per hand (landmarks, score,
/// handedness, world landmarks).
const PALM_READBACK: usize = ANCHOR_COUNT * 18 + ANCHOR_COUNT;
const HAND_READBACK: usize = LANDMARK_COUNT * 3 + 2 + LANDMARK_COUNT * 3;
const HAND_WORLD_OFFSET: usize = LANDMARK_COUNT * 3 + 2;
const HAND_INPUT_LEN: usize = HAND_INPUT_SIZE * HAND_INPUT_SIZE * 3;

/// A job on the GPU: what it was submitted with, and whether its read-back is mapped.
struct Job {
    palm: bool,
    rois: Vec<HandRoi>,
    frame_size: (usize, usize),
    started_ms: f64,
    mapped: Arc<AtomicBool>,
}

pub struct GpuTracker {
    palm: GpuNet,
    hand: GpuNet,
    /// The landmark model input of each hand of a job, copied into `hand.input()` before its run
    hand_inputs: wgpu::Buffer,
    readback: wgpu::Buffer,
    job: Option<Job>,
    hands: Vec<[Vector2f; LANDMARK_COUNT]>,
    /// Metric 3D landmarks of `hands` (`HandRoi::world_landmarks`)
    world_hands: Vec<[Vector3f; LANDMARK_COUNT]>,
    /// Crops from the last palm detection, run in the next job
    palm_rois: Vec<HandRoi>,
    /// Submit → read-back of the last finished job, milliseconds
    pub latency_ms: f32,
}

/// Milliseconds on a monotonic clock (the page's `performance.now()` in the browser).
pub fn now_ms() -> f64 {
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

fn bounds(points: &[Vector2f]) -> (Vector2f, Vector2f) {
    let min = points.iter().copied().reduce(Vector2f::min).unwrap_or_default();
    let max = points.iter().copied().reduce(Vector2f::max).unwrap_or_default();
    (min, max)
}

impl GpuTracker {
    pub fn new(device: &wgpu::Device, palm_onnx: &[u8], hand_onnx: &[u8]) -> Result<Self, String> {
        let palm = GpuNet::new(
            device,
            &crate::onnx::parse(palm_onnx)?,
            &[1, PALM_INPUT_SIZE, PALM_INPUT_SIZE, 3],
        )?;
        let hand = GpuNet::new(
            device,
            &crate::onnx::parse(hand_onnx)?,
            &[1, HAND_INPUT_SIZE, HAND_INPUT_SIZE, 3],
        )?;
        let hand_inputs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hand_inputs"),
            size: (MAX_HANDS * HAND_INPUT_LEN * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hand_tracking_readback"),
            size: ((PALM_READBACK + MAX_HANDS * HAND_READBACK) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            palm,
            hand,
            hand_inputs,
            readback,
            job: None,
            hands: Vec::new(),
            world_hands: Vec::new(),
            palm_rois: Vec::new(),
            latency_ms: 0.0,
        })
    }

    /// Landmarks of the tracked hands, in the pixels of the frames given to `step`.
    pub fn hands(&self) -> &[[Vector2f; LANDMARK_COUNT]] {
        &self.hands
    }

    /// Metric 3D landmarks of `hands()`, same order.
    pub fn world_hands(&self) -> &[[Vector3f; LANDMARK_COUNT]] {
        &self.world_hands
    }

    /// Collects a finished job, then submits the next one on `frame()` when idle. Returns whether
    /// a job finished.
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: impl FnOnce() -> Option<Image>,
    ) -> bool {
        // Native needs polling to fire map callbacks; the browser fires them itself
        let _ = device.poll(wgpu::PollType::Poll);
        let finished = self
            .job
            .as_ref()
            .is_some_and(|job| job.mapped.load(Ordering::Acquire));
        if finished {
            self.collect();
        }
        if self.job.is_none() {
            if let Some(image) = frame() {
                self.submit(device, queue, &image);
            }
        }
        finished
    }

    fn collect(&mut self) {
        let Some(job) = self.job.take() else {
            return;
        };
        {
            let view = self.readback.slice(..).get_mapped_range();
            let data: &[f32] = bytemuck::cast_slice(&view);
            (self.hands, self.world_hands) = job
                .rois
                .iter()
                .enumerate()
                .filter_map(|(i, roi)| {
                    let result = &data[PALM_READBACK + i * HAND_READBACK..][..HAND_READBACK];
                    (result[LANDMARK_COUNT * 3] >= hands::HAND_CONFIDENCE_THRESHOLD).then(|| {
                        (roi.landmarks(result), roi.world_landmarks(&result[HAND_WORLD_OFFSET..]))
                    })
                })
                .unzip();
            self.palm_rois.clear();
            if job.palm {
                let tracked: Vec<(Vector2f, Vector2f)> =
                    self.hands.iter().map(|landmarks| bounds(landmarks)).collect();
                let (width, height) = job.frame_size;
                let palms = hands::decode_palms(
                    &data[..ANCHOR_COUNT * 18],
                    &data[ANCHOR_COUNT * 18..PALM_READBACK],
                    width,
                    height,
                );
                for palm in palms {
                    let overlaps = tracked
                        .iter()
                        .any(|&(min, max)| palm.max.cmpge(min).all() && palm.min.cmple(max).all());
                    if !overlaps && self.hands.len() + self.palm_rois.len() < MAX_HANDS {
                        self.palm_rois
                            .push(HandRoi::from_palm(palm.min, palm.max, &palm.keypoints));
                    }
                }
            }
        }
        self.readback.unmap();
        self.latency_ms = (now_ms() - job.started_ms) as f32;
    }

    fn submit(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) {
        let mut rois: Vec<HandRoi> = self.hands.iter().map(HandRoi::from_landmarks).collect();
        rois.extend(self.palm_rois.drain(..));
        rois.truncate(MAX_HANDS);
        let palm = rois.len() < MAX_HANDS;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hand_tracking"),
        });
        let copy_outputs = |net: &GpuNet, encoder: &mut wgpu::CommandEncoder, offset: usize| {
            let mut offset = offset;
            for (buffer, len) in net.outputs() {
                encoder.copy_buffer_to_buffer(buffer, 0, &self.readback, (offset * 4) as u64, (len * 4) as u64);
                offset += len;
            }
        };
        if palm {
            queue.write_buffer(
                self.palm.input(),
                0,
                bytemuck::cast_slice(&hands::palm_input(image)),
            );
            self.palm.encode(&mut encoder);
            copy_outputs(&self.palm, &mut encoder, 0);
        }
        for (i, roi) in rois.iter().enumerate() {
            let offset = (i * HAND_INPUT_LEN * 4) as u64;
            queue.write_buffer(&self.hand_inputs, offset, bytemuck::cast_slice(&roi.input(image)));
            encoder.copy_buffer_to_buffer(
                &self.hand_inputs,
                offset,
                self.hand.input(),
                0,
                (HAND_INPUT_LEN * 4) as u64,
            );
            self.hand.encode(&mut encoder);
            copy_outputs(&self.hand, &mut encoder, PALM_READBACK + i * HAND_READBACK);
        }
        queue.submit([encoder.finish()]);

        let mapped = Arc::new(AtomicBool::new(false));
        let flag = mapped.clone();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    flag.store(true, Ordering::Release);
                }
            });
        self.job = Some(Job {
            palm,
            rois,
            frame_size: (image.width, image.height),
            started_ms: now_ms(),
            mapped,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::{test_device, test_image, test_tracker};

    #[test]
    fn step_victory_image_finds_then_tracks_one_hand() {
        let (device, queue) = test_device();
        let mut tracker = test_tracker(&device);
        let image = test_image("victory");
        // Job 1 detects the palm, job 2 finds its landmarks, job 3 tracks them
        let mut jobs = 0;
        while jobs < 3 {
            tracker.step(&device, &queue, || Some(image.clone()));
            let _ = device.poll(wgpu::PollType::Wait);
            if tracker.step(&device, &queue, || None) {
                jobs += 1;
            }
        }
        assert_eq!(tracker.hands().len(), 1);
        let (min, max) = bounds(&tracker.hands()[0]);
        assert!(min.cmpge(Vector2f::ZERO).all() && max.cmple(Vector2f::new(342.0, 362.0)).all());
        // The victory hand is upright: its metric wrist → middle finger base points up the image
        let up = crate::interaction::hand_axes(&tracker.world_hands()[0]).0.normalize();
        assert!(up.y < -0.8, "{up}");
    }
}
