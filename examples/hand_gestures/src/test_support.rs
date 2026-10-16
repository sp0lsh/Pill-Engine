//! Shared helpers of the tests: the models, the MediaPipe test images and a wgpu device.

use crate::gpu_tracker::GpuTracker;
use crate::hands::Image;
use pill_engine::game::wgpu;

pub const PALM_ONNX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/res/models/palm_detection_mediapipe_2023feb.onnx"
);
pub const HAND_ONNX: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/res/models/handpose_estimation_mediapipe_2023feb.onnx"
);

/// `tests/<name>.png`: MediaPipe's test images (storage.googleapis.com/mediapipe-assets, Apache 2.0).
pub fn test_image(name: &str) -> Image {
    let path = format!("{}/tests/{name}.png", env!("CARGO_MANIFEST_DIR"));
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).expect("png")));
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

pub fn test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("device")
}

pub fn test_tracker(device: &wgpu::Device) -> GpuTracker {
    GpuTracker::new(
        device,
        &std::fs::read(PALM_ONNX).expect("palm model"),
        &std::fs::read(HAND_ONNX).expect("hand model"),
    )
    .expect("tracker")
}
