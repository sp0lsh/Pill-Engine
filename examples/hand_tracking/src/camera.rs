//! Browser camera (web build only): the first `videoinput` device through `getUserMedia` into a
//! hidden `<video>`, and its frames read back through an OffscreenCanvas for inference. JS
//! handles aren't `Send`, so they live in a thread-local.

use crate::hands::Image;
use std::cell::RefCell;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    HtmlVideoElement, MediaDeviceInfo, MediaDeviceKind, MediaStream, MediaStreamConstraints,
    OffscreenCanvas, OffscreenCanvasRenderingContext2d,
};

/// Longest side of the frames given to the tracker, pixels.
const MAX_FRAME_SIZE: u32 = 640;

struct Camera {
    video: HtmlVideoElement,
    canvas: OffscreenCanvas,
    context: OffscreenCanvasRenderingContext2d,
}

thread_local! {
    static CAMERA: RefCell<Option<Camera>> = const { RefCell::new(None) };
}

fn js_error(context: &str, error: JsValue) -> String {
    format!("{context}: {}", error.as_string().unwrap_or_else(|| format!("{error:?}")))
}

/// `getUserMedia` constraints for the first camera, or any camera when none is listed.
async fn first_camera_constraints() -> Result<MediaStreamConstraints, String> {
    let window = web_sys::window().ok_or("no window")?;
    let devices = window
        .navigator()
        .media_devices()
        .map_err(|error| js_error("camera API unavailable", error))?;
    let listed = JsFuture::from(
        devices
            .enumerate_devices()
            .map_err(|error| js_error("enumerateDevices", error))?,
    )
    .await
    .map_err(|error| js_error("enumerateDevices", error))?;
    let first_id = js_sys::Array::from(&listed)
        .iter()
        .filter_map(|device| device.dyn_into::<MediaDeviceInfo>().ok())
        .find(|device| device.kind() == MediaDeviceKind::Videoinput)
        .map(|device| device.device_id())
        .filter(|id| !id.is_empty());

    let video = js_sys::Object::new();
    let set = |key: &str, value: JsValue| js_sys::Reflect::set(&video, &key.into(), &value);
    if let Some(id) = first_id {
        set("deviceId", id.into()).map_err(|error| js_error("constraints", error))?;
    }
    set("width", 640.into()).map_err(|error| js_error("constraints", error))?;
    set("height", 480.into()).map_err(|error| js_error("constraints", error))?;
    let constraints = MediaStreamConstraints::new();
    constraints.set_video(&video);
    Ok(constraints)
}

/// Opens the camera and starts the hidden video; `Ok` once it plays.
pub async fn open() -> Result<(), String> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let constraints = first_camera_constraints().await?;
    let stream: MediaStream = JsFuture::from(
        window
            .navigator()
            .media_devices()
            .map_err(|error| js_error("camera API unavailable", error))?
            .get_user_media_with_constraints(&constraints)
            .map_err(|error| js_error("getUserMedia", error))?,
    )
    .await
    .map_err(|error| js_error("camera access", error))?
    .dyn_into()
    .map_err(|_| "getUserMedia: not a MediaStream".to_string())?;

    let video: HtmlVideoElement = document
        .create_element("video")
        .map_err(|error| js_error("create video", error))?
        .dyn_into()
        .map_err(|_| "create video: not a video element".to_string())?;
    video.set_muted(true);
    video
        .set_attribute("playsinline", "")
        .map_err(|error| js_error("video", error))?;
    // Kept in the page (some browsers stop decoding detached or display:none videos), unseen
    video
        .style()
        .set_css_text("position:fixed;left:0;top:0;width:1px;height:1px;opacity:0;pointer-events:none");
    document
        .body()
        .ok_or("no body")?
        .append_child(&video)
        .map_err(|error| js_error("append video", error))?;
    video.set_src_object(Some(&stream));
    JsFuture::from(video.play().map_err(|error| js_error("play", error))?)
        .await
        .map_err(|error| js_error("play", error))?;

    let canvas = OffscreenCanvas::new(1, 1).map_err(|error| js_error("OffscreenCanvas", error))?;
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"willReadFrequently".into(), &true.into())
        .map_err(|error| js_error("canvas options", error))?;
    let context: OffscreenCanvasRenderingContext2d = canvas
        .get_context_with_context_options("2d", &options)
        .map_err(|error| js_error("canvas 2d", error))?
        .ok_or("canvas 2d unavailable")?
        .dyn_into()
        .map_err(|_| "canvas 2d: wrong context type".to_string())?;
    CAMERA.with(|camera| {
        *camera.borrow_mut() = Some(Camera {
            video,
            canvas,
            context,
        })
    });
    Ok(())
}

/// The playing video, if it has a frame.
pub fn video() -> Option<HtmlVideoElement> {
    CAMERA.with(|camera| {
        camera
            .borrow()
            .as_ref()
            .map(|camera| camera.video.clone())
            .filter(|video| video.ready_state() >= 2 && video.video_width() > 0)
    })
}

/// The current frame, downscaled to at most `MAX_FRAME_SIZE` on its longest side.
pub fn frame() -> Option<Image> {
    let video = video()?;
    CAMERA.with(|camera| {
        let camera = camera.borrow();
        let camera = camera.as_ref()?;
        let (video_width, video_height) = (video.video_width(), video.video_height());
        let scale = (MAX_FRAME_SIZE as f32 / video_width.max(video_height) as f32).min(1.0);
        let width = ((video_width as f32 * scale) as u32).max(1);
        let height = ((video_height as f32 * scale) as u32).max(1);
        if camera.canvas.width() != width || camera.canvas.height() != height {
            camera.canvas.set_width(width);
            camera.canvas.set_height(height);
        }
        camera
            .context
            .draw_image_with_html_video_element_and_dw_and_dh(
                &video,
                0.0,
                0.0,
                width as f64,
                height as f64,
            )
            .ok()?;
        let data = camera
            .context
            .get_image_data(0.0, 0.0, width as f64, height as f64)
            .ok()?;
        Some(Image {
            width: width as usize,
            height: height as usize,
            rgba: data.data().0,
        })
    })
}

/// GET `url` (relative to the page) as bytes.
pub async fn fetch(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let response: web_sys::Response = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|error| js_error(url, error))?
        .dyn_into()
        .map_err(|_| format!("{url}: not a Response"))?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|error| js_error(url, error))?,
    )
    .await
    .map_err(|error| js_error(url, error))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}
