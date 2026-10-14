//! HTTP byte fetch. WASM uses the browser `fetch` API (cross-origin is allowed because
//! raw.githubusercontent.com serves `access-control-allow-origin: *`); native uses `ureq`.
//! Responses are cached so iteration doesn't refetch the ~50 MB each run: in the browser's Cache
//! Storage (`CACHE_NAME`, clear via DevTools → Application) and natively in `build/fetch_cache/`
//! of this example (`rm -rf` it to refetch). The assets are versioned upstream, never revalidated.

/// Base URL of the multi-file Sponza glTF. Buffer (`Sponza.bin`) and texture URIs in the
/// document are relative to this, exactly as three.js's GLTFLoader resolves them.
pub const SPONZA_BASE: &str =
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Sponza/glTF/";

/// CC0 studio HDR for image-based lighting (Poly Haven also serves permissive CORS headers).
pub const HDR_URL: &str =
    "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_08_1k.hdr";

#[cfg(target_arch = "wasm32")]
const CACHE_NAME: &str = "pill-sponza";

/// Fetches all `urls` concurrently; results are in `urls` order.
#[cfg(target_arch = "wasm32")]
pub async fn get_all(urls: &[String]) -> Result<Vec<Vec<u8>>, String> {
    futures::future::join_all(urls.iter().map(|url| get(url)))
        .await
        .into_iter()
        .collect()
}

/// Fetches all `urls` concurrently (one thread each); results are in `urls` order.
#[cfg(not(target_arch = "wasm32"))]
pub async fn get_all(urls: &[String]) -> Result<Vec<Vec<u8>>, String> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = urls
            .iter()
            .map(|url| scope.spawn(move || get(url)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .map_err(|_| "fetch thread panicked".to_string())?
            })
            .collect()
    })
}

#[cfg(target_arch = "wasm32")]
async fn get(url: &str) -> Result<Vec<u8>, String> {
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Cache, Request, Response};

    let window = web_sys::window().ok_or_else(|| "no window".to_string())?;
    // Cache Storage exists only in secure contexts (https, localhost); without it, plain fetch
    let cache: Option<Cache> = match window.caches() {
        Ok(storage) => JsFuture::from(storage.open(CACHE_NAME))
            .await
            .ok()
            .and_then(|cache| cache.dyn_into().ok()),
        Err(_) => None,
    };
    let cached: Option<Response> = match &cache {
        Some(cache) => JsFuture::from(cache.match_with_str(url))
            .await
            .ok()
            .and_then(|hit| hit.dyn_into().ok()),
        None => None,
    };
    let resp = match cached {
        Some(resp) => resp,
        None => {
            // A plain GET request; cross-origin requests default to CORS mode, which is what we want.
            let request =
                Request::new_with_str(url).map_err(|e| format!("build request {url}: {e:?}"))?;
            let resp: Response = JsFuture::from(window.fetch_with_request(&request))
                .await
                .map_err(|e| format!("fetch {url}: {e:?}"))?
                .dyn_into()
                .map_err(|_| format!("fetch {url}: response was not a Response"))?;
            if !resp.ok() {
                return Err(format!("HTTP {} for {url}", resp.status()));
            }
            // A failed store only costs a refetch next run
            if let (Some(cache), Ok(copy)) = (&cache, resp.clone()) {
                if let Err(error) = JsFuture::from(cache.put_with_str(url, &copy)).await {
                    log::warn!("Sponza: caching {url} failed: {error:?}");
                }
            }
            resp
        }
    };
    let buf = JsFuture::from(
        resp.array_buffer()
            .map_err(|e| format!("array_buffer {url}: {e:?}"))?,
    )
    .await
    .map_err(|e| format!("array_buffer await {url}: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

#[cfg(not(target_arch = "wasm32"))]
fn get(url: &str) -> Result<Vec<u8>, String> {
    let cache_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("build/fetch_cache");
    // One file per URL, named after it with path separators flattened
    let file_name: String = url
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = cache_dir.join(file_name);
    if let Ok(bytes) = std::fs::read(&path) {
        return Ok(bytes);
    }
    let bytes = download(url)?;
    // Written to a temporary name, then renamed, so an interrupted run never leaves a truncated
    // file behind; a failed store only costs a refetch next run
    let partial = path.with_extension("partial");
    let stored = std::fs::create_dir_all(&cache_dir)
        .and_then(|()| std::fs::write(&partial, &bytes))
        .and_then(|()| std::fs::rename(&partial, &path));
    if let Err(error) = stored {
        log::warn!("Sponza: caching {url} failed: {error}");
    }
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn download(url: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let resp = ureq::get(url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| format!("read {url}: {e}"))?;
    Ok(buf)
}
