//! HTTP byte fetch with an on-disk cache (`build/fetch_cache/`, mirroring the repository paths;
//! `rm -rf` it to refetch). The assets are versioned upstream and never revalidated.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The Bistro glTF + KTX2 conversion (qian-o/GLTF-Assets, CC BY 4.0). Plain files.
pub const BISTRO_RAW: &str = "https://raw.githubusercontent.com/qian-o/GLTF-Assets/main/Bistro/";
/// The same repository's Git LFS objects (the `.bin` buffers).
pub const BISTRO_LFS: &str =
    "https://media.githubusercontent.com/media/qian-o/GLTF-Assets/main/Bistro/";
/// Downloads in flight at once.
const PARALLEL_DOWNLOADS: usize = 16;

/// Directory under this example's `build/` (created on demand).
pub fn build_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("build")
        .join(name)
}

/// Bytes of `path` (relative to the Bistro folder), from the cache or downloaded from `base`.
pub fn get(base: &str, path: &str) -> Result<Vec<u8>, String> {
    let cached = build_dir("fetch_cache").join(path);
    if let Ok(bytes) = std::fs::read(&cached) {
        return Ok(bytes);
    }
    let bytes = download(&format!("{base}{path}"))?;
    store(&cached, &bytes);
    Ok(bytes)
}

/// `get` for every path, `PARALLEL_DOWNLOADS` at a time; results are in `paths` order.
pub fn get_all(base: &str, paths: &[String]) -> Result<Vec<Vec<u8>>, String> {
    map_parallel(paths, PARALLEL_DOWNLOADS, |path| get(base, path))
}

/// Runs `work` over `items` on `threads` threads; results in `items` order, the first error wins.
pub fn map_parallel<T: Sync, R: Send>(
    items: &[T],
    threads: usize,
    work: impl Fn(&T) -> Result<R, String> + Sync,
) -> Result<Vec<R>, String> {
    let next = AtomicUsize::new(0);
    let mut results: Vec<(usize, Result<R, String>)> =
        std::thread::scope(|scope| -> Result<_, String> {
            let workers: Vec<_> = (0..threads.min(items.len()).max(1))
                .map(|_| {
                    scope.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(item) = items.get(index) else {
                                return done;
                            };
                            done.push((index, work(item)));
                        }
                    })
                })
                .collect();
            // A worker that panicked lost the results it had: fail the whole call rather than return
            // fewer results than items
            let mut results = Vec::with_capacity(items.len());
            for worker in workers {
                results.extend(
                    worker
                        .join()
                        .map_err(|_| "worker thread panicked".to_string())?,
                );
            }
            Ok(results)
        })?;
    results.sort_unstable_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

/// Writes to a temporary name, then renames, so an interrupted run never leaves a truncated file;
/// a failed store only costs redoing the work next run.
pub fn store(path: &Path, bytes: &[u8]) {
    let partial = path.with_extension("partial");
    let stored = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&partial, bytes))
        .and_then(|()| std::fs::rename(&partial, path));
    if let Err(error) = stored {
        log::warn!("Bistro: caching {path:?} failed: {error}");
    }
}

fn download(url: &str) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {url}: {e}"))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_parallel_worker_panic_is_an_error_not_fewer_results() {
        let result = map_parallel(&[1, 2, 3, 4], 2, |&item| {
            if item == 2 {
                panic!("cook failed");
            }
            Ok(item)
        });
        assert!(result.is_err());
    }
}
