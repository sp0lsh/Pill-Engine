//! Files a page mounts at runtime, such as uploaded models.

// Standard library
use std::cell::RefCell;

// External crates
use pill_core::telemetry::telemetry_target;
use pill_core::warn;
use pill_engine_core::asset_store::{self, mount_runtime_file};
use pill_engine_core::{AssetDatabase, World};
use wasm_bindgen::prelude::wasm_bindgen;

thread_local! {
    /// Source paths whose `.meta` was mounted and is not indexed yet.
    static PENDING_INDEX: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Makes `bytes` loadable at `path`, replacing an earlier file there; mount a
/// mesh's `.meta` beside it. Returns false for an empty, absolute or `..` path.
#[wasm_bindgen]
pub fn assets_mount(path: String, bytes: Vec<u8>) -> bool {
    if !mount_runtime_file(&path, bytes) {
        warn!(
            target: telemetry_target::EDITOR,
            path = path.as_str(),
            "[editor] asset mount refused"
        );
        return false;
    }
    if let Some(source) = path.strip_suffix(".meta") {
        PENDING_INDEX.with(|pending| pending.borrow_mut().push(source.to_owned()));
    }
    true
}

/// Indexes the guids of `.meta` files mounted since the last frame, so references
/// by guid find them.
pub fn apply_runtime_mounts(world: &mut World) {
    let sources = PENDING_INDEX.with(|pending| std::mem::take(&mut *pending.borrow_mut()));
    if sources.is_empty() {
        return;
    }
    let Some(database) = world.get_resource_mut::<AssetDatabase>() else {
        return;
    };
    for source in sources {
        let indexed = asset_store::read(std::path::Path::new(&format!("{source}.meta")))
            .is_ok_and(|metadata| database.index_sourced(&source, &metadata));
        if !indexed {
            warn!(
                target: telemetry_target::EDITOR,
                path = source.as_str(),
                "[editor] mounted asset not indexed"
            );
        }
    }
}
