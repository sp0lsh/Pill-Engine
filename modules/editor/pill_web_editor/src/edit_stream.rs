//! The edit stream to the page: applied edits and selection out to a listener,
//! ops and highlighted entities in.

// Standard library
use std::cell::RefCell;

// External crates
use pill_core::telemetry::telemetry_target;
use pill_core::warn;
use pill_editor_core::commands::PendingEditorCommands;
use pill_editor_core::edit_stream::{from_edit_op, EditOp, EditOutbox};
use pill_editor_core::editor_state::EditorSelection;
use pill_engine_core::{Engine, EntityId};
use serde::Serialize;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

thread_local! {
    static LISTENER: RefCell<Option<js_sys::Function>> = const { RefCell::new(None) };
    static INBOX: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    static HIGHLIGHTED: RefCell<Vec<EntityId>> = const { RefCell::new(Vec::new()) };
    static LAST_SELECTION: RefCell<Vec<EntityId>> = const { RefCell::new(Vec::new()) };
}

/// Calls `listener(json)` with each frame's `{"kind":"ops","ops":[..]}` and each
/// selection change, `{"kind":"selection","ids":[..]}`.
#[wasm_bindgen]
pub fn edits_subscribe(listener: js_sys::Function) {
    LISTENER.with(|slot| *slot.borrow_mut() = Some(listener));
}

/// Queues ops, a JSON array of `EditOp`, for the next frame. They are applied
/// with no history entry and are not handed back to the listener.
#[wasm_bindgen]
pub fn edits_apply(json: &str) {
    INBOX.with(|inbox| inbox.borrow_mut().push(json.to_owned()));
}

/// Replaces the entities outlined beside the selection, a JSON array of ids.
#[wasm_bindgen]
pub fn highlight_set(json: &str) {
    match serde_json::from_str::<Vec<EntityId>>(json) {
        Ok(ids) => HIGHLIGHTED.with(|highlighted| *highlighted.borrow_mut() = ids),
        Err(error) => refuse("highlight", &error.to_string()),
    }
}

/// Runs `read` over the highlighted entity ids.
pub fn with_highlighted<R>(read: impl FnOnce(&[EntityId]) -> R) -> R {
    HIGHLIGHTED.with(|highlighted| read(&highlighted.borrow()))
}

/// Queues the page's ops as remote batches. The outbox exists only once a
/// listener does, so an unsubscribed editor converts nothing.
pub fn before_drain(engine: &mut Engine) {
    let world = engine.world_mut();
    let subscribed = LISTENER.with(|slot| slot.borrow().is_some());
    if subscribed && world.get_resource::<EditOutbox>().is_none() {
        world.register_resource::<EditOutbox>();
        world.insert_resource(EditOutbox::default());
    }
    for json in INBOX.with(|inbox| std::mem::take(&mut *inbox.borrow_mut())) {
        queue_remote_ops(engine, &json);
    }
}

/// Emits this frame's ops and a changed selection.
pub fn after_drain(engine: &mut Engine) {
    emit_outbox(engine);
    let world = engine.world();
    let Some(selection) = world.get_resource::<EditorSelection>() else {
        return;
    };
    let ids = selection
        .entities()
        .iter()
        .filter_map(|entity| world.entity_id(*entity));
    let changed = LAST_SELECTION.with(|last| !last.borrow().iter().copied().eq(ids.clone()));
    if changed {
        let ids: Vec<EntityId> = ids.collect();
        emit(&Message::Selection { ids: &ids });
        LAST_SELECTION.with(|last| *last.borrow_mut() = ids);
    }
}

/// Emits the ops waiting in the outbox.
pub fn emit_outbox(engine: &mut Engine) {
    let ops = engine
        .world_mut()
        .get_resource_mut::<EditOutbox>()
        .map(|outbox| std::mem::take(&mut outbox.ops))
        .unwrap_or_default();
    if !ops.is_empty() {
        emit(&Message::Ops { ops: &ops });
    }
}

fn queue_remote_ops(engine: &mut Engine, json: &str) {
    let ops: Vec<EditOp> = match serde_json::from_str(json) {
        Ok(ops) => ops,
        Err(error) => return refuse("ops", &error.to_string()),
    };
    let mut commands = Vec::with_capacity(ops.len());
    for op in ops {
        match from_edit_op(engine, op) {
            Ok(Some(command)) => commands.push(command),
            Ok(None) => {}
            Err(reason) => refuse("op", &reason),
        }
    }
    if commands.is_empty() {
        return;
    }
    if let Some(pending) = engine
        .world_mut()
        .get_resource_mut::<PendingEditorCommands>()
    {
        pending.push_remote(commands);
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Message<'a> {
    Ops { ops: &'a [EditOp] },
    Selection { ids: &'a [EntityId] },
}

fn emit(message: &Message) {
    let Some(listener) = LISTENER.with(|slot| slot.borrow().clone()) else {
        return;
    };
    let text = match serde_json::to_string(message) {
        Ok(text) => text,
        Err(error) => return refuse("message", &error.to_string()),
    };
    if let Err(error) = listener.call1(&JsValue::NULL, &JsValue::from_str(&text)) {
        refuse("listener", &format!("{error:?}"));
    }
}

fn refuse(what: &str, reason: &str) {
    warn!(
        target: telemetry_target::EDITOR,
        what,
        reason,
        "[editor] edit stream input refused"
    );
}
