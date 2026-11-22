//! The editor's applied world edits as portable ops, and ops applied from
//! outside: entities by `EntityId`, values in the scene document's encoding.

// Standard library
use std::collections::{BTreeMap, HashSet};

// External crates
use pill_engine::{
    decode_document_field, encode_document_field, Engine, EntityId, FieldValue, World,
};
use serde::{Deserialize, Serialize};

// Current crate
use crate::editor_state::{ComponentSeed, EditorCommand};

/// One component of a created entity, with its fields document-encoded.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditComponent {
    pub type_name: String,
    pub fields: BTreeMap<String, serde_json::Value>,
}

/// One world edit, portable between worlds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum EditOp {
    CreateEntity {
        id: EntityId,
        components: Vec<EditComponent>,
    },
    DestroyEntity {
        id: EntityId,
    },
    AddComponent {
        id: EntityId,
        component: String,
    },
    RemoveComponent {
        id: EntityId,
        component: String,
    },
    RestoreComponent {
        id: EntityId,
        component: String,
        fields: BTreeMap<String, serde_json::Value>,
    },
    SetField {
        id: EntityId,
        component: String,
        field: String,
        value: serde_json::Value,
    },
    SetFieldAsset {
        id: EntityId,
        component: String,
        field: String,
        reference: Option<String>,
    },
    SetFieldEntity {
        id: EntityId,
        component: String,
        field: String,
        reference: Option<EntityId>,
    },
    SetArrayElement {
        id: EntityId,
        component: String,
        field: String,
        index: usize,
        value: serde_json::Value,
    },
}

impl EditOp {
    /// The stable id of the entity the op edits.
    pub fn id(&self) -> EntityId {
        match self {
            Self::CreateEntity { id, .. }
            | Self::DestroyEntity { id }
            | Self::AddComponent { id, .. }
            | Self::RemoveComponent { id, .. }
            | Self::RestoreComponent { id, .. }
            | Self::SetField { id, .. }
            | Self::SetFieldAsset { id, .. }
            | Self::SetFieldEntity { id, .. }
            | Self::SetArrayElement { id, .. } => *id,
        }
    }
}

/// The ops this world's applied edits produced, in apply order.
#[derive(Debug, Default)]
pub struct EditOutbox {
    pub ops: Vec<EditOp>,
}

impl pill_engine::Resource for EditOutbox {
    /// Infrastructure, not game state: Play leaves this value alone and
    /// Stop does not restore it.
    fn play_state() -> pill_engine::resource::PlayState {
        pill_engine::resource::PlayState::Keep
    }
}

/// Gives every id-less create a fresh id, then converts the batch into indexed ops.
/// Empty when the world has no [`EditOutbox`].
pub fn take_outbound(world: &World, commands: &mut [EditorCommand]) -> Vec<(usize, EditOp)> {
    if world.get_resource::<EditOutbox>().is_none() {
        return Vec::new();
    }
    let mut minted = HashSet::new();
    for command in commands.iter_mut() {
        if let EditorCommand::CreateEntity { id: id @ None, .. } = command {
            *id = fresh_id(world, &mut minted);
            if id.is_none() {
                pill_core::warn!(
                    target: pill_core::telemetry::telemetry_target::EDITOR,
                    "[editor] create not synced: no random source for its entity id"
                );
            }
        }
    }
    commands
        .iter()
        .enumerate()
        .filter_map(|(index, command)| to_edit_op(world, command).map(|op| (index, op)))
        .collect()
}

/// Moves the ops whose commands did not fail into the world's [`EditOutbox`].
pub fn publish_outbound(
    world: &mut World,
    outbound: Vec<(usize, EditOp)>,
    failed_command_indices: &HashSet<usize>,
) {
    if outbound.is_empty() {
        return;
    }
    let Some(outbox) = world.get_resource_mut::<EditOutbox>() else {
        return;
    };
    outbox.ops.extend(
        outbound
            .into_iter()
            .filter(|(index, _)| !failed_command_indices.contains(index))
            .map(|(_, op)| op),
    );
}

/// The op for one command, or `None` when the command is not a world edit or
/// names something with no portable form.
pub fn to_edit_op(world: &World, command: &EditorCommand) -> Option<EditOp> {
    let id_of = |entity| world.entity_id(entity);
    Some(match command {
        EditorCommand::CreateEntity {
            components,
            id: Some(id),
        }
        | EditorCommand::RestoreEntity {
            components,
            id: Some(id),
            ..
        } => EditOp::CreateEntity {
            id: *id,
            components: components
                .iter()
                .map(|seed| EditComponent {
                    type_name: seed.type_name.clone(),
                    fields: encode_fields(world, &seed.fields),
                })
                .collect(),
        },
        EditorCommand::DestroyEntity { entity } => EditOp::DestroyEntity {
            id: id_of(*entity)?,
        },
        EditorCommand::AddComponent { entity, component } => EditOp::AddComponent {
            id: id_of(*entity)?,
            component: component.clone(),
        },
        EditorCommand::RemoveComponent { entity, component } => EditOp::RemoveComponent {
            id: id_of(*entity)?,
            component: component.clone(),
        },
        EditorCommand::RestoreComponent {
            entity,
            component,
            fields,
        } => EditOp::RestoreComponent {
            id: id_of(*entity)?,
            component: component.clone(),
            fields: encode_fields(world, fields),
        },
        EditorCommand::SetField {
            entity,
            component,
            field,
            value,
        } => EditOp::SetField {
            id: id_of(*entity)?,
            component: component.clone(),
            field: field.clone(),
            value: encode_document_field(world, value)?,
        },
        EditorCommand::SetFieldAsset {
            entity,
            component,
            field,
            reference,
        } => EditOp::SetFieldAsset {
            id: id_of(*entity)?,
            component: component.clone(),
            field: field.clone(),
            reference: reference.clone(),
        },
        EditorCommand::SetFieldEntity {
            entity,
            component,
            field,
            reference,
        } => EditOp::SetFieldEntity {
            id: id_of(*entity)?,
            component: component.clone(),
            field: field.clone(),
            reference: match reference {
                Some(target) => Some(id_of(*target)?),
                None => None,
            },
        },
        EditorCommand::SetArrayElement {
            entity,
            component,
            field,
            index,
            value,
        } => EditOp::SetArrayElement {
            id: id_of(*entity)?,
            component: component.clone(),
            field: field.clone(),
            index: *index,
            value: encode_document_field(world, value)?,
        },
        _ => return None,
    })
}

/// Encodes every field that has a document form; a heap container has none
/// and is left out.
fn encode_fields(
    world: &World,
    fields: &[(String, FieldValue)],
) -> BTreeMap<String, serde_json::Value> {
    fields
        .iter()
        .filter_map(|(name, value)| Some((name.clone(), encode_document_field(world, value)?)))
        .collect()
}

/// A random id no live entity and no earlier create of this batch holds.
fn fresh_id(world: &World, minted: &mut HashSet<EntityId>) -> Option<EntityId> {
    for _ in 0..4 {
        let id = EntityId::random().ok()?;
        if !world.entity_id_is_taken(id) && minted.insert(id) {
            return Some(id);
        }
    }
    None
}

/// The command a received op asks for, or `None` when this world already has
/// its result or lacks its entity. Errors name an unknown field or a value that does not read.
pub fn from_edit_op(engine: &mut Engine, op: EditOp) -> Result<Option<EditorCommand>, String> {
    if let EditOp::CreateEntity { id, components } = op {
        if engine.world().entity_id_is_taken(id) {
            return Ok(None);
        }
        let mut seeds = Vec::with_capacity(components.len());
        for component in components {
            let fields = decode_fields(engine, &component.type_name, component.fields)?;
            seeds.push(ComponentSeed {
                type_name: component.type_name,
                fields,
            });
        }
        return Ok(Some(EditorCommand::CreateEntity {
            components: seeds,
            id: Some(id),
        }));
    }
    let Some(entity) = engine.world().entity_by_id(op.id()) else {
        return Ok(None);
    };
    let has = |engine: &Engine, component: &str| {
        engine
            .world()
            .resolve_entity_component_id(entity, component)
            .is_some()
    };
    Ok(Some(match op {
        EditOp::CreateEntity { .. } => unreachable!("handled above"),
        EditOp::DestroyEntity { .. } => EditorCommand::DestroyEntity { entity },
        EditOp::AddComponent { component, .. } if !has(engine, &component) => {
            EditorCommand::AddComponent { entity, component }
        }
        EditOp::RemoveComponent { component, .. } if has(engine, &component) => {
            EditorCommand::RemoveComponent { entity, component }
        }
        EditOp::RestoreComponent {
            component, fields, ..
        } if !has(engine, &component) => EditorCommand::RestoreComponent {
            fields: decode_fields(engine, &component, fields)?,
            entity,
            component,
        },
        EditOp::AddComponent { .. }
        | EditOp::RemoveComponent { .. }
        | EditOp::RestoreComponent { .. } => return Ok(None),
        EditOp::SetField {
            component,
            field,
            value,
            ..
        } => EditorCommand::SetField {
            value: decode_field(engine, &component, &field, None, &value)?,
            entity,
            component,
            field,
        },
        EditOp::SetFieldAsset {
            component,
            field,
            reference,
            ..
        } => EditorCommand::SetFieldAsset {
            entity,
            component,
            field,
            reference,
        },
        EditOp::SetFieldEntity {
            component,
            field,
            reference,
            ..
        } => {
            let reference = match reference {
                Some(target_id) => match engine.world().entity_by_id(target_id) {
                    Some(target) => Some(target),
                    None => return Ok(None),
                },
                None => None,
            };
            EditorCommand::SetFieldEntity {
                entity,
                component,
                field,
                reference,
            }
        }
        EditOp::SetArrayElement {
            component,
            field,
            index,
            value,
            ..
        } => EditorCommand::SetArrayElement {
            value: decode_field(engine, &component, &field, Some(index), &value)?,
            entity,
            component,
            field,
            index,
        },
    }))
}

/// Decodes one encoded value for a field, or for one element of an array field.
fn decode_field(
    engine: &mut Engine,
    component: &str,
    field: &str,
    index: Option<usize>,
    value: &serde_json::Value,
) -> Result<FieldValue, String> {
    let tag = field_tag(engine.world(), component, field)?;
    let tag = match index {
        Some(_) => tag
            .strip_prefix("array:")
            .ok_or_else(|| format!("{component}.{field} is not an array"))?,
        None => tag,
    };
    decode_document_field(engine, tag, value).map_err(|reason| match index {
        Some(index) => format!("{component}.{field}[{index}]: {reason}"),
        None => format!("{component}.{field}: {reason}"),
    })
}

/// Decodes a component's encoded fields against its registered layout.
fn decode_fields(
    engine: &mut Engine,
    component: &str,
    fields: BTreeMap<String, serde_json::Value>,
) -> Result<Vec<(String, FieldValue)>, String> {
    let mut decoded = Vec::with_capacity(fields.len());
    for (field, value) in fields {
        let value = decode_field(engine, component, &field, None, &value)?;
        decoded.push((field, value));
    }
    Ok(decoded)
}

/// The type tag of one registered component field.
fn field_tag(world: &World, component: &str, field: &str) -> Result<&'static str, String> {
    let component_id = world
        .resolve_component_id_by_name_any(component)
        .map_err(|error| format!("{component}: {error}"))?
        .ok_or_else(|| format!("`{component}` is not registered"))?;
    world
        .component_field_layout(component_id)
        .and_then(|layout| layout.iter().find(|row| row.name == field).map(|row| row.type_tag))
        .ok_or_else(|| format!("{component} has no field `{field}`"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::PendingEditorCommands;
    use crate::drain::{apply_pending, step_history, HistoryDirection};
    use crate::editor_state::EditorSelection;
    use crate::history::CommandHistory;
    use pill_engine::component_registry::ComponentFieldDescriptor;

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Probe {
        strength: f32,
        enabled: bool,
    }
    impl pill_engine::Component for Probe {}

    static PROBE_FIELDS: &[ComponentFieldDescriptor] = &[
        ComponentFieldDescriptor {
            name: "strength",
            type_tag: "f32",
            offset: 0,
            size: 4,
            align: 4,
            element_count: 0,
            decorators: std::borrow::Cow::Borrowed(&[]),
        },
        ComponentFieldDescriptor {
            name: "enabled",
            type_tag: "bool",
            offset: 4,
            size: 1,
            align: 1,
            element_count: 0,
            decorators: std::borrow::Cow::Borrowed(&[]),
        },
    ];

    fn probe_name() -> String {
        pill_engine::component::ComponentRegistry::registered_name::<Probe>()
    }

    /// An editor world that syncs: command resources, the outbox, one component.
    fn peer() -> Engine {
        let mut engine = Engine::new();
        let world = engine.world_mut();
        world.register_component_with_layout::<Probe>(PROBE_FIELDS);
        world.register_resource::<PendingEditorCommands>();
        world.insert_resource(PendingEditorCommands::default());
        world.register_resource::<CommandHistory>();
        world.insert_resource(CommandHistory::default());
        world.register_resource::<EditorSelection>();
        world.insert_resource(EditorSelection::default());
        world.register_resource::<EditOutbox>();
        world.insert_resource(EditOutbox::default());
        engine
    }

    fn queue(engine: &mut Engine, commands: Vec<EditorCommand>) {
        engine
            .world_mut()
            .get_resource_mut::<PendingEditorCommands>()
            .expect("pending")
            .push_recorded(commands, "edit", None);
        apply_pending(engine, 0);
    }

    fn take_ops(engine: &mut Engine) -> Vec<EditOp> {
        std::mem::take(
            &mut engine
                .world_mut()
                .get_resource_mut::<EditOutbox>()
                .expect("outbox")
                .ops,
        )
    }

    /// Converts `ops` on `engine` and applies them as one remote batch.
    fn receive(engine: &mut Engine, ops: Vec<EditOp>) {
        let mut commands = Vec::new();
        for op in ops {
            if let Some(command) = from_edit_op(engine, op).expect("op converts") {
                commands.push(command);
            }
        }
        engine
            .world_mut()
            .get_resource_mut::<PendingEditorCommands>()
            .expect("pending")
            .push_remote(commands);
        apply_pending(engine, 0);
    }

    fn create_probe(strength: f32) -> EditorCommand {
        EditorCommand::CreateEntity {
            components: vec![ComponentSeed {
                type_name: probe_name(),
                fields: vec![
                    ("strength".to_string(), FieldValue::F32(strength)),
                    ("enabled".to_string(), FieldValue::Bool(true)),
                ],
            }],
            id: None,
        }
    }

    fn strength(engine: &Engine, id: EntityId) -> Option<FieldValue> {
        let entity = engine.world().entity_by_id(id)?;
        engine
            .world()
            .read_component_fields(entity, &probe_name())
            .ok()
            .map(|values| values[0].clone())
    }

    #[test]
    fn op_json_round_trip() {
        let id = EntityId::new(7);
        let ops = vec![
            EditOp::CreateEntity {
                id,
                components: vec![EditComponent {
                    type_name: "Probe".into(),
                    fields: BTreeMap::from([("strength".into(), serde_json::json!(1.5))]),
                }],
            },
            EditOp::DestroyEntity { id },
            EditOp::SetFieldEntity {
                id,
                component: "Probe".into(),
                field: "target".into(),
                reference: Some(EntityId::new(9)),
            },
        ];
        let text = serde_json::to_string(&ops).expect("serializes");
        let back: Vec<EditOp> = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, ops);
        assert!(text.contains(r#""op":"create_entity""#), "{text}");
    }

    #[test]
    fn create_on_a_replays_on_b_under_the_same_id() {
        let mut a = peer();
        let mut b = peer();
        queue(&mut a, vec![create_probe(2.5)]);
        let ops = take_ops(&mut a);
        let [EditOp::CreateEntity { id, .. }] = ops.as_slice() else {
            panic!("one create op, got {ops:?}");
        };
        let id = *id;
        assert!(a.world().entity_by_id(id).is_some(), "a holds the id it sent");
        receive(&mut b, ops);
        assert_eq!(strength(&b, id), Some(FieldValue::F32(2.5)));
    }

    #[test]
    fn remote_batch_is_not_recorded_or_sent() {
        let mut a = peer();
        let mut b = peer();
        queue(&mut a, vec![create_probe(1.0)]);
        receive(&mut b, take_ops(&mut a));
        let history = b.world().get_resource::<CommandHistory>().expect("history");
        assert_eq!(history.undo_depth(), 0);
        assert!(take_ops(&mut b).is_empty());
    }

    #[test]
    fn own_create_echo_converts_to_nothing() {
        let mut a = peer();
        queue(&mut a, vec![create_probe(1.0)]);
        let op = take_ops(&mut a).remove(0);
        assert_eq!(from_edit_op(&mut a, op).expect("converts"), None);
    }

    #[test]
    fn set_field_replays_on_b() {
        let mut a = peer();
        let mut b = peer();
        queue(&mut a, vec![create_probe(1.0)]);
        receive(&mut b, take_ops(&mut a));
        let entity = a.world().entity_rows()[0].entity;
        let id = a.world().entity_id(entity).expect("id");
        queue(
            &mut a,
            vec![EditorCommand::SetField {
                entity,
                component: probe_name(),
                field: "strength".into(),
                value: FieldValue::F32(4.0),
            }],
        );
        receive(&mut b, take_ops(&mut a));
        assert_eq!(strength(&b, id), Some(FieldValue::F32(4.0)));
    }

    #[test]
    fn undo_of_destroy_recreates_the_same_id_on_b() {
        let mut a = peer();
        let mut b = peer();
        queue(&mut a, vec![create_probe(3.0)]);
        receive(&mut b, take_ops(&mut a));
        let entity = a.world().entity_rows()[0].entity;
        let id = a.world().entity_id(entity).expect("id");
        queue(&mut a, vec![EditorCommand::DestroyEntity { entity }]);
        receive(&mut b, take_ops(&mut a));
        assert!(b.world().entity_by_id(id).is_none(), "b destroyed it");
        step_history(&mut a, HistoryDirection::Undo);
        apply_pending(&mut a, 0);
        assert!(a.world().entity_by_id(id).is_some(), "a restored the same id");
        receive(&mut b, take_ops(&mut a));
        assert_eq!(strength(&b, id), Some(FieldValue::F32(3.0)));
    }
}
