//! Minimal ONNX reader for `gpu_net`: the graph's nodes (op, inputs, outputs, int / int list /
//! string attributes), its initializers (f32 / i64 tensors) and the graph inputs and outputs,
//! decoded straight from the protobuf wire format (onnx.proto field numbers).

// Used by the web build and the tests
#![cfg_attr(not(any(target_arch = "wasm32", test)), allow(dead_code))]

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum Attribute {
    Int(i64),
    Ints(Vec<i64>),
    String(String),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Attribute>,
}

impl Node {
    pub fn int(&self, name: &str, default: i64) -> i64 {
        match self.attributes.get(name) {
            Some(Attribute::Int(value)) => *value,
            _ => default,
        }
    }

    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.attributes.get(name) {
            Some(Attribute::Ints(values)) => Some(values),
            _ => None,
        }
    }

    pub fn string(&self, name: &str) -> Option<&str> {
        match self.attributes.get(name) {
            Some(Attribute::String(value)) => Some(value),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

#[derive(Clone, Debug)]
pub struct Initializer {
    pub dims: Vec<usize>,
    pub data: TensorData,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, Initializer>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Wire-format field: number and value (varint, fixed or length-delimited bytes).
enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
}

struct Fields<'a> {
    data: &'a [u8],
}

fn varint(data: &[u8], index: &mut usize) -> Result<u64, String> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *data.get(*index).ok_or("onnx: truncated varint")?;
        *index += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
    }
    Err("onnx: varint too long".to_string())
}

impl<'a> Iterator for Fields<'a> {
    type Item = Result<(u64, Value<'a>), String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }
        let mut index = 0;
        let result = (|| {
            let key = varint(self.data, &mut index)?;
            let value = match key & 7 {
                0 => Value::Varint(varint(self.data, &mut index)?),
                1 => {
                    index += 8;
                    Value::Varint(0)
                }
                2 => {
                    let length = varint(self.data, &mut index)? as usize;
                    let bytes = self
                        .data
                        .get(index..index + length)
                        .ok_or("onnx: truncated field")?;
                    index += length;
                    Value::Bytes(bytes)
                }
                5 => {
                    let bytes = self.data.get(index..index + 4).ok_or("onnx: truncated field")?;
                    index += 4;
                    Value::Fixed32(u32::from_le_bytes(bytes.try_into().expect("4 bytes")))
                }
                wire => return Err(format!("onnx: unsupported wire type {wire}")),
            };
            Ok((key >> 3, value))
        })();
        self.data = self.data.get(index..).unwrap_or_default();
        if result.is_err() {
            self.data = &[];
        }
        Some(result)
    }
}

fn fields(data: &[u8]) -> Fields<'_> {
    Fields { data }
}

fn string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Packed (bytes) or unpacked (one varint) repeated int64 field.
fn push_ints(value: &Value, out: &mut Vec<i64>) -> Result<(), String> {
    match value {
        Value::Varint(v) => out.push(*v as i64),
        Value::Bytes(bytes) => {
            let mut index = 0;
            while index < bytes.len() {
                out.push(varint(bytes, &mut index)? as i64);
            }
        }
        Value::Fixed32(_) => return Err("onnx: fixed32 in an int64 field".to_string()),
    }
    Ok(())
}

fn attribute(data: &[u8]) -> Result<(String, Attribute), String> {
    let (mut name, mut int, mut text, mut ints) = (String::new(), None, None, Vec::new());
    for field in fields(data) {
        match field? {
            (1, Value::Bytes(bytes)) => name = string(bytes),
            (3, Value::Varint(v)) => int = Some(v as i64),
            (4, Value::Bytes(bytes)) => text = Some(string(bytes)),
            (8, value) => push_ints(&value, &mut ints)?,
            _ => {}
        }
    }
    let value = match (int, text) {
        (Some(i), _) => Attribute::Int(i),
        (_, Some(s)) => Attribute::String(s),
        _ => Attribute::Ints(ints),
    };
    Ok((name, value))
}

fn node(data: &[u8]) -> Result<Node, String> {
    let mut node = Node {
        op: String::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: HashMap::new(),
    };
    for field in fields(data) {
        match field? {
            (1, Value::Bytes(bytes)) => node.inputs.push(string(bytes)),
            (2, Value::Bytes(bytes)) => node.outputs.push(string(bytes)),
            (4, Value::Bytes(bytes)) => node.op = string(bytes),
            (5, Value::Bytes(bytes)) => {
                let (name, value) = attribute(bytes)?;
                node.attributes.insert(name, value);
            }
            _ => {}
        }
    }
    Ok(node)
}

fn initializer(data: &[u8]) -> Result<(String, Initializer), String> {
    let (mut name, mut dims, mut data_type, mut raw, mut floats, mut int64s) =
        (String::new(), Vec::new(), 0, &[][..], Vec::new(), Vec::new());
    for field in fields(data) {
        match field? {
            (1, value) => push_ints(&value, &mut dims)?,
            (2, Value::Varint(v)) => data_type = v,
            (4, Value::Bytes(bytes)) => floats.extend(
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes"))),
            ),
            (4, Value::Fixed32(bits)) => floats.push(f32::from_bits(bits)),
            (7, value) => push_ints(&value, &mut int64s)?,
            (8, Value::Bytes(bytes)) => name = string(bytes),
            (9, Value::Bytes(bytes)) => raw = bytes,
            _ => {}
        }
    }
    let data = match data_type {
        1 if !raw.is_empty() => TensorData::F32(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")))
                .collect(),
        ),
        1 => TensorData::F32(floats),
        7 if !raw.is_empty() => TensorData::I64(
            raw.chunks_exact(8)
                .map(|b| i64::from_le_bytes(b.try_into().expect("8 bytes")))
                .collect(),
        ),
        7 => TensorData::I64(int64s),
        other => return Err(format!("onnx: initializer {name}: unsupported data type {other}")),
    };
    Ok((
        name,
        Initializer {
            dims: dims.into_iter().map(|d| d as usize).collect(),
            data,
        },
    ))
}

fn value_info_name(data: &[u8]) -> Result<String, String> {
    for field in fields(data) {
        if let (1, Value::Bytes(bytes)) = field? {
            return Ok(string(bytes));
        }
    }
    Err("onnx: value info without a name".to_string())
}

pub fn parse(model: &[u8]) -> Result<Graph, String> {
    let mut graph_bytes = None;
    for field in fields(model) {
        if let (7, Value::Bytes(bytes)) = field? {
            graph_bytes = Some(bytes);
        }
    }
    let graph_bytes = graph_bytes.ok_or("onnx: no graph")?;
    let mut graph = Graph {
        nodes: Vec::new(),
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
    };
    for field in fields(graph_bytes) {
        match field? {
            (1, Value::Bytes(bytes)) => graph.nodes.push(node(bytes)?),
            (5, Value::Bytes(bytes)) => {
                let (name, init) = initializer(bytes)?;
                graph.initializers.insert(name, init);
            }
            (11, Value::Bytes(bytes)) => graph.inputs.push(value_info_name(bytes)?),
            (12, Value::Bytes(bytes)) => graph.outputs.push(value_info_name(bytes)?),
            _ => {}
        }
    }
    // Initializers may also be listed as graph inputs
    graph
        .inputs
        .retain(|name| !graph.initializers.contains_key(name));
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PALM_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/palm_detection_mediapipe_2023feb.onnx"
    );
    const HAND_ONNX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/res/models/handpose_estimation_mediapipe_2023feb.onnx"
    );

    fn op_count(graph: &Graph, op: &str) -> usize {
        graph.nodes.iter().filter(|node| node.op == op).count()
    }

    #[test]
    fn parse_palm_model_ops_io_and_initializers() {
        let graph = parse(&std::fs::read(PALM_ONNX).expect("palm model")).expect("parse");
        assert_eq!(graph.inputs, ["input_1"]);
        assert_eq!(graph.outputs, ["Identity", "Identity_1"]);
        assert_eq!(op_count(&graph, "Conv"), 53);
        assert_eq!(op_count(&graph, "PRelu"), 26);
        assert_eq!(graph.initializers.len(), 121);
        let conv = graph.nodes.iter().find(|node| node.op == "Conv").expect("conv");
        assert_eq!(conv.ints("kernel_shape"), Some(&[5, 5][..]));
        assert_eq!(conv.ints("pads"), Some(&[1, 1, 2, 2][..]));
    }

    #[test]
    fn parse_hand_model_clip_bounds_are_relu6() {
        let graph = parse(&std::fs::read(HAND_ONNX).expect("hand model")).expect("parse");
        assert_eq!(graph.outputs.len(), 4);
        let clip = graph.nodes.iter().find(|node| node.op == "Clip").expect("clip");
        let bound = |name: &str| match &graph.initializers[name].data {
            TensorData::F32(values) => values[0],
            TensorData::I64(_) => panic!("float bound"),
        };
        assert_eq!(bound(&clip.inputs[1]), 0.0);
        assert_eq!(bound(&clip.inputs[2]), 6.0);
    }
}
