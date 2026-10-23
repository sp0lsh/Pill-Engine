//! Quake `.map` loader (feature `quake_map`; `pill_engine` re-exports it under `quake_map_loading`).
//!
//! Turns the brush source of a Quake level into the same uploadable [`GltfSceneData`] the glTF
//! loader produces, plus what the level's entities describe: point/spot lights with their light
//! styles, the sun, sky light, fog, a sky panorama for the background and IBL, and spawn points.
//!
//! - Parsing: Standard (id) and Valve 220 face formats; Quake 2 surface flags after a face are
//!   skipped. Brush primitives and patches (Quake 3) are rejected.
//! - Geometry: every brush plane is clipped by the brush's other planes into a convex face. Faces
//!   inside or on a solid world brush are dropped (what qbsp's CSG removes), tool faces (`clip`,
//!   `skip`, `trigger`, ...) and sky faces are not drawn. Triangles are grouped per texture and
//!   per 32 m cell, so frustum culling still works on large levels.
//! - Materials: the albedo is the texture; a normal map (from luminance) and a roughness/metallic
//!   map are derived from it, and Quake's fullbright palette colors become the emissive map.
//!   Liquids (`*`), fences (`{`) and animated textures (`+0`..`+9`) get their Quake behavior.
//!
//! Wasm-safe like `gltf_scene`: the caller fetches the map text and the texture PNGs.

use std::collections::HashMap;

use anyhow::{bail, Context, Error, Result};

use crate::convert::{self, PrimitiveAttributes};
use crate::gltf_scene::{
    decode_image, GltfAlphaMode, GltfMaterialData, GltfMeshData, GltfSceneData, GltfTextureData,
};
use crate::ibl::Equirect;
use crate::util::color::srgb_to_linear;
use crate::TextureType;

/// Quake units per meter (a 56-unit player is 1.75 m tall).
pub const UNITS_PER_METER: f64 = 32.0;
/// Edge of the cells triangles are grouped into, in Quake units (32 m).
const CELL_SIZE: f64 = 1024.0;
/// Half-size of the winding a face starts from before clipping, in Quake units.
const WORLD_EXTENT: f64 = 131_072.0;
/// Distance (Quake units) under which a point counts as on a plane.
const PLANE_EPSILON: f64 = 0.01;
/// Emissive factor of fullbright texels (HDR, picked up by bloom).
const FULLBRIGHT_EMISSIVE: f32 = 3.0;
/// Width and height of the sky panorama.
const SKY_WIDTH: u32 = 1024;
const SKY_HEIGHT: u32 = 512;

/// Quake's built-in light styles (`world.qc`), 10 frames per second: `a` is dark, `m` normal and
/// `z` double brightness. Styles 32 and up are switchable and stay at `m`.
pub const LIGHT_STYLES: [&str; 12] = [
    "m",
    "mmnmmommommnonmmonqnmmo",
    "abcdefghijklmnopqrstuvwxyzyxwvutsrqponmlkjihgfedcba",
    "mmmmmaaaaammmmmaaaaaabcdefgabcdefg",
    "mamamamamama",
    "jklmnopqrstuvwxyzyxwvutsrqponmlkj",
    "nmonqnmomnmomomno",
    "mmmaaaabcdefgmmmmaaaammmaamm",
    "mmmaaammmaaammmabcdefaaaammmmabcdefmmmaaaa",
    "aaaaaaaazzzzzzzz",
    "mmamammmmammamamaaamammma",
    "abcdefghijklmnopqrrqponmlkjihgfedcba",
];

/// Brightness of light style `style` at `time` seconds (1 = the light's own intensity).
pub fn light_style_brightness(style: u32, time: f32) -> f32 {
    let pattern = LIGHT_STYLES.get(style as usize).copied().unwrap_or("m");
    let frames = pattern.as_bytes();
    let frame = frames[((time.max(0.0) * 10.0) as usize) % frames.len()];
    (frame.saturating_sub(b'a')) as f32 / (b'm' - b'a') as f32
}

/// Frames Quake cycles an animated texture (`+0name`) through: `+0name` .. `+9name`. The caller
/// fetches the ones that exist; [`build_quake_level`] animates the contiguous run from `+0`.
pub fn animation_frame_names(texture: &str) -> Vec<String> {
    match animation_base(texture) {
        Some(base) => (0..10).map(|frame| format!("+{frame}{base}")).collect(),
        None => Vec::new(),
    }
}

fn animation_base(texture: &str) -> Option<&str> {
    let bytes = texture.as_bytes();
    (bytes.len() > 2 && bytes[0] == b'+' && bytes[1].is_ascii_digit()).then(|| &texture[2..])
}

// --- Parsed map ---

/// A whole `.map` file: entities in file order, the first being `worldspawn`.
pub struct QuakeMap {
    pub entities: Vec<MapEntity>,
}

/// One entity: its key/value pairs in file order and its brushes (brush entities only).
pub struct MapEntity {
    pub properties: Vec<(String, String)>,
    pub brushes: Vec<MapBrush>,
}

/// A convex solid: the intersection of the half-spaces behind its faces' planes.
pub struct MapBrush {
    pub faces: Vec<MapFace>,
}

/// One brush plane, through three points (clockwise seen from outside the brush), and its texture
/// mapping.
pub struct MapFace {
    pub points: [[f64; 3]; 3],
    pub texture: String,
    pub projection: TextureProjection,
    /// Degrees; already part of the axes in the Valve format.
    pub rotation: f64,
    pub scale: [f64; 2],
}

pub enum TextureProjection {
    /// id format: texture axes from the plane's nearest axial plane, then rotated.
    Standard { offset: [f64; 2] },
    /// Valve 220: explicit texture axes `[x y z offset]`.
    Valve { u: [f64; 4], v: [f64; 4] },
}

impl MapEntity {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.properties
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    pub fn classname(&self) -> &str {
        self.get("classname").unwrap_or("")
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.get(key)?.trim().parse().ok()
    }

    fn numbers(&self, key: &str) -> Option<Vec<f64>> {
        let value = self.get(key)?;
        let numbers: Vec<f64> = value
            .split_whitespace()
            .map_while(|word| word.parse().ok())
            .collect();
        (!numbers.is_empty()).then_some(numbers)
    }

    fn vector(&self, key: &str) -> Option<[f64; 3]> {
        match self.numbers(key)?.as_slice() {
            [x, y, z, ..] => Some([*x, *y, *z]),
            _ => None,
        }
    }
}

impl QuakeMap {
    pub fn worldspawn(&self) -> Option<&MapEntity> {
        self.entities
            .iter()
            .find(|entity| entity.classname() == "worldspawn")
    }

    /// WAD names the map's `wad` key lists, in order, without directory and extension; texture
    /// lookups should prefer them in this order.
    pub fn wads(&self) -> Vec<String> {
        let Some(list) = self.worldspawn().and_then(|world| world.get("wad")) else {
            return Vec::new();
        };
        list.split(';')
            .filter_map(|path| {
                let file = path.trim().rsplit(['/', '\\']).next()?;
                let stem = file.strip_suffix(".wad").unwrap_or(file);
                (!stem.is_empty()).then(|| stem.to_ascii_lowercase())
            })
            .collect()
    }

    /// Lowercase names of the textures on drawn faces (sky included), sorted; animated ones also
    /// need their [`animation_frame_names`].
    pub fn texture_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .drawn_entities()
            .flat_map(|entity| &entity.brushes)
            .flat_map(|brush| &brush.faces)
            .filter(|face| texture_class(&face.texture) != TextureClass::Tool)
            .map(|face| face.texture.to_ascii_lowercase())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Brush entities whose brushes are drawn: not triggers, not in a TrenchBroom layer omitted
    /// from export.
    fn drawn_entities(&self) -> impl Iterator<Item = &MapEntity> {
        let omitted_layers: Vec<&str> = self
            .entities
            .iter()
            .filter(|entity| entity.get("_tb_layer_omit_from_export") == Some("1"))
            .filter_map(|entity| entity.get("_tb_id"))
            .collect();
        self.entities.iter().filter(move |entity| {
            let classname = entity.classname();
            !classname.starts_with("trigger_")
                && entity.get("_tb_layer_omit_from_export") != Some("1")
                && !entity
                    .get("_tb_layer")
                    .is_some_and(|layer| omitted_layers.contains(&layer))
        })
    }
}

// --- Tokenizer + parser ---

#[derive(Clone, Copy, Debug, PartialEq)]
enum Token<'a> {
    Quoted(&'a str),
    Bare(&'a str),
}

struct Lexer<'a> {
    text: &'a str,
    position: usize,
    line: usize,
    peeked: Option<Option<Token<'a>>>,
}

impl<'a> Lexer<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            position: 0,
            line: 1,
            peeked: None,
        }
    }

    fn peek(&mut self) -> Option<Token<'a>> {
        if self.peeked.is_none() {
            self.peeked = Some(self.read());
        }
        self.peeked.flatten()
    }

    fn next(&mut self) -> Option<Token<'a>> {
        match self.peeked.take() {
            Some(token) => token,
            None => self.read(),
        }
    }

    fn read(&mut self) -> Option<Token<'a>> {
        let bytes = self.text.as_bytes();
        loop {
            while self.position < bytes.len() && bytes[self.position].is_ascii_whitespace() {
                if bytes[self.position] == b'\n' {
                    self.line += 1;
                }
                self.position += 1;
            }
            if bytes[self.position..].starts_with(b"//") {
                while self.position < bytes.len() && bytes[self.position] != b'\n' {
                    self.position += 1;
                }
                continue;
            }
            break;
        }
        if self.position >= bytes.len() {
            return None;
        }
        if bytes[self.position] == b'"' {
            let start = self.position + 1;
            let end = self.text[start..]
                .find('"')
                .map_or(bytes.len(), |offset| start + offset);
            self.line += self.text[start..end].matches('\n').count();
            self.position = (end + 1).min(bytes.len());
            return Some(Token::Quoted(&self.text[start..end]));
        }
        let start = self.position;
        while self.position < bytes.len() && !bytes[self.position].is_ascii_whitespace() {
            self.position += 1;
        }
        Some(Token::Bare(&self.text[start..self.position]))
    }

    fn error(&self, message: &str) -> Error {
        Error::msg(format!("line {}: {message}", self.line))
    }

    fn expect(&mut self, symbol: &str) -> Result<()> {
        match self.next() {
            Some(Token::Bare(found)) if found == symbol => Ok(()),
            found => Err(self.error(&format!("expected `{symbol}`, found {found:?}"))),
        }
    }

    fn number(&mut self) -> Result<f64> {
        match self.next() {
            Some(Token::Bare(word)) => word
                .parse()
                .map_err(|_| self.error(&format!("expected a number, found `{word}`"))),
            found => Err(self.error(&format!("expected a number, found {found:?}"))),
        }
    }
}

/// Parses a `.map` file (Standard or Valve 220 faces).
pub fn parse_map(text: &str) -> Result<QuakeMap> {
    let mut lexer = Lexer::new(text);
    let mut entities = Vec::new();
    while lexer.peek().is_some() {
        lexer.expect("{")?;
        entities.push(parse_entity(&mut lexer)?);
    }
    Ok(QuakeMap { entities })
}

fn parse_entity(lexer: &mut Lexer) -> Result<MapEntity> {
    let mut entity = MapEntity {
        properties: Vec::new(),
        brushes: Vec::new(),
    };
    loop {
        match lexer.next() {
            Some(Token::Bare("}")) => return Ok(entity),
            Some(Token::Bare("{")) => entity.brushes.push(parse_brush(lexer)?),
            Some(Token::Quoted(key)) => match lexer.next() {
                Some(Token::Quoted(value)) => {
                    entity.properties.push((key.to_string(), value.to_string()))
                }
                found => return Err(lexer.error(&format!("expected a value, found {found:?}"))),
            },
            None => return Err(lexer.error("unterminated entity")),
            found => return Err(lexer.error(&format!("unexpected {found:?} in entity"))),
        }
    }
}

fn parse_brush(lexer: &mut Lexer) -> Result<MapBrush> {
    let mut faces = Vec::new();
    loop {
        match lexer.peek() {
            Some(Token::Bare("}")) => {
                lexer.next();
                return Ok(MapBrush { faces });
            }
            Some(Token::Bare("(")) => faces.push(parse_face(lexer)?),
            Some(Token::Bare(word @ ("brushDef" | "brushDef3" | "patchDef2" | "patchDef3"))) => {
                return Err(lexer.error(&format!("`{word}` (Quake 3 / Doom 3) is not supported")))
            }
            None => return Err(lexer.error("unterminated brush")),
            found => return Err(lexer.error(&format!("unexpected {found:?} in brush"))),
        }
    }
}

fn parse_face(lexer: &mut Lexer) -> Result<MapFace> {
    let mut points = [[0.0; 3]; 3];
    for point in &mut points {
        lexer.expect("(")?;
        for coordinate in point.iter_mut() {
            *coordinate = lexer.number()?;
        }
        lexer.expect(")")?;
    }
    let texture = match lexer.next() {
        Some(Token::Bare(name) | Token::Quoted(name)) => name.to_string(),
        None => return Err(lexer.error("expected a texture name")),
    };
    let projection = if lexer.peek() == Some(Token::Bare("[")) {
        let mut axes = [[0.0; 4]; 2];
        for axis in &mut axes {
            lexer.expect("[")?;
            for value in axis.iter_mut() {
                *value = lexer.number()?;
            }
            lexer.expect("]")?;
        }
        TextureProjection::Valve {
            u: axes[0],
            v: axes[1],
        }
    } else {
        TextureProjection::Standard {
            offset: [lexer.number()?, lexer.number()?],
        }
    };
    let rotation = lexer.number()?;
    let scale = [lexer.number()?, lexer.number()?];
    // Quake 2 / Hexen 2 extras: content flags, surface flags, value
    while let Some(Token::Bare(word)) = lexer.peek() {
        if word.parse::<f64>().is_err() {
            break;
        }
        lexer.next();
    }
    Ok(MapFace {
        points,
        texture,
        projection,
        rotation,
        scale,
    })
}

// --- Texture classes ---

#[derive(Clone, Copy, Debug, PartialEq)]
enum TextureClass {
    /// Compiler-only (clip, skip, trigger, hint, ...): never drawn.
    Tool,
    Sky,
    Liquid(Liquid),
    /// `{name`: alpha-tested (palette index 255 is transparent).
    Fence,
    Solid,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Liquid {
    Water,
    Slime,
    Lava,
    Teleport,
}

fn texture_class(texture: &str) -> TextureClass {
    let name = texture.to_ascii_lowercase();
    let tool = matches!(
        name.as_str(),
        "clip" | "skip" | "hint" | "hintskip" | "origin" | "nodraw" | "null" | "caulk" | "noclip"
    ) || name.ends_with("skip")
        || name.contains("trigger")
        || name.starts_with("common/")
        || name.starts_with("tools/");
    if tool {
        TextureClass::Tool
    } else if name.starts_with("sky") {
        TextureClass::Sky
    } else if let Some(liquid) = name.strip_prefix('*') {
        TextureClass::Liquid(if liquid.contains("lava") {
            Liquid::Lava
        } else if liquid.contains("slime") {
            Liquid::Slime
        } else if liquid.contains("tele") {
            Liquid::Teleport
        } else {
            Liquid::Water
        })
    } else if name.starts_with('{') {
        TextureClass::Fence
    } else {
        TextureClass::Solid
    }
}

// --- Vector math (f64, Quake space) ---

type Vec3 = [f64; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: Vec3, factor: f64) -> Vec3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: Vec3) -> Vec3 {
    let len = length(a);
    if len > 0.0 {
        scale(a, 1.0 / len)
    } else {
        a
    }
}

/// Quake (Z up, units) → engine (Y up, meters, right-handed): a rotation, so windings keep their
/// orientation.
fn to_engine(point: Vec3) -> [f32; 3] {
    let factor = 1.0 / UNITS_PER_METER;
    [
        (point[0] * factor) as f32,
        (point[2] * factor) as f32,
        (-point[1] * factor) as f32,
    ]
}

fn direction_to_engine(direction: Vec3) -> [f32; 3] {
    [
        direction[0] as f32,
        direction[2] as f32,
        -direction[1] as f32,
    ]
}

/// Quake angles (degrees) → forward vector: yaw about +Z from +X, pitch up from the horizon.
fn angles_to_direction(yaw: f64, pitch: f64) -> Vec3 {
    let (yaw, pitch) = (yaw.to_radians(), pitch.to_radians());
    [
        pitch.cos() * yaw.cos(),
        pitch.cos() * yaw.sin(),
        pitch.sin(),
    ]
}

// --- Brush → faces ---

#[derive(Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance: f64,
}

impl Plane {
    /// Quake's `PlaneFromPoints`: the normal points out of the brush.
    fn from_points(points: &[Vec3; 3]) -> Option<Plane> {
        let normal = cross(sub(points[0], points[1]), sub(points[2], points[1]));
        let len = length(normal);
        if len < 1e-9 {
            return None;
        }
        let normal = scale(normal, 1.0 / len);
        Some(Plane {
            normal,
            distance: dot(points[0], normal),
        })
    }

    fn side(&self, point: Vec3) -> f64 {
        dot(self.normal, point) - self.distance
    }

    fn same_as(&self, other: &Plane) -> bool {
        dot(self.normal, other.normal) > 0.999_999
            && (self.distance - other.distance).abs() < PLANE_EPSILON
    }
}

/// A huge square on `plane` (qbsp's `BaseWindingForPlane`).
fn base_winding(plane: &Plane) -> Vec<Vec3> {
    let normal = plane.normal;
    let axis = (0..3)
        .max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
        .unwrap_or(2);
    let mut up = if axis == 2 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    up = normalize(sub(up, scale(normal, dot(up, normal))));
    let right = scale(cross(up, normal), WORLD_EXTENT);
    let up = scale(up, WORLD_EXTENT);
    let origin = scale(normal, plane.distance);
    vec![
        add(sub(origin, right), up),
        add(add(origin, right), up),
        sub(add(origin, right), up),
        sub(sub(origin, right), up),
    ]
}

/// Keeps the part of `winding` behind `plane` (Sutherland–Hodgman).
fn clip_winding(winding: &[Vec3], plane: &Plane) -> Vec<Vec3> {
    let sides: Vec<f64> = winding.iter().map(|&point| plane.side(point)).collect();
    if sides.iter().all(|&side| side <= PLANE_EPSILON) {
        return winding.to_vec();
    }
    let mut output = Vec::with_capacity(winding.len() + 1);
    for index in 0..winding.len() {
        let next = (index + 1) % winding.len();
        let (point, side) = (winding[index], sides[index]);
        let next_side = sides[next];
        if side <= PLANE_EPSILON {
            output.push(point);
        }
        let crosses = (side > PLANE_EPSILON && next_side < -PLANE_EPSILON)
            || (side < -PLANE_EPSILON && next_side > PLANE_EPSILON);
        if crosses {
            let t = side / (side - next_side);
            output.push(add(point, scale(sub(winding[next], point), t)));
        }
    }
    output
}

/// Snaps coordinates within 1/100 unit of an integer onto it (removes clipping noise).
fn snap(point: Vec3) -> Vec3 {
    point.map(|value| {
        let rounded = value.round();
        if (value - rounded).abs() < 0.01 {
            rounded
        } else {
            value
        }
    })
}

struct BrushFace {
    plane: Plane,
    /// Convex polygon, counter-clockwise seen from outside the brush.
    polygon: Vec<Vec3>,
    face_index: usize,
}

struct BuiltBrush<'a> {
    brush: &'a MapBrush,
    planes: Vec<Plane>,
    faces: Vec<BrushFace>,
    min: Vec3,
    max: Vec3,
    /// Hides faces inside it (opaque world geometry).
    occluder: bool,
}

fn build_brush<'a>(entity: &'a MapEntity, brush: &'a MapBrush) -> Option<BuiltBrush<'a>> {
    let planes: Vec<Plane> = brush
        .faces
        .iter()
        .map(|face| Plane::from_points(&face.points))
        .collect::<Option<_>>()?;
    let mut faces = Vec::with_capacity(planes.len());
    for (face_index, plane) in planes.iter().enumerate() {
        // A repeated plane only contributes its first face
        if planes[..face_index]
            .iter()
            .any(|other| other.same_as(plane))
        {
            continue;
        }
        let mut winding = base_winding(plane);
        for (other_index, other) in planes.iter().enumerate() {
            if other_index == face_index || other.same_as(plane) {
                continue;
            }
            winding = clip_winding(&winding, other);
            if winding.len() < 3 {
                break;
            }
        }
        let mut polygon: Vec<Vec3> = Vec::with_capacity(winding.len());
        for point in winding.into_iter().map(snap) {
            if polygon
                .last()
                .is_none_or(|&last| length(sub(point, last)) > PLANE_EPSILON)
            {
                polygon.push(point);
            }
        }
        while polygon.len() > 1
            && length(sub(polygon[0], polygon[polygon.len() - 1])) <= PLANE_EPSILON
        {
            polygon.pop();
        }
        if polygon.len() < 3 {
            continue;
        }
        // Counter-clockwise about the outward normal (the engine's front faces)
        let mut area = [0.0; 3];
        for index in 1..polygon.len() - 1 {
            area = add(
                area,
                cross(
                    sub(polygon[index], polygon[0]),
                    sub(polygon[index + 1], polygon[0]),
                ),
            );
        }
        let signed_area = dot(area, plane.normal);
        if signed_area.abs() < 1e-6 {
            continue;
        }
        if signed_area < 0.0 {
            polygon.reverse();
        }
        faces.push(BrushFace {
            plane: *plane,
            polygon,
            face_index,
        });
    }
    if faces.len() < 4 {
        return None;
    }
    let mut min = [f64::MAX; 3];
    let mut max = [f64::MIN; 3];
    for point in faces.iter().flat_map(|face| &face.polygon) {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    let occluder = matches!(
        entity.classname(),
        "worldspawn" | "func_group" | "func_detail" | "func_detail_wall"
    ) && brush.faces.iter().all(|face| {
        matches!(
            texture_class(&face.texture),
            TextureClass::Solid | TextureClass::Sky
        )
    });
    Some(BuiltBrush {
        brush,
        planes,
        faces,
        min,
        max,
        occluder,
    })
}

fn inside_brush(points: &[Vec3], brush: &BuiltBrush) -> bool {
    points.iter().all(|&point| {
        brush
            .planes
            .iter()
            .all(|plane| plane.side(point) <= PLANE_EPSILON)
    })
}

/// Uniform grid over the occluder brushes' bounds, for the hidden-face test.
struct OccluderGrid {
    cells: HashMap<[i32; 3], Vec<usize>>,
}

const GRID_CELL: f64 = 512.0;

fn grid_range(min: Vec3, max: Vec3) -> ([i32; 3], [i32; 3]) {
    let cell = |value: f64| (value / GRID_CELL).floor() as i32;
    (
        [cell(min[0]), cell(min[1]), cell(min[2])],
        [cell(max[0]), cell(max[1]), cell(max[2])],
    )
}

impl OccluderGrid {
    fn new(brushes: &[BuiltBrush]) -> Self {
        let mut cells: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
        for (index, brush) in brushes
            .iter()
            .enumerate()
            .filter(|(_, brush)| brush.occluder)
        {
            let (low, high) = grid_range(brush.min, brush.max);
            for x in low[0]..=high[0] {
                for y in low[1]..=high[1] {
                    for z in low[2]..=high[2] {
                        cells.entry([x, y, z]).or_default().push(index);
                    }
                }
            }
        }
        Self { cells }
    }

    /// Occluders whose bounds contain the box `min`..`max`. A brush containing the box is listed
    /// in every cell the box overlaps, so the cell of `min` is the only one searched.
    fn candidates(&self, brushes: &[BuiltBrush], min: Vec3, max: Vec3, found: &mut Vec<usize>) {
        found.clear();
        let (low, _) = grid_range(min, min);
        if let Some(indices) = self.cells.get(&low) {
            for &index in indices {
                let brush = &brushes[index];
                let contains = (0..3).all(|axis| {
                    brush.min[axis] - PLANE_EPSILON <= min[axis]
                        && max[axis] <= brush.max[axis] + PLANE_EPSILON
                });
                if contains {
                    found.push(index);
                }
            }
        }
    }
}

/// Faces of `brushes[brush_index]` that are seen: not inside or on another solid brush.
fn visible_faces(brushes: &[BuiltBrush], grid: &OccluderGrid, brush_index: usize) -> Vec<usize> {
    let brush = &brushes[brush_index];
    let mut candidates = Vec::new();
    let mut visible = Vec::with_capacity(brush.faces.len());
    'faces: for (face_slot, face) in brush.faces.iter().enumerate() {
        let mut min = [f64::MAX; 3];
        let mut max = [f64::MIN; 3];
        for point in &face.polygon {
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        grid.candidates(brushes, min, max, &mut candidates);
        for &other_index in &candidates {
            if other_index == brush_index {
                continue;
            }
            let other = &brushes[other_index];
            if !inside_brush(&face.polygon, other) {
                continue;
            }
            // On the other brush's surface facing the same way: a coincident duplicate keeps the
            // earlier brush's face, otherwise the other brush's (larger) face is the one drawn
            let coplanar = other
                .faces
                .iter()
                .find(|other_face| other_face.plane.same_as(&face.plane));
            if let Some(other_face) = coplanar {
                let coincident = inside_brush(&other_face.polygon, brush);
                if coincident && brush_index < other_index {
                    continue;
                }
            }
            continue 'faces;
        }
        visible.push(face_slot);
    }
    visible
}

// --- Texture mapping ---

/// Texture axes (scaled, so `dot(point, axis) + offset` is in texels) of a face.
fn texture_axes(face: &MapFace, normal: Vec3) -> ([f64; 4], [f64; 4]) {
    let scale_or_one = |value: f64| if value.abs() < 1e-9 { 1.0 } else { value };
    let (sx, sy) = (scale_or_one(face.scale[0]), scale_or_one(face.scale[1]));
    match &face.projection {
        TextureProjection::Valve { u, v } => (
            [u[0] / sx, u[1] / sx, u[2] / sx, u[3]],
            [v[0] / sy, v[1] / sy, v[2] / sy, v[3]],
        ),
        TextureProjection::Standard { offset } => {
            // qbsp's TextureAxisFromPlane, then the rotation about the axial plane's normal
            const BASE_AXES: [[Vec3; 3]; 6] = [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            ];
            let mut best = 0;
            let mut best_dot = 0.0;
            for (index, axes) in BASE_AXES.iter().enumerate() {
                let d = dot(normal, axes[0]);
                if d > best_dot {
                    best_dot = d;
                    best = index;
                }
            }
            let mut vectors = [BASE_AXES[best][1], BASE_AXES[best][2]];
            let (sin, cos) = match face.rotation.rem_euclid(360.0) {
                angle if angle == 0.0 => (0.0, 1.0),
                angle if angle == 90.0 => (1.0, 0.0),
                angle if angle == 180.0 => (0.0, -1.0),
                angle if angle == 270.0 => (-1.0, 0.0),
                angle => angle.to_radians().sin_cos(),
            };
            let nonzero = |vector: Vec3| (0..3).find(|&axis| vector[axis] != 0.0).unwrap_or(2);
            let (s_axis, t_axis) = (nonzero(vectors[0]), nonzero(vectors[1]));
            for vector in &mut vectors {
                let s = cos * vector[s_axis] - sin * vector[t_axis];
                let t = sin * vector[s_axis] + cos * vector[t_axis];
                vector[s_axis] = s;
                vector[t_axis] = t;
            }
            let [s, t] = vectors;
            (
                [s[0] / sx, s[1] / sx, s[2] / sx, offset[0]],
                [t[0] / sy, t[1] / sy, t[2] / sy, offset[1]],
            )
        }
    }
}

// --- Textures → materials ---

/// The map's textures, decoded, and the material each one becomes.
struct TextureSet {
    textures: Vec<GltfTextureData>,
    materials: Vec<GltfMaterialData>,
    /// Lowercase texture name → (material index, width, height).
    by_name: HashMap<String, (usize, u32, u32)>,
    animations: Vec<QuakeTextureAnimation>,
    missing: Vec<String>,
}

struct DecodedTexture {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    /// Texels that are Quake fullbright palette colors.
    fullbright: Vec<bool>,
}

/// Palette index 255: transparent in fence (`{`) textures.
const TRANSPARENT_COLOR: [u8; 3] = [159, 91, 83];

fn decode_texture(
    name: &str,
    png: Option<&Vec<u8>>,
    palette: Option<&[u8]>,
) -> Option<DecodedTexture> {
    let image = match png.map(|bytes| decode_image(bytes)) {
        Some(Ok(image)) => image,
        Some(Err(error)) => {
            log::warn!("Quake map: texture `{name}`: {error}");
            return None;
        }
        None => return None,
    };
    let mut rgba = image.rgba;
    let fence = texture_class(name) == TextureClass::Fence;
    for texel in rgba.as_chunks_mut::<4>().0 {
        texel[3] = if fence && (texel[3] < 128 || texel[..3] == TRANSPARENT_COLOR) {
            0
        } else {
            255
        };
    }
    let fullbright = fullbright_mask(&rgba, palette);
    Some(DecodedTexture {
        rgba,
        width: image.w,
        height: image.h,
        fullbright,
    })
}

/// Marks texels drawn in Quake's fullbright colors (palette 224..254, minus any that also appear
/// in the lit range), only for textures quantized to the palette (truecolor art would match by
/// chance).
fn fullbright_mask(rgba: &[u8], palette: Option<&[u8]>) -> Vec<bool> {
    let texel_count = rgba.len() / 4;
    let Some(palette) = palette.filter(|palette| palette.len() >= 768) else {
        return vec![false; texel_count];
    };
    let color = |index: usize| {
        [
            palette[index * 3],
            palette[index * 3 + 1],
            palette[index * 3 + 2],
        ]
    };
    let lit: Vec<[u8; 3]> = (0..224).map(color).collect();
    let bright: Vec<[u8; 3]> = (224..255)
        .map(color)
        .filter(|candidate| !lit.contains(candidate))
        .collect();
    let mut in_palette = 0;
    let mut mask = Vec::with_capacity(texel_count);
    for texel in rgba.as_chunks::<4>().0 {
        let rgb = [texel[0], texel[1], texel[2]];
        let is_bright = bright.contains(&rgb);
        if is_bright || lit.contains(&rgb) || rgb == TRANSPARENT_COLOR {
            in_palette += 1;
        }
        mask.push(is_bright && texel[3] > 0);
    }
    if in_palette * 10 < texel_count * 9 {
        mask.iter_mut().for_each(|bright| *bright = false);
    }
    mask
}

fn luminance(texel: &[u8]) -> f32 {
    let linear = |channel: u8| srgb_to_linear(channel as f32 / 255.0);
    0.2126 * linear(texel[0]) + 0.7152 * linear(texel[1]) + 0.0722 * linear(texel[2])
}

/// Tangent-space normal map (glTF convention, +Y up the image) from the texture's luminance as
/// height; wraps at the edges like the tiling texture.
fn normal_map(texture: &DecodedTexture, strength: f32) -> Vec<u8> {
    let (width, height) = (texture.width as usize, texture.height as usize);
    let heights: Vec<f32> = texture
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|texel| luminance(texel).sqrt())
        .collect();
    let at = |x: usize, y: usize| heights[(y % height) * width + (x % width)];
    let mut output = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let dx = at(x + 1, y) - at(x + width - 1, y);
            let dy = at(x, y + 1) - at(x, y + height - 1);
            let normal = normalize([(-dx * strength) as f64, (dy * strength) as f64, 1.0]);
            output.extend_from_slice(&[
                ((normal[0] * 0.5 + 0.5) * 255.0).round() as u8,
                ((normal[1] * 0.5 + 0.5) * 255.0).round() as u8,
                ((normal[2] * 0.5 + 0.5) * 255.0).round() as u8,
                255,
            ]);
        }
    }
    output
}

/// glTF metallic-roughness map (G roughness, B metallic): `roughness` rises in dark crevices,
/// `metallic` fades out of dark (rusted, dirty) texels.
fn metallic_roughness_map(texture: &DecodedTexture, roughness: f32, metallic: f32) -> Vec<u8> {
    texture
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|texel| {
            let brightness = luminance(texel).sqrt();
            let texel_roughness = (roughness + (0.45 - brightness) * 0.35).clamp(0.04, 1.0);
            let texel_metallic = metallic * ((brightness - 0.08) / 0.3).clamp(0.0, 1.0);
            [
                255,
                (texel_roughness * 255.0).round() as u8,
                (texel_metallic * 255.0).round() as u8,
                255,
            ]
        })
        .collect()
}

/// Base roughness and metallic of a solid texture, from its name.
fn surface_from_name(name: &str) -> (f32, f32) {
    let has = |words: &[&str]| words.iter().any(|word| name.contains(word));
    if has(&["glass", "window"]) {
        (0.1, 0.0)
    } else if has(&[
        "met", "tech", "steel", "iron", "copper", "brass", "gold", "chrome", "pipe",
    ]) {
        (0.4, 0.75)
    } else if has(&["wood", "wd_", "plank", "plk"]) {
        (0.7, 0.0)
    } else if has(&["marble", "grk", "tile", "flr"]) {
        (0.45, 0.0)
    } else if has(&["snow", "ice"]) {
        (0.35, 0.0)
    } else if has(&[
        "grass", "leaf", "dirt", "ground", "mud", "rock", "terra", "sand",
    ]) {
        (0.9, 0.0)
    } else {
        (0.8, 0.0)
    }
}

fn checker_texture() -> DecodedTexture {
    let size = 64usize;
    let rgba = (0..size * size)
        .flat_map(|index| {
            let (x, y) = (index % size, index / size);
            if (x / 8 + y / 8) % 2 == 0 {
                [200, 0, 200, 255]
            } else {
                [40, 40, 40, 255]
            }
        })
        .collect();
    DecodedTexture {
        rgba,
        width: size as u32,
        height: size as u32,
        fullbright: vec![false; size * size],
    }
}

fn push_texture(
    textures: &mut Vec<GltfTextureData>,
    name: String,
    kind: TextureType,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
) -> usize {
    textures.push(GltfTextureData {
        name,
        kind,
        rtex: convert::rtex_from_rgba(width, height, rgba),
    });
    textures.len() - 1
}

fn emissive_texels(texture: &DecodedTexture, whole: bool) -> Option<Vec<u8>> {
    if !whole && !texture.fullbright.contains(&true) {
        return None;
    }
    Some(
        texture
            .rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(&texture.fullbright)
            .flat_map(|(texel, &bright)| {
                if whole || bright {
                    [texel[0], texel[1], texel[2], 255]
                } else {
                    [0, 0, 0, 255]
                }
            })
            .collect(),
    )
}

fn build_textures(
    prefix: &str,
    names: &[String],
    images: &HashMap<String, Vec<u8>>,
    palette: Option<&[u8]>,
    water_alpha: f32,
) -> TextureSet {
    let mut set = TextureSet {
        textures: Vec::new(),
        materials: Vec::new(),
        by_name: HashMap::new(),
        animations: Vec::new(),
        missing: Vec::new(),
    };
    for name in names {
        let class = texture_class(name);
        if matches!(class, TextureClass::Tool | TextureClass::Sky) {
            continue;
        }
        let texture = match decode_texture(name, images.get(name), palette) {
            Some(texture) => texture,
            None => {
                set.missing.push(name.clone());
                checker_texture()
            }
        };
        let (width, height) = (texture.width, texture.height);
        let (roughness, metallic, emissive_whole, normal_strength) = match class {
            TextureClass::Liquid(Liquid::Water) => (0.05, 0.0, false, 0.6),
            TextureClass::Liquid(Liquid::Slime) => (0.15, 0.0, true, 0.6),
            TextureClass::Liquid(Liquid::Lava | Liquid::Teleport) => (0.6, 0.0, true, 1.0),
            _ => {
                let (roughness, metallic) = surface_from_name(name);
                (roughness, metallic, false, 2.5)
            }
        };
        let texture_name = |role: &str| format!("{prefix}_{name}_{role}");
        let normal = push_texture(
            &mut set.textures,
            texture_name("normal"),
            TextureType::Normal,
            width,
            height,
            normal_map(&texture, normal_strength),
        );
        let metallic_roughness = push_texture(
            &mut set.textures,
            texture_name("metallic_roughness"),
            TextureType::MetallicRoughness,
            width,
            height,
            metallic_roughness_map(&texture, roughness, metallic),
        );
        let emissive = emissive_texels(&texture, emissive_whole).map(|texels| {
            push_texture(
                &mut set.textures,
                texture_name("emissive"),
                TextureType::Emissive,
                width,
                height,
                texels,
            )
        });
        let albedo = push_texture(
            &mut set.textures,
            texture_name("albedo"),
            TextureType::Color,
            width,
            height,
            texture.rgba,
        );
        let emissive_strength = match class {
            TextureClass::Liquid(Liquid::Slime) => 0.6,
            TextureClass::Liquid(Liquid::Lava | Liquid::Teleport) => 2.0,
            _ => FULLBRIGHT_EMISSIVE,
        };
        let translucent = matches!(class, TextureClass::Liquid(Liquid::Water | Liquid::Slime))
            && water_alpha < 1.0;
        let alpha_mode = match class {
            TextureClass::Fence => GltfAlphaMode::Mask(0.5),
            _ if translucent => GltfAlphaMode::Blend,
            _ => GltfAlphaMode::Opaque,
        };
        set.materials.push(GltfMaterialData {
            name: format!("{prefix}_{name}"),
            albedo: Some(albedo),
            normal: Some(normal),
            metallic_roughness: Some(metallic_roughness),
            emissive,
            specular: None,
            base_color: [1.0, 1.0, 1.0, if translucent { water_alpha } else { 1.0 }],
            metallic: 1.0,
            roughness: 1.0,
            emissive_color: if emissive.is_some() {
                [emissive_strength; 3]
            } else {
                [0.0; 3]
            },
            alpha_mode,
            double_sided: matches!(class, TextureClass::Liquid(_) | TextureClass::Fence),
            transmission: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
        });
        let material = set.materials.len() - 1;
        set.by_name.insert(name.clone(), (material, width, height));

        // Animated: frame 0 is the material's own albedo/emissive; later frames of the same size
        if let Some(base) = animation_base(name) {
            let mut frames = vec![QuakeAnimationFrame { albedo, emissive }];
            for frame in 1..10 {
                let frame_name = format!("+{frame}{base}");
                let Some(decoded) = decode_texture(&frame_name, images.get(&frame_name), palette)
                else {
                    break;
                };
                if (decoded.width, decoded.height) != (width, height) {
                    break;
                }
                let emissive = emissive_texels(&decoded, emissive_whole).map(|texels| {
                    push_texture(
                        &mut set.textures,
                        format!("{prefix}_{name}_frame{frame}_emissive"),
                        TextureType::Emissive,
                        width,
                        height,
                        texels,
                    )
                });
                let albedo = push_texture(
                    &mut set.textures,
                    format!("{prefix}_{name}_frame{frame}_albedo"),
                    TextureType::Color,
                    width,
                    height,
                    decoded.rgba,
                );
                frames.push(QuakeAnimationFrame { albedo, emissive });
            }
            if frames.len() > 1 {
                set.animations
                    .push(QuakeTextureAnimation { material, frames });
            }
        }
    }
    set
}

// --- Sky panorama ---

/// Equirect panorama (engine directions, as `ibl` samples it) from a Quake sky texture: the right
/// half is the far layer, the left half the near layer (black is see-through), both projected on
/// Quake's flattened sky dome; the horizon fades to the sky's mean color and below it darkens.
fn sky_panorama(texture: &DecodedTexture) -> Equirect {
    let (width, height) = (texture.width as usize, texture.height as usize);
    let half = (width / 2).max(1);
    let linear = |texel: &[u8]| {
        [
            srgb_to_linear(texel[0] as f32 / 255.0),
            srgb_to_linear(texel[1] as f32 / 255.0),
            srgb_to_linear(texel[2] as f32 / 255.0),
        ]
    };
    let texel = |x: usize, y: usize| {
        let index = ((y % height) * width + x) * 4;
        &texture.rgba[index..index + 4]
    };
    let mut mean = [0.0f32; 3];
    for y in 0..height {
        for x in half..width {
            let color = linear(texel(x, y));
            for channel in 0..3 {
                mean[channel] += color[channel];
            }
        }
    }
    let count = ((width - half) * height).max(1) as f32;
    mean = mean.map(|channel| channel / count);

    let sample = |s: f32, t: f32| -> [f32; 3] {
        let x = (s * half as f32).rem_euclid(half as f32) as usize;
        let y = (t * height as f32).rem_euclid(height as f32) as usize;
        let back = linear(texel(half + x.min(half - 1), y));
        let front = texel(x.min(half - 1), y);
        if front[0] as u32 + front[1] as u32 + front[2] as u32 > 0 {
            linear(front)
        } else {
            back
        }
    };
    let mut pixels = Vec::with_capacity((SKY_WIDTH * SKY_HEIGHT * 4) as usize);
    for row in 0..SKY_HEIGHT {
        let elevation = (0.5 - (row as f32 + 0.5) / SKY_HEIGHT as f32) * std::f32::consts::PI;
        for column in 0..SKY_WIDTH {
            let azimuth =
                ((column as f32 + 0.5) / SKY_WIDTH as f32 - 0.5) * 2.0 * std::f32::consts::PI;
            // Engine direction (ibl's equirect_uv_to_dir) → Quake (x, -z, y)
            let engine = [
                elevation.cos() * azimuth.cos(),
                elevation.sin(),
                elevation.cos() * azimuth.sin(),
            ];
            let quake = [engine[0], -engine[2], engine[1].abs()];
            // R_DrawSkyChain: the dome is flattened 3x, 6*63 texels across
            let stretched = [quake[0], quake[1], quake[2] * 3.0];
            let factor = 6.0 * 63.0
                / (stretched[0] * stretched[0]
                    + stretched[1] * stretched[1]
                    + stretched[2] * stretched[2])
                    .sqrt();
            let color = sample(
                quake[0] * factor / half as f32,
                quake[1] * factor / half as f32,
            );
            let horizon = (quake[2] / 0.2).clamp(0.0, 1.0);
            let below = if engine[1] < 0.0 { 0.35 } else { 1.0 };
            for channel in 0..3 {
                let value = mean[channel] + (color[channel] - mean[channel]) * horizon;
                pixels.push(value * below);
            }
            pixels.push(1.0);
        }
    }
    Equirect {
        pixels,
        width: SKY_WIDTH,
        height: SKY_HEIGHT,
    }
}

// --- Collision ---

/// Distance (Quake units) a trace stops short of the surface it hits, so the next trace from its
/// end starts outside the brush.
const TRACE_EPSILON: f64 = 1.0 / 32.0;
/// Edge of the collision grid cells, in Quake units.
const COLLISION_CELL: f64 = 256.0;

struct CollisionPlane {
    normal: Vec3,
    distance: f64,
    /// Index into `QuakeCollision::textures`; `u32::MAX` for the added bevel planes.
    texture: u32,
}

struct CollisionBrush {
    planes: Vec<CollisionPlane>,
    min: Vec3,
    max: Vec3,
}

/// Solid brushes of a map for swept-box traces (player movement, hitscan), in Quake space
/// (Z up, units; see [`quake_to_engine`]): what blocks the player in Quake. `clip` and sky brushes
/// block, liquids, `func_illusionary` and triggers don't.
pub struct QuakeCollision {
    brushes: Vec<CollisionBrush>,
    cells: HashMap<[i32; 3], Vec<u32>>,
    textures: Vec<String>,
}

/// Result of [`QuakeCollision::trace_box`], in Quake space.
#[derive(Clone, Copy, Debug)]
pub struct QuakeTrace {
    /// Part of the move completed (1 = nothing hit).
    pub fraction: f32,
    pub end: [f32; 3],
    /// Normal of the surface hit (zero when nothing was).
    pub normal: [f32; 3],
    /// The box started inside a brush; `all_solid`: and never left it.
    pub start_solid: bool,
    pub all_solid: bool,
    /// Texture of the face hit (see [`QuakeCollision::texture_name`]).
    pub texture: Option<u32>,
}

/// Quake space (Z up, units) → engine space (Y up, meters).
pub fn quake_to_engine(point: [f32; 3]) -> [f32; 3] {
    to_engine(point.map(f64::from))
}

/// Engine space (Y up, meters) → Quake space (Z up, units).
pub fn engine_to_quake(point: [f32; 3]) -> [f32; 3] {
    let factor = UNITS_PER_METER as f32;
    [point[0] * factor, -point[2] * factor, point[1] * factor]
}

/// Quake direction → engine direction (no scaling).
pub fn quake_direction_to_engine(direction: [f32; 3]) -> [f32; 3] {
    [direction[0], direction[2], -direction[1]]
}

/// Brushes that don't block: liquids (volumes), and compile-only brushes made of hint/origin
/// faces.
fn brush_is_solid(brush: &MapBrush) -> bool {
    let mut blocking = false;
    for face in &brush.faces {
        let name = face.texture.to_ascii_lowercase();
        if name.starts_with('*') {
            return false;
        }
        if !matches!(name.as_str(), "hint" | "hintskip" | "origin") && !name.contains("trigger") {
            blocking = true;
        }
    }
    blocking
}

impl QuakeCollision {
    pub fn from_map(map: &QuakeMap) -> Self {
        let mut textures: Vec<String> = Vec::new();
        let mut texture_ids: HashMap<String, u32> = HashMap::new();
        let mut brushes = Vec::new();
        for entity in map.drawn_entities() {
            if matches!(
                entity.classname(),
                "func_illusionary" | "func_detail_illusionary"
            ) {
                continue;
            }
            for brush in entity.brushes.iter().filter(|brush| brush_is_solid(brush)) {
                let Some(built) = build_brush(entity, brush) else {
                    continue;
                };
                let mut planes: Vec<CollisionPlane> = built
                    .faces
                    .iter()
                    .map(|face| {
                        let name = brush.faces[face.face_index].texture.to_ascii_lowercase();
                        let next = textures.len() as u32;
                        let texture = *texture_ids.entry(name.clone()).or_insert_with(|| {
                            textures.push(name);
                            next
                        });
                        CollisionPlane {
                            normal: face.plane.normal,
                            distance: face.plane.distance,
                            texture,
                        }
                    })
                    .collect();
                // Axial bevels: without them a box sweeping along a slanted face's edge would
                // be stopped too early or too late at the brush's corners
                for axis in 0..3 {
                    for sign in [-1.0, 1.0] {
                        let mut normal = [0.0; 3];
                        normal[axis] = sign;
                        let distance = if sign > 0.0 {
                            built.max[axis]
                        } else {
                            -built.min[axis]
                        };
                        let exists = planes
                            .iter()
                            .any(|plane| dot(plane.normal, normal) > 0.999_999);
                        if !exists {
                            planes.push(CollisionPlane {
                                normal,
                                distance,
                                texture: u32::MAX,
                            });
                        }
                    }
                }
                brushes.push(CollisionBrush {
                    planes,
                    min: built.min,
                    max: built.max,
                });
            }
        }
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (index, brush) in brushes.iter().enumerate() {
            let (low, high) = collision_cells(brush.min, brush.max);
            for x in low[0]..=high[0] {
                for y in low[1]..=high[1] {
                    for z in low[2]..=high[2] {
                        cells.entry([x, y, z]).or_default().push(index as u32);
                    }
                }
            }
        }
        Self {
            brushes,
            cells,
            textures,
        }
    }

    /// Lowercase name of a texture from [`QuakeTrace::texture`].
    pub fn texture_name(&self, texture: u32) -> &str {
        self.textures
            .get(texture as usize)
            .map_or("", String::as_str)
    }

    /// Sweeps an axis-aligned box with half size `half_extents`, centered on `start`, to `end`
    /// (Quake space) and stops it at the first solid brush.
    pub fn trace_box(&self, start: [f32; 3], end: [f32; 3], half_extents: [f32; 3]) -> QuakeTrace {
        let start = start.map(f64::from);
        let end = end.map(f64::from);
        let extents = half_extents.map(f64::from);
        let mut swept_min = [0.0; 3];
        let mut swept_max = [0.0; 3];
        for axis in 0..3 {
            swept_min[axis] = start[axis].min(end[axis]) - extents[axis] - 1.0;
            swept_max[axis] = start[axis].max(end[axis]) + extents[axis] + 1.0;
        }
        let mut candidates: Vec<u32> = Vec::new();
        let (low, high) = collision_cells(swept_min, swept_max);
        for x in low[0]..=high[0] {
            for y in low[1]..=high[1] {
                for z in low[2]..=high[2] {
                    if let Some(indices) = self.cells.get(&[x, y, z]) {
                        candidates.extend_from_slice(indices);
                    }
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();

        let mut fraction = 1.0;
        let mut normal = [0.0; 3];
        let mut texture = None;
        let mut start_solid = false;
        let mut all_solid = false;
        for index in candidates {
            let brush = &self.brushes[index as usize];
            let overlaps = (0..3).all(|axis| {
                brush.min[axis] <= swept_max[axis] && swept_min[axis] <= brush.max[axis]
            });
            if !overlaps {
                continue;
            }
            match clip_against_brush(brush, start, end, extents) {
                BrushClip::Missed => {}
                BrushClip::StartsInside { never_leaves } => {
                    start_solid = true;
                    all_solid |= never_leaves;
                }
                BrushClip::Hit {
                    fraction: hit,
                    plane,
                } => {
                    if hit < fraction {
                        fraction = hit;
                        normal = brush.planes[plane].normal;
                        let id = brush.planes[plane].texture;
                        texture = (id != u32::MAX).then_some(id);
                    }
                }
            }
        }
        if all_solid {
            fraction = 0.0;
        }
        let end_point: [f64; 3] =
            std::array::from_fn(|axis| start[axis] + (end[axis] - start[axis]) * fraction);
        QuakeTrace {
            fraction: fraction as f32,
            end: end_point.map(|value| value as f32),
            normal: normal.map(|value| value as f32),
            start_solid,
            all_solid,
            texture,
        }
    }

    /// [`QuakeCollision::trace_box`] of a point.
    pub fn trace_ray(&self, start: [f32; 3], end: [f32; 3]) -> QuakeTrace {
        self.trace_box(start, end, [0.0; 3])
    }
}

fn collision_cells(min: Vec3, max: Vec3) -> ([i32; 3], [i32; 3]) {
    let cell = |value: f64| (value / COLLISION_CELL).floor() as i32;
    (min.map(cell), max.map(cell))
}

enum BrushClip {
    Missed,
    StartsInside { never_leaves: bool },
    Hit { fraction: f64, plane: usize },
}

/// Clips the segment `start`..`end` against the brush grown by the box (each plane pushed out by
/// the box's extent along its normal): the fraction where it enters, if it does.
fn clip_against_brush(brush: &CollisionBrush, start: Vec3, end: Vec3, extents: Vec3) -> BrushClip {
    let mut enter = -1.0;
    let mut leave = 1.0;
    let mut enter_plane = None;
    let mut starts_outside = false;
    let mut ends_outside = false;
    for (index, plane) in brush.planes.iter().enumerate() {
        let grow = plane.normal[0].abs() * extents[0]
            + plane.normal[1].abs() * extents[1]
            + plane.normal[2].abs() * extents[2];
        let distance = plane.distance + grow;
        let start_side = dot(plane.normal, start) - distance;
        let end_side = dot(plane.normal, end) - distance;
        if start_side > 0.0 {
            starts_outside = true;
        }
        if end_side > 0.0 {
            ends_outside = true;
        }
        // Entirely in front of this plane: outside the brush
        if start_side > 0.0 && (end_side >= TRACE_EPSILON || end_side >= start_side) {
            return BrushClip::Missed;
        }
        // Entirely behind it: this plane doesn't limit the segment
        if start_side <= 0.0 && end_side <= 0.0 {
            continue;
        }
        if start_side > end_side {
            // Entering through this plane; stop TRACE_EPSILON in front of it
            let fraction = ((start_side - TRACE_EPSILON) / (start_side - end_side)).max(0.0);
            if fraction > enter {
                enter = fraction;
                enter_plane = Some(index);
            }
        } else {
            let fraction = ((start_side + TRACE_EPSILON) / (start_side - end_side)).min(1.0);
            if fraction < leave {
                leave = fraction;
            }
        }
    }
    if !starts_outside {
        return BrushClip::StartsInside {
            never_leaves: !ends_outside,
        };
    }
    match enter_plane {
        Some(plane) if enter < leave => BrushClip::Hit {
            fraction: enter.max(0.0),
            plane,
        },
        _ => BrushClip::Missed,
    }
}

// --- Level ---

/// A point or spot light, in engine space.
pub struct QuakeLight {
    pub classname: String,
    pub position: [f32; 3],
    /// Linear RGB, max component 1.
    pub color: [f32; 3],
    /// Engine `LightComponent::intensity` at light style brightness 1.
    pub intensity: f32,
    /// Meters at which the light reaches zero.
    pub range: f32,
    pub spot: Option<QuakeSpot>,
    /// Light style ([`light_style_brightness`]).
    pub style: u32,
    /// Switchable light that starts off (spawnflag 1).
    pub start_off: bool,
}

pub struct QuakeSpot {
    /// Engine-space unit direction the cone points along.
    pub direction: [f32; 3],
    /// Half-angles, degrees.
    pub inner_angle: f32,
    pub outer_angle: f32,
}

/// Directional light, in engine space (`direction` points from the sun toward the scene).
pub struct QuakeSun {
    pub direction: [f32; 3],
    pub color: [f32; 3],
    /// The map's `_sunlight` value (Quake light units).
    pub light: f32,
}

pub struct QuakeFog {
    /// Linear RGB.
    pub color: [f32; 3],
    /// Per meter, exponential-squared (as `RenderStateComponent::fog_density`).
    pub density: f32,
}

/// Where the camera can start: a spawn point's eye position, looking along its angles.
pub struct QuakeSpawn {
    pub classname: String,
    pub position: [f32; 3],
    /// Quake yaw (degrees, counter-clockwise from +X seen from above) and pitch (up positive).
    pub yaw: f32,
    pub pitch: f32,
}

pub struct QuakeAnimationFrame {
    pub albedo: usize,
    pub emissive: Option<usize>,
}

/// An animated texture's material and its frames (indices into the scene's textures), cycled at
/// 5 frames per second like Quake.
pub struct QuakeTextureAnimation {
    pub material: usize,
    pub frames: Vec<QuakeAnimationFrame>,
}

pub struct QuakeLevelStats {
    pub entities: usize,
    pub brushes: usize,
    pub faces: usize,
    pub hidden_faces: usize,
    pub triangles: usize,
}

/// A level ready to upload: the scene (as the glTF loader returns it) and what its entities
/// describe.
pub struct QuakeLevel {
    pub scene: GltfSceneData,
    pub animations: Vec<QuakeTextureAnimation>,
    pub lights: Vec<QuakeLight>,
    pub sun: Option<QuakeSun>,
    /// Sky dome light (`_sunlight2`, Quake light units) and its color.
    pub sky_light: f32,
    pub sky_light_color: [f32; 3],
    /// Minimum light of every surface (worldspawn `light` / `_minlight`, Quake light units).
    pub minimum_light: f32,
    pub fog: Option<QuakeFog>,
    /// Panorama from the map's sky texture, for the background and image-based lighting.
    pub sky: Option<Equirect>,
    pub spawns: Vec<QuakeSpawn>,
    /// Engine-space bounds of the drawn geometry.
    pub bounds: ([f32; 3], [f32; 3]),
    /// Texture names without an image (drawn with a checker).
    pub missing_textures: Vec<String>,
    pub stats: QuakeLevelStats,
}

/// Quake light value at which a surface shows its texture's own color.
const NORMAL_LIGHT: f32 = 128.0;

impl QuakeLevel {
    /// Environment for image-based lighting, as the light compilers light a level: a uniform sky
    /// dome of `_sunlight2` above the horizon (ericw-tools' sky light), a fifth of it from below
    /// (bounce, until GI adds it), and the minimum light everywhere. 1 = a surface open to the
    /// whole dome shows its texture color, so use it at `ibl_intensity` 1; the sky texture itself
    /// is only the background ([`QuakeLevel::sky`]).
    pub fn lighting_environment(&self) -> Equirect {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let dome = self.sky_light / NORMAL_LIGHT;
        let minimum = self.minimum_light / NORMAL_LIGHT;
        let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
        for row in 0..HEIGHT {
            let above = row < HEIGHT / 2;
            let sky = if above { dome } else { dome * 0.2 };
            for _ in 0..WIDTH {
                for channel in 0..3 {
                    pixels.push(sky * self.sky_light_color[channel] + minimum);
                }
                pixels.push(1.0);
            }
        }
        Equirect {
            pixels,
            width: WIDTH,
            height: HEIGHT,
        }
    }
}

/// Color key: three components, 0..1 or 0..255.
fn parse_color(entity: &MapEntity, keys: &[&str]) -> Option<[f32; 3]> {
    let color = keys.iter().find_map(|key| entity.vector(key))?;
    let divisor = if color.iter().any(|&channel| channel > 1.0) {
        255.0
    } else {
        1.0
    };
    Some(color.map(|channel| (channel / divisor).max(0.0) as f32))
}

/// Max component 1 (brightness stays in `intensity`); black becomes white.
fn normalize_color(color: [f32; 3]) -> [f32; 3] {
    let max = color[0].max(color[1]).max(color[2]);
    if max <= 0.0 {
        [1.0; 3]
    } else {
        color.map(|channel| channel / max)
    }
}

/// Converts the entities' Quake light into the engine's inverse-square lights: the range is where
/// Quake's (linear by default) falloff ends, and the intensity matches Quake's brightness halfway
/// there. `scale` is light units → `LightComponent::intensity` at 1 m.
fn build_lights(map: &QuakeMap, light_scale: f32) -> (Vec<QuakeLight>, Option<QuakeSun>) {
    let targets: HashMap<&str, Vec3> = map
        .entities
        .iter()
        .filter_map(|entity| Some((entity.get("targetname")?, entity.vector("origin")?)))
        .collect();
    let mut lights = Vec::new();
    let mut sun = None;
    for entity in &map.entities {
        let classname = entity.classname();
        if !classname.starts_with("light") {
            continue;
        }
        let Some(origin) = entity.vector("origin") else {
            continue;
        };
        let light = entity
            .number("light")
            .or_else(|| entity.number("_light"))
            .unwrap_or(300.0);
        if light <= 0.0 {
            continue; // negative: an "antilight", not representable
        }
        let color = normalize_color(parse_color(entity, &["_color", "color"]).unwrap_or([1.0; 3]));
        let mangle = entity.vector("mangle").or_else(|| entity.vector("_mangle"));
        let target_direction = entity
            .get("target")
            .and_then(|target| targets.get(target))
            .map(|&target| normalize(sub(target, origin)));
        // `mangle` is yaw pitch roll on lights
        let direction = target_direction
            .or_else(|| mangle.map(|[yaw, pitch, _]| angles_to_direction(yaw, pitch)));

        if entity.number("_sun") == Some(1.0) {
            sun = Some(QuakeSun {
                direction: direction_to_engine(direction.unwrap_or([0.0, 0.0, -1.0])),
                color,
                light: light as f32,
            });
            continue;
        }

        let wait = entity
            .number("wait")
            .filter(|&wait| wait > 0.0)
            .unwrap_or(1.0);
        let falloff = entity.number("delay").unwrap_or(0.0) as i32;
        let range_units = match falloff {
            0 => entity
                .number("_falloff")
                .filter(|&distance| distance > 0.0)
                .unwrap_or(light / wait),
            // Inverse and inverse-square falloffs reach further than linear
            1 | 2 | 5 => 2.0 * light / wait,
            3 => 4096.0,
            _ => continue, // 4: local minlight, no light of its own
        };
        let range = (range_units / UNITS_PER_METER) as f32;
        // Quake brightness at half range is light/2 (128 = "normal"); match it there
        let half_range = range * 0.5;
        let intensity = light_scale * (light as f32 / 256.0) * half_range * half_range;

        let spot = direction.map(|direction| {
            let cone = entity
                .number("angle")
                .filter(|&angle| angle > 0.0)
                .unwrap_or(40.0);
            let outer = (cone * 0.5).min(89.0) as f32;
            let inner = entity
                .number("_softangle")
                .filter(|&angle| angle > 0.0)
                .map(|angle| (angle * 0.5) as f32)
                .unwrap_or(outer * 0.7)
                .min(outer);
            QuakeSpot {
                direction: direction_to_engine(direction),
                inner_angle: inner,
                outer_angle: outer,
            }
        });
        let style = entity.number("style").unwrap_or(0.0).max(0.0) as u32;
        let switchable = entity.get("targetname").is_some();
        let start_off = switchable && (entity.number("spawnflags").unwrap_or(0.0) as i64) & 1 == 1;
        lights.push(QuakeLight {
            classname: classname.to_string(),
            position: to_engine(origin),
            color,
            intensity,
            range,
            spot,
            style,
            start_off,
        });
    }

    if sun.is_none() {
        if let Some(world) = map.worldspawn() {
            let sunlight = world.number("_sunlight").unwrap_or(0.0);
            if sunlight > 0.0 {
                let [yaw, pitch, _] = world
                    .vector("_sun_mangle")
                    .or_else(|| world.vector("_sunlight_mangle"))
                    .unwrap_or([0.0, -90.0, 0.0]);
                sun = Some(QuakeSun {
                    direction: direction_to_engine(angles_to_direction(yaw, pitch)),
                    color: normalize_color(
                        parse_color(world, &["_sunlight_color"]).unwrap_or([1.0; 3]),
                    ),
                    light: sunlight as f32,
                });
            }
        }
    }
    (lights, sun)
}

fn build_spawns(map: &QuakeMap) -> Vec<QuakeSpawn> {
    let mut spawns = Vec::new();
    for entity in &map.entities {
        let classname = entity.classname();
        let player = matches!(
            classname,
            "info_player_start"
                | "info_player_start2"
                | "info_player_deathmatch"
                | "info_player_coop"
        );
        if !player && classname != "info_intermission" {
            continue;
        }
        let Some(origin) = entity.vector("origin") else {
            continue;
        };
        let (yaw, pitch) = match entity.vector("mangle").or_else(|| entity.vector("angles")) {
            // pitch yaw roll; Quake pitch is positive looking down
            Some([pitch, yaw, _]) => (yaw, -pitch),
            None => match entity.number("angle").unwrap_or(0.0) {
                angle if angle == -1.0 => (0.0, 90.0),
                angle if angle == -2.0 => (0.0, -90.0),
                angle => (angle, 0.0),
            },
        };
        let eye = if player {
            add(
                origin,
                [0.0, 0.0, f64::from(crate::quake_player::EYE_HEIGHT)],
            )
        } else {
            origin
        };
        spawns.push(QuakeSpawn {
            classname: classname.to_string(),
            position: to_engine(eye),
            yaw: yaw as f32,
            pitch: pitch as f32,
        });
    }
    // Single-player start first
    spawns.sort_by_key(|spawn| match spawn.classname.as_str() {
        "info_player_start" => 0,
        "info_player_deathmatch" => 1,
        "info_player_coop" | "info_player_start2" => 2,
        _ => 3,
    });
    spawns
}

/// Triangles of one material in one cell.
#[derive(Default)]
struct MeshBatch {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    texture_coordinates: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

/// Builds the level: geometry, materials from `images` (lowercase texture name → PNG bytes;
/// see [`QuakeMap::texture_names`] and [`animation_frame_names`]), lights, sun, fog, sky and
/// spawns. `palette` is Quake's `gfx/palette.lmp` (768 bytes), which marks the fullbright texels;
/// `light_scale` converts Quake light units into engine light intensity (see [`QuakeLight`]).
pub fn build_quake_level(
    name_prefix: &str,
    map: &QuakeMap,
    images: &HashMap<String, Vec<u8>>,
    palette: Option<&[u8]>,
    light_scale: f32,
) -> Result<QuakeLevel> {
    let world = map
        .worldspawn()
        .context("the map has no worldspawn entity")?;
    let water_alpha = world
        .number("_wateralpha")
        .map(|alpha| alpha.clamp(0.0, 1.0) as f32)
        .unwrap_or(1.0);

    let names = map.texture_names();
    let set = build_textures(name_prefix, &names, images, palette, water_alpha);

    // --- Brushes → visible faces → batches ---
    let mut brushes: Vec<BuiltBrush> = Vec::new();
    let mut brush_count = 0;
    for entity in map.drawn_entities() {
        for brush in &entity.brushes {
            brush_count += 1;
            if let Some(built) = build_brush(entity, brush) {
                brushes.push(built);
            }
        }
    }
    let grid = OccluderGrid::new(&brushes);

    let mut batches: HashMap<(usize, [i32; 3]), MeshBatch> = HashMap::new();
    let mut sky_faces: HashMap<String, usize> = HashMap::new();
    let mut face_count = 0;
    let mut hidden_faces = 0;
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for brush_index in 0..brushes.len() {
        let brush = &brushes[brush_index];
        let visible = visible_faces(&brushes, &grid, brush_index);
        for (slot, face) in brush.faces.iter().enumerate() {
            let source = &brush.brush.faces[face.face_index];
            let class = texture_class(&source.texture);
            if class == TextureClass::Tool {
                continue;
            }
            face_count += 1;
            if !visible.contains(&slot) {
                hidden_faces += 1;
                continue;
            }
            let name = source.texture.to_ascii_lowercase();
            if class == TextureClass::Sky {
                *sky_faces.entry(name).or_default() += 1;
                continue;
            }
            let Some(&(material, width, height)) = set.by_name.get(&name) else {
                continue;
            };
            let centroid = scale(
                face.polygon
                    .iter()
                    .fold([0.0; 3], |sum, &point| add(sum, point)),
                1.0 / face.polygon.len() as f64,
            );
            let cell = centroid.map(|value| (value / CELL_SIZE).floor() as i32);
            let batch = batches.entry((material, cell)).or_default();
            let (u_axis, v_axis) = texture_axes(source, face.plane.normal);
            let normal = direction_to_engine(face.plane.normal);
            let first = batch.positions.len() as u32;
            for &point in &face.polygon {
                let position = to_engine(point);
                for axis in 0..3 {
                    min[axis] = min[axis].min(position[axis]);
                    max[axis] = max[axis].max(position[axis]);
                }
                batch.positions.push(position);
                batch.normals.push(normal);
                let u = dot(point, [u_axis[0], u_axis[1], u_axis[2]]) + u_axis[3];
                let v = dot(point, [v_axis[0], v_axis[1], v_axis[2]]) + v_axis[3];
                batch
                    .texture_coordinates
                    .push([(u / width as f64) as f32, (v / height as f64) as f32]);
            }
            for index in 1..face.polygon.len() as u32 - 1 {
                batch
                    .indices
                    .extend_from_slice(&[first, first + index, first + index + 1]);
            }
        }
    }

    let mut keys: Vec<(usize, [i32; 3])> = batches.keys().copied().collect();
    keys.sort();
    let mut meshes = Vec::with_capacity(keys.len());
    let mut triangles = 0;
    for key in keys {
        let Some(batch) = batches.remove(&key) else {
            continue;
        };
        triangles += batch.indices.len() / 3;
        let (vertices, indices) = convert::build_primitive(
            PrimitiveAttributes {
                positions: batch.positions,
                normals: Some(batch.normals),
                texture_coordinates: Some(batch.texture_coordinates),
                tangents: Vec::new(),
                indices: Some(batch.indices),
            },
            convert::IDENTITY,
        )?;
        let (material, [x, y, z]) = key;
        meshes.push(GltfMeshData {
            name: format!("{}_{x}_{y}_{z}", set.materials[material].name),
            rmsh: convert::rmsh_bytes(&vertices, &indices),
            material,
        });
    }
    if meshes.is_empty() {
        bail!("the map has no drawable faces");
    }

    // --- Sky: the texture on the most sky faces ---
    let sky = sky_faces
        .into_iter()
        .max_by_key(|(name, count)| (*count, std::cmp::Reverse(name.clone())))
        .and_then(|(name, _)| decode_texture(&name, images.get(&name), None))
        .map(|texture| sky_panorama(&texture));

    let (lights, sun) = build_lights(map, light_scale);
    let fog = world
        .numbers("fog")
        .or_else(|| world.numbers("_fog"))
        .and_then(|values| {
            let density = *values.first()?;
            (density > 0.0).then(|| {
                let color = match values.as_slice() {
                    [_, r, g, b, ..] => {
                        [*r, *g, *b].map(|channel| srgb_to_linear((channel as f32).clamp(0.0, 1.0)))
                    }
                    _ => [0.3; 3],
                };
                QuakeFog {
                    color,
                    // FitzQuake: exp2 fog of density/64 per unit
                    density: (density / 64.0 * UNITS_PER_METER) as f32,
                }
            })
        });

    Ok(QuakeLevel {
        animations: set.animations,
        lights,
        sun,
        sky_light: world.number("_sunlight2").unwrap_or(0.0) as f32,
        sky_light_color: normalize_color(
            parse_color(world, &["_sunlight2_color", "_sunlight_color2"]).unwrap_or([1.0; 3]),
        ),
        minimum_light: world
            .number("_minlight")
            .or_else(|| world.number("light"))
            .unwrap_or(0.0) as f32,
        fog,
        sky,
        spawns: build_spawns(map),
        bounds: (min, max),
        missing_textures: set.missing,
        stats: QuakeLevelStats {
            entities: map.entities.len(),
            brushes: brush_count,
            faces: face_count,
            hidden_faces,
            triangles,
        },
        scene: GltfSceneData {
            textures: set.textures,
            materials: set.materials,
            meshes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned box brush `min`..`max` in Valve 220 format: per axis, the min face's three
    /// points (outward normal) and the same points reversed on the max side.
    fn valve_box(min: [i32; 3], max: [i32; 3], texture: &str) -> String {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        type Points = [[i32; 3]; 3];
        let x_face = |x: i32| -> Points { [[x, y0, z0], [x, y1, z0], [x, y0, z1]] };
        let y_face = |y: i32| -> Points { [[x0, y, z0], [x0, y, z1], [x1, y, z0]] };
        let z_face = |z: i32| -> Points { [[x0, y0, z], [x1, y0, z], [x0, y1, z]] };
        let faces: [(Points, &str); 6] = [
            (x_face(x0), "[ 0 1 0 0 ] [ 0 0 -1 0 ]"),
            (y_face(y0), "[ 1 0 0 0 ] [ 0 0 -1 0 ]"),
            (z_face(z0), "[ 1 0 0 0 ] [ 0 -1 0 0 ]"),
            (reversed(x_face(x1)), "[ 0 1 0 0 ] [ 0 0 -1 0 ]"),
            (reversed(y_face(y1)), "[ 1 0 0 0 ] [ 0 0 -1 0 ]"),
            (reversed(z_face(z1)), "[ 1 0 0 0 ] [ 0 -1 0 0 ]"),
        ];
        let mut brush = String::from("{\n");
        for (points, axes) in faces {
            for [x, y, z] in points {
                brush.push_str(&format!("( {x} {y} {z} ) "));
            }
            brush.push_str(&format!("{texture} {axes} 0 1 1\n"));
        }
        brush.push_str("}\n");
        brush
    }

    fn reversed(points: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
        [points[2], points[1], points[0]]
    }

    fn world(brushes: &str, extra: &str) -> String {
        format!(
            "// Game: Quake\n// Format: Valve\n{{\n\"classname\" \"worldspawn\"\n{brushes}}}\n{extra}"
        )
    }

    fn level(text: &str) -> QuakeLevel {
        let map = parse_map(text).unwrap();
        build_quake_level("test", &map, &HashMap::new(), None, 1.0).unwrap()
    }

    fn triangle_count(level: &QuakeLevel) -> usize {
        level
            .scene
            .meshes
            .iter()
            .map(|mesh| u32::from_le_bytes(mesh.rmsh[12..16].try_into().unwrap()) as usize / 3)
            .sum()
    }

    #[test]
    fn box_brush_six_faces_twelve_triangles_meter_bounds() {
        let level = level(&world(&valve_box([0, 0, 0], [64, 64, 32], "stone"), ""));
        assert_eq!(level.stats.brushes, 1);
        assert_eq!(level.stats.faces, 6);
        assert_eq!(level.stats.hidden_faces, 0);
        assert_eq!(triangle_count(&level), 12);
        // Quake (x, y, z) → engine (x, z, -y) / 32
        assert_eq!(level.bounds, ([0.0, 0.0, -2.0], [2.0, 1.0, 0.0]));
        assert_eq!(level.missing_textures, vec!["stone".to_string()]);
    }

    #[test]
    fn box_brush_triangles_face_outward() {
        let level = level(&world(&valve_box([0, 0, 0], [64, 64, 64], "stone"), ""));
        let rmsh = &level.scene.meshes[0].rmsh;
        let vertex_count = u32::from_le_bytes(rmsh[8..12].try_into().unwrap()) as usize;
        let vertices: &[convert::CookedVertex] =
            bytemuck::cast_slice(&rmsh[16..16 + vertex_count * 56]);
        let index_bytes = &rmsh[16 + vertex_count * 56..];
        let indices: &[u32] = bytemuck::cast_slice(index_bytes);
        let center = [1.0f32, 1.0, -1.0];
        for triangle in indices.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| vertices[triangle[i] as usize].position);
            let edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            let outward = [a[0] - center[0], a[1] - center[1], a[2] - center[2]];
            let facing = normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2];
            assert!(facing > 0.0, "counter-clockwise seen from outside");
            let stored = vertices[triangle[0] as usize].normal;
            assert!(stored[0] * normal[0] + stored[1] * normal[1] + stored[2] * normal[2] > 0.0);
        }
    }

    #[test]
    fn touching_brushes_shared_faces_hidden() {
        let brushes = valve_box([0, 0, 0], [64, 64, 64], "stone")
            + &valve_box([64, 0, 0], [128, 64, 64], "stone");
        let level = level(&world(&brushes, ""));
        assert_eq!(level.stats.faces, 12);
        assert_eq!(level.stats.hidden_faces, 2);
        assert_eq!(triangle_count(&level), 20);
    }

    #[test]
    fn coincident_brushes_one_copy_drawn() {
        let brushes = valve_box([0, 0, 0], [64, 64, 64], "stone")
            + &valve_box([0, 0, 0], [64, 64, 64], "stone");
        let level = level(&world(&brushes, ""));
        assert_eq!(triangle_count(&level), 12);
    }

    #[test]
    fn tool_and_sky_faces_not_drawn() {
        let brushes = valve_box([0, 0, 0], [64, 64, 64], "clip")
            + &valve_box([0, 0, 128], [64, 64, 192], "sky1")
            + &valve_box([0, 0, 256], [64, 64, 320], "stone");
        let level = level(&world(&brushes, ""));
        assert_eq!(triangle_count(&level), 12);
        assert!(level.sky.is_none(), "no sky image given");
    }

    #[test]
    fn trigger_entity_brushes_not_drawn() {
        let trigger = format!(
            "{{\n\"classname\" \"trigger_once\"\n{}}}\n",
            valve_box([0, 0, 256], [64, 64, 320], "stone")
        );
        let level = level(&world(
            &valve_box([0, 0, 0], [64, 64, 64], "stone"),
            &trigger,
        ));
        assert_eq!(level.stats.brushes, 1);
    }

    #[test]
    fn standard_format_floor_uv_from_axial_projection() {
        let text = "{\n\"classname\" \"worldspawn\"\n{\n\
            ( 0 0 0 ) ( 0 64 0 ) ( 0 0 64 ) stone 0 0 0 1 1\n\
            ( 0 0 0 ) ( 0 0 64 ) ( 64 0 0 ) stone 0 0 0 1 1\n\
            ( 0 0 0 ) ( 64 0 0 ) ( 0 64 0 ) stone 0 0 0 1 1\n\
            ( 0 0 64 ) ( 0 64 64 ) ( 64 0 64 ) stone 16 0 0 2 1\n\
            ( 64 64 0 ) ( 0 64 64 ) ( 64 64 64 ) stone 0 0 0 1 1\n\
            ( 64 0 64 ) ( 64 64 0 ) ( 64 64 64 ) stone 0 0 0 1 1\n\
            }\n}\n";
        let map = parse_map(text).unwrap();
        let top = &map.entities[0].brushes[0].faces[3];
        let plane = Plane::from_points(&top.points).unwrap();
        assert_eq!(plane.normal, [0.0, 0.0, 1.0]);
        let (u, v) = texture_axes(top, plane.normal);
        // Floor: s = x / scale_x + offset, t = -y
        assert_eq!(u, [0.5, 0.0, 0.0, 16.0]);
        assert_eq!(v, [0.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn quake2_surface_flags_after_face_skipped() {
        let text = "{\n\"classname\" \"worldspawn\"\n{\n\
            ( 0 0 0 ) ( 0 64 0 ) ( 0 0 64 ) stone 0 0 0 1 1 0 0 0\n\
            }\n}\n";
        let map = parse_map(text).unwrap();
        assert_eq!(map.entities[0].brushes[0].faces.len(), 1);
    }

    #[test]
    fn brush_primitives_rejected_with_line() {
        let text = "{\n\"classname\" \"worldspawn\"\n{\nbrushDef\n{\n}\n}\n}\n";
        let error = parse_map(text).err().unwrap().to_string();
        assert!(error.starts_with("line 4:"), "{error}");
    }

    #[test]
    fn fence_texture_name_starting_with_brace_parsed_as_texture() {
        let text = format!(
            "{{\n\"classname\" \"worldspawn\"\n{}}}\n",
            valve_box([0, 0, 0], [64, 64, 64], "{grate")
        );
        let map = parse_map(&text).unwrap();
        assert_eq!(map.texture_names(), vec!["{grate".to_string()]);
    }

    #[test]
    fn light_linear_300_range_and_half_range_intensity() {
        let light =
            "{\n\"classname\" \"light\"\n\"origin\" \"0 32 64\"\n\"_color\" \"255 128 0\"\n}\n";
        let level = level(&world(&valve_box([0, 0, 0], [64, 64, 32], "stone"), light));
        let light = &level.lights[0];
        assert_eq!(light.position, [0.0, 2.0, -1.0]);
        assert_eq!(light.range, 300.0 / 32.0);
        let half = light.range * 0.5;
        assert!((light.intensity - 300.0 / 256.0 * half * half).abs() < 1e-4);
        assert_eq!(light.color, [1.0, 128.0 / 255.0, 0.0]);
        assert!(light.spot.is_none());
    }

    #[test]
    fn light_with_target_is_spot_toward_target() {
        let entities = "{\n\"classname\" \"light\"\n\"origin\" \"0 0 128\"\n\"target\" \"t1\"\n\"angle\" \"60\"\n}\n\
            {\n\"classname\" \"info_null\"\n\"origin\" \"0 0 0\"\n\"targetname\" \"t1\"\n}\n";
        let level = level(&world(
            &valve_box([0, 0, 0], [64, 64, 32], "stone"),
            entities,
        ));
        let spot = level.lights[0].spot.as_ref().unwrap();
        assert_eq!(spot.direction, [0.0, -1.0, 0.0]);
        assert_eq!(spot.outer_angle, 30.0);
    }

    #[test]
    fn worldspawn_sun_mangle_fog_and_spawn() {
        let text = "{\n\"classname\" \"worldspawn\"\n\"_sunlight\" \"120\"\n\"_sun_mangle\" \"0 -90 0\"\n\"fog\" \"0.02 1 1 1\"\n"
            .to_string()
            + &valve_box([0, 0, 0], [64, 64, 32], "stone")
            + "}\n{\n\"classname\" \"info_player_start\"\n\"origin\" \"32 32 56\"\n\"angle\" \"90\"\n}\n";
        let level = level(&text);
        let sun = level.sun.as_ref().unwrap();
        assert!((sun.direction[1] + 1.0).abs() < 1e-6, "{:?}", sun.direction);
        assert_eq!(sun.light, 120.0);
        let fog = level.fog.as_ref().unwrap();
        assert!((fog.density - 0.01).abs() < 1e-6);
        assert_eq!(fog.color, [1.0; 3]);
        let spawn = &level.spawns[0];
        assert_eq!(spawn.position, [1.0, 2.4375, -1.0]);
        assert_eq!(spawn.yaw, 90.0);
    }

    fn collision(brushes: &str, extra: &str) -> QuakeCollision {
        QuakeCollision::from_map(&parse_map(&world(brushes, extra)).unwrap())
    }

    #[test]
    fn trace_ray_down_onto_box_top_stops_epsilon_above_with_up_normal() {
        let world = collision(&valve_box([0, 0, 0], [64, 64, 32], "stone"), "");
        let trace = world.trace_ray([32.0, 32.0, 100.0], [32.0, 32.0, -100.0]);
        assert!(
            (trace.end[2] - (32.0 + 1.0 / 32.0)).abs() < 1e-3,
            "{:?}",
            trace.end
        );
        assert_eq!(trace.normal, [0.0, 0.0, 1.0]);
        assert_eq!(world.texture_name(trace.texture.unwrap()), "stone");
        assert!(!trace.start_solid);
    }

    #[test]
    fn trace_box_grown_by_extents_hits_side() {
        let world = collision(&valve_box([0, 0, 0], [64, 64, 64], "stone"), "");
        let trace = world.trace_box(
            [-100.0, 32.0, 32.0],
            [100.0, 32.0, 32.0],
            [16.0, 16.0, 16.0],
        );
        assert!(
            (trace.end[0] - (-16.0 - 1.0 / 32.0)).abs() < 1e-3,
            "{:?}",
            trace.end
        );
        assert_eq!(trace.normal, [-1.0, 0.0, 0.0]);
    }

    #[test]
    fn trace_box_starting_inside_is_start_solid() {
        let world = collision(&valve_box([0, 0, 0], [64, 64, 64], "stone"), "");
        let trace = world.trace_box([32.0, 32.0, 32.0], [32.0, 32.0, 40.0], [4.0; 3]);
        assert!(trace.start_solid && trace.all_solid);
        assert_eq!(trace.fraction, 0.0);
    }

    #[test]
    fn trace_box_sliding_along_top_surface_not_blocked() {
        let world = collision(&valve_box([0, 0, 0], [256, 64, 32], "stone"), "");
        let z = 32.0 + 28.0 + 1.0 / 32.0;
        let trace = world.trace_box([32.0, 32.0, z], [200.0, 32.0, z], [16.0, 16.0, 28.0]);
        assert_eq!(trace.fraction, 1.0);
        assert!(!trace.start_solid);
    }

    #[test]
    fn clip_brush_blocks_liquid_and_trigger_do_not() {
        let blocking = collision(&valve_box([0, 0, 0], [64, 64, 64], "clip"), "");
        let water = collision(&valve_box([0, 0, 0], [64, 64, 64], "*water0"), "");
        let trigger = format!(
            "{{\n\"classname\" \"trigger_once\"\n{}}}\n",
            valve_box([0, 0, 0], [64, 64, 64], "trigger")
        );
        let trigger = collision("", &trigger);
        let ray =
            |world: &QuakeCollision| world.trace_ray([32.0, 32.0, 100.0], [32.0, 32.0, -100.0]);
        assert!(ray(&blocking).fraction < 1.0);
        assert_eq!(ray(&water).fraction, 1.0);
        assert_eq!(ray(&trigger).fraction, 1.0);
    }

    #[test]
    fn quake_engine_point_conversion_round_trips() {
        let quake = [64.0, -32.0, 96.0];
        assert_eq!(quake_to_engine(quake), [2.0, 3.0, 1.0]);
        assert_eq!(engine_to_quake(quake_to_engine(quake)), quake);
    }

    #[test]
    fn light_style_m_is_one_and_a_is_zero() {
        assert_eq!(light_style_brightness(0, 3.7), 1.0);
        assert_eq!(light_style_brightness(4, 0.0), 1.0);
        assert_eq!(light_style_brightness(4, 0.1), 0.0);
        assert_eq!(light_style_brightness(99, 0.0), 1.0);
    }

    #[test]
    fn animation_frames_listed_for_plus_textures_only() {
        assert_eq!(animation_frame_names("+0lava")[9], "+9lava");
        assert!(animation_frame_names("lava").is_empty());
    }
}
