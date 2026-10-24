//! What a map gives the deathmatch: spawn spots, health items, other item spots worth visiting.

use pill_engine::game::{
    quake_map::{parse_map, QuakeCollision, QuakeMap},
    quake_player::trace_player,
};

use super::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Spawn {
    /// Player origin standing on the floor.
    pub origin: Vec3,
    pub yaw: f32,
}

pub struct Arena {
    pub spawns: Vec<Spawn>,
    /// Standing spots of the health items.
    pub health: Vec<Vec3>,
    /// Standing spots of the other items and weapons: places to roam through.
    pub spots: Vec<Vec3>,
}

/// The player box dropped onto the floor below `origin`, if it isn't stuck in a wall.
pub fn drop_to_floor(collision: &QuakeCollision, origin: Vec3) -> Option<Vec3> {
    let down = trace_player(
        collision,
        origin,
        [origin[0], origin[1], origin[2] - 4096.0],
    );
    (!down.start_solid && down.fraction < 1.0).then_some(down.end)
}

fn vector(text: &str) -> Option<Vec3> {
    let mut parts = text.split_whitespace().map(|part| part.parse::<f32>().ok());
    Some([parts.next()??, parts.next()??, parts.next()??])
}

impl Arena {
    /// Deathmatch spawns (else the single player start), items.
    pub fn build(map: &QuakeMap, collision: &QuakeCollision) -> Self {
        let mut deathmatch = Vec::new();
        let mut starts = Vec::new();
        let mut health = Vec::new();
        let mut spots = Vec::new();
        for entity in &map.entities {
            let classname = entity.classname();
            let Some(origin) = entity.get("origin").and_then(vector) else {
                continue;
            };
            let yaw = entity
                .get("angle")
                .and_then(|angle| angle.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            match classname {
                "info_player_deathmatch" | "info_player_start" => {
                    let raised = [origin[0], origin[1], origin[2] + 8.0];
                    if let Some(origin) = drop_to_floor(collision, raised) {
                        let spawn = Spawn { origin, yaw };
                        if classname == "info_player_deathmatch" {
                            deathmatch.push(spawn);
                        } else {
                            starts.push(spawn);
                        }
                    }
                }
                _ if classname.starts_with("item_") || classname.starts_with("weapon_") => {
                    // Item origins sit at the item's bottom: stand the player box over it
                    let raised = [origin[0], origin[1], origin[2] + 40.0];
                    if let Some(spot) = drop_to_floor(collision, raised) {
                        if classname.starts_with("item_health") {
                            health.push(spot);
                        } else {
                            spots.push(spot);
                        }
                    }
                }
                _ => {}
            }
        }
        let spawns = if deathmatch.is_empty() {
            starts
        } else {
            deathmatch
        };
        Self::from_points(spawns, health, spots)
    }

    pub fn from_points(spawns: Vec<Spawn>, health: Vec<Vec3>, spots: Vec<Vec3>) -> Self {
        Self {
            spawns,
            health,
            spots,
        }
    }
}

/// Axis-aligned box brushes (Quake units) in a worldspawn, as a collision world: test levels and
/// the calibration range. Per axis the min face's three points (outward normal) and the same
/// points reversed on the max side.
pub fn box_world(boxes: &[([i32; 3], [i32; 3])]) -> QuakeCollision {
    let mut text = String::from("{\n\"classname\" \"worldspawn\"\n");
    for &([x0, y0, z0], [x1, y1, z1]) in boxes {
        text.push_str(&format!(
            "{{\n\
             ( {x0} {y0} {z0} ) ( {x0} {y1} {z0} ) ( {x0} {y0} {z1} ) stone 0 0 0 1 1\n\
             ( {x0} {y0} {z0} ) ( {x0} {y0} {z1} ) ( {x1} {y0} {z0} ) stone 0 0 0 1 1\n\
             ( {x0} {y0} {z0} ) ( {x1} {y0} {z0} ) ( {x0} {y1} {z0} ) stone 0 0 0 1 1\n\
             ( {x1} {y0} {z1} ) ( {x1} {y1} {z0} ) ( {x1} {y0} {z0} ) stone 0 0 0 1 1\n\
             ( {x1} {y1} {z0} ) ( {x0} {y1} {z1} ) ( {x0} {y1} {z0} ) stone 0 0 0 1 1\n\
             ( {x0} {y1} {z1} ) ( {x1} {y0} {z1} ) ( {x0} {y0} {z1} ) stone 0 0 0 1 1\n\
             }}\n"
        ));
    }
    text.push_str("}\n");
    QuakeCollision::from_map(&parse_map(&text).expect("box world map"))
}
