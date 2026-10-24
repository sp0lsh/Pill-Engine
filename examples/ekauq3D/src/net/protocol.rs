//! Messages between the game's clients and the server (listen or dedicated), bincode over the
//! engine's renet channels: commands up (unreliable, the last few repeated against loss),
//! snapshots down (unreliable, every few ticks), events and joining reliable.

use pill_engine::game::quake_player::PlayerState;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::sim::{combat::Event, combat::Fighter, record::WireCommand, Vec3};

/// Default UDP port.
pub const PORT: u16 = 27500;
/// Remote players a server takes.
pub const MAX_CLIENTS: usize = 7;
/// A snapshot every this many ticks (24 per second).
pub const SNAPSHOT_TICKS: u64 = 3;
/// Commands repeated in each message (lost packets).
pub const REPEATED_COMMANDS: usize = 3;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMessage {
    Hello {
        build: String,
        weapons: u64,
        map: String,
    },
    /// Command `seq` first, then the ones before it.
    Commands { seq: u32, recent: Vec<WireCommand> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerMessage {
    Welcome {
        slot: u8,
        fighters: u8,
        players: u8,
        level: f32,
    },
    Refused {
        reason: String,
    },
    /// Everyone's state after `tick`; `ack` the last command of this client applied.
    Snapshot {
        tick: u64,
        ack: u32,
        clock: f32,
        fighters: Vec<WireFighter>,
    },
    Events {
        tick: u64,
        events: Vec<Event>,
    },
    Message {
        text: String,
    },
}

/// A fighter's state on the wire (enough to predict one's own movement from it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WireFighter {
    pub origin: Vec3,
    pub velocity: Vec3,
    pub on_ground: bool,
    pub jump_held: bool,
    pub jump_buffered: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub health: f32,
    pub weapon: u8,
    pub alive: bool,
    pub active: bool,
    pub frags: i16,
}

impl WireFighter {
    pub fn of(fighter: &Fighter) -> Self {
        Self {
            origin: fighter.state.origin,
            velocity: fighter.state.velocity,
            on_ground: fighter.state.on_ground,
            jump_held: fighter.state.jump_held,
            jump_buffered: fighter.state.jump_buffered,
            yaw: fighter.yaw,
            pitch: fighter.pitch,
            health: fighter.health,
            weapon: fighter.weapon as u8,
            alive: fighter.alive,
            active: fighter.active,
            frags: fighter.frags as i16,
        }
    }

    pub fn state(&self) -> PlayerState {
        PlayerState {
            origin: self.origin,
            velocity: self.velocity,
            on_ground: self.on_ground,
            jump_held: self.jump_held,
            jump_buffered: self.jump_buffered,
        }
    }

    /// Copies this onto a fighter (a client's mirror of the match).
    pub fn apply(&self, fighter: &mut Fighter) {
        fighter.state = self.state();
        fighter.yaw = self.yaw;
        fighter.pitch = self.pitch;
        fighter.health = self.health;
        fighter.weapon = self.weapon as usize;
        fighter.alive = self.alive;
        fighter.active = self.active;
        fighter.frags = self.frags as i32;
    }
}

pub fn encode(message: &impl Serialize) -> Vec<u8> {
    bincode::serialize(message).expect("messages serialize")
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Option<T> {
    bincode::deserialize(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::combat::ImpactKind;

    #[test]
    fn messages_round_trip() {
        let messages = [
            ServerMessage::Snapshot {
                tick: 7,
                ack: 3,
                clock: 1.5,
                fighters: vec![WireFighter {
                    origin: [1.0, 2.0, 3.0],
                    health: 75.0,
                    alive: true,
                    ..WireFighter::default()
                }],
            },
            ServerMessage::Events {
                tick: 8,
                events: vec![Event::Impact {
                    by: 1,
                    point: [0.0; 3],
                    normal: [0.0, 0.0, 1.0],
                    texture: Some(4),
                    kind: ImpactKind::Wall,
                }],
            },
        ];
        for message in messages {
            assert_eq!(decode::<ServerMessage>(&encode(&message)), Some(message));
        }
        let command = ClientMessage::Commands {
            seq: 9,
            recent: vec![WireCommand {
                forward: 1.0,
                yaw: 90.0,
                fire: true,
                ..WireCommand::default()
            }],
        };
        assert_eq!(decode::<ClientMessage>(&encode(&command)), Some(command));
    }
}
