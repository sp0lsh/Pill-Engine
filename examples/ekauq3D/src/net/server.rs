//! The server side, in a listen server (the hosting game) or the dedicated server: accepts
//! players into free slots of the session, applies their latest commands each tick, sends
//! events every tick and snapshots every `SNAPSHOT_TICKS`.

use std::time::Duration;

use pill_core::{
    server_flush, server_start, server_update, NetworkServer, RELIABLE_CHANNEL_ID,
    UNRELIABLE_CHANNEL_ID,
};

use super::protocol::{
    decode, encode, ClientMessage, ServerMessage, WireFighter, MAX_CLIENTS, SNAPSHOT_TICKS,
};
use crate::sim::{
    arena::Arena,
    combat::{Command, Event},
    session::{Session, SlotChange},
    weapons::{hash, WEAPONS},
};

struct Remote {
    id: u64,
    slot: Option<usize>,
    /// The last command applied, and its sequence number.
    command: Command,
    ack: u32,
}

pub struct Host {
    net: NetworkServer,
    remotes: Vec<Remote>,
    map: String,
}

impl Host {
    /// Listens on UDP `port` (all interfaces) for the match on `map`.
    pub fn start(port: u16, map: &str) -> Result<Self, String> {
        let net =
            server_start(&format!("0.0.0.0:{port}"), MAX_CLIENTS).map_err(|e| e.to_string())?;
        log::info!("ekauq3D: hosting {map} on UDP port {port}");
        Ok(Self {
            net,
            remotes: Vec::new(),
            map: map.to_string(),
        })
    }

    fn send(&mut self, id: u64, channel: u8, message: &ServerMessage) {
        self.net.server.send_message(id, channel, encode(message));
    }

    /// Network transport, players connecting (into free player slots after a matching hello),
    /// leaving, and their commands. The slots that changed.
    pub fn receive(
        &mut self,
        dt: Duration,
        session: &mut Session,
        arena: &Arena,
    ) -> Vec<SlotChange> {
        let mut changes = Vec::new();
        if let Err(error) = server_update(&mut self.net, dt) {
            log::warn!("ekauq3D: server: {error}");
        }
        let connected = self.net.server.clients_id();
        self.remotes.retain(|remote| {
            let stays = connected.contains(&remote.id);
            if !stays {
                if let Some(slot) = remote.slot {
                    session.leave(slot);
                    changes.push(SlotChange::Leave(slot));
                    log::info!("ekauq3D: player left slot {slot}");
                }
            }
            stays
        });
        for id in connected {
            if !self.remotes.iter().any(|remote| remote.id == id) {
                self.remotes.push(Remote {
                    id,
                    slot: None,
                    command: Command::default(),
                    ack: 0,
                });
            }
        }
        for index in 0..self.remotes.len() {
            let id = self.remotes[index].id;
            while let Some(bytes) = self.net.server.receive_message(id, RELIABLE_CHANNEL_ID) {
                let Some(ClientMessage::Hello {
                    build,
                    weapons,
                    map,
                }) = decode(&bytes)
                else {
                    continue;
                };
                let refused = if map != self.map {
                    Some(format!(
                        "the server plays {}, not {map} (EKAUQ_MAP)",
                        self.map
                    ))
                } else if weapons != hash(&WEAPONS) || build != env!("CARGO_PKG_VERSION") {
                    Some("a different game build".to_string())
                } else {
                    None
                };
                let free = (1..session.players).find(|&slot| {
                    !session.fight.fighters[slot].active
                        && !self.remotes.iter().any(|remote| remote.slot == Some(slot))
                });
                match (refused, free) {
                    (None, Some(slot)) => {
                        session.join(arena, slot, None);
                        changes.push(SlotChange::Join(slot));
                        self.remotes[index].slot = Some(slot);
                        log::info!("ekauq3D: player joined slot {slot}");
                        let welcome = ServerMessage::Welcome {
                            slot: slot as u8,
                            fighters: session.fight.fighters.len() as u8,
                            players: session.players as u8,
                            level: session.level,
                        };
                        self.send(id, RELIABLE_CHANNEL_ID, &welcome);
                    }
                    (refused, _) => {
                        let reason = refused.unwrap_or_else(|| "the server is full".to_string());
                        self.send(id, RELIABLE_CHANNEL_ID, &ServerMessage::Refused { reason });
                    }
                }
            }
            while let Some(bytes) = self.net.server.receive_message(id, UNRELIABLE_CHANNEL_ID) {
                let Some(ClientMessage::Commands { seq, recent }) = decode(&bytes) else {
                    continue;
                };
                let remote = &mut self.remotes[index];
                if seq <= remote.ack {
                    continue;
                }
                // A weapon switch in a lost command still counts
                let select = recent
                    .iter()
                    .take((seq - remote.ack) as usize)
                    .find_map(|command| command.select);
                if let Some(latest) = recent.first() {
                    remote.command = Command::from(latest);
                    remote.command.select = select.map(usize::from);
                    remote.ack = seq;
                }
            }
        }
        changes
    }

    /// The remote players' commands for this tick (a weapon switch only once).
    pub fn take_commands(&mut self) -> Vec<(usize, Command)> {
        self.remotes
            .iter_mut()
            .filter_map(|remote| {
                let slot = remote.slot?;
                let command = remote.command;
                remote.command.select = None;
                Some((slot, command))
            })
            .collect()
    }

    /// The players in the match: (client id, last command applied).
    fn joined(&self) -> Vec<(u64, u32)> {
        self.remotes
            .iter()
            .filter(|remote| remote.slot.is_some())
            .map(|remote| (remote.id, remote.ack))
            .collect()
    }

    /// One reliable message to every player in the match (encoded once).
    fn broadcast(&mut self, message: &ServerMessage) {
        let bytes = encode(message);
        for (id, _) in self.joined() {
            self.net
                .server
                .send_message(id, RELIABLE_CHANNEL_ID, bytes.clone());
        }
    }

    /// After a tick: its events to everyone (reliable), a snapshot every `SNAPSHOT_TICKS`.
    pub fn send_tick(&mut self, session: &Session, events: &[Event]) {
        if !events.is_empty() {
            self.broadcast(&ServerMessage::Events {
                tick: session.tick,
                events: events.to_vec(),
            });
        }
        if session.tick % SNAPSHOT_TICKS != 0 {
            return;
        }
        let fighters: Vec<WireFighter> =
            session.fight.fighters.iter().map(WireFighter::of).collect();
        for (id, ack) in self.joined() {
            let snapshot = ServerMessage::Snapshot {
                tick: session.tick,
                ack,
                clock: session.clock,
                fighters: fighters.clone(),
            };
            self.send(id, UNRELIABLE_CHANNEL_ID, &snapshot);
        }
    }

    /// A text for everyone (match results).
    pub fn announce(&mut self, text: &str) {
        self.broadcast(&ServerMessage::Message {
            text: text.to_string(),
        });
    }

    pub fn flush(&mut self) {
        let _ = server_flush(&mut self.net);
    }

    pub fn players(&self) -> usize {
        self.joined().len()
    }
}
