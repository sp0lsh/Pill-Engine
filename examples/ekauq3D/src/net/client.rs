//! The client side: joins a server, sends the player's command every tick, predicts the player's
//! own movement (the same `player_move`, replaying the commands the server hasn't applied yet on
//! top of its latest state) and shows the other fighters interpolated between snapshots.

use std::collections::VecDeque;
use std::time::Duration;

use pill_core::{
    client_connect, client_flush, client_update, NetworkClient, RELIABLE_CHANNEL_ID,
    UNRELIABLE_CHANNEL_ID,
};
use pill_engine::game::{
    quake_map::QuakeCollision,
    quake_player::{player_move, MoveEvents, MoveSettings, PlayerState},
};

use super::protocol::{
    decode, encode, ClientMessage, ServerMessage, WireFighter, REPEATED_COMMANDS,
};
use crate::sim::{
    combat::{Command, Event, Match},
    record::WireCommand,
    weapons::{hash, WEAPONS},
    STEP,
};

/// Commands kept for prediction while the server hasn't applied them (2 s).
const MAX_UNACKED: usize = 144;
/// Other fighters are shown this far in the past (seconds), between two snapshots.
const INTERPOLATION_DELAY: f32 = 0.1;
const KEPT_SNAPSHOTS: usize = 16;

pub struct Welcome {
    pub slot: usize,
    pub fighters: usize,
    pub players: usize,
    pub level: f32,
}

pub struct Connection {
    net: NetworkClient,
    map: String,
    hello_sent: bool,
    pub welcome: Option<Welcome>,
    pub refused: Option<String>,
    /// Commands sent, not yet applied by the server (sequence, command).
    unacked: VecDeque<(u32, Command)>,
    seq: u32,
    /// (arrival time, fighters) of recent snapshots, oldest first.
    snapshots: VecDeque<(f32, Vec<WireFighter>)>,
    last_tick: u64,
    pub clock: f32,
    /// The player's own movement, predicted.
    pub predicted: PlayerState,
    pub moved: MoveEvents,
    pub events: Vec<Event>,
    pub messages: Vec<String>,
    time: f32,
    settings: MoveSettings,
}

impl Connection {
    pub fn connect(address: &str, map: &str) -> Result<Self, String> {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |time| time.as_nanos() as u64 | 1);
        let net = client_connect(address, id).map_err(|e| e.to_string())?;
        log::info!("ekauq3D: connecting to {address}");
        Ok(Self {
            net,
            map: map.to_string(),
            hello_sent: false,
            welcome: None,
            refused: None,
            unacked: VecDeque::new(),
            seq: 0,
            snapshots: VecDeque::new(),
            last_tick: 0,
            clock: 0.0,
            predicted: PlayerState::default(),
            moved: MoveEvents::default(),
            events: Vec::new(),
            messages: Vec::new(),
            time: 0.0,
            settings: MoveSettings::default(),
        })
    }

    pub fn slot(&self) -> Option<usize> {
        self.welcome.as_ref().map(|welcome| welcome.slot)
    }

    fn own(&self) -> Option<&WireFighter> {
        let slot = self.slot()?;
        self.snapshots
            .back()
            .and_then(|(_, fighters)| fighters.get(slot))
    }

    /// Network transport and the server's messages; then the player's predicted movement redone
    /// from the newest snapshot.
    pub fn receive(&mut self, dt: f32, collision: &QuakeCollision) {
        self.time += dt;
        if let Err(error) = client_update(&mut self.net, Duration::from_secs_f32(dt)) {
            log::warn!("ekauq3D: client: {error}");
        }
        if self.net.client.is_connected() && !self.hello_sent {
            let hello = ClientMessage::Hello {
                build: env!("CARGO_PKG_VERSION").to_string(),
                weapons: hash(&WEAPONS),
                map: self.map.clone(),
            };
            self.net
                .client
                .send_message(RELIABLE_CHANNEL_ID, encode(&hello));
            self.hello_sent = true;
        }
        while let Some(bytes) = self.net.client.receive_message(RELIABLE_CHANNEL_ID) {
            match decode::<ServerMessage>(&bytes) {
                Some(ServerMessage::Welcome {
                    slot,
                    fighters,
                    players,
                    level,
                }) => {
                    log::info!("ekauq3D: joined as slot {slot} of {fighters}");
                    self.welcome = Some(Welcome {
                        slot: slot as usize,
                        fighters: fighters as usize,
                        players: players as usize,
                        level,
                    });
                }
                Some(ServerMessage::Refused { reason }) => {
                    log::warn!("ekauq3D: server refused: {reason}");
                    self.refused = Some(reason);
                }
                Some(ServerMessage::Events { events, .. }) => self.events.extend(events),
                Some(ServerMessage::Message { text }) => self.messages.push(text),
                _ => {}
            }
        }
        let mut newest = None;
        while let Some(bytes) = self.net.client.receive_message(UNRELIABLE_CHANNEL_ID) {
            if let Some(ServerMessage::Snapshot {
                tick,
                ack,
                clock,
                fighters,
            }) = decode(&bytes)
            {
                if tick > self.last_tick {
                    self.last_tick = tick;
                    self.clock = clock;
                    newest = Some(ack);
                    self.snapshots.push_back((self.time, fighters));
                    while self.snapshots.len() > KEPT_SNAPSHOTS {
                        self.snapshots.pop_front();
                    }
                }
            }
        }
        if let Some(ack) = newest {
            self.reconcile(ack, collision);
        }
    }

    /// The server's state of the player plus the commands it hasn't applied yet.
    fn reconcile(&mut self, ack: u32, collision: &QuakeCollision) {
        while self.unacked.front().is_some_and(|(seq, _)| *seq <= ack) {
            self.unacked.pop_front();
        }
        let Some(own) = self.own().copied() else {
            return;
        };
        self.predicted = own.state();
        if !own.alive {
            return;
        }
        for (_, command) in &self.unacked {
            player_move(
                &mut self.predicted,
                &command.input,
                STEP,
                collision,
                &self.settings,
            );
        }
    }

    /// Sends this tick's command (with the last few, against loss) and moves the prediction.
    pub fn send_command(&mut self, command: Command, collision: &QuakeCollision) {
        self.seq += 1;
        self.unacked.push_back((self.seq, command));
        if self.unacked.len() > MAX_UNACKED {
            self.unacked.pop_front();
        }
        let recent: Vec<WireCommand> = self
            .unacked
            .iter()
            .rev()
            .take(REPEATED_COMMANDS)
            .map(|(_, command)| WireCommand::from(command))
            .collect();
        let message = ClientMessage::Commands {
            seq: self.seq,
            recent,
        };
        self.net
            .client
            .send_message(UNRELIABLE_CHANNEL_ID, encode(&message));
        if self.own().is_some_and(|own| own.alive) {
            self.moved = player_move(
                &mut self.predicted,
                &command.input,
                STEP,
                collision,
                &self.settings,
            );
        }
    }

    /// The match as this client shows it: the others `INTERPOLATION_DELAY` in the past between
    /// two snapshots, the player itself predicted.
    pub fn mirror(&self, fight: &mut Match) {
        let Some(slot) = self.slot() else {
            return;
        };
        let shown = self.time - INTERPOLATION_DELAY;
        let after = self.snapshots.iter().position(|(time, _)| *time >= shown);
        let (from, to, t) = match after {
            Some(0) | None => {
                let last = self.snapshots.len().saturating_sub(1);
                (last, last, 0.0)
            }
            Some(index) => {
                let (t0, t1) = (self.snapshots[index - 1].0, self.snapshots[index].0);
                (
                    index - 1,
                    index,
                    ((shown - t0) / (t1 - t0).max(1e-4)).clamp(0.0, 1.0),
                )
            }
        };
        let (Some((_, older)), Some((_, newer))) =
            (self.snapshots.get(from), self.snapshots.get(to))
        else {
            return;
        };
        for (index, fighter) in fight.fighters.iter_mut().enumerate() {
            let (Some(a), Some(b)) = (older.get(index), newer.get(index)) else {
                continue;
            };
            if index == slot {
                // Its own state from the newest snapshot, its movement predicted
                if let Some((_, latest)) = self.snapshots.back() {
                    latest[index].apply(fighter);
                }
                fighter.state = self.predicted;
                continue;
            }
            b.apply(fighter);
            // A respawn jumps, no sliding across the map
            if crate::sim::distance(a.origin, b.origin) < 128.0 {
                fighter.state.origin = std::array::from_fn(|axis| {
                    a.origin[axis] + (b.origin[axis] - a.origin[axis]) * t
                });
                fighter.yaw = a.yaw + crate::sim::angle_delta(a.yaw, b.yaw) * t;
            }
        }
    }

    pub fn flush(&mut self) {
        let _ = client_flush(&mut self.net);
    }
}
