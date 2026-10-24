//! Session recordings: the header (map, seed, slots, weapons hash, build), then per tick every
//! fighter's command (bots included), players joining and leaving, and a state checksum every
//! second. Replaying the commands through a `Session` rebuilds the whole match (positions, hits,
//! frags) on the same build; the checksums tell when it doesn't. Bincode stream.

use std::io::{Read, Write};

use pill_engine::game::{quake_map::QuakeCollision, quake_player::PlayerInput};
use serde::{Deserialize, Serialize};

use super::{
    arena::Arena,
    combat::Command,
    session::{Session, SlotChange},
    weapons::{hash, WEAPONS},
    Vec3,
};

pub const VERSION: u32 = 1;
/// Ticks between checksums (a second).
pub const CHECKSUM_TICKS: u64 = 72;

/// A command as recorded and sent over the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WireCommand {
    pub forward: f32,
    pub side: f32,
    pub jump: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub fire: bool,
    pub select: Option<u8>,
}

impl From<&Command> for WireCommand {
    fn from(command: &Command) -> Self {
        Self {
            forward: command.input.forward,
            side: command.input.side,
            jump: command.input.jump,
            yaw: command.input.yaw,
            pitch: command.pitch,
            fire: command.fire,
            select: command.select.map(|slot| slot as u8),
        }
    }
}

impl From<&WireCommand> for Command {
    fn from(wire: &WireCommand) -> Self {
        Command {
            input: PlayerInput {
                forward: wire.forward,
                side: wire.side,
                jump: wire.jump,
                yaw: wire.yaw,
            },
            pitch: wire.pitch,
            fire: wire.fire,
            select: wire.select.map(usize::from),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub version: u32,
    /// The game's crate version: replays are exact on the same build.
    pub build: String,
    pub weapons: u64,
    pub map: String,
    pub seed: u32,
    pub players: usize,
    pub bots: usize,
    pub level: f32,
}

impl Header {
    pub fn of(session: &Session, map: &str) -> Self {
        Self {
            version: VERSION,
            build: env!("CARGO_PKG_VERSION").to_string(),
            weapons: hash(&WEAPONS),
            map: map.to_string(),
            seed: session.seed,
            players: session.players,
            bots: session.fight.fighters.len() - session.players,
            level: session.level,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Entry {
    Join {
        slot: u8,
        place: Option<(Vec3, f32)>,
    },
    Leave {
        slot: u8,
    },
    Tick {
        commands: Vec<WireCommand>,
    },
    Checksum {
        tick: u64,
        value: u64,
    },
}

pub struct Recorder<W: Write> {
    writer: W,
    failed: bool,
}

impl<W: Write> Recorder<W> {
    pub fn new(mut writer: W, header: &Header) -> Self {
        let failed = bincode::serialize_into(&mut writer, header).is_err();
        Self { writer, failed }
    }

    pub fn into_inner(self) -> W {
        self.writer
    }

    fn write(&mut self, entry: &Entry) {
        if self.failed {
            return;
        }
        if let Err(error) = bincode::serialize_into(&mut self.writer, entry) {
            log::warn!("recording stopped: {error}");
            self.failed = true;
        }
    }

    pub fn join(&mut self, slot: usize, place: Option<(Vec3, f32)>) {
        self.write(&Entry::Join {
            slot: slot as u8,
            place,
        });
    }

    pub fn leave(&mut self, slot: usize) {
        self.write(&Entry::Leave { slot: slot as u8 });
    }

    /// A remote player joined (at a spawn) or left.
    pub fn slot_change(&mut self, change: SlotChange) {
        match change {
            SlotChange::Join(slot) => self.join(slot, None),
            SlotChange::Leave(slot) => self.leave(slot),
        }
    }

    /// The commands of a tick just advanced; every `CHECKSUM_TICKS` the session's checksum.
    pub fn tick(&mut self, commands: &[Command], session: &Session) {
        self.write(&Entry::Tick {
            commands: commands.iter().map(WireCommand::from).collect(),
        });
        if session.tick % CHECKSUM_TICKS == 0 {
            self.write(&Entry::Checksum {
                tick: session.tick,
                value: session.checksum(),
            });
            let _ = self.writer.flush();
        }
    }
}

/// A recorder writing `build/sessions/<unix seconds>_<map>.ekr` (native).
#[cfg(not(target_arch = "wasm32"))]
pub fn record_to_file(header: &Header) -> Option<Recorder<std::io::BufWriter<std::fs::File>>> {
    let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/build/sessions");
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |time| time.as_secs());
    let path = format!("{directory}/{seconds}_{}.ekr", header.map);
    let file = std::fs::create_dir_all(directory)
        .and_then(|_| std::fs::File::create(&path))
        .inspect_err(|error| log::warn!("recording {path}: {error}"))
        .ok()?;
    log::info!("recording {path}");
    Some(Recorder::new(std::io::BufWriter::new(file), header))
}

/// What a replay found.
#[derive(Clone, Debug, PartialEq)]
pub struct Replayed {
    pub ticks: u64,
    pub checksums: usize,
    /// The first tick whose checksum differed, if any.
    pub diverged: Option<u64>,
    pub frags: Vec<i32>,
    pub damage: Vec<f32>,
}

pub fn read_header(reader: &mut impl Read) -> Result<Header, String> {
    let header: Header = bincode::deserialize_from(&mut *reader).map_err(|e| e.to_string())?;
    if header.version != VERSION {
        return Err(format!(
            "recording version {} (this build reads {VERSION})",
            header.version
        ));
    }
    Ok(header)
}

/// Replays the entries after `header` through a fresh session on the same map.
pub fn replay(
    header: &Header,
    reader: &mut impl Read,
    collision: &QuakeCollision,
    arena: &Arena,
) -> Result<Replayed, String> {
    let mut session = Session::new(
        arena,
        header.players,
        header.bots,
        header.level,
        header.seed,
    );
    let mut checksums = 0;
    let mut diverged = None;
    loop {
        let entry: Entry = match bincode::deserialize_from(&mut *reader) {
            Ok(entry) => entry,
            Err(error) => match *error {
                bincode::ErrorKind::Io(ref io)
                    if io.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break
                }
                _ => return Err(error.to_string()),
            },
        };
        match entry {
            Entry::Join { slot, place } => session.join(arena, slot as usize, place),
            Entry::Leave { slot } => session.leave(slot as usize),
            Entry::Tick { commands } => {
                let commands: Vec<Command> = commands.iter().map(Command::from).collect();
                session.advance(collision, arena, &commands);
            }
            Entry::Checksum { tick, value } => {
                checksums += 1;
                if diverged.is_none() && (tick != session.tick || value != session.checksum()) {
                    diverged = Some(tick);
                }
            }
        }
    }
    Ok(Replayed {
        ticks: session.tick,
        checksums,
        diverged,
        frags: session
            .fight
            .fighters
            .iter()
            .map(|fighter| fighter.frags)
            .collect(),
        damage: session
            .fight
            .fighters
            .iter()
            .map(|fighter| fighter.damage_dealt)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        arena::{box_world, Spawn},
        STEP,
    };
    use super::*;
    use crate::bots::{
        calibrate::calibrate,
        genome::{Genome, Knowledge},
        nav::NavGraph,
        Bots,
    };

    #[test]
    fn recorded_session_replays_with_equal_checksums() {
        let collision = box_world(&[
            ([-1024, -1024, -64], [1024, 1024, 0]),
            ([-64, -256, 0], [64, 256, 128]),
        ]);
        let z = 24.0 + 1.0 / 32.0;
        let spawns = [
            [-512.0, -512.0],
            [512.0, 512.0],
            [-512.0, 512.0],
            [512.0, -512.0],
        ]
        .iter()
        .map(|&[x, y]| Spawn {
            origin: [x, y, z],
            yaw: 0.0,
        })
        .collect();
        let arena = Arena::from_points(spawns, Vec::new(), Vec::new());
        let knowledge = Knowledge {
            weapons: hash(&WEAPONS),
            calibration: calibrate(&WEAPONS, 1),
            genome: Genome::default(),
        };
        let mut session = Session::new(&arena, 1, 3, 1.0, 9);
        let nav = NavGraph::of(&collision, &arena);
        let mut bots = Bots::new(&session, &arena, nav, knowledge);
        let header = Header::of(&session, "test");
        let mut recorder = Recorder::new(Vec::new(), &header);
        // 60 s; the player joins after a second and runs in a circle, firing
        for tick in 0..(60.0 / STEP) as u64 {
            if tick == 72 {
                let place = Some(([0.0, -600.0, z], 90.0));
                session.join(&arena, 0, place);
                recorder.join(0, place);
            }
            let mut command = Command::default();
            command.input.forward = 1.0;
            command.input.yaw = (tick as f32 * 2.0).rem_euclid(360.0);
            command.fire = true;
            let commands = bots.commands(&session, &collision, &arena, &[(0, command)]);
            session.advance(&collision, &arena, &commands);
            recorder.tick(&commands, &session);
        }
        let bytes = recorder.into_inner();
        let mut reader = bytes.as_slice();
        let read = read_header(&mut reader).unwrap();
        assert_eq!(read, header);
        let replayed = replay(&read, &mut reader, &collision, &arena).unwrap();
        assert_eq!(replayed.ticks, session.tick);
        assert_eq!(replayed.checksums, 60);
        assert_eq!(replayed.diverged, None);
        let frags: Vec<i32> = session.fight.fighters.iter().map(|f| f.frags).collect();
        assert_eq!(replayed.frags, frags);
        assert!(frags.iter().any(|&f| f != 0), "nobody scored: {frags:?}");
    }
}
