//! Dedicated server (native, no window): the match with `EKAUQ_BOTS` bots (default 3) on
//! `EKAUQ_MAP` (default lqdm1), players joining over UDP `EKAUQ_HOST` (default 27500); every
//! session recorded to `build/sessions`. The match waits while nobody is connected.
//!
//! Run from `engine/`: `EKAUQ_HOST=27500 cargo run --release -p pill_game --example ekauq_server`

use std::time::{Duration, Instant};

use pill_game::{
    bots::Bots,
    loader::{bot_knowledge, load_arena, requested_map},
    net::{
        protocol::{MAX_CLIENTS, PORT},
        server::Host,
    },
    sim::{
        record::{record_to_file, Header},
        session::Session,
        STEP,
    },
};

/// Prints `log` records of level info and above.
struct Logger;

impl log::Log for Logger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("{} {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

fn setting<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let _ = log::set_logger(&Logger).map(|_| log::set_max_level(log::LevelFilter::Info));
    let map = requested_map();
    let bots: usize = setting("EKAUQ_BOTS", 3);
    let port: u16 = setting("EKAUQ_HOST", PORT).max(2);
    let (collision, arena, nav) = load_arena(&map).expect("map");
    let bots = bots.min(arena.spawns.len().saturating_sub(1));
    let players = 1 + MAX_CLIENTS;
    let mut session = Session::new(&arena, players, bots, 0.5, 1);
    let mut brains = Bots::new(&session, &arena, nav, bot_knowledge());
    let mut recorder = record_to_file(&Header::of(&session, &map));
    let mut host = Host::start(port, &map).expect("host");
    println!("ekauq_server: {map}, {bots} bots, UDP port {port}");
    let tick = Duration::from_secs_f32(STEP);
    let mut next = Instant::now();
    loop {
        for change in host.receive(tick, &mut session, &arena) {
            if let Some(recorder) = recorder.as_mut() {
                recorder.slot_change(change);
            }
        }
        if host.players() > 0 {
            let players = host.take_commands();
            let commands = brains.commands(&session, &collision, &arena, &players);
            let (events, ended) = session.advance(&collision, &arena, &commands);
            if let Some(recorder) = recorder.as_mut() {
                recorder.tick(&commands, &session);
            }
            host.send_tick(&session, &events);
            if let Some(ended) = ended {
                brains.set_level(ended.level);
                let text = format!(
                    "Match over: best player {}, best bot {}",
                    ended.best_player, ended.best_bot
                );
                println!("{text}");
                host.announce(&text);
            }
        }
        host.flush();
        next += tick;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else {
            next = now;
        }
    }
}
