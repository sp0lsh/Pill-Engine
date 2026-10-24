//! Bot tools (native), run from `engine/`:
//! `cargo run --release -p pill_game --example ekauq_tune -- <command>`
//!
//! - `nav [maps]`: builds and measures the navigation of maps
//! - `bench [map]`: simulation speed of a bot duel
//! - `calibrate`: measures the weapons, writes `src/bots/knowledge.txt`
//! - `tune [generations]`: evolves the genome by self-play on the tuning maps, writes the
//!   knowledge file after every generation
//! - `eval`: the knowledge's genome against the default one on the held-out maps
//! - `replay <file.ekr>`: re-simulates a recorded session, checks its checksums

use std::time::Instant;

use pill_game::{
    bots::{
        genome::{Genome, Knowledge, Skill, KNOWLEDGE},
        tune::{duel, tune, Settings},
    },
    loader::{bot_knowledge, load_arena},
    sim::{
        record::{read_header, replay},
        weapons::{hash, WEAPONS},
        STEP,
    },
};

const MAPS: [&str; 9] = [
    "lqdm1", "lqdm2", "lqdm3", "lqdm4", "lqdm5", "lqdm6", "lqdm7", "lqdm8", "lqdm9",
];
// Not lqdm4: its parts are joined by teleporters, which the game lacks
const TUNING_MAPS: [&str; 6] = ["lqdm1", "lqdm2", "lqdm3", "lqdm6", "lqdm7", "lqdm9"];
const HELD_OUT_MAPS: [&str; 2] = ["lqdm5", "lqdm8"];

fn knowledge_path() -> String {
    format!("{}/src/bots/knowledge.txt", env!("CARGO_MANIFEST_DIR"))
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |count| count.get())
}

fn nav(arguments: &[String]) {
    let maps: Vec<String> = if arguments.is_empty() {
        MAPS.iter().map(|map| map.to_string()).collect()
    } else {
        arguments.to_vec()
    };
    println!("map     spawns health  nodes  edges jumps  sims   traces    moves   seconds");
    for map in maps {
        let started = Instant::now();
        match load_arena(&map) {
            Ok((_, arena, nav)) => {
                let stats = &nav.stats;
                println!(
                    "{map:<7} {:>6} {:>6} {:>6} {:>6} {:>5} {:>5} {:>8} {:>8} {:>8.2}",
                    arena.spawns.len(),
                    arena.health.len(),
                    stats.nodes,
                    stats.edges,
                    stats.jumps,
                    stats.jump_sims,
                    stats.traces,
                    stats.moves,
                    started.elapsed().as_secs_f32(),
                );
            }
            Err(error) => println!("{map}: {error}"),
        }
    }
}

fn bench(arguments: &[String]) {
    let map = arguments.first().map_or("lqdm1", String::as_str);
    let loaded = load_arena(map).expect("map");
    let knowledge = bot_knowledge();
    let seconds = 180.0;
    let started = Instant::now();
    let outcome = duel(
        &loaded,
        &WEAPONS,
        &knowledge.calibration,
        [knowledge.genome, Genome::default()],
        Skill::at(1.0),
        seconds,
        1,
    );
    let wall = started.elapsed().as_secs_f32();
    println!(
        "{map}: {seconds} s duel in {wall:.2} s wall: {:.0}× real time on one core; frags {:?}, damage {:?}",
        seconds / wall,
        outcome.kills,
        outcome.dealt
    );
}

fn calibrate_command() {
    let started = Instant::now();
    let mut knowledge = Knowledge::parse(KNOWLEDGE);
    knowledge.recalibrate(&WEAPONS);
    for (weapon, curve) in WEAPONS.iter().zip(&knowledge.calibration.dps) {
        println!("{:<8} {curve:?}", weapon.name);
    }
    let text = knowledge.write(&WEAPONS, "calibrated");
    std::fs::write(knowledge_path(), text).expect("write knowledge");
    println!(
        "{} in {:.1} s",
        knowledge_path(),
        started.elapsed().as_secs_f32()
    );
}

fn tune_command(arguments: &[String]) {
    let generations = arguments
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(60);
    let maps: Vec<_> = TUNING_MAPS
        .iter()
        .map(|map| load_arena(map).expect("map"))
        .collect();
    let mut knowledge = bot_knowledge();
    let settings = Settings {
        generations,
        population: 20,
        parents: 5,
        seconds: 120.0,
        seeds: 2,
        threads: threads(),
        skill: Skill::at(1.0),
    };
    let started = Instant::now();
    let calibration = knowledge.calibration.clone();
    let path = knowledge_path();
    tune(
        &maps,
        &WEAPONS,
        &calibration,
        knowledge.genome,
        &settings,
        7,
        |generation, best, mean| {
            knowledge.genome = *mean;
            let provenance = format!(
                "generation {} of {generations}, best score {best:.2}, maps {}",
                generation + 1,
                TUNING_MAPS.join(" ")
            );
            std::fs::write(&path, knowledge.write(&WEAPONS, &provenance)).expect("write knowledge");
            println!(
                "generation {:>3}: best {best:+.2}  ({:.0} s)",
                generation + 1,
                started.elapsed().as_secs_f32()
            );
        },
    );
}

fn evaluate() {
    let knowledge = bot_knowledge();
    let mut wins = 0;
    let mut duels = 0;
    for map in HELD_OUT_MAPS {
        let loaded = load_arena(map).expect("map");
        for seed in 0..10 {
            let outcome = duel(
                &loaded,
                &WEAPONS,
                &knowledge.calibration,
                [knowledge.genome, Genome::default()],
                Skill::at(1.0),
                120.0,
                1000 + seed,
            );
            duels += 1;
            if outcome.score() > 0.0 {
                wins += 1;
            }
            println!(
                "{map} seed {seed}: frags {:+}, damage {:+.0}",
                outcome.frags, outcome.damage
            );
        }
    }
    println!(
        "tuned beats default in {wins} of {duels} duels ({:.0}%)",
        100.0 * wins as f32 / duels as f32
    );
}

fn replay_command(arguments: &[String]) {
    let Some(path) = arguments.first() else {
        eprintln!("usage: ekauq_tune replay <file.ekr>");
        return;
    };
    let bytes = std::fs::read(path).expect("recording");
    let mut reader = bytes.as_slice();
    let header = read_header(&mut reader).expect("header");
    println!(
        "{path}: map {}, {} players + {} bots, seed {}, build {}{}",
        header.map,
        header.players,
        header.bots,
        header.seed,
        header.build,
        if header.weapons == hash(&WEAPONS) {
            ""
        } else {
            ", weapons changed since"
        }
    );
    let (collision, arena, _) = load_arena(&header.map).expect("map");
    let started = Instant::now();
    let replayed = replay(&header, &mut reader, &collision, &arena).expect("replay");
    println!(
        "{} ticks ({:.0} s) in {:.2} s; checksums {} {}; frags {:?}",
        replayed.ticks,
        replayed.ticks as f32 * STEP,
        started.elapsed().as_secs_f32(),
        replayed.checksums,
        match replayed.diverged {
            None => "all equal".to_string(),
            Some(tick) => format!("diverged at tick {tick}"),
        },
        replayed.frags
    );
}

fn main() {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let rest = arguments.get(1..).unwrap_or(&[]);
    match arguments.first().map(String::as_str) {
        Some("nav") => nav(rest),
        Some("bench") => bench(rest),
        Some("calibrate") => calibrate_command(),
        Some("tune") => tune_command(rest),
        Some("eval") => evaluate(),
        Some("replay") => replay_command(rest),
        _ => eprintln!("usage: ekauq_tune nav [maps] | bench [map] | calibrate | tune [generations] | eval | replay <file>"),
    }
}
