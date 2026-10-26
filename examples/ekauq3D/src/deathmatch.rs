//! The deathmatch in the game: the player (fighter 0, in player mode) against `EKAUQ_BOTS` bots
//! (`?bots=` on the web), all under `sim::combat`'s rules; the bots drawn with LibreQuake's
//! `player.mdl`, a HUD, matches to 10 frags or 5 minutes, and the bots' difficulty adapting to
//! the player after each match (native: kept in `build/bot_skill.txt`).

use std::{ops::Range, sync::Mutex};

use pill_engine::{
    define_component,
    game::{
        quake_map::{quake_direction_to_engine, quake_to_engine, QuakeCollision},
        quake_mdl::QuakeModel,
        quake_player::PlayerInput,
        *,
    },
};

use crate::{
    bots::{genome::Knowledge, nav::NavGraph, Bots},
    game::LevelState,
    player::{euler_from_columns, DeathView, Mode, PlayerEvents, PlayerRig},
    sim::{
        arena::Arena,
        combat::{Command, Event, ImpactKind},
        distance, dot, horizontal_distance,
        session::{Session, TIME_LIMIT},
        sub, view_basis,
        weapons::{Kind, WEAPONS},
        STEP,
    },
    sounds::{self, Sfx},
    weapon::{self, is_metal, Impact, Weapon, WeaponModel},
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{
    net::{client::Connection, protocol::MAX_CLIENTS, server::Host},
    sim::record::{record_to_file, Header, Recorder},
};

/// Bodies created for the other fighters (players and bots).
const MAX_BODIES: usize = 24;
/// Ticks simulated at most per frame (a long frame drops the rest).
const MAX_TICKS_PER_FRAME: usize = 8;
const DEFAULT_LEVEL: f32 = 0.5;
const MAX_BOTS: usize = 15;
const ANIMATION_RATE: f32 = 10.0;
/// Seconds a body without death frames takes to tip over.
const TIP_OVER_TIME: f32 = 0.3;
const MESSAGE_TIME: f32 = 6.0;
/// Corpses left lying (the oldest is reused), seconds before one sinks away, sinking time.
const MAX_CORPSES: usize = 16;
const CORPSE_TIME: f32 = 30.0;
const CORPSE_SINK_TIME: f32 = 1.0;
/// Pain sounds at most this often; the hit tick this long after your hit (seconds).
const HIT_TICK_DELAY: f32 = 0.08;
const PAIN_INTERVAL: f32 = 0.4;
/// The view kick per point of damage (degrees per second).
const HIT_KICK: f32 = 3.0;
/// Blasts closer than this shake the view (units); kick at the blast (degrees per second).
const BLAST_SHAKE_RADIUS: f32 = 400.0;
const BLAST_KICK: f32 = 150.0;
/// A killing hit this strong tears the victim apart (gib sound).
const GIB_DAMAGE: f32 = 40.0;
/// Kill feed: lines kept, seconds shown.
const FEED_LINES: usize = 4;
const FEED_TIME: f32 = 4.0;
/// The health number: five times the HUD's text.
const HEALTH_FONT_SIZE: f32 = 80.0;

/// The HUD of this frame (the egui overlay has no engine access).
#[derive(Clone, Default)]
pub struct Hud {
    pub playing: bool,
    pub health: f32,
    pub weapon: &'static str,
    pub dead: bool,
    /// (name, frags), best first.
    pub scores: Vec<(String, i32)>,
    pub time_left: f32,
    pub level: f32,
    pub message: Option<String>,
    /// Seconds the message has been shown (it slides in).
    pub message_age: f32,
    /// 0..1, just hurt / just fragged someone: the health and score pop, the screen reddens.
    pub damage_pulse: f32,
    pub frag_pulse: f32,
    /// Kill feed: text, involving the player, seconds shown.
    pub feed: Vec<(String, bool, f32)>,
}

pub static HUD: Mutex<Option<Hud>> = Mutex::new(None);

// A bot's body: fighter `index`, seconds in its life or death, its deaths (which death
// animation), the mesh shown
define_component!(BotModel {
    index: usize,
    time: f32,
    dead: bool,
    deaths: usize,
    frame: usize,
    // Where it died (its corpse stays there after the respawn)
    died_at: [f32; 3],
    died_yaw: f32,
    // A corpse (`index` is `CORPSE`): seconds lying, `None` while unused
    corpse: Option<f32>,
});

/// `BotModel::index` of corpses: bodies left lying where fighters died, after their respawn.
const CORPSE: usize = usize::MAX;

/// The player model's frames by animation (indices into its uploaded meshes).
#[derive(Default)]
struct Animations {
    run: Vec<usize>,
    stand: Vec<usize>,
    shoot: Vec<usize>,
    swing: Vec<usize>,
    pain: Vec<usize>,
    /// Several deaths, all ending lying on the ground.
    deaths: Vec<Vec<usize>>,
}

// id's player.mdl frames, for a model whose frames have no animation names (LibreQuake's are
// `frame1`..`frame143`, in id's order)
const RUN_FRAMES: Range<usize> = 6..12;
const STAND_FRAMES: Range<usize> = 12..17;
const SHOOT_FRAMES: Range<usize> = 113..119;
const SWING_FRAMES: Range<usize> = 119..125;
const PAIN_FRAMES: Range<usize> = 35..41;
const DEATH_FRAMES: [Range<usize>; 5] = [50..61, 61..70, 70..85, 85..94, 94..103];
const DEATH_PREFIXES: [&str; 6] = ["deatha", "deathb", "deathc", "deathd", "deathe", "axdeth"];

impl Animations {
    /// Frames named `<prefix><number>`, for the first of `prefixes` that has any.
    fn sequence(names: &[String], prefixes: &[&str]) -> Option<Vec<usize>> {
        for prefix in prefixes {
            let frames: Vec<usize> = names
                .iter()
                .enumerate()
                .filter(|(_, name)| {
                    name.strip_prefix(prefix).is_some_and(|rest| {
                        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
                    })
                })
                .map(|(index, _)| index)
                .collect();
            if !frames.is_empty() {
                return Some(frames);
            }
        }
        None
    }

    fn new(names: &[String]) -> Self {
        // By name, else id's frame range, else the first frame
        let named = |prefixes: &[&str], frames: Range<usize>| {
            Self::sequence(names, prefixes).unwrap_or_else(|| {
                if frames.end <= names.len() {
                    frames.collect()
                } else {
                    vec![0]
                }
            })
        };
        let mut deaths: Vec<Vec<usize>> = DEATH_PREFIXES
            .iter()
            .filter_map(|prefix| Self::sequence(names, &[prefix]))
            .collect();
        if deaths.is_empty() {
            deaths = DEATH_FRAMES
                .iter()
                .filter(|frames| frames.end <= names.len())
                .map(|frames| frames.clone().collect())
                .collect();
        }
        if deaths.is_empty() {
            deaths.push(vec![0]);
        }
        Self {
            run: named(&["rockrun", "run", "axrun"], RUN_FRAMES),
            stand: named(&["stand", "axstnd"], STAND_FRAMES),
            shoot: named(&["shotatt", "nailatt", "attack"], SHOOT_FRAMES),
            swing: named(&["axatt", "axattb"], SWING_FRAMES),
            pain: named(&["pain", "axpain"], PAIN_FRAMES),
            deaths,
        }
    }
}

pub struct Deathmatch {
    pub session: Session,
    pub arena: Arena,
    bots: Bots,
    /// Seconds not yet simulated (fixed ticks).
    accumulator: f32,
    /// The player's origin before the last tick: drawn in between.
    previous_origin: [f32; 3],
    message: Option<(String, f32)>,
    model: Option<(WeaponModel, Animations)>,
    /// The local player's fighter.
    me: usize,
    /// Per fighter, seconds since it was last hit (flinch, pain sounds).
    since_hit: Vec<f32>,
    damage_pulse: f32,
    frag_pulse: f32,
    message_age: f32,
    feed: Vec<(String, bool, f32)>,
    /// Seconds until the hit tick of your last hit plays (after the shot's own sound).
    hit_tick_in: Option<f32>,
    #[cfg(not(target_arch = "wasm32"))]
    recorder: Option<Recorder<std::io::BufWriter<std::fs::File>>>,
    /// Listen server: remote players join this match.
    #[cfg(not(target_arch = "wasm32"))]
    host: Option<Host>,
    /// Client: the match is the server's, this shows it.
    #[cfg(not(target_arch = "wasm32"))]
    connection: Option<Connection>,
}

/// An effect of this frame's events, positions resolved (Quake space).
enum Juice {
    /// Your shot hit someone.
    HitConfirm,
    /// Someone was hurt: you (`None`) or someone there.
    Pain(Option<[f32; 3]>),
    /// Your view jolts (degrees per second of pitch and roll).
    Kick {
        pitch: f32,
        roll: f32,
    },
    Death {
        at: [f32; 3],
        gib: bool,
    },
    Burst([f32; 3]),
    /// Your pellets' streaks to these points.
    Tracers(Vec<[f32; 3]>),
    Mark {
        point: [f32; 3],
        normal: [f32; 3],
    },
}

/// Where the match runs.
pub enum Network {
    Local,
    /// Listen server on this UDP port.
    Host(u16),
    /// Join the server at this address.
    Connect(String),
}

#[cfg(not(target_arch = "wasm32"))]
fn skill_path() -> String {
    format!("{}/build/bot_skill.txt", env!("CARGO_MANIFEST_DIR"))
}

fn load_level() -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(text) = std::fs::read_to_string(skill_path()) {
        if let Some(level) = text
            .split_whitespace()
            .skip_while(|word| *word != "level")
            .nth(1)
            .and_then(|value| value.parse::<f32>().ok())
        {
            return level.clamp(0.0, 1.0);
        }
    }
    DEFAULT_LEVEL
}

fn save_level(level: f32) {
    #[cfg(not(target_arch = "wasm32"))]
    if let Err(error) = std::fs::write(skill_path(), format!("level {level:.3}\n")) {
        log::warn!("ekauq3D: {}: {error}", skill_path());
    }
    #[cfg(target_arch = "wasm32")]
    let _ = level;
}

fn fighter_name(session: &Session, index: usize, me: usize) -> String {
    if index == me {
        "you".to_string()
    } else if index == 0 {
        "host".to_string()
    } else if index < session.players {
        format!("player{index}")
    } else {
        format!("bot{}", index + 1 - session.players)
    }
}

/// Uploads the frames of `player.mdl` the animations use (meshes are a budget).
fn upload_player(engine: &mut Engine, model: QuakeModel) -> Result<(WeaponModel, Animations)> {
    let names: Vec<String> = model
        .frames
        .iter()
        .map(|frame| frame.name.clone())
        .collect();
    let animations = Animations::new(&names);
    log::info!(
        "ekauq3D: player.mdl {} frames; run {} stand {} shoot {} swing {} deaths {:?}",
        names.len(),
        animations.run.len(),
        animations.stand.len(),
        animations.shoot.len(),
        animations.swing.len(),
        animations.deaths.iter().map(Vec::len).collect::<Vec<_>>()
    );
    let mut used: Vec<usize> = [
        &animations.run,
        &animations.stand,
        &animations.shoot,
        &animations.swing,
        &animations.pain,
    ]
    .into_iter()
    .chain(&animations.deaths)
    .flatten()
    .copied()
    .collect();
    used.sort_unstable();
    used.dedup();
    let remap = |frames: &Vec<usize>| -> Vec<usize> {
        frames
            .iter()
            .map(|frame| used.binary_search(frame).unwrap_or(0))
            .collect()
    };
    let animations = Animations {
        run: remap(&animations.run),
        stand: remap(&animations.stand),
        shoot: remap(&animations.shoot),
        swing: remap(&animations.swing),
        pain: remap(&animations.pain),
        deaths: animations.deaths.iter().map(remap).collect(),
    };
    let model = QuakeModel {
        frames: model
            .frames
            .into_iter()
            .enumerate()
            .filter(|(frame, _)| used.binary_search(frame).is_ok())
            .map(|(_, frame)| frame)
            .collect(),
        ..model
    };
    Ok((weapon::upload_model(engine, "player", model)?, animations))
}

/// Sets up the deathmatch once the level is loaded: the player's slot (joined in player mode)
/// and the bots at the spawns, a body for every other fighter.
pub fn spawn_deathmatch(
    engine: &mut Engine,
    scene: SceneHandle,
    map: &str,
    arena: Arena,
    nav: NavGraph,
    knowledge: Knowledge,
    player: Option<QuakeModel>,
    bots: usize,
    network: Network,
) -> Result<Deathmatch> {
    let bots = bots.min(MAX_BOTS).min(arena.spawns.len().saturating_sub(1));
    let level = load_level();
    #[cfg(not(target_arch = "wasm32"))]
    let (host, connection) = match &network {
        Network::Local => (None, None),
        Network::Host(port) => (
            Host::start(*port, map)
                .inspect_err(|error| log::error!("ekauq3D: can't host: {error}"))
                .ok(),
            None,
        ),
        Network::Connect(address) => (
            None,
            Connection::connect(address, map)
                .inspect_err(|error| log::error!("ekauq3D: can't connect: {error}"))
                .ok(),
        ),
    };
    #[cfg(not(target_arch = "wasm32"))]
    let (players, bots) = match (&host, &connection) {
        (Some(_), _) => (1 + MAX_CLIENTS, bots),
        // The server's match arrives with its welcome
        (_, Some(_)) => (1, 0),
        _ => (1, bots),
    };
    #[cfg(target_arch = "wasm32")]
    let (players, bots) = {
        let _ = &network;
        (1, bots)
    };
    let session = Session::new(&arena, players, bots, level, 1);
    let brains = Bots::new(&session, &arena, nav, knowledge);
    let model = player
        .map(|model| upload_player(engine, model))
        .transpose()?;
    // A body per fighter (a client's match arrives later: as many as there may be); the local
    // player's stays hidden
    #[cfg(not(target_arch = "wasm32"))]
    let bodies = if connection.is_some() {
        MAX_BODIES
    } else {
        session.fight.fighters.len()
    };
    #[cfg(target_arch = "wasm32")]
    let bodies = session.fight.fighters.len();
    if let Some((model, _)) = model.as_ref() {
        for index in 0..bodies {
            engine
                .build_entity(scene)
                .with_component(TransformComponent::builder().build())
                .with_component(
                    MeshRenderingComponent::builder()
                        .mesh(&model.frames[0])
                        .material(&model.material)
                        .build(),
                )
                .with_component(BotModel {
                    index,
                    time: 0.0,
                    dead: false,
                    deaths: 0,
                    frame: 0,
                    died_at: [0.0; 3],
                    died_yaw: 0.0,
                    corpse: None,
                })
                .build();
        }
        for _ in 0..MAX_CORPSES {
            engine
                .build_entity(scene)
                .with_component(TransformComponent::builder().scale(Vector3f::ZERO).build())
                .with_component(
                    MeshRenderingComponent::builder()
                        .mesh(&model.frames[0])
                        .material(&model.material)
                        .build(),
                )
                .with_component(BotModel {
                    index: CORPSE,
                    time: 0.0,
                    dead: false,
                    deaths: 0,
                    frame: 0,
                    died_at: [0.0; 3],
                    died_yaw: 0.0,
                    corpse: None,
                })
                .build();
        }
    }
    log::info!("ekauq3D: deathmatch with {bots} bots, difficulty {level:.2}");
    #[cfg(not(target_arch = "wasm32"))]
    let recorder = connection
        .is_none()
        .then(|| record_to_file(&Header::of(&session, map)));
    #[cfg(target_arch = "wasm32")]
    let _ = map;
    Ok(Deathmatch {
        session,
        arena,
        bots: brains,
        accumulator: 0.0,
        previous_origin: [0.0; 3],
        message: None,
        model,
        me: 0,
        since_hit: Vec::new(),
        damage_pulse: 0.0,
        frag_pulse: 0.0,
        message_age: 0.0,
        feed: Vec::new(),
        hit_tick_in: None,
        #[cfg(not(target_arch = "wasm32"))]
        recorder: recorder.flatten(),
        #[cfg(not(target_arch = "wasm32"))]
        host,
        #[cfg(not(target_arch = "wasm32"))]
        connection,
    })
}

/// The player's trigger and weapon keys (or a capture script's).
fn player_command(engine: &mut Engine) -> Result<(bool, Option<usize>)> {
    let (fire, select) = {
        let input = engine.get_global_component::<InputComponent>()?;
        let fire = input.cursor_locked() && input.get_mouse_button(MouseButton::Left);
        let keys = [KeyboardKey::Digit1, KeyboardKey::Digit2];
        let select = keys
            .iter()
            .position(|&key| input.get_key_pressed(key))
            .filter(|&slot| slot < WEAPONS.len());
        (fire, select)
    };
    let (scripted_fire, scripted_select) =
        engine.get_global_component_mut::<Weapon>()?.take_script();
    Ok((fire || scripted_fire, select.or(scripted_select)))
}

/// What the deathmatch takes of the player each frame.
struct PlayerFrame {
    playing: bool,
    origin: [f32; 3],
    yaw: f32,
    pitch: f32,
    input: PlayerInput,
    fire: bool,
    select: Option<usize>,
}

impl PlayerFrame {
    /// The player's command for the frame's `tick`-th tick (a weapon switch only on the first).
    fn command(&self, tick: usize) -> Command {
        Command {
            input: self.input,
            pitch: self.pitch,
            fire: self.fire && self.playing,
            select: self.select.filter(|_| tick == 0),
        }
    }
}

/// The fixed ticks due after `dt` more seconds (a long frame drops the rest).
fn due_ticks(accumulator: &mut f32, dt: f32) -> usize {
    *accumulator = (*accumulator + dt).min(MAX_TICKS_PER_FRAME as f32 * STEP);
    let ticks = (*accumulator / STEP) as usize;
    *accumulator -= ticks as f32 * STEP;
    ticks
}

impl Deathmatch {
    /// The fixed ticks this frame owes: the player joins or leaves with player mode, everyone's
    /// commands, the rules. What happened, and the player's move events.
    fn update(
        &mut self,
        collision: &QuakeCollision,
        player: &PlayerFrame,
        dt: f32,
    ) -> (Vec<Event>, PlayerEvents) {
        if let Some((_, left)) = self.message.as_mut() {
            *left -= dt;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if self.connection.is_some() {
            return self.update_client(collision, player, dt);
        }
        let active = self.session.fight.fighters[0].active;
        if player.playing && !active {
            let place = Some((player.origin, player.yaw));
            self.session.join(&self.arena, 0, place);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(recorder) = self.recorder.as_mut() {
                recorder.join(0, place);
            }
        } else if !player.playing && active {
            self.session.leave(0);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(recorder) = self.recorder.as_mut() {
                recorder.leave(0);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(host) = self.host.as_mut() {
            let changes = host.receive(
                std::time::Duration::from_secs_f32(dt),
                &mut self.session,
                &self.arena,
            );
            if let Some(recorder) = self.recorder.as_mut() {
                changes
                    .into_iter()
                    .for_each(|change| recorder.slot_change(change));
            }
        }
        let mut events = Vec::new();
        let mut moved = PlayerEvents::default();
        for tick in 0..due_ticks(&mut self.accumulator, dt) {
            #[allow(unused_mut)]
            let mut players = vec![(0, player.command(tick))];
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(host) = self.host.as_mut() {
                players.extend(host.take_commands());
            }
            let commands = self
                .bots
                .commands(&self.session, collision, &self.arena, &players);
            self.previous_origin = self.session.fight.fighters[0].state.origin;
            let (mut happened, ended) = self.session.advance(collision, &self.arena, &commands);
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(recorder) = self.recorder.as_mut() {
                recorder.tick(&commands, &self.session);
            }
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(host) = self.host.as_mut() {
                host.send_tick(&self.session, &happened);
            }
            let human = &self.session.fight.fighters[0];
            moved.merge(&human.moved);
            events.append(&mut happened);
            if let Some(ended) = ended {
                let mut text = format!(
                    "Match over: you {}, best bot {}",
                    ended.best_player, ended.best_bot
                );
                if ended.level != ended.level_before {
                    self.bots.set_level(ended.level);
                    save_level(ended.level);
                    text.push_str(&format!(
                        "; difficulty {:.2} → {:.2}",
                        ended.level_before, ended.level
                    ));
                }
                log::info!("ekauq3D: {text}");
                #[cfg(not(target_arch = "wasm32"))]
                if let Some(host) = self.host.as_mut() {
                    host.announce(&text);
                }
                self.message = Some((text, MESSAGE_TIME));
                self.message_age = 0.0;
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(host) = self.host.as_mut() {
            host.flush();
        }
        (events, moved)
    }

    /// Client: the server's messages, this frame's commands sent and the player's movement
    /// predicted, the match mirrored from the snapshots.
    #[cfg(not(target_arch = "wasm32"))]
    fn update_client(
        &mut self,
        collision: &QuakeCollision,
        player: &PlayerFrame,
        dt: f32,
    ) -> (Vec<Event>, PlayerEvents) {
        let mut moved = PlayerEvents::default();
        let Some(connection) = self.connection.as_mut() else {
            return (Vec::new(), moved);
        };
        connection.receive(dt, collision);
        if let Some(reason) = connection.refused.take() {
            self.message = Some((format!("Refused: {reason}"), MESSAGE_TIME * 3.0));
            self.message_age = 0.0;
        }
        for text in connection.messages.drain(..) {
            self.message = Some((text, MESSAGE_TIME));
            self.message_age = 0.0;
        }
        let Some(welcome) = connection.welcome.as_ref() else {
            connection.flush();
            return (Vec::new(), moved);
        };
        self.me = welcome.slot;
        if self.session.fight.fighters.len() != welcome.fighters {
            // A mirror of the server's match: its bots never think here
            self.session = Session::new(
                &self.arena,
                welcome.players,
                welcome.fighters.saturating_sub(welcome.players),
                welcome.level,
                1,
            );
        }
        for tick in 0..due_ticks(&mut self.accumulator, dt) {
            self.previous_origin = connection.predicted.origin;
            connection.send_command(player.command(tick), collision);
            moved.merge(&connection.moved);
        }
        connection.mirror(&mut self.session.fight);
        self.session.clock = connection.clock;
        let events = std::mem::take(&mut connection.events);
        self.session.age_attacks(&events, dt);
        connection.flush();
        (events, moved)
    }

    /// The player drawn between the last two ticks.
    fn player_origin(&self) -> [f32; 3] {
        let current = self.session.fight.fighters[self.me].state.origin;
        let alpha = self.accumulator / STEP;
        // A teleport (respawn) isn't smoothed
        if distance(current, self.previous_origin) > 64.0 {
            return current;
        }
        std::array::from_fn(|axis| {
            self.previous_origin[axis] + (current[axis] - self.previous_origin[axis]) * alpha
        })
    }

    /// The effects of this frame's events; ages the hit timers, pulses and kill feed.
    fn juice(&mut self, events: &[Event], dt: f32) -> Vec<Juice> {
        let count = self.session.fight.fighters.len();
        self.since_hit.resize(count, f32::MAX);
        for since in &mut self.since_hit {
            *since += dt;
        }
        self.damage_pulse = (self.damage_pulse - 1.5 * dt).max(0.0);
        self.frag_pulse = (self.frag_pulse - 2.0 * dt).max(0.0);
        self.message_age += dt;
        for line in &mut self.feed {
            line.2 += dt;
        }
        self.feed.retain(|line| line.2 < FEED_TIME);

        let me = self.me;
        let fighters = &self.session.fight.fighters;
        let my_eye = fighters[me].eye();
        let mut juice = Vec::new();
        // Per fighter: damage taken this frame, by whom last
        let mut damage = vec![(0.0f32, None); count];
        let mut hit_confirm = false;
        let mut tracer_hits = Vec::new();
        let mut kills = Vec::new();
        for event in events {
            match *event {
                Event::Hit {
                    attacker,
                    target,
                    damage: dealt,
                } => {
                    if let Some(taken) = damage.get_mut(target) {
                        *taken = (taken.0 + dealt, Some(attacker));
                    }
                    hit_confirm |= attacker == me && target != me;
                }
                Event::Impact {
                    by,
                    point,
                    normal,
                    kind,
                    ..
                } => {
                    if by == me && matches!(kind, ImpactKind::Wall | ImpactKind::Flesh) {
                        tracer_hits.push(point);
                    }
                    if kind == ImpactKind::Wall {
                        juice.push(Juice::Mark { point, normal });
                    }
                    let near = 1.0 - distance(point, my_eye) / BLAST_SHAKE_RADIUS;
                    if kind == ImpactKind::Explosion && near > 0.0 {
                        juice.push(Juice::Kick {
                            pitch: BLAST_KICK * near,
                            roll: BLAST_KICK * near * 0.5,
                        });
                    }
                }
                Event::Killed { attacker, target } => kills.push((attacker, target)),
                _ => {}
            }
        }
        // The tick after the shot's own sound, which would drown it
        if hit_confirm && self.hit_tick_in.is_none() {
            self.hit_tick_in = Some(HIT_TICK_DELAY);
        }
        if let Some(left) = self.hit_tick_in.as_mut() {
            *left -= dt;
            if *left <= 0.0 {
                self.hit_tick_in = None;
                juice.push(Juice::HitConfirm);
            }
        }
        for (target, &(dealt, attacker)) in damage.iter().enumerate() {
            if dealt <= 0.0 {
                continue;
            }
            if self.since_hit[target] >= PAIN_INTERVAL && fighters[target].alive {
                let at = (target != me).then(|| fighters[target].center());
                juice.push(Juice::Pain(at));
            }
            self.since_hit[target] = 0.0;
            if target == me {
                self.damage_pulse = (self.damage_pulse + dealt / 40.0).min(1.0);
                // Pitched up, rolled away from the side the shot came from
                let (_, right, _) = view_basis(fighters[me].yaw, 0.0);
                let side = attacker
                    .filter(|&attacker| attacker != me)
                    .map_or(0.0, |attacker| {
                        let to = sub(fighters[attacker].state.origin, fighters[me].state.origin);
                        dot(to, right).signum()
                    });
                juice.push(Juice::Kick {
                    pitch: dealt * HIT_KICK,
                    roll: -side * dealt * HIT_KICK,
                });
            }
        }
        for (attacker, target) in kills {
            let at = fighters[target].center();
            juice.push(Juice::Death {
                at,
                gib: damage
                    .get(target)
                    .is_some_and(|taken| taken.0 >= GIB_DAMAGE),
            });
            juice.push(Juice::Burst(at));
            let victim = fighter_name(&self.session, target, me);
            let text = match attacker.filter(|&attacker| attacker != target) {
                Some(attacker) => {
                    format!(
                        "{} fragged {victim}",
                        fighter_name(&self.session, attacker, me)
                    )
                }
                None => format!("{victim} died"),
            };
            if attacker == Some(me) && target != me {
                self.frag_pulse = 1.0;
            }
            self.feed
                .push((text, attacker == Some(me) || target == me, 0.0));
            if self.feed.len() > FEED_LINES {
                self.feed.remove(0);
            }
        }
        if !tracer_hits.is_empty() && WEAPONS[fighters[me].weapon].kind == Kind::Hitscan {
            juice.push(Juice::Tracers(tracer_hits));
        }
        juice
    }

    fn hud(&self, playing: bool) -> Hud {
        let session = &self.session;
        let human = &session.fight.fighters[self.me];
        let mut scores: Vec<(String, i32)> = session
            .fight
            .fighters
            .iter()
            .enumerate()
            .filter(|(_, fighter)| fighter.active)
            .map(|(index, fighter)| (fighter_name(session, index, self.me), fighter.frags))
            .collect();
        scores.sort_by(|a, b| b.1.cmp(&a.1));
        Hud {
            playing,
            health: human.health.max(0.0),
            weapon: WEAPONS[human.weapon].name,
            dead: !human.alive,
            scores,
            time_left: (TIME_LIMIT - session.clock).max(0.0),
            level: session.level,
            message: self
                .message
                .as_ref()
                .filter(|(_, left)| *left > 0.0)
                .map(|(text, _)| text.clone()),
            message_age: self.message_age,
            damage_pulse: self.damage_pulse,
            frag_pulse: self.frag_pulse,
            feed: self.feed.clone(),
        }
    }
}

/// Runs the deathmatch: the player and the bots at the fixed tick, then the player's position,
/// the effects and the player's weapon of what happened.
pub fn deathmatch_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    let (fire, select) = player_command(engine)?;
    let player = {
        let rig = engine.get_global_component::<PlayerRig>()?;
        PlayerFrame {
            playing: rig.mode == Mode::Player,
            origin: rig.state.origin,
            yaw: rig.yaw,
            pitch: rig.pitch,
            input: rig.input,
            fire,
            select,
        }
    };
    let killer = engine
        .get_global_component::<PlayerRig>()?
        .death
        .and_then(|death| death.killer);
    let (events, moved, impacts, hud, human, origin, me, killer, juice) = {
        let state = engine.get_global_component_mut::<LevelState>()?;
        let LevelState {
            collision,
            deathmatch,
            ..
        } = state;
        let (Some(collision), Some(deathmatch)) = (collision.as_ref(), deathmatch.as_mut()) else {
            return Ok(());
        };
        let (events, moved) = deathmatch.update(collision, &player, delta_time);
        let juice = deathmatch.juice(&events, delta_time);
        let impacts: Vec<Impact> = events
            .iter()
            .filter_map(|event| match *event {
                Event::Impact {
                    point,
                    normal,
                    texture,
                    kind,
                    ..
                } => Some(Impact {
                    point,
                    normal,
                    metal: texture.is_some_and(|id| is_metal(collision.texture_name(id))),
                    kind,
                }),
                _ => None,
            })
            .collect();
        let me = deathmatch.me;
        let fighters = &deathmatch.session.fight.fighters;
        let human = fighters[me].clone();
        // The killer's origin and velocity, for the dead player's view
        let killer = killer
            .and_then(|killer| fighters.get(killer))
            .map(|killer| (killer.state.origin, killer.state.velocity));
        (
            events,
            moved,
            impacts,
            deathmatch.hud(player.playing),
            human,
            deathmatch.player_origin(),
            me,
            killer,
            juice,
        )
    };

    // --- The player: where the deathmatch moved them, died, came back
    {
        let rig = engine.get_global_component_mut::<PlayerRig>()?;
        if player.playing {
            rig.state = human.state;
            rig.origin = origin;
            rig.dead = !human.alive;
            rig.events.jumped = moved.jumped;
            rig.events.landed_speed = moved.landed_speed;
            if human.alive {
                rig.death = None;
            }
        }
        // The dead player's head turns to where the killer runs (else toward the killer)
        if let (Some(death), Some((killer_origin, velocity))) = (rig.death.as_mut(), killer) {
            let toward = if horizontal_distance(velocity, [0.0; 3]) > 50.0 {
                [velocity[0], velocity[1]]
            } else {
                [killer_origin[0] - origin[0], killer_origin[1] - origin[1]]
            };
            death.look_toward(toward[1].atan2(toward[0]).to_degrees());
        }
    }
    for event in &events {
        match *event {
            Event::Killed { attacker, target } if target == me && player.playing => {
                let rig = engine.get_global_component_mut::<PlayerRig>()?;
                let killer = attacker.filter(|&attacker| attacker != me);
                rig.death = Some(DeathView::new(rig.yaw, rig.pitch, killer, rig.bob_time));
            }
            Event::Respawned { fighter } if fighter == me => {
                let rig = engine.get_global_component_mut::<PlayerRig>()?;
                rig.yaw = human.yaw;
                rig.pitch = 0.0;
            }
            Event::Fired { fighter, weapon } if fighter == me => {
                engine
                    .get_global_component_mut::<Weapon>()?
                    .player_fired(weapon);
            }
            Event::Switched { fighter, weapon } if fighter == me => {
                engine
                    .get_global_component_mut::<Weapon>()?
                    .player_switched(weapon);
            }
            Event::Impact {
                by,
                kind: ImpactKind::Melee | ImpactKind::Flesh,
                ..
            } if by == me && WEAPONS[human.weapon].kind == Kind::Melee => {
                engine
                    .get_global_component_mut::<Weapon>()?
                    .player_axe_hit();
            }
            Event::Fired { fighter, weapon } if WEAPONS[weapon].kind != Kind::Melee => {
                let position = {
                    let state = engine.get_global_component::<LevelState>()?;
                    state
                        .deathmatch
                        .as_ref()
                        .map(|deathmatch| deathmatch.session.fight.fighters[fighter].eye())
                };
                if let Some(eye) = position {
                    let at = Vector3f::from_array(quake_to_engine(eye));
                    sounds::play(engine, Sfx::BotFire, Some(at))?;
                }
            }
            _ => {}
        }
    }
    for impact in &impacts {
        weapon::spawn_impact(engine, impact)?;
    }
    play_juice(engine, juice, player.playing)?;
    if let Ok(mut shown) = HUD.lock() {
        *shown = Some(hud);
    }
    Ok(())
}

/// Sounds, view kicks, bursts, tracers and marks of this frame's events.
fn play_juice(engine: &mut Engine, juice: Vec<Juice>, playing: bool) -> Result<()> {
    let at = |point: [f32; 3]| Some(Vector3f::from_array(quake_to_engine(point)));
    for effect in juice {
        match effect {
            Juice::HitConfirm => sounds::play(engine, Sfx::HitConfirm, None)?,
            Juice::Pain(None) => sounds::play(engine, Sfx::Pain, None)?,
            Juice::Pain(Some(point)) => sounds::play(engine, Sfx::OtherPain, at(point))?,
            Juice::Kick { pitch, roll } if playing => {
                engine
                    .get_global_component_mut::<PlayerRig>()?
                    .kick(pitch, roll);
            }
            Juice::Kick { .. } => {}
            Juice::Death { at: point, gib } => {
                let sound = if gib { Sfx::Gib } else { Sfx::Death };
                sounds::play(engine, sound, at(point))?;
            }
            Juice::Burst(point) => weapon::spawn_kill_burst(engine, point)?,
            Juice::Tracers(points) => {
                let view = engine.get_global_component::<PlayerRig>()?.view;
                weapon::spawn_tracers(engine, &view, &points)?;
            }
            Juice::Mark { point, normal } => weapon::spawn_decal(engine, point, normal)?,
        }
    }
    Ok(())
}

/// What a body shows of its fighter this frame.
#[derive(Clone, Copy)]
struct Pose {
    shown: bool,
    alive: bool,
    origin: [f32; 3],
    yaw: f32,
    speed: f32,
    since_attack: f32,
    since_hit: f32,
    melee: bool,
}

/// A body's rotation (degrees) facing `yaw`, tipped over backward by `tip` (0..1).
fn body_rotation(yaw: f32, tip: f32) -> Vector3f {
    let (sin, cos) = yaw.to_radians().sin_cos();
    let (tip_sin, tip_cos) = (tip * std::f32::consts::FRAC_PI_2).sin_cos();
    let forward = [cos * tip_cos, sin * tip_cos, tip_sin];
    let up = [-cos * tip_sin, -sin * tip_sin, tip_cos];
    // Mesh axes: +X the model's forward, +Y up, +Z its right (engine-converted Quake)
    euler_from_columns([
        quake_direction_to_engine(forward),
        quake_direction_to_engine(up),
        quake_direction_to_engine([sin, -cos, 0.0]),
    ])
}

/// Places, turns and animates the fighters' bodies: running, standing, attacking, flinching and
/// when hit, dying; a respawn leaves a corpse where the body lay.
pub fn bot_render_system(engine: &mut Engine) -> Result<()> {
    let delta_time = engine
        .get_global_component::<TimeComponent>()?
        .delta_time
        .clamp(0.0, 0.1);
    let (poses, meshes, [run, stand, shoot, swing, pain], deaths) = {
        let state = engine.get_global_component::<LevelState>()?;
        let Some(deathmatch) = state.deathmatch.as_ref() else {
            return Ok(());
        };
        let Some((model, animations)) = deathmatch.model.as_ref() else {
            return Ok(());
        };
        let session = &deathmatch.session;
        let me = deathmatch.me;
        let poses: Vec<Pose> = session
            .fight
            .fighters
            .iter()
            .zip(&session.since_attack)
            .enumerate()
            .map(|(index, (fighter, &since_attack))| Pose {
                shown: fighter.active && index != me,
                alive: fighter.alive,
                origin: fighter.state.origin,
                yaw: fighter.yaw,
                speed: horizontal_distance(fighter.state.velocity, [0.0; 3]),
                since_attack,
                since_hit: deathmatch.since_hit.get(index).copied().unwrap_or(f32::MAX),
                melee: WEAPONS[fighter.weapon].kind == Kind::Melee,
            })
            .collect();
        (
            poses,
            model.frames.clone(),
            [
                animations.run.clone(),
                animations.stand.clone(),
                animations.shoot.clone(),
                animations.swing.clone(),
                animations.pain.clone(),
            ],
            animations.deaths.clone(),
        )
    };
    // (where, yaw, death animation) of bodies that just respawned
    let mut left_behind = Vec::new();
    for (_, transform, rendering, body) in engine
        .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, BotModel>()?
    {
        if body.index == CORPSE {
            continue;
        }
        let Some(pose) = poses.get(body.index).copied() else {
            transform.set_scale(Vector3f::ZERO);
            continue;
        };
        let death = (body.index + body.deaths) % deaths.len();
        if body.dead == pose.alive {
            body.dead = !pose.alive;
            body.time = 0.0;
            if pose.alive {
                left_behind.push((body.died_at, body.died_yaw, death));
                body.deaths += 1;
            } else {
                body.died_at = pose.origin;
                body.died_yaw = pose.yaw;
            }
        }
        body.time += delta_time;
        if !pose.shown {
            transform.set_scale(Vector3f::ZERO);
            continue;
        }
        let step = (body.time * ANIMATION_RATE) as usize;
        let attack = if pose.melee { &swing } else { &shoot };
        let attack_step = (pose.since_attack * ANIMATION_RATE) as usize;
        let pain_step = (pose.since_hit * ANIMATION_RATE) as usize;
        // Dead: a death (varying per body and death), then lying on its last frame until respawn
        let dying = &deaths[death];
        let frame = if !pose.alive {
            dying[step.min(dying.len() - 1)]
        } else if attack_step < attack.len() {
            attack[attack_step]
        } else if pain_step < pain.len() {
            pain[pain_step]
        } else if pose.speed > 40.0 {
            run[step % run.len()]
        } else {
            stand[step % stand.len()]
        };
        // A model without death frames tips over backward about its right axis instead
        let tip = if !pose.alive && dying.len() == 1 {
            (body.time / TIP_OVER_TIME).min(1.0)
        } else {
            0.0
        };
        transform.set_position(Vector3f::from_array(quake_to_engine(pose.origin)));
        transform.set_rotation(body_rotation(pose.yaw, tip));
        transform.set_scale(Vector3f::ONE);
        if frame != body.frame {
            if let Some(mesh) = meshes.get(frame) {
                rendering.set_mesh(mesh);
            }
            body.frame = frame;
        }
    }

    // Corpses: new ones replace free or the oldest; they lie, then sink away
    for (origin, yaw, death) in left_behind {
        let mut chosen = None;
        let mut oldest = f32::MIN;
        for (entity, corpse) in engine.iterate_one_component::<BotModel>()? {
            if corpse.index != CORPSE {
                continue;
            }
            let age = corpse.corpse.unwrap_or(f32::MAX);
            if age > oldest {
                oldest = age;
                chosen = Some(entity);
            }
        }
        for (entity, transform, rendering, corpse) in engine
            .iterate_three_components_mut::<TransformComponent, MeshRenderingComponent, BotModel>(
            )?
        {
            if Some(entity) != chosen {
                continue;
            }
            // Lying as the body lay: the death's last frame, or tipped over without one
            let dying = &deaths[death];
            corpse.corpse = Some(0.0);
            if let Some(mesh) = meshes.get(dying[dying.len() - 1]) {
                rendering.set_mesh(mesh);
            }
            let tip = if dying.len() == 1 { 1.0 } else { 0.0 };
            transform.set_position(Vector3f::from_array(quake_to_engine(origin)));
            transform.set_rotation(body_rotation(yaw, tip));
            transform.set_scale(Vector3f::ONE);
        }
    }
    for (_, transform, corpse) in
        engine.iterate_two_components_mut::<TransformComponent, BotModel>()?
    {
        let Some(age) = corpse.corpse.as_mut() else {
            continue;
        };
        *age += delta_time;
        let sunk = (*age - CORPSE_TIME) / CORPSE_SINK_TIME;
        if sunk >= 1.0 {
            corpse.corpse = None;
            transform.set_scale(Vector3f::ZERO);
            continue;
        }
        if sunk > 0.0 {
            // 24 units (0.75 m) down into the floor
            let sink = Vector3f::Y * (0.75 * delta_time / CORPSE_SINK_TIME);
            transform.set_position(transform.position - sink);
        }
    }
    Ok(())
}

/// HUD: health and weapon, the scores, time left, the match message.
pub fn hud(ctx: &egui::Context) {
    let Some(hud) = HUD.lock().ok().and_then(|hud| hud.clone()) else {
        return;
    };
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("ekauq3d.hud"),
    ));
    let screen = ctx.screen_rect();
    let font = egui::FontId::monospace(16.0);
    let color = egui::Color32::from_rgb(235, 225, 200);
    let hurt = egui::Color32::from_rgb(230, 40, 30);
    let ease_out = |x: f32| 1.0 - (1.0 - x.clamp(0.0, 1.0)).powi(3);
    if hud.playing {
        // Damage: the screen's edges redden, more when hurt badly
        let low = if hud.dead {
            0.0
        } else {
            (1.0 - hud.health / 25.0).clamp(0.0, 1.0) * 0.5
        };
        let red = hud.damage_pulse.max(low);
        if red > 0.0 {
            for ring in 0..12 {
                let inset = ring as f32 * 6.0;
                let alpha = (red * 110.0 * (1.0 - ring as f32 / 12.0)) as u8;
                painter.rect_stroke(
                    screen.shrink(inset + 3.0),
                    0.0,
                    egui::Stroke::new(6.0, egui::Color32::from_rgba_unmultiplied(200, 0, 0, alpha)),
                    egui::StrokeKind::Middle,
                );
            }
        }
        let status = if hud.dead { "dead" } else { hud.weapon };
        painter.text(
            screen.left_bottom() + egui::vec2(16.0, -16.0),
            egui::Align2::LEFT_BOTTOM,
            status,
            font.clone(),
            color,
        );
        // Health: big at the bottom center, bold by painting it a few times slightly shifted
        // (egui's default fonts have no bold face)
        let health = format!("{}", hud.health.ceil() as i32);
        let at = screen.center_bottom() + egui::vec2(0.0, -16.0);
        // Hurt: the number swells and reddens, easing back
        let big = egui::FontId::monospace(HEALTH_FONT_SIZE * (1.0 + 0.35 * hud.damage_pulse));
        let color = lerp_color(color, hurt, hud.damage_pulse);
        for dx in -2..=2 {
            for dy in -2..=2 {
                painter.text(
                    at + egui::vec2(dx as f32, dy as f32),
                    egui::Align2::CENTER_BOTTOM,
                    &health,
                    big.clone(),
                    color,
                );
            }
        }
    }
    // Time and scores top right; your line pops when you frag someone
    let mut y = 40.0;
    let mut line = |text: String, size: f32, color: egui::Color32| {
        painter.text(
            screen.right_top() + egui::vec2(-16.0, y),
            egui::Align2::RIGHT_TOP,
            text,
            egui::FontId::monospace(size),
            color,
        );
        y += size * 1.25;
    };
    line(
        format!(
            "{}:{:02}  difficulty {:.2}",
            hud.time_left as i32 / 60,
            hud.time_left as i32 % 60,
            hud.level
        ),
        16.0,
        color,
    );
    let bright = egui::Color32::from_rgb(255, 230, 120);
    for (name, frags) in &hud.scores {
        let (size, shade) = if name == "you" {
            (
                16.0 * (1.0 + 0.5 * hud.frag_pulse),
                lerp_color(color, bright, hud.frag_pulse),
            )
        } else {
            (16.0, color)
        };
        line(format!("{name:<6} {frags:>3}"), size, shade);
    }
    // Kill feed top left, under the engine panels; yours bright, fading out at the end
    for (index, (text, mine, age)) in hud.feed.iter().enumerate() {
        let fade = (1.0 - (age - (FEED_TIME - 0.5)).max(0.0) / 0.5).clamp(0.0, 1.0);
        let base = if *mine { bright } else { color };
        painter.text(
            screen.left_top() + egui::vec2(16.0, 190.0 + index as f32 * 20.0),
            egui::Align2::LEFT_TOP,
            text,
            font.clone(),
            base.gamma_multiply(fade),
        );
    }
    if let Some(message) = hud.message {
        // Slides down into place
        let slide = 60.0 * (1.0 - ease_out(hud.message_age / 0.35));
        painter.text(
            screen.center() - egui::vec2(0.0, 80.0 + slide),
            egui::Align2::CENTER_CENTER,
            message,
            egui::FontId::proportional(22.0),
            color,
        );
    }
}

/// `a` toward `b` by `t` (0..1).
fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t) as u8;
    egui::Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}
