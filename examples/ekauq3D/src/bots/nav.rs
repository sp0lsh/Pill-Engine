//! Navigation from the level alone (no waypoints): floor nodes flood-filled from the spawns with
//! the player's box, walk and drop edges mirroring Quake's step-up movement, and jump edges found
//! by simulating the player's own movement (`player_move`), so a jump that works here works in
//! play. After the Area Awareness System's reachabilities (van Waveren 2001, §6.4), measured
//! instead of derived from geometry. Count budgets only (no clocks): native and web build the
//! same graph.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use pill_engine::game::{
    quake_map::{QuakeCollision, QuakeTrace},
    quake_player::{player_move, trace_player, MoveSettings, PlayerInput, PlayerState},
};

use crate::sim::{angles_of, arena::Arena, dot, horizontal_distance, scale, sub, Vec3, STEP};

/// Floor sampling (units): nodes sit at cell centers.
pub const CELL: f32 = 32.0;
const COARSE_CELL: f32 = 48.0;
const MAX_NODES: usize = 12_000;
const STEP_HEIGHT: f32 = 18.0;
const MAX_DROP: f32 = 512.0;
/// Steepest walkable ground (normal's up component), as the movement code.
const MIN_GROUND_NORMAL: f32 = 0.7;
const RUN_SPEED: f32 = 320.0;
const GRAVITY: f32 = 800.0;
/// Jump candidates: horizontal reach, how much higher or lower the landing may be.
const JUMP_MIN: f32 = 48.0;
const JUMP_MAX: f32 = 288.0;
const JUMP_RISE: f32 = 40.0;
const JUMP_FALL: f32 = 256.0;
/// Jumps toward an edge node are also tried from this many cells behind it.
const TAKE_OFF_CELLS: i32 = 4;
const MAX_JUMP_SIMS: usize = 40_000;
/// A jump is only tried where walking there takes this much longer than the straight line.
const JUMP_GAIN: f32 = 2.0;
/// Running jumps start this far behind the take-off node.
pub const RUN_UP: f32 = 48.0;
const JUMP_TIME: f32 = 1.2;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum EdgeKind {
    Walk,
    /// Walking off a ledge: one way.
    Drop,
    /// Running jump along `yaw`: run up from `RUN_UP` behind the node, jump at the node.
    Jump {
        yaw: f32,
    },
    /// Jump from standing still at the node, pushing along `yaw`.
    StandingJump {
        yaw: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub to: u32,
    /// Seconds.
    pub time: f32,
    pub kind: EdgeKind,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NavStats {
    pub cell: f32,
    pub nodes: usize,
    pub edges: usize,
    pub jumps: usize,
    pub jump_sims: usize,
    pub traces: usize,
    pub moves: usize,
}

pub struct NavGraph {
    /// Player origins standing on the floor (Quake space).
    pub nodes: Vec<Vec3>,
    first: Vec<u32>,
    edges: Vec<Edge>,
    reverse_first: Vec<u32>,
    reverse: Vec<(u32, f32)>,
    columns: HashMap<(i32, i32), Vec<u32>>,
    cell: f32,
    pub stats: NavStats,
}

struct Builder<'a> {
    world: &'a QuakeCollision,
    cell: f32,
    nodes: Vec<Vec3>,
    keys: HashMap<(i32, i32, i32), u32>,
    edges: Vec<(u32, Edge)>,
    /// Per node, the directions (bits of `DIRECTIONS`) that are blocked, fall off or drop
    edge_directions: Vec<u8>,
    queue: VecDeque<u32>,
    traces: usize,
    moves: usize,
    sims: usize,
}

// prettier-ignore
const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

impl<'a> Builder<'a> {
    fn trace(&mut self, from: Vec3, to: Vec3) -> QuakeTrace {
        self.traces += 1;
        trace_player(self.world, from, to)
    }

    fn column(&self, point: Vec3) -> (i32, i32) {
        (
            (point[0] / self.cell).floor() as i32,
            (point[1] / self.cell).floor() as i32,
        )
    }

    fn key(&self, point: Vec3) -> (i32, i32, i32) {
        let (x, y) = self.column(point);
        (x, y, (point[2] / 16.0).round() as i32)
    }

    fn center(&self, column: (i32, i32)) -> (f32, f32) {
        (
            (column.0 as f32 + 0.5) * self.cell,
            (column.1 as f32 + 0.5) * self.cell,
        )
    }

    /// Standing spot below `point` (within a step plus the longest drop), on walkable ground.
    fn ground(&mut self, point: Vec3) -> Option<Vec3> {
        let down = self.trace(
            point,
            [point[0], point[1], point[2] - STEP_HEIGHT - MAX_DROP],
        );
        (!down.start_solid && down.fraction < 1.0 && down.normal[2] >= MIN_GROUND_NORMAL)
            .then_some(down.end)
    }

    /// The node at `point`'s cell and height, added (and queued for the flood) if new; `None`
    /// once the node budget is spent.
    fn node(&mut self, point: Vec3) -> Option<u32> {
        let key = self.key(point);
        if let Some(&index) = self.keys.get(&key) {
            return Some(index);
        }
        if self.nodes.len() >= MAX_NODES {
            return None;
        }
        let index = self.nodes.len() as u32;
        self.nodes.push(point);
        self.edge_directions.push(0);
        self.keys.insert(key, index);
        self.queue.push_back(index);
        Some(index)
    }

    /// Flood fill of the queued nodes; false when the node budget runs out.
    fn flood(&mut self) -> bool {
        while let Some(index) = self.queue.pop_front() {
            let point = self.nodes[index as usize];
            let column = self.column(point);
            for (direction, (dx, dy)) in DIRECTIONS.into_iter().enumerate() {
                let (x, y) = self.center((column.0 + dx, column.1 + dy));
                // Quake's step: up a step, across, back down
                let up = self.trace(point, [point[0], point[1], point[2] + STEP_HEIGHT]);
                let start = up.end;
                let mut across = self.trace(start, [x, y, start[2]]);
                if across.start_solid || across.fraction < 1.0 {
                    // Ramps steeper than a step per cell: up to 45°
                    across = self.trace(start, [x, y, start[2] + self.cell]);
                }
                if across.start_solid || across.fraction < 1.0 {
                    self.edge_directions[index as usize] |= 1 << direction;
                    continue;
                }
                let Some(landing) = self.ground(across.end) else {
                    self.edge_directions[index as usize] |= 1 << direction;
                    continue;
                };
                let rise = landing[2] - point[2];
                if rise > STEP_HEIGHT + 0.5 {
                    // Climbed more than a step: only a ramp, rising evenly (half way at the middle)
                    let middle = [
                        (point[0] + x) * 0.5,
                        (point[1] + y) * 0.5,
                        landing[2] + STEP_HEIGHT,
                    ];
                    let even = self.ground(middle).is_some_and(|ground| {
                        (ground[2] - (point[2] + landing[2]) * 0.5).abs() < rise * 0.25
                    });
                    if !even {
                        self.edge_directions[index as usize] |= 1 << direction;
                        continue;
                    }
                }
                let horizontal = horizontal_distance(point, landing);
                let (kind, time) = if rise >= -STEP_HEIGHT - 0.5 {
                    (EdgeKind::Walk, horizontal / RUN_SPEED)
                } else {
                    (
                        EdgeKind::Drop,
                        horizontal / RUN_SPEED + (2.0 * -rise / GRAVITY).sqrt(),
                    )
                };
                if kind == EdgeKind::Drop {
                    self.edge_directions[index as usize] |= 1 << direction;
                }
                let Some(to) = self.node(landing) else {
                    return false;
                };
                if to != index {
                    self.edges.push((index, Edge { to, time, kind }));
                }
            }
        }
        true
    }

    /// Jump edges toward the edge of the walkable space: for each node with a direction that is
    /// blocked, falls off or drops, a running jump (else a standing one) along that direction
    /// from it or from a node up to `TAKE_OFF_CELLS` behind it (onto a ledge the jump must
    /// already be high when it reaches the wall). The edge leads wherever the jump lands, if
    /// walking there takes much longer (or is impossible). New landing spots become nodes and
    /// are queued for the flood.
    fn jumps(&mut self, start: usize) {
        let count = self.nodes.len();
        let (first, edges) = compact(count, self.edges.iter().copied());
        let mut columns: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (index, &point) in self.nodes.iter().enumerate() {
            columns
                .entry(self.column(point))
                .or_default()
                .push(index as u32);
        }
        let settings = MoveSettings::default();
        let limit = JUMP_GAIN * 2.0 * JUMP_MAX / RUN_SPEED;
        let mut tried = HashSet::new();
        let mut walked: HashMap<u32, HashMap<u32, f32>> = HashMap::new();
        let mut found = Vec::new();
        for edge_node in start..count {
            let edge_directions = self.edge_directions[edge_node];
            if edge_directions == 0 {
                continue;
            }
            let edge_point = self.nodes[edge_node];
            let column = self.column(edge_point);
            for (direction_index, (dx, dy)) in DIRECTIONS.iter().enumerate() {
                if edge_directions & (1 << direction_index) == 0 {
                    continue;
                }
                let (yaw, _) = angles_of([*dx as f32, *dy as f32, 0.0]);
                let mut jumped = false;
                for back in 0..=TAKE_OFF_CELLS {
                    if self.sims >= MAX_JUMP_SIMS {
                        break;
                    }
                    let behind = (column.0 - dx * back, column.1 - dy * back);
                    let Some(from) = columns.get(&behind).and_then(|nodes| {
                        nodes.iter().copied().min_by(|&a, &b| {
                            let za = (self.nodes[a as usize][2] - edge_point[2]).abs();
                            let zb = (self.nodes[b as usize][2] - edge_point[2]).abs();
                            za.total_cmp(&zb)
                        })
                    }) else {
                        break;
                    };
                    let point = self.nodes[from as usize];
                    if (point[2] - edge_point[2]).abs() > JUMP_RISE
                        || !tried.insert((from, direction_index))
                    {
                        continue;
                    }
                    for kind in [EdgeKind::Jump { yaw }, EdgeKind::StandingJump { yaw }] {
                        self.sims += 1;
                        let (landed, traces, moves) =
                            simulate_jump(self.world, &settings, point, kind);
                        self.traces += traces;
                        self.moves += moves;
                        let Some((landing, time)) = landed else {
                            continue;
                        };
                        let rise = landing[2] - point[2];
                        let horizontal = horizontal_distance(point, landing);
                        if !(JUMP_MIN..=JUMP_MAX).contains(&horizontal)
                            || !(-JUMP_FALL..=JUMP_RISE + 8.0).contains(&rise)
                        {
                            continue;
                        }
                        let times = walked
                            .entry(from)
                            .or_insert_with(|| bounded_times(&first, &edges, from, limit));
                        let walking = self
                            .keys
                            .get(&self.key(landing))
                            .and_then(|to| times.get(to).copied())
                            .unwrap_or(f32::INFINITY);
                        if walking <= JUMP_GAIN * horizontal / RUN_SPEED + 0.2 {
                            continue;
                        }
                        found.push((from, landing, time, kind));
                        jumped = true;
                        break;
                    }
                    if jumped {
                        break;
                    }
                }
            }
        }
        for (from, landing, time, kind) in found {
            if let Some(to) = self.node(landing).filter(|&to| to != from) {
                self.edges.push((from, Edge { to, time, kind }));
            }
        }
    }
}

/// Compressed adjacency of `(from, edge)` pairs: `first[node]..first[node + 1]` index `edges`.
fn compact(count: usize, pairs: impl Iterator<Item = (u32, Edge)>) -> (Vec<u32>, Vec<Edge>) {
    let mut pairs: Vec<(u32, Edge)> = pairs.collect();
    pairs.sort_by_key(|(from, _)| *from);
    let mut first = vec![0u32; count + 1];
    for (from, _) in &pairs {
        first[*from as usize + 1] += 1;
    }
    for index in 0..count {
        first[index + 1] += first[index];
    }
    (first, pairs.into_iter().map(|(_, edge)| edge).collect())
}

/// Travel times from `from` to nodes within `limit` seconds.
fn bounded_times(first: &[u32], edges: &[Edge], from: u32, limit: f32) -> HashMap<u32, f32> {
    let mut times = HashMap::new();
    let mut heap = BinaryHeap::new();
    times.insert(from, 0.0f32);
    heap.push(Reverse((0u32, from)));
    while let Some(Reverse((bits, node))) = heap.pop() {
        let time = f32::from_bits(bits);
        if time > times[&node] {
            continue;
        }
        for edge in &edges[first[node as usize] as usize..first[node as usize + 1] as usize] {
            let next = time + edge.time;
            if next > limit {
                continue;
            }
            if times.get(&edge.to).map_or(true, |&known| next < known) {
                times.insert(edge.to, next);
                heap.push(Reverse((next.to_bits(), edge.to)));
            }
        }
    }
    times
}

/// Simulates the jump of a jump edge from the node `from`, always pushing forward along its yaw:
/// running (run up from `RUN_UP` behind, jump at the node) or standing. Where and
/// after how many seconds the player lands, if within `JUMP_TIME`; traces and moves spent.
pub fn simulate_jump(
    world: &QuakeCollision,
    settings: &MoveSettings,
    from: Vec3,
    kind: EdgeKind,
) -> (Option<(Vec3, f32)>, usize, usize) {
    let (yaw, running) = match kind {
        EdgeKind::Jump { yaw } => (yaw, true),
        EdgeKind::StandingJump { yaw } => (yaw, false),
        EdgeKind::Walk | EdgeKind::Drop => return (None, 0, 0),
    };
    let (direction, _, _) = crate::sim::view_basis(yaw, 0.0);
    let mut traces = 0;
    let mut state = PlayerState {
        origin: from,
        on_ground: true,
        ..PlayerState::default()
    };
    if running {
        let Some(start) = run_up_start(world, from, kind) else {
            return (None, 2, 0);
        };
        traces += 2;
        state.origin = start;
    }
    let run_steps = if running { (0.8 / STEP) as usize } else { 0 };
    let steps = run_steps + (JUMP_TIME / STEP) as usize;
    let mut jumped = false;
    let mut airborne = false;
    for step in 0..steps {
        let along = dot(sub(state.origin, from), direction);
        let jump = !jumped && (!running || along >= -2.0);
        if !jumped && running && step >= run_steps {
            return (None, traces, step);
        }
        let input = PlayerInput {
            forward: 1.0,
            side: 0.0,
            jump,
            yaw,
        };
        let moved = player_move(&mut state, &input, STEP, world, settings);
        jumped |= moved.jumped;
        if jumped && !state.on_ground {
            airborne = true;
        }
        if airborne && state.on_ground {
            return (
                Some((state.origin, (step + 1) as f32 * STEP)),
                traces,
                step + 1,
            );
        }
    }
    (None, traces, steps)
}

/// Where a running jump's run-up starts: `RUN_UP` behind the node along the jump, if the box
/// gets there in a straight line and stands on the floor about level with the node.
pub fn run_up_start(world: &QuakeCollision, from: Vec3, kind: EdgeKind) -> Option<Vec3> {
    let EdgeKind::Jump { yaw } = kind else {
        return None;
    };
    let (direction, _, _) = crate::sim::view_basis(yaw, 0.0);
    let start = sub(from, scale(direction, RUN_UP));
    let line = trace_player(world, from, start);
    if line.start_solid || line.fraction < 1.0 {
        return None;
    }
    let down = trace_player(
        world,
        start,
        [start[0], start[1], start[2] - STEP_HEIGHT - 1.0],
    );
    (down.fraction < 1.0 && down.normal[2] >= MIN_GROUND_NORMAL).then_some(down.end)
}

impl NavGraph {
    /// The graph of an arena: flooded from its spawns.
    pub fn of(world: &QuakeCollision, arena: &Arena) -> Self {
        let seeds: Vec<Vec3> = arena.spawns.iter().map(|spawn| spawn.origin).collect();
        Self::build(world, &seeds)
    }

    /// Builds the graph from standing spots (the spawns, on the floor).
    pub fn build(world: &QuakeCollision, seeds: &[Vec3]) -> Self {
        let mut flooded = None;
        for cell in [CELL, COARSE_CELL] {
            let mut builder = Builder {
                world,
                cell,
                nodes: Vec::new(),
                keys: HashMap::new(),
                edges: Vec::new(),
                edge_directions: Vec::new(),
                queue: VecDeque::new(),
                traces: 0,
                moves: 0,
                sims: 0,
            };
            let mut complete = seeds.iter().all(|&seed| builder.node(seed).is_some());
            // Walk everywhere, then jump from the new edge of it, until nothing new is reached
            let mut checked = 0;
            while complete {
                complete = builder.flood();
                let before = builder.nodes.len();
                if !complete || checked == before || builder.sims >= MAX_JUMP_SIMS {
                    break;
                }
                builder.jumps(checked);
                checked = before;
            }
            if complete || cell == COARSE_CELL {
                if !complete {
                    log::warn!("bots: navigation node budget reached at cell {cell}");
                }
                flooded = Some(builder);
                break;
            }
        }
        let builder = flooded.expect("flood ran");
        let sims = builder.sims;
        let jumps = builder
            .edges
            .iter()
            .filter(|(_, edge)| {
                matches!(
                    edge.kind,
                    EdgeKind::Jump { .. } | EdgeKind::StandingJump { .. }
                )
            })
            .count();

        // All of it, also places without a way back (routes only lead where the goal is reached)
        let nodes = builder.nodes.clone();
        let (first, edges) = compact(nodes.len(), builder.edges.iter().copied());
        let reversed = (0..nodes.len()).flat_map(|from| {
            edges[first[from] as usize..first[from + 1] as usize]
                .iter()
                .map(move |edge| {
                    (
                        edge.to,
                        Edge {
                            to: from as u32,
                            ..*edge
                        },
                    )
                })
        });
        let (reverse_first, reverse_edges) = compact(nodes.len(), reversed);
        let reverse = reverse_edges
            .into_iter()
            .map(|edge| (edge.to, edge.time))
            .collect();
        let mut columns: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (index, &point) in nodes.iter().enumerate() {
            columns
                .entry(builder.column(point))
                .or_default()
                .push(index as u32);
        }
        let stats = NavStats {
            cell: builder.cell,
            nodes: nodes.len(),
            edges: edges.len(),
            jumps,
            jump_sims: sims,
            traces: builder.traces,
            moves: builder.moves,
        };
        Self {
            nodes,
            first,
            edges,
            reverse_first,
            reverse,
            columns,
            cell: builder.cell,
            stats,
        }
    }

    pub fn outgoing(&self, node: u32) -> &[Edge] {
        &self.edges[self.first[node as usize] as usize..self.first[node as usize + 1] as usize]
    }

    pub fn edge(&self, from: u32, to: u32) -> Option<Edge> {
        self.outgoing(from)
            .iter()
            .copied()
            .find(|edge| edge.to == to)
    }

    fn incoming(&self, node: u32) -> &[(u32, f32)] {
        &self.reverse[self.reverse_first[node as usize] as usize
            ..self.reverse_first[node as usize + 1] as usize]
    }

    /// The node a player at `origin` stands on: the highest node of its column not above its
    /// feet, else the nearest in the columns around.
    pub fn nearest(&self, origin: Vec3) -> Option<u32> {
        let column = (
            (origin[0] / self.cell).floor() as i32,
            (origin[1] / self.cell).floor() as i32,
        );
        let below = |nodes: &Vec<u32>| {
            nodes
                .iter()
                .copied()
                .filter(|&node| {
                    let z = self.nodes[node as usize][2];
                    z <= origin[2] + STEP_HEIGHT + 1.0 && z >= origin[2] - 160.0
                })
                .max_by(|&a, &b| self.nodes[a as usize][2].total_cmp(&self.nodes[b as usize][2]))
        };
        if let Some(node) = self.columns.get(&column).and_then(below) {
            return Some(node);
        }
        let mut best = None;
        let mut best_distance = f32::MAX;
        for ring in 1..=3 {
            for x in column.0 - ring..=column.0 + ring {
                for y in column.1 - ring..=column.1 + ring {
                    let Some(nodes) = self.columns.get(&(x, y)) else {
                        continue;
                    };
                    for &node in nodes {
                        let point = self.nodes[node as usize];
                        if (point[2] - origin[2]).abs() > 96.0 {
                            continue;
                        }
                        let distance = crate::sim::distance(point, origin);
                        if distance < best_distance {
                            best_distance = distance;
                            best = Some(node);
                        }
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        best
    }
}

/// Routes toward goals, cached per goal (most goals stay put: items, spawns): travel time of every
/// node to the goal and the next node on the way (van Waveren 2001, §6.5, without clusters).
pub struct RouteCache {
    goals: Vec<CachedGoal>,
    clock: u64,
    capacity: usize,
}

struct CachedGoal {
    goal: u32,
    used: u64,
    times: Vec<f32>,
    next: Vec<u32>,
}

impl RouteCache {
    /// Room for every goal of the arena (health, items, spawns) and a few more.
    pub fn for_arena(arena: &Arena) -> Self {
        Self::new(arena.health.len() + arena.spots.len() + arena.spawns.len() + 8)
    }

    pub fn new(capacity: usize) -> Self {
        Self {
            goals: Vec::new(),
            clock: 0,
            capacity: capacity.max(1),
        }
    }

    fn goal(&mut self, nav: &NavGraph, goal: u32) -> &CachedGoal {
        self.clock += 1;
        let clock = self.clock;
        if let Some(index) = self.goals.iter().position(|cached| cached.goal == goal) {
            self.goals[index].used = clock;
            return &self.goals[index];
        }
        let count = nav.nodes.len();
        let mut times = vec![f32::INFINITY; count];
        let mut next = vec![u32::MAX; count];
        let mut heap = BinaryHeap::new();
        times[goal as usize] = 0.0;
        next[goal as usize] = goal;
        heap.push(Reverse((0u32, goal)));
        while let Some(Reverse((bits, node))) = heap.pop() {
            let time = f32::from_bits(bits);
            if time > times[node as usize] {
                continue;
            }
            for &(from, edge_time) in nav.incoming(node) {
                let total = time + edge_time;
                if total < times[from as usize] {
                    times[from as usize] = total;
                    next[from as usize] = node;
                    heap.push(Reverse((total.to_bits(), from)));
                }
            }
        }
        let cached = CachedGoal {
            goal,
            used: clock,
            times,
            next,
        };
        if self.goals.len() >= self.capacity {
            let oldest = (0..self.goals.len())
                .min_by_key(|&index| self.goals[index].used)
                .expect("capacity ≥ 1");
            self.goals[oldest] = cached;
            return &self.goals[oldest];
        }
        self.goals.push(cached);
        self.goals.last().expect("pushed")
    }

    /// Seconds from `from` to `goal` and the next node on the way (`goal` itself once there).
    pub fn route(&mut self, nav: &NavGraph, from: u32, goal: u32) -> Option<(f32, u32)> {
        if from as usize >= nav.nodes.len() || goal as usize >= nav.nodes.len() {
            return None;
        }
        let cached = self.goal(nav, goal);
        let time = cached.times[from as usize];
        time.is_finite().then(|| (time, cached.next[from as usize]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::arena::box_world;

    /// Standing origin on a floor whose top is at `z`.
    fn standing(x: f32, y: f32, z: f32) -> Vec3 {
        [x + 16.0, y + 16.0, z + 24.0 + 1.0 / 32.0]
    }

    fn has_edge(nav: &NavGraph, from: Vec3, to: Vec3, kinds: &[EdgeKind]) -> bool {
        let (Some(from), Some(to)) = (nav.nearest(from), nav.nearest(to)) else {
            return false;
        };
        nav.outgoing(from)
            .iter()
            .any(|edge| edge.to == to && kinds.contains(&edge.kind))
    }

    fn route_time(nav: &NavGraph, from: Vec3, to: Vec3) -> Option<f32> {
        let (from, to) = (nav.nearest(from)?, nav.nearest(to)?);
        RouteCache::new(4)
            .route(nav, from, to)
            .map(|(time, _)| time)
    }

    #[test]
    fn flat_floor_cells_link_by_walk_edges() {
        let world = box_world(&[([-256, -256, -64], [256, 256, 0])]);
        let nav = NavGraph::build(&world, &[standing(0.0, 0.0, 0.0)]);
        // 16 × 16 cells, the outer ring is too close to the void edge for the box? no: all inside
        assert!(nav.nodes.len() >= 200, "nodes {}", nav.nodes.len());
        assert!(has_edge(
            &nav,
            standing(0.0, 0.0, 0.0),
            standing(32.0, 0.0, 0.0),
            &[EdgeKind::Walk]
        ));
        let time = route_time(&nav, standing(-192.0, 0.0, 0.0), standing(192.0, 0.0, 0.0)).unwrap();
        assert!((time - 384.0 / 320.0).abs() < 0.05, "time {time}");
    }

    #[test]
    fn step_18_is_walked_up_and_step_40_needs_a_jump() {
        // Floor, then a platform 18 high (x 128..256) and one 40 high (x 384..512)
        let world = box_world(&[
            ([-128, -128, -64], [640, 128, 0]),
            ([128, -128, 0], [256, 128, 18]),
            ([384, -128, 0], [512, 128, 40]),
        ]);
        let nav = NavGraph::build(&world, &[standing(0.0, 0.0, 0.0)]);
        assert!(has_edge(
            &nav,
            standing(96.0, 0.0, 0.0),
            standing(128.0, 0.0, 18.0),
            &[EdgeKind::Walk]
        ));
        let high = nav.nearest(standing(448.0, 0.0, 40.0)).unwrap();
        assert!((nav.nodes[high as usize][2] - standing(0.0, 0.0, 40.0)[2]).abs() < 1.0);
        let jumped_up = (0..nav.nodes.len() as u32).any(|from| {
            nav.outgoing(from).iter().any(|edge| {
                nav.nodes[edge.to as usize][2] > 60.0
                    && matches!(
                        edge.kind,
                        EdgeKind::Jump { .. } | EdgeKind::StandingJump { .. }
                    )
            })
        });
        assert!(jumped_up, "no jump edge onto the 40 platform");
        assert!(route_time(&nav, standing(0.0, 0.0, 0.0), standing(448.0, 0.0, 40.0)).is_some());
    }

    #[test]
    fn ledge_drop_is_one_way() {
        // Upper floor (x < 0, top 128) and lower floor (x ≥ 0, top 0), no way back up
        let world = box_world(&[
            ([-512, -128, -64], [0, 128, 128]),
            ([0, -128, -64], [512, 128, 0]),
        ]);
        let (upper, lower) = (standing(-64.0, 0.0, 128.0), standing(64.0, 0.0, 0.0));
        let nav = NavGraph::build(&world, &[upper]);
        assert!(route_time(&nav, upper, lower).is_some());
        assert!(route_time(&nav, lower, upper).is_none());
    }

    #[test]
    fn gap_of_128_is_jumped_and_gap_of_320_is_not() {
        let gap = |width: i32| {
            box_world(&[
                ([-512, -128, -64], [0, 128, 0]),
                ([width, -128, -64], [width + 512, 128, 0]),
                // A deep pit floor far below with walls, so falling in is a dead end
                ([-512, -128, -1024], [width + 512, 128, -960]),
            ])
        };
        let near = gap(128);
        let nav = NavGraph::build(&near, &[standing(-64.0, 0.0, 0.0)]);
        assert!(
            route_time(&nav, standing(-64.0, 0.0, 0.0), standing(192.0, 0.0, 0.0)).is_some(),
            "128 gap not crossed"
        );
        let far = gap(320);
        let nav = NavGraph::build(&far, &[standing(-64.0, 0.0, 0.0)]);
        assert!(
            nav.nearest(standing(384.0, 0.0, 0.0)).is_none()
                || route_time(&nav, standing(-64.0, 0.0, 0.0), standing(384.0, 0.0, 0.0)).is_none(),
            "320 gap crossed"
        );
    }

    #[test]
    fn route_prefers_lower_travel_time() {
        // A wall with a gap near one end: the route goes through the gap, longer than straight
        let world = box_world(&[
            ([-512, -512, -64], [512, 512, 0]),
            ([-16, -512, 0], [16, 384, 128]),
        ]);
        let nav = NavGraph::build(&world, &[standing(-256.0, 0.0, 0.0)]);
        let time = route_time(&nav, standing(-256.0, 0.0, 0.0), standing(224.0, 0.0, 0.0)).unwrap();
        let straight = 480.0 / 320.0;
        assert!(time > straight * 1.3, "time {time} vs straight {straight}");
    }

    #[test]
    fn nearest_node_picks_floor_below_not_floor_above() {
        // Two floors stacked: z 0 and a slab at 128..144 over half of it
        let world = box_world(&[
            ([-256, -256, -64], [256, 256, 0]),
            ([-256, -256, 128], [0, 256, 144]),
        ]);
        let nav = NavGraph::build(
            &world,
            &[standing(-128.0, 0.0, 0.0), standing(-128.0, 0.0, 144.0)],
        );
        let node = nav.nearest(standing(-128.0, 0.0, 0.0)).unwrap();
        assert!(nav.nodes[node as usize][2] < 50.0);
    }
}
