//! Fly-through path of a level for the tour capture: from a start past the level's lights
//! (nearest first), A* on a grid of free space (the level's collision, a camera-sized box),
//! shortened by line of sight, rounded at the corners and sampled at a constant speed. All in Quake units; `CameraSpot`s come out in engine
//! space.

use crate::game::CameraSpot;

use pill_engine::game::{
    quake_map::{engine_to_quake, quake_to_engine, QuakeCollision},
    Vector3f,
};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Grid spacing and the half extent of the box kept free around the camera (units): wide enough
/// to stay off walls, narrow enough for doorways.
const CELL: f32 = 32.0;
const CAMERA_HALF: [f32; 3] = [24.0, 24.0, 20.0];
/// Cells farther than this above the floor cost more, so the path stays at walking height.
const FLOOR_REACH: f32 = 96.0;
const HIGH_COST: f32 = 4.0;
/// Expanded cells per leg before giving up (the leg is then a straight line).
const MAX_EXPANDED: usize = 200_000;
/// How far ahead (units) the camera looks along the path.
const LOOK_AHEAD: f32 = 64.0;
/// Free cell searched around a waypoint inside a wall (cells), and the spacing of the lights
/// visited (units; lights closer to a visited one are skipped).
const SNAP_CELLS: i32 = 4;
const LIGHT_SPACING: f32 = 192.0;

type Cell = [i32; 3];

struct Grid<'a> {
    collision: &'a QuakeCollision,
    origin: [f32; 3],
    free: HashMap<Cell, bool>,
}

impl Grid<'_> {
    fn center(&self, cell: Cell) -> [f32; 3] {
        [0, 1, 2].map(|axis| self.origin[axis] + cell[axis] as f32 * CELL)
    }

    fn cell_of(&self, point: [f32; 3]) -> Cell {
        [0, 1, 2].map(|axis| ((point[axis] - self.origin[axis]) / CELL).round() as i32)
    }

    fn is_free(&mut self, cell: Cell) -> bool {
        if let Some(&free) = self.free.get(&cell) {
            return free;
        }
        let point = self.center(cell);
        let free = !self
            .collision
            .trace_box(point, point, CAMERA_HALF)
            .start_solid;
        self.free.insert(cell, free);
        free
    }

    fn clear(&self, from: [f32; 3], to: [f32; 3]) -> bool {
        let trace = self.collision.trace_box(from, to, CAMERA_HALF);
        trace.fraction >= 1.0 && !trace.start_solid
    }

    fn near_floor(&self, cell: Cell) -> bool {
        let point = self.center(cell);
        let below = [point[0], point[1], point[2] - FLOOR_REACH];
        self.collision.trace_ray(point, below).fraction < 1.0
    }

    /// Cell centers from `start` to `goal`, or `None` when no path was found in time.
    fn find_path(&mut self, start: [f32; 3], goal: [f32; 3]) -> Option<Vec<[f32; 3]>> {
        let (start_cell, goal_cell) = (self.cell_of(start), self.cell_of(goal));
        let heuristic = |cell: Cell| {
            let d = [0, 1, 2].map(|axis| (cell[axis] - goal_cell[axis]) as f32);
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        // Costs are scaled to integers for the heap
        let mut open = BinaryHeap::new();
        let mut cost: HashMap<Cell, f32> = HashMap::new();
        let mut parent: HashMap<Cell, Cell> = HashMap::new();
        cost.insert(start_cell, 0.0);
        open.push(Reverse((
            (heuristic(start_cell) * 1000.0) as u64,
            start_cell,
        )));
        let mut expanded = 0;
        while let Some(Reverse((_, cell))) = open.pop() {
            if cell == goal_cell {
                let mut path = vec![goal];
                let mut current = cell;
                while let Some(&previous) = parent.get(&current) {
                    path.push(self.center(previous));
                    current = previous;
                }
                path.pop();
                path.push(start);
                path.reverse();
                return Some(path);
            }
            expanded += 1;
            if expanded > MAX_EXPANDED {
                return None;
            }
            let base = cost[&cell];
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let next = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                        if next != goal_cell && !self.is_free(next) {
                            continue;
                        }
                        if !self.clear(self.center(cell), self.center(next)) {
                            continue;
                        }
                        let step = ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
                        let penalty = if self.near_floor(next) {
                            1.0
                        } else {
                            HIGH_COST
                        };
                        let total = base + step * penalty;
                        if cost.get(&next).is_some_and(|&known| known <= total) {
                            continue;
                        }
                        cost.insert(next, total);
                        parent.insert(next, cell);
                        let priority = ((total + heuristic(next)) * 1000.0) as u64;
                        open.push(Reverse((priority, next)));
                    }
                }
            }
        }
        None
    }

    /// The nearest free cell center to `point` reachable in a straight line from it, within
    /// `SNAP_CELLS`.
    fn snap(&mut self, point: [f32; 3]) -> Option<[f32; 3]> {
        let center = self.cell_of(point);
        let mut best: Option<(f32, [f32; 3])> = None;
        for dx in -SNAP_CELLS..=SNAP_CELLS {
            for dy in -SNAP_CELLS..=SNAP_CELLS {
                for dz in -SNAP_CELLS..=SNAP_CELLS {
                    let cell = [center[0] + dx, center[1] + dy, center[2] + dz];
                    if !self.is_free(cell) {
                        continue;
                    }
                    let candidate = self.center(cell);
                    let gap = distance(point, candidate);
                    if best.is_some_and(|(nearest, _)| nearest <= gap) {
                        continue;
                    }
                    if self.collision.trace_ray(candidate, point).fraction >= 1.0 {
                        best = Some((gap, candidate));
                    }
                }
            }
        }
        best.map(|(_, candidate)| candidate)
    }

    /// Drops points the camera can skip in a straight line (string pulling).
    fn shorten(&self, path: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let mut out = vec![path[0]];
        let mut index = 0;
        while index + 1 < path.len() {
            let mut next = index + 1;
            for candidate in (index + 2..path.len()).rev() {
                if self.clear(path[index], path[candidate]) {
                    next = candidate;
                    break;
                }
            }
            out.push(path[next]);
            index = next;
        }
        out
    }
}

/// Chaikin corner cutting: rounds the turns, keeps the ends.
fn round_corners(points: &[[f32; 3]], iterations: usize) -> Vec<[f32; 3]> {
    let mut points = points.to_vec();
    for _ in 0..iterations {
        if points.len() < 3 {
            break;
        }
        let mut rounded = vec![points[0]];
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            rounded.push([0, 1, 2].map(|axis| a[axis] * 0.75 + b[axis] * 0.25));
            rounded.push([0, 1, 2].map(|axis| a[axis] * 0.25 + b[axis] * 0.75));
        }
        rounded.push(points[points.len() - 1]);
        points = rounded;
    }
    points
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = [0, 1, 2].map(|axis| b[axis] - a[axis]);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Point at arc length `at` along `points`, with the cumulative lengths `lengths`.
fn point_at(points: &[[f32; 3]], lengths: &[f32], at: f32) -> [f32; 3] {
    let at = at.clamp(0.0, lengths[lengths.len() - 1]);
    let segment = lengths
        .windows(2)
        .position(|pair| at <= pair[1])
        .unwrap_or(lengths.len() - 2);
    let span = (lengths[segment + 1] - lengths[segment]).max(1e-3);
    let t = (at - lengths[segment]) / span;
    let (a, b) = (points[segment], points[segment + 1]);
    [0, 1, 2].map(|axis| a[axis] + (b[axis] - a[axis]) * t)
}

/// Up to `count` lights to visit from `start` (engine space): each next one the nearest
/// unvisited, skipping lights within `LIGHT_SPACING` of a visited one.
pub fn light_waypoints(start: Vector3f, lights: &[Vector3f], count: usize) -> Vec<Vector3f> {
    let spacing = LIGHT_SPACING / engine_to_quake([1.0, 0.0, 0.0])[0];
    let mut left = lights.to_vec();
    let mut visited = Vec::new();
    let mut current = start;
    while visited.len() < count {
        left.retain(|light: &Vector3f| {
            visited
                .iter()
                .all(|seen: &Vector3f| seen.distance(*light) >= spacing)
        });
        let Some(nearest) = left
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance(current).total_cmp(&b.1.distance(current)))
            .map(|(index, _)| index)
        else {
            break;
        };
        current = left.swap_remove(nearest);
        visited.push(current);
    }
    visited
}

/// Camera spots along a collision-free path from `start` through `waypoints` (engine space;
/// each moved to free space beside it, dropped when there's none), `step` units apart; a leg
/// without a path found is flown straight.
pub fn fly_through(
    collision: &QuakeCollision,
    start: Vector3f,
    waypoints: &[Vector3f],
    step: f32,
) -> Vec<CameraSpot> {
    let first = engine_to_quake(start.to_array());
    let mut grid = Grid {
        collision,
        origin: first,
        free: HashMap::new(),
    };
    let mut quake = vec![first];
    quake.extend(
        waypoints
            .iter()
            .filter_map(|point| grid.snap(engine_to_quake(point.to_array()))),
    );
    let mut points = vec![first];
    for leg in quake.windows(2) {
        let path = grid
            .find_path(leg[0], leg[1])
            .map(|path| grid.shorten(&path))
            .unwrap_or_else(|| vec![leg[0], leg[1]]);
        points.extend_from_slice(&path[1..]);
    }
    let points = round_corners(&points, 3);
    let mut lengths = vec![0.0];
    for pair in points.windows(2) {
        lengths.push(lengths[lengths.len() - 1] + distance(pair[0], pair[1]));
    }
    let total = lengths[lengths.len() - 1];
    let count = (total / step.max(1e-3)).ceil() as usize + 1;
    let mut previous = (0.0, 0.0);
    (0..count)
        .map(|index| {
            let at = index as f32 * step;
            let position = Vector3f::from(quake_to_engine(point_at(&points, &lengths, at)));
            let ahead = Vector3f::from(quake_to_engine(point_at(
                &points,
                &lengths,
                at + LOOK_AHEAD,
            )));
            let direction = (ahead - position).normalize_or_zero();
            // FlyCamera: yaw 0 looks down −Z, forward = (−sin yaw, ·, −cos yaw)
            // At the end, where nothing is ahead, the last direction is kept
            let (yaw, pitch) = if direction.length_squared() > 0.0 {
                (
                    (-direction.x).atan2(-direction.z).to_degrees(),
                    direction.y.asin().to_degrees().clamp(-30.0, 30.0),
                )
            } else {
                previous
            };
            previous = (yaw, pitch);
            CameraSpot {
                position,
                yaw,
                pitch,
            }
        })
        .collect()
}
