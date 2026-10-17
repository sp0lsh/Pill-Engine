//! Triangle BVH for tracing DDGI probe rays in compute (no ray-tracing hardware in WebGPU).
//!
//! Built on the CPU with binned SAH, flattened depth-first: an interior node's left child is the
//! next node, its right child is stored. The GPU layout is `BvhNode` / `BvhTriangle` of
//! include/bvh_trace.hlsl; traversal follows WickedEngine's `TraceRay_Closest` (stack of node
//! indices, Möller–Trumbore triangle test).

use crate::graphics::shader_bindings::structs::{BvhNode, BvhTriangle};

use pill_core::Vector3f;

/// Triangles per leaf below which a node is not split.
const LEAF_SIZE: usize = 4;
/// SAH bins per split axis.
const BIN_COUNT: usize = 12;
/// Node-index stack of the traversal in include/bvh_trace.hlsl (`BVH_STACK_SIZE`). It pushes
/// both children of an interior node, so the stack holds at most one entry per level plus one,
/// plus the static tree's root while the dynamic tree is traversed.
pub(crate) const BVH_STACK_SIZE: usize = 32;
/// Deepest node level (root = 0); deeper subtrees become one leaf so traversal never needs more
/// than `BVH_STACK_SIZE` entries.
pub(crate) const MAX_DEPTH: usize = BVH_STACK_SIZE - 3;

/// A world-space triangle with the diffuse albedo of its material (linear RGB) and the share of
/// rays it stops (1 = opaque; less for alpha-tested materials, whose texture has holes).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Triangle {
    pub(crate) vertices: [Vector3f; 3],
    pub(crate) albedo: [f32; 3],
    pub(crate) coverage: f32,
}

/// A built BVH, ready to upload; no nodes when there are no triangles.
pub(crate) struct Bvh {
    pub(crate) nodes: Vec<BvhNode>,
    pub(crate) triangles: Vec<BvhTriangle>,
    /// Deepest node level (root = 0).
    pub(crate) depth: usize,
}

#[derive(Clone, Copy)]
struct Bounds {
    min: Vector3f,
    max: Vector3f,
}

impl Bounds {
    const EMPTY: Self = Self {
        min: Vector3f::splat(f32::INFINITY),
        max: Vector3f::splat(f32::NEG_INFINITY),
    };

    fn grow(&mut self, point: Vector3f) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    fn merge(&mut self, other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn half_area(&self) -> f32 {
        let size = (self.max - self.min).max(Vector3f::ZERO);
        size.x * size.y + size.y * size.z + size.z * size.x
    }
}

impl Bvh {
    /// Binned-SAH BVH over `triangles`; degenerate (zero-area) triangles are dropped.
    pub(crate) fn build(triangles: &[Triangle]) -> Self {
        let triangles: Vec<&Triangle> = triangles
            .iter()
            .filter(|triangle| {
                let [a, b, c] = triangle.vertices;
                (b - a).cross(c - a).length_squared() > 1e-20
            })
            .collect();
        let centroids: Vec<Vector3f> = triangles
            .iter()
            .map(|triangle| triangle.vertices.iter().copied().sum::<Vector3f>() / 3.0)
            .collect();
        let mut order: Vec<u32> = (0..triangles.len() as u32).collect();
        let mut builder = Builder {
            triangles: &triangles,
            centroids: &centroids,
            nodes: Vec::with_capacity(triangles.len() * 2 / LEAF_SIZE + 1),
            depth: 0,
        };
        if !triangles.is_empty() {
            builder.build(&mut order, 0, 0);
        }
        let (nodes, depth) = (builder.nodes, builder.depth);
        let triangles = order
            .iter()
            .map(|&index| {
                let triangle = triangles[index as usize];
                let [a, b, c] = triangle.vertices;
                BvhTriangle {
                    v0: a.to_array(),
                    albedo: pack_rgba8(triangle.albedo, triangle.coverage),
                    e1: (b - a).to_array(),
                    pad0: 0,
                    e2: (c - a).to_array(),
                    pad1: 0,
                }
            })
            .collect();
        Self {
            nodes,
            triangles,
            depth,
        }
    }

    /// As a tree stored after `node_offset` nodes and `triangle_offset` triangles of another (the
    /// dynamic tree after the static one, traversed from root `node_offset`).
    pub(crate) fn offset(&mut self, node_offset: u32, triangle_offset: u32) {
        for node in &mut self.nodes {
            node.left_or_first += if node.count > 0 {
                triangle_offset
            } else {
                node_offset
            };
        }
    }

    /// Bounds of the whole scene, `None` when empty.
    pub(crate) fn bounds(&self) -> Option<(Vector3f, Vector3f)> {
        (!self.triangles.is_empty()).then(|| {
            let root = &self.nodes[0];
            (
                Vector3f::from_array(root.min),
                Vector3f::from_array(root.max),
            )
        })
    }
}

/// Linear RGB and alpha in 0..1 → RGBA8 (r in the low byte), as `bvh_triangle_albedo` and
/// `bvh_blocks` in bvh_trace.hlsl read it.
fn pack_rgba8([r, g, b]: [f32; 3], alpha: f32) -> u32 {
    let byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    byte(r) | byte(g) << 8 | byte(b) << 16 | byte(alpha) << 24
}

struct Builder<'a> {
    triangles: &'a [&'a Triangle],
    centroids: &'a [Vector3f],
    nodes: Vec<BvhNode>,
    depth: usize,
}

impl Builder<'_> {
    /// Builds the subtree of `order` (triangle indices, reordered in place) whose first
    /// triangle has index `first` in the final triangle array, at node level `depth`; returns
    /// its node index.
    fn build(&mut self, order: &mut [u32], first: usize, depth: usize) -> usize {
        let mut bounds = Bounds::EMPTY;
        let mut centroid_bounds = Bounds::EMPTY;
        for &index in order.iter() {
            for &vertex in &self.triangles[index as usize].vertices {
                bounds.grow(vertex);
            }
            centroid_bounds.grow(self.centroids[index as usize]);
        }
        let node_index = self.nodes.len();
        self.nodes.push(BvhNode {
            min: bounds.min.to_array(),
            left_or_first: first as u32,
            max: bounds.max.to_array(),
            count: order.len() as u32,
        });
        self.depth = self.depth.max(depth);
        if order.len() <= LEAF_SIZE || depth == MAX_DEPTH {
            return node_index;
        }
        let Some(split) = self.split(order, &bounds, &centroid_bounds) else {
            return node_index;
        };
        let (left, right) = order.split_at_mut(split);
        self.build(left, first, depth + 1);
        let right_index = self.build(right, first + split, depth + 1);
        self.nodes[node_index].left_or_first = right_index as u32;
        self.nodes[node_index].count = 0;
        node_index
    }

    /// Partitions `order` at the cheapest SAH bin boundary of the widest centroid axis; returns
    /// the split position, or `None` when a leaf is cheaper (or the centroids coincide).
    fn split(&self, order: &mut [u32], bounds: &Bounds, centroid_bounds: &Bounds) -> Option<usize> {
        let extent = centroid_bounds.max - centroid_bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        if extent[axis] <= 1e-9 {
            // All centroids at one point: split in the middle so huge leaves don't form
            return (order.len() > LEAF_SIZE * 4).then_some(order.len() / 2);
        }
        let bin_of = |index: u32| {
            let offset = self.centroids[index as usize][axis] - centroid_bounds.min[axis];
            ((offset / extent[axis] * BIN_COUNT as f32) as usize).min(BIN_COUNT - 1)
        };
        let mut bins = [(Bounds::EMPTY, 0usize); BIN_COUNT];
        for &index in order.iter() {
            let bin = &mut bins[bin_of(index)];
            for &vertex in &self.triangles[index as usize].vertices {
                bin.0.grow(vertex);
            }
            bin.1 += 1;
        }
        // Sweep from the right, then from the left: cost of splitting after each bin
        let mut right_area = [0.0f32; BIN_COUNT];
        let mut accumulated = (Bounds::EMPTY, 0usize);
        for bin in (1..BIN_COUNT).rev() {
            accumulated.0.merge(&bins[bin].0);
            accumulated.1 += bins[bin].1;
            right_area[bin - 1] = accumulated.0.half_area() * accumulated.1 as f32;
        }
        let mut best = (f32::INFINITY, 0usize);
        let mut accumulated = (Bounds::EMPTY, 0usize);
        for bin in 0..BIN_COUNT - 1 {
            accumulated.0.merge(&bins[bin].0);
            accumulated.1 += bins[bin].1;
            let cost = accumulated.0.half_area() * accumulated.1 as f32 + right_area[bin];
            if accumulated.1 > 0 && accumulated.1 < order.len() && cost < best.0 {
                best = (cost, bin);
            }
        }
        let leaf_cost = bounds.half_area() * order.len() as f32;
        if best.0 >= leaf_cost && order.len() <= LEAF_SIZE * 4 {
            return None;
        }
        if best.0.is_infinite() {
            return Some(order.len() / 2);
        }
        let mut split = 0;
        for position in 0..order.len() {
            if bin_of(order[position]) <= best.1 {
                order.swap(position, split);
                split += 1;
            }
        }
        Some(split)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closest hit distance by Möller–Trumbore, as bvh_trace.hlsl tests triangles.
    fn intersect(origin: Vector3f, direction: Vector3f, triangle: &BvhTriangle) -> Option<f32> {
        let (v0, e1, e2) = (
            Vector3f::from_array(triangle.v0),
            Vector3f::from_array(triangle.e1),
            Vector3f::from_array(triangle.e2),
        );
        let pvec = direction.cross(e2);
        let det = e1.dot(pvec);
        if det.abs() < 1e-8 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = origin - v0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(e1);
        let v = direction.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = e2.dot(qvec) * inv_det;
        (t > 0.0).then_some(t)
    }

    fn intersect_box(origin: Vector3f, inv_direction: Vector3f, node: &BvhNode) -> bool {
        let t0 = (Vector3f::from_array(node.min) - origin) * inv_direction;
        let t1 = (Vector3f::from_array(node.max) - origin) * inv_direction;
        let near = t0.min(t1).max_element();
        let far = t0.max(t1).min_element();
        far >= near.max(0.0)
    }

    /// Stack traversal in the shader's order, with its fixed stack: panics where the shader
    /// would have to drop a subtree.
    fn trace(bvh: &Bvh, origin: Vector3f, direction: Vector3f) -> Option<f32> {
        trace_roots(bvh, origin, direction, 0)
    }

    /// `trace` from root 0 and, unless 0, `dynamic_root` (traversed first, as the shader does).
    fn trace_roots(
        bvh: &Bvh,
        origin: Vector3f,
        direction: Vector3f,
        dynamic_root: usize,
    ) -> Option<f32> {
        let inv_direction = direction.recip();
        let mut best: Option<f32> = None;
        let mut stack = vec![0usize];
        if dynamic_root != 0 {
            stack.push(dynamic_root);
        }
        while let Some(index) = stack.pop() {
            assert!(
                stack.len() + 2 <= BVH_STACK_SIZE,
                "traversal stack overflow"
            );
            let node = &bvh.nodes[index];
            if !intersect_box(origin, inv_direction, node) {
                continue;
            }
            if node.count > 0 {
                let first = node.left_or_first as usize;
                for triangle in &bvh.triangles[first..first + node.count as usize] {
                    if let Some(t) = intersect(origin, direction, triangle) {
                        best = Some(best.map_or(t, |current: f32| current.min(t)));
                    }
                }
            } else {
                stack.push(node.left_or_first as usize);
                stack.push(index + 1);
            }
        }
        best
    }

    fn random_triangles(count: usize) -> Vec<Triangle> {
        let mut seed = 12345u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        (0..count)
            .map(|_| {
                let center = Vector3f::new(
                    next() * 20.0 - 10.0,
                    next() * 20.0 - 10.0,
                    next() * 20.0 - 10.0,
                );
                let mut vertex =
                    || center + Vector3f::new(next() - 0.5, next() - 0.5, next() - 0.5) * 2.0;
                Triangle {
                    vertices: [vertex(), vertex(), vertex()],
                    albedo: [0.5; 3],
                    coverage: 1.0,
                }
            })
            .collect()
    }

    #[test]
    fn closest_hit_matches_brute_force() {
        let triangles = random_triangles(2000);
        let bvh = Bvh::build(&triangles);
        let mut hits = 0;
        for index in 0..500 {
            let t = index as f32;
            let origin = Vector3f::new(
                (t * 0.37).sin() * 12.0,
                (t * 0.71).cos() * 12.0,
                (t * 1.3).sin() * 12.0,
            );
            let direction =
                Vector3f::new((t * 2.1).cos(), (t * 0.9).sin(), (t * 1.7).cos()).normalize();
            let brute = bvh
                .triangles
                .iter()
                .filter_map(|triangle| intersect(origin, direction, triangle))
                .reduce(f32::min);
            let traced = trace(&bvh, origin, direction);
            assert_eq!(brute.is_some(), traced.is_some(), "ray {index}");
            if let (Some(a), Some(b)) = (brute, traced) {
                assert!((a - b).abs() < 1e-4, "ray {index}: {a} vs {b}");
                hits += 1;
            }
        }
        assert!(hits > 50, "only {hits} hits");
    }

    #[test]
    fn every_node_bounds_its_triangles_and_leaves_cover_all_triangles() {
        let triangles = random_triangles(1000);
        let bvh = Bvh::build(&triangles);
        let mut covered = vec![false; bvh.triangles.len()];
        // Each node's box contains the boxes of its subtree's triangles
        fn check(bvh: &Bvh, index: usize, covered: &mut [bool]) -> (Vector3f, Vector3f) {
            let node = bvh.nodes[index];
            let (min, max) = (
                Vector3f::from_array(node.min),
                Vector3f::from_array(node.max),
            );
            let (sub_min, sub_max) = if node.count > 0 {
                let first = node.left_or_first as usize;
                let mut bounds = Bounds::EMPTY;
                for (offset, triangle) in bvh.triangles[first..first + node.count as usize]
                    .iter()
                    .enumerate()
                {
                    covered[first + offset] = true;
                    let v0 = Vector3f::from_array(triangle.v0);
                    for vertex in [
                        v0,
                        v0 + Vector3f::from_array(triangle.e1),
                        v0 + Vector3f::from_array(triangle.e2),
                    ] {
                        bounds.grow(vertex);
                    }
                }
                (bounds.min, bounds.max)
            } else {
                let left = check(bvh, index + 1, covered);
                let right = check(bvh, node.left_or_first as usize, covered);
                (left.0.min(right.0), left.1.max(right.1))
            };
            assert!(min.cmple(sub_min + 1e-4).all() && max.cmpge(sub_max - 1e-4).all());
            (min, max)
        }
        check(&bvh, 0, &mut covered);
        assert!(covered.iter().all(|&c| c));
        assert!(bvh
            .nodes
            .iter()
            .filter(|node| node.count > 0)
            .all(|leaf| leaf.count as usize <= LEAF_SIZE * 4));
    }

    #[test]
    fn empty_scene_has_no_nodes_and_no_bounds() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.nodes.is_empty());
        assert!(bvh.bounds().is_none());
    }

    #[test]
    fn deep_input_is_capped_at_max_depth_and_traces_within_the_stack() {
        // Triangles at geometrically shrinking offsets: each SAH split peels off a few, so an
        // uncapped tree would be about as deep as the triangle count
        let triangles: Vec<Triangle> = (0..400)
            .map(|index| {
                let x = 0.9f32.powi(index) * 100.0;
                Triangle {
                    vertices: [
                        Vector3f::new(x, 0.0, 0.0),
                        Vector3f::new(x, 1.0, 0.0),
                        Vector3f::new(x, 0.0, 1.0),
                    ],
                    albedo: [0.5; 3],
                    coverage: 1.0,
                }
            })
            .collect();
        let bvh = Bvh::build(&triangles);
        assert!(bvh.depth <= MAX_DEPTH, "depth {}", bvh.depth);
        for index in 0..100 {
            let y = 0.1 + index as f32 * 0.003;
            let origin = Vector3f::new(-1.0, y, 0.2);
            let brute = bvh
                .triangles
                .iter()
                .filter_map(|triangle| intersect(origin, Vector3f::X, triangle))
                .reduce(f32::min);
            let traced = trace(&bvh, origin, Vector3f::X);
            assert_eq!(brute.is_some(), traced.is_some(), "ray {index}");
        }
    }

    #[test]
    fn albedo_packs_red_in_the_low_byte() {
        assert_eq!(pack_rgba8([1.0, 0.0, 0.0], 1.0), 0xFF0000FF);
        assert_eq!(pack_rgba8([0.0, 0.0, 1.0], 1.0) & 0x00FF0000, 0x00FF0000);
        assert_eq!(pack_rgba8([0.0; 3], 0.5) >> 24, 128);
    }

    #[test]
    fn bvh_dynamic_tree_after_static_offset_hit() {
        let triangle = |z: f32| Triangle {
            vertices: [
                Vector3f::new(-1.0, -1.0, z),
                Vector3f::new(1.0, -1.0, z),
                Vector3f::new(0.0, 1.0, z),
            ],
            albedo: [1.0; 3],
            coverage: 1.0,
        };
        // Static: a triangle at z = -10; dynamic: one at z = -5, nearer to a ray from the origin
        let mut combined = Bvh::build(&[triangle(-10.0)]);
        let mut dynamic = Bvh::build(&[triangle(-5.0)]);
        let root = combined.nodes.len();
        dynamic.offset(root as u32, combined.triangles.len() as u32);
        combined.nodes.extend(dynamic.nodes);
        combined.triangles.extend(dynamic.triangles);
        let direction = Vector3f::new(0.0, 0.0, -1.0);
        let hit = trace_roots(&combined, Vector3f::ZERO, direction, root).unwrap();
        assert!((hit - 5.0).abs() < 1e-4, "{hit}");
        // Without the dynamic root only the static triangle is found
        assert!((trace(&combined, Vector3f::ZERO, direction).unwrap() - 10.0).abs() < 1e-4);
    }
}
