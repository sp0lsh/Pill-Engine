#ifndef BVH_TRACE_HLSL
#define BVH_TRACE_HLSL
// Triangle BVH traversal in compute (no ray-tracing hardware), after WickedEngine's
// TraceRay_Closest / TraceRay_Any (raytracingHF.hlsli): a stack of node indices, slab test per
// node, Möller–Trumbore per triangle. Nodes and triangles are built by ddgi/bvh.rs.
// The includer declares `bvh_nodes` and `bvh_triangles`.
//
// Alpha-tested triangles (coverage below 1 in the albedo alpha) stop a ray with probability
// `coverage`, decided by a hash of the ray's `seed` and the triangle: partial occlusion that the
// probes' averaging over rays and frames turns into the texture's average opacity. Their hits
// are never backfaces (a leaf card has no inside).

struct BvhNode {
    float3 min;
    uint   left_or_first; // interior: right child (left = this + 1); leaf: first triangle
    float3 max;
    uint   count;         // 0 = interior
};
struct BvhTriangle {
    float3 v0;
    uint   albedo; // RGBA8, r in the low byte; alpha = coverage
    float3 e1;
    uint   pad0;
    float3 e2;
    uint   pad1;
};
struct BvhHit {
    float distance; // ray length to the hit; `found` false = miss
    uint  triangle;
    bool  found;
    bool  backface;
};

// Mirrors bvh.rs BVH_STACK_SIZE: the builder caps the tree depth so both children of every
// interior node fit
static const uint BVH_STACK_SIZE = 32;

// Entry distance of the ray into the box, or a huge value when it misses or starts beyond `tmax`
float bvh_box_entry(BvhNode node, float3 origin, float3 inv_direction, float tmax) {
    float3 t0   = (node.min - origin) * inv_direction;
    float3 t1   = (node.max - origin) * inv_direction;
    float3 near = min(t0, t1);
    float3 far  = max(t0, t1);
    float  t_in  = max(max(near.x, near.y), max(near.z, 0.0));
    float  t_out = min(min(far.x, far.y), min(far.z, tmax));
    return t_in <= t_out ? t_in : 3.4e38;
}

uint bvh_hash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Whether the triangle stops this ray (always, unless alpha-tested)
bool bvh_blocks(uint triangle_index, uint seed) {
    uint coverage = bvh_triangles[triangle_index].albedo >> 24;
    return coverage == 255u || (bvh_hash(seed ^ bvh_hash(triangle_index)) & 255u) < coverage;
}

// Möller–Trumbore; returns the ray length to the hit or a huge value, and whether the hit side
// faces away from the ray (counter-clockwise front faces, as the PBR pipeline culls)
float bvh_triangle(BvhTriangle triangle, float3 origin, float3 direction, out bool backface) {
    backface = false;
    float3 pvec = cross(direction, triangle.e2);
    float  det  = dot(triangle.e1, pvec);
    if (abs(det) < 1e-9) {
        return 3.4e38;
    }
    float  inv_det = 1.0 / det;
    float3 tvec    = origin - triangle.v0;
    float  u       = dot(tvec, pvec) * inv_det;
    float3 qvec    = cross(tvec, triangle.e1);
    float  v       = dot(direction, qvec) * inv_det;
    float  t       = dot(triangle.e2, qvec) * inv_det;
    backface = det < 0.0;
    return (u < 0.0 || u > 1.0 || v < 0.0 || u + v > 1.0 || t <= 0.0) ? 3.4e38 : t;
}

BvhHit bvh_trace_closest(float3 origin, float3 direction, float tmax, uint seed) {
    BvhHit hit;
    hit.distance = tmax;
    hit.triangle = 0;
    hit.found    = false;
    hit.backface = false;
    float3 inv_direction = 1.0 / direction;
    uint stack[BVH_STACK_SIZE];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        uint    index = stack[--depth];
        BvhNode node  = bvh_nodes[index];
        if (bvh_box_entry(node, origin, inv_direction, hit.distance) > hit.distance) {
            continue;
        }
        if (node.count > 0) {
            for (uint i = 0; i < node.count; i++) {
                uint triangle_index = node.left_or_first + i;
                bool backface;
                float t = bvh_triangle(bvh_triangles[triangle_index], origin, direction, backface);
                if (t < hit.distance && bvh_blocks(triangle_index, seed)) {
                    hit.distance = t;
                    hit.triangle = triangle_index;
                    hit.found    = true;
                    hit.backface = backface && (bvh_triangles[triangle_index].albedo >> 24) == 255u;
                }
            }
        } else {
            stack[depth++] = node.left_or_first;
            stack[depth++] = index + 1;
        }
    }
    return hit;
}

// Whether anything blocks the ray before `tmax` (shadow rays)
bool bvh_trace_any(float3 origin, float3 direction, float tmax, uint seed) {
    float3 inv_direction = 1.0 / direction;
    uint stack[BVH_STACK_SIZE];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        uint    index = stack[--depth];
        BvhNode node  = bvh_nodes[index];
        if (bvh_box_entry(node, origin, inv_direction, tmax) > tmax) {
            continue;
        }
        if (node.count > 0) {
            for (uint i = 0; i < node.count; i++) {
                uint triangle_index = node.left_or_first + i;
                bool backface;
                if (bvh_triangle(bvh_triangles[triangle_index], origin, direction, backface) < tmax
                    && bvh_blocks(triangle_index, seed)) {
                    return true;
                }
            }
        } else {
            stack[depth++] = node.left_or_first;
            stack[depth++] = index + 1;
        }
    }
    return false;
}

float3 bvh_triangle_albedo(uint triangle_index) {
    uint packed = bvh_triangles[triangle_index].albedo;
    return float3(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF) / 255.0;
}

float3 bvh_triangle_normal(uint triangle_index) {
    BvhTriangle triangle = bvh_triangles[triangle_index];
    return normalize(cross(triangle.e1, triangle.e2));
}
#endif
