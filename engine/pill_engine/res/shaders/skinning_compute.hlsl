// GPU skin cache (Unreal's GPU Skin Cache model): skins one instance of a source mesh into that
// instance's own vertex streams (RendererMesh position + packed attribute buffers), which every
// later pass of the frame draws unchanged. Port of pill_assets::skinning::skin and
// MeshVertex::packed_attributes (resources/mesh.rs), validated against them by
// pass_skinning.rs tests.
//
// S = Σ wᵢ·J[jᵢ] (row-major 3x4 joint matrices = joint global · inverse bind), p' = S·p,
// n' = inverse-transpose(S)·n, t' = S·t, b' = S·b. All-zero weights keep the rest vertex.
// Flat sources (no NORMAL, one triangle per vertex) use one thread per triangle and the face
// normal of the skinned triangle, like convert::face_normals for one-triangle vertices.

struct SkinInstance {
    uint joint_offset;
    uint vertex_count;
    uint flat;
    uint padding;
};

// Source (shared by every instance of one glTF primitive)
[[vk::binding(0, 0)]] StructuredBuffer<float>   rest_positions; // xyz per vertex
[[vk::binding(1, 0)]] StructuredBuffer<float4>  rest_frames;    // normal, tangent, bitangent per vertex
[[vk::binding(2, 0)]] StructuredBuffer<uint4>   skin_vertices;  // joints 0|1, joints 2|3 (u16), uv (f16x2), unused
[[vk::binding(3, 0)]] StructuredBuffer<float4>  skin_weights;

// Instance: its joint range and its RendererMesh streams (12 B per vertex each, hence flat arrays)
[[vk::binding(0, 1)]] ConstantBuffer<SkinInstance> instance;
[[vk::binding(1, 1)]] RWStructuredBuffer<float>    out_positions;
[[vk::binding(2, 1)]] RWStructuredBuffer<uint>     out_attributes; // normal, tangent (unorm10x3 + 2-bit code), uv

// Frame: every instance's joint matrices, 3 rows each
[[vk::binding(0, 2)]] StructuredBuffer<float4> joint_matrices;

static const uint THREADS = 64;
// resources/mesh.rs TANGENT_CODE_*
static const uint TANGENT_CODE_NEGATIVE = 0;
static const uint TANGENT_CODE_NO_FRAME = 1;
static const uint TANGENT_CODE_POSITIVE = 3;

struct Skinning {
    float4 rows[3];
};

Skinning blend(uint vertex) {
    float4 weights = skin_weights[vertex];
    Skinning s;
    if (all(weights == 0.0)) {
        s.rows[0] = float4(1.0, 0.0, 0.0, 0.0);
        s.rows[1] = float4(0.0, 1.0, 0.0, 0.0);
        s.rows[2] = float4(0.0, 0.0, 1.0, 0.0);
        return s;
    }
    uint4 packed = skin_vertices[vertex];
    uint joints[4] = { packed.x & 0xFFFFu, packed.x >> 16, packed.y & 0xFFFFu, packed.y >> 16 };
    s.rows[0] = float4(0.0, 0.0, 0.0, 0.0);
    s.rows[1] = float4(0.0, 0.0, 0.0, 0.0);
    s.rows[2] = float4(0.0, 0.0, 0.0, 0.0);
    for (uint i = 0; i < 4; i++) {
        uint base = (instance.joint_offset + joints[i]) * 3;
        s.rows[0] += weights[i] * joint_matrices[base];
        s.rows[1] += weights[i] * joint_matrices[base + 1];
        s.rows[2] += weights[i] * joint_matrices[base + 2];
    }
    return s;
}

float3 rest_position(uint vertex) {
    return float3(rest_positions[vertex * 3], rest_positions[vertex * 3 + 1], rest_positions[vertex * 3 + 2]);
}

float3 transform_point(Skinning s, float3 p) {
    float4 p1 = float4(p, 1.0);
    return float3(dot(s.rows[0], p1), dot(s.rows[1], p1), dot(s.rows[2], p1));
}

float3 transform_vector(Skinning s, float3 v) {
    return float3(dot(s.rows[0].xyz, v), dot(s.rows[1].xyz, v), dot(s.rows[2].xyz, v));
}

// inverse-transpose(L)·n up to a positive factor. With rows r0..r2 of L, det(L)·L⁻¹ has the
// columns r1×r2, r2×r0, r0×r1, so they are the rows of det(L)·L⁻ᵀ
float3 transform_normal(Skinning s, float3 n) {
    float3 r0 = s.rows[0].xyz;
    float3 r1 = s.rows[1].xyz;
    float3 r2 = s.rows[2].xyz;
    float3 c0 = cross(r1, r2);
    float3 c = float3(dot(c0, n), dot(cross(r2, r0), n), dot(cross(r0, r1), n));
    return dot(r0, c0) < 0.0 ? -c : c;
}

// glam normalize_or_zero
float3 normalize_or_zero(float3 v) {
    float len = length(v);
    return len > 0.0 ? v / len : float3(0.0, 0.0, 0.0);
}

// resources/mesh.rs pack_unit_vector; floor(x + 0.5) is Rust's round for x ≥ 0
uint pack_unit_vector(float3 v, uint alpha) {
    uint3 u = uint3(floor((clamp(v, -1.0, 1.0) * 0.5 + 0.5) * 1023.0 + 0.5));
    return u.x | (u.y << 10) | (u.z << 20) | (alpha << 30);
}

void write_vertex(uint vertex, float3 position, float3 normal, float3 tangent, float3 bitangent) {
    out_positions[vertex * 3]     = position.x;
    out_positions[vertex * 3 + 1] = position.y;
    out_positions[vertex * 3 + 2] = position.z;
    uint code = dot(tangent, tangent) <= 1e-12 ? TANGENT_CODE_NO_FRAME
              : dot(cross(normal, tangent), bitangent) < 0.0 ? TANGENT_CODE_NEGATIVE
              : TANGENT_CODE_POSITIVE;
    out_attributes[vertex * 3]     = pack_unit_vector(normal, 0);
    out_attributes[vertex * 3 + 1] = pack_unit_vector(tangent, code);
    out_attributes[vertex * 3 + 2] = skin_vertices[vertex].z;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    if (instance.flat == 0) {
        uint vertex = id.x;
        if (vertex >= instance.vertex_count) {
            return;
        }
        Skinning s = blend(vertex);
        write_vertex(
            vertex,
            transform_point(s, rest_position(vertex)),
            normalize_or_zero(transform_normal(s, rest_frames[vertex * 3].xyz)),
            normalize_or_zero(transform_vector(s, rest_frames[vertex * 3 + 1].xyz)),
            normalize_or_zero(transform_vector(s, rest_frames[vertex * 3 + 2].xyz)));
        return;
    }
    uint first = id.x * 3;
    if (first >= instance.vertex_count) {
        return;
    }
    Skinning s[3] = { blend(first), blend(first + 1), blend(first + 2) };
    float3 p[3] = {
        transform_point(s[0], rest_position(first)),
        transform_point(s[1], rest_position(first + 1)),
        transform_point(s[2], rest_position(first + 2)),
    };
    float3 normal = normalize_or_zero(cross(p[1] - p[0], p[2] - p[0]));
    for (uint corner = 0; corner < 3; corner++) {
        uint vertex = first + corner;
        write_vertex(
            vertex,
            p[corner],
            normal,
            normalize_or_zero(transform_vector(s[corner], rest_frames[vertex * 3 + 1].xyz)),
            normalize_or_zero(transform_vector(s[corner], rest_frames[vertex * 3 + 2].xyz)));
    }
}
