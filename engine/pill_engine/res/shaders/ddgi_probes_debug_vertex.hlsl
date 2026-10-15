// DDGI probe debug view: one small sphere per probe at its relocated position, shaded with the probe's irradiance in the
// sphere's normal direction as a 50 % grey ball; probes without weight yet (new) are magenta. Instanced: instance =
// probe index, vertices generated from the vertex index (RINGS × SEGMENTS quads).

#include "include/ddgi_common.hlsl"

struct Camera {
    column_major float4x4 view_projection;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera>        camera;
[[vk::binding(1, 0)]] ConstantBuffer<DdgiParams>    ddgi;
[[vk::binding(2, 0)]] StructuredBuffer<DdgiProbe>   ddgi_probes;

static const uint  RINGS    = 8;
static const uint  SEGMENTS = 12;
// Sphere radius (m)
static const float RADIUS   = 0.1;

struct VertexOutput {
    float4 position : SV_Position;
    float3 normal   : NORMAL;
    nointerpolation uint probe : PROBE;
};

float3 sphere_point(uint ring, uint segment) {
    float theta = DDGI_PI * float(ring) / float(RINGS);
    float phi   = 2.0 * DDGI_PI * float(segment) / float(SEGMENTS);
    return float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
}

[shader("vertex")]
VertexOutput vs_main(uint vertex : SV_VertexID, uint instance : SV_InstanceID) {
    // Two triangles per quad: corners (0,0) (1,0) (1,1) (0,0) (1,1) (0,1)
    const uint2 corners[6] = { uint2(0, 0), uint2(1, 0), uint2(1, 1), uint2(0, 0), uint2(1, 1), uint2(0, 1) };
    uint  quad    = vertex / 6;
    uint2 corner  = corners[vertex % 6];
    float3 normal = sphere_point(quad / SEGMENTS + corner.x, quad % SEGMENTS + corner.y);
    DdgiProbe probe = ddgi_probes[instance];
    // Instance = probe index over all cascades
    float3 center = ddgi_probe_position(ddgi_probe_coord(instance % ddgi.probe_count), instance / ddgi.probe_count)
                  + probe.offset.xyz;
    VertexOutput output;
    output.position = mul(camera.view_projection, float4(center + normal * RADIUS, 1.0));
    output.normal   = normal;
    output.probe    = instance;
    return output;
}
