// Fragment stage of the DDGI probe debug view (ddgi_probes_debug_vertex.hlsl).
// DDGI probe debug view: one small sphere per probe at its relocated position, shaded with the probe's irradiance in the
// sphere's normal direction as a 50 % grey ball; probes without weight yet (new) are magenta. Instanced: instance =
// probe index, vertices generated from the vertex index (RINGS × SEGMENTS quads).

#include "include/ddgi_common.hlsl"

struct ProbeDebugCamera {
    column_major float4x4 view_projection;
};

[[vk::binding(0, 0)]] ConstantBuffer<ProbeDebugCamera> camera;
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

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target0 {
    DdgiProbe probe = ddgi_probes[input.probe];
    if (probe.sh[0].w <= 0.0) {
        return float4(1.0, 0.0, 1.0, 1.0);
    }
    float3 N = normalize(input.normal);
    // L1 SH convolved with the cosine lobe, evaluated in N, divided by π (as ddgi_irradiance)
    const float basis_l0 = 0.2820948;
    const float basis_l1 = 0.4886025;
    float3 irradiance = basis_l0 * DDGI_PI * probe.sh[0].xyz
                      + basis_l1 * (2.0 * DDGI_PI / 3.0) * (N.y * probe.sh[1].xyz + N.z * probe.sh[2].xyz + N.x * probe.sh[3].xyz);
    // A 50 % grey diffuse ball lit by the probe: raw irradiance mostly tonemaps to white
    return float4(max(irradiance / DDGI_PI, 0.0) * ddgi.intensity * 0.5, 1.0);
}
