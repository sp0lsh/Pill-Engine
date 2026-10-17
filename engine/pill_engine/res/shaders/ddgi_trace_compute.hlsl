// DDGI probe rays (method: WickedEngine ddgi_raytraceCS, compute BVH path): one thread per ray of
// every probe. Each frame traces one slice of a larger pattern: a spherical Fibonacci set of
// RAY_CYCLE × rays_per_probe directions, slice (frame + per-probe offset) mod RAY_CYCLE, in a
// random rotation per probe and cycle. Every probe covers the whole set every RAY_CYCLE frames,
// probes don't sample the same directions in the same frame (a shared pattern makes all probes
// catch more or less sky together), and the rotation changing every cycle reaches every depth
// texel's lobe over time. A hit is shaded with
// its triangle's albedo × (key light + one random point/spot light, both with BVH shadow rays) and
// last frame's probe irradiance for further bounces; a miss returns the IBL irradiance.

#define DDGI_SAMPLER ddgi_sampler
#include "include/ddgi_common.hlsl"
#include "include/bvh_trace.hlsl"
#include "include/punctual_light.hlsl"
#include "include/oct_ibl.hlsl"

// Radiance and length of one probe ray
struct DdgiRay {
    float4 direction_depth; // xyz = direction, w = hit distance (−1 = miss)
    float4 radiance;        // rgb
};

[[vk::binding(0, 0)]] ConstantBuffer<DdgiParams>        ddgi;
[[vk::binding(1, 0)]] StructuredBuffer<BvhNode>         bvh_nodes;
[[vk::binding(2, 0)]] StructuredBuffer<BvhTriangle>     bvh_triangles;
[[vk::binding(3, 0)]] StructuredBuffer<DdgiProbe>       ddgi_probes;
[[vk::binding(4, 0)]] Texture2D                         ddgi_depth;
[[vk::binding(5, 0)]] SamplerState                      ddgi_sampler;
[[vk::binding(6, 0)]] StructuredBuffer<PunctualLight>   punctual_lights;
[[vk::binding(7, 0)]] Texture2D                         sky_irradiance;
[[vk::binding(8, 0)]] RWStructuredBuffer<DdgiRay>       ddgi_rays;

static const uint THREADS = 128;
static const uint DISPATCH_WIDTH = 32768;

static const uint RAY_CYCLE = 16;

// Spherical Fibonacci point i of n
float3 spherical_fibonacci(float i, float n) {
    const float golden = 1.61803398875;
    float phi       = 2.0 * DDGI_PI * frac(i * (golden - 1.0));
    float cos_theta = 1.0 - (2.0 * i + 1.0) / n;
    float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
}

uint pcg_hash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random rotation of a probe's ray pattern for one cycle: unit quaternion from three hashed
// uniforms (Shoemake's method)
float4 probe_rotation(uint probe_index, uint cycle) {
    uint   seed = pcg_hash(probe_index ^ pcg_hash(cycle));
    uint   a  = pcg_hash(seed * 3u);
    uint   b  = pcg_hash(seed * 3u + 1u);
    uint   c  = pcg_hash(seed * 3u + 2u);
    float3 u  = float3(a, b, c) / 4294967295.0;
    float  r1 = sqrt(1.0 - u.x);
    float  r2 = sqrt(u.x);
    return float4(r1 * sin(2.0 * DDGI_PI * u.y), r1 * cos(2.0 * DDGI_PI * u.y),
                  r2 * sin(2.0 * DDGI_PI * u.z), r2 * cos(2.0 * DDGI_PI * u.z));
}

float3 rotate(float4 q, float3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Irradiance (not / π) a surface point receives directly: key light plus one point/spot light
// picked at random, scaled by the light count; each with a BVH shadow ray
float3 direct_irradiance(float3 P, float3 N, uint seed) {
    float3 irradiance = float3(0.0, 0.0, 0.0);
    float3 origin = P + N * 0.01;
    float3 to_sun = -ddgi.sun_direction;
    float  sun_dot = dot(N, to_sun);
    if (sun_dot > 0.0 && any(ddgi.sun_radiance > 0.0) && !bvh_trace_any(origin, to_sun, 1.0e30, pcg_hash(seed + 1u), ddgi.dynamic_root)) {
        irradiance += ddgi.sun_radiance * sun_dot;
    }
    if (ddgi.light_count > 0) {
        PunctualLight light = punctual_lights[pcg_hash(seed) % ddgi.light_count];
        float3 incoming;
        float  distance_square;
        float  attenuation = punctual_attenuation(light, P, incoming, distance_square);
        float  light_dot = dot(N, incoming);
        if (attenuation > 0.0 && light_dot > 0.0
            && !bvh_trace_any(origin, incoming, sqrt(distance_square) - 0.02, pcg_hash(seed + 2u), ddgi.dynamic_root)) {
            irradiance += light.color_intensity.rgb * light.color_intensity.w * attenuation * light_dot
                        * float(ddgi.light_count);
        }
    }
    return irradiance;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 dispatch_id : SV_DispatchThreadID) {
    // Rows of DISPATCH_WIDTH workgroups (pass_ddgi.rs TRACE_DISPATCH_WIDTH)
    uint3 id = uint3(dispatch_id.x + dispatch_id.y * DISPATCH_WIDTH * THREADS, 0, 0);
    uint ray_count = ddgi.rays_per_probe;
    if (id.x >= ddgi.probe_count * ddgi.cascade_count * ray_count) {
        return;
    }
    uint   probe_index = id.x / ray_count;
    uint   ray         = id.x % ray_count;
    // Relocated position (ddgi_relocate_compute.hlsl)
    uint   cascade     = probe_index / ddgi.probe_count;
    if (!ddgi_cascade_updates(cascade)) {
        return;
    }
    float3 probe_pos   = ddgi_probe_position(ddgi_probe_coord(probe_index % ddgi.probe_count), cascade)
                       + ddgi_probes[probe_index].offset.xyz;
    uint   step        = ddgi.frame + pcg_hash(probe_index) % RAY_CYCLE;
    uint   slice       = step % RAY_CYCLE;
    uint   cycle       = step / RAY_CYCLE;
    {
        float3 fibonacci = spherical_fibonacci(float(ray * RAY_CYCLE + slice), float(ray_count * RAY_CYCLE));
        float3 direction = normalize(rotate(probe_rotation(probe_index, cycle), fibonacci));
        BvhHit hit = bvh_trace_closest(probe_pos, direction, 1.0e30, pcg_hash(id.x ^ pcg_hash(ddgi.frame)), ddgi.dynamic_root);
        DdgiRay result;
        if (!hit.found) {
            result.direction_depth = float4(direction, -1.0);
            result.radiance = float4(sample_ibl_irradiance(sky_irradiance, ddgi_sampler, direction)
                                     * ddgi.sky_intensity, 0.0);
        } else {
            float3 P = probe_pos + direction * hit.distance;
            // Shade the side the ray arrived at
            float3 N = bvh_triangle_normal(hit.triangle);
            N = dot(N, direction) > 0.0 ? -N : N;
            float3 albedo   = bvh_triangle_albedo(hit.triangle);
            float3 radiance = albedo * direct_irradiance(P, N, probe_index * 1031u + ray * 7919u + ddgi.frame * 104729u) / DDGI_PI;
            // Further bounces from last frame's probes (energy conservation 0.95)
            if (ddgi.frame > 0) {
                float4 bounce = ddgi_irradiance(P, N);
                radiance += albedo * bounce.rgb * bounce.a * 0.95;
            }
            // A back face means the probe sees the inside of geometry: shorten the depth so the
            // visibility test treats the probe as occluded there
            float depth = hit.backface ? hit.distance * 0.9 : hit.distance;
            result.direction_depth = float4(direction, depth);
            // w: backface hit (the color pass ignores those)
            result.radiance = float4(min(radiance, 65000.0), hit.backface ? 1.0 : 0.0);
        }
        ddgi_rays[id.x] = result;
    }
}
