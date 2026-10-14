// DDGI probe rays (port of WickedEngine ddgi_raytraceCS, compute BVH path): one workgroup per probe
// updated this frame, rays on a randomly rotated spherical Fibonacci pattern. A hit is shaded with
// its triangle's albedo × (key light + one random point/spot light, both with BVH shadow rays) and
// last frame's probe irradiance for further bounces; a miss returns the IBL irradiance.

#define DDGI_SAMPLER ddgi_sampler
#include "include/ddgi_common.hlsl"
#include "include/bvh_trace.hlsl"
#include "include/punctual_light.hlsl"
#include "include/equirect.hlsl"

// Radiance and length of one probe ray — mirrors the ray buffer layout in pass_ddgi.rs
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

static const uint THREADS = 64;

// Spherical Fibonacci point i of n (hybrid-rendering, via Wicked)
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

// Irradiance (not / π) a surface point receives directly: key light plus one point/spot light
// picked at random, scaled by the light count; each with a BVH shadow ray
float3 direct_irradiance(float3 P, float3 N, uint seed) {
    float3 irradiance = float3(0.0, 0.0, 0.0);
    float3 origin = P + N * 0.01;
    float3 to_sun = -ddgi.sun_direction;
    float  sun_dot = dot(N, to_sun);
    if (sun_dot > 0.0 && any(ddgi.sun_radiance > 0.0) && !bvh_trace_any(origin, to_sun, 1.0e30)) {
        irradiance += ddgi.sun_radiance * sun_dot;
    }
    if (ddgi.light_count > 0) {
        PunctualLight light = punctual_lights[pcg_hash(seed) % ddgi.light_count];
        float3 incoming;
        float  distance_square;
        float  attenuation = punctual_attenuation(light, P, incoming, distance_square);
        float  light_dot = dot(N, incoming);
        if (attenuation > 0.0 && light_dot > 0.0
            && !bvh_trace_any(origin, incoming, sqrt(distance_square) - 0.02)) {
            irradiance += light.color_intensity.rgb * light.color_intensity.w * attenuation * light_dot
                        * float(ddgi.light_count);
        }
    }
    return irradiance;
}

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    uint   slot        = group.x;
    uint   probe_index = (ddgi.probe_offset + slot) % ddgi.probe_count;
    float3 probe_pos   = ddgi_probe_position(ddgi_probe_coord(probe_index));
    uint   ray_count   = ddgi.rays_per_probe;
    for (uint ray = thread; ray < ray_count; ray += THREADS) {
        float3 fibonacci = spherical_fibonacci(float(ray), float(ray_count));
        float3 direction = normalize(ddgi.rotation[0].xyz * fibonacci.x
                                   + ddgi.rotation[1].xyz * fibonacci.y
                                   + ddgi.rotation[2].xyz * fibonacci.z);
        BvhHit hit = bvh_trace_closest(probe_pos, direction, 1.0e30);
        DdgiRay result;
        if (!hit.found) {
            result.direction_depth = float4(direction, -1.0);
            result.radiance = float4(sky_irradiance.SampleLevel(ddgi_sampler, dir_to_equirect_uv(direction), 0.0).rgb
                                     * ddgi.sky_intensity, 0.0);
        } else {
            float3 P = probe_pos + direction * hit.distance;
            // Shade the side the ray arrived at
            float3 N = bvh_triangle_normal(hit.triangle);
            N = dot(N, direction) > 0.0 ? -N : N;
            float3 albedo   = bvh_triangle_albedo(hit.triangle);
            float3 radiance = albedo * direct_irradiance(P, N, probe_index * 1031u + ray * 7919u + ddgi.frame * 104729u) / DDGI_PI;
            // Further bounces from last frame's probes (Wicked's energy conservation 0.95)
            if (ddgi.frame > 0) {
                radiance += albedo * ddgi_irradiance(P, N) * 0.95;
            }
            // A back face means the probe sees the inside of geometry: shorten the depth so the
            // visibility test treats the probe as occluded there (Wicked)
            float depth = hit.backface ? hit.distance * 0.9 : hit.distance;
            result.direction_depth = float4(direction, depth);
            result.radiance = float4(min(radiance, 65000.0), 0.0);
        }
        ddgi_rays[slot * ray_count + ray] = result;
    }
}
