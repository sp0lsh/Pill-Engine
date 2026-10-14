// Screen-space reflections against the depth buffer, without a depth pyramid: a port of Bevy's
// SSR ray march (bevy_pbr/src/ssr/raymarch.wesl, Tomasz Stachowiak, after
// https://gist.github.com/h3r2tic/9c8356bdaefbe80b1a22ae0aaee192db): linear steps along the ray
// in NDC, bisection of the first crossing, one secant step. Bevy samples stochastic GGX lobes
// and relies on temporal filtering; here the ray is the mirror direction, and reflections fade
// out with roughness instead. Pill's depth is standard [0, 1] (not reversed, not infinite), so
// depths are compared as linear view distances.
//
// out = color + W · lerp(ibl_specular, reflected, confidence), with W the specular weight the
// opaque pass left out of `color` (pbr_opaque_fragment.hlsl).
#include "include/octahedral.hlsl"
#include "include/equirect.hlsl"

// Mirrors SsrParams in pass_ssr.rs
struct SsrParams {
    column_major float4x4 view_projection;
    column_major float4x4 inverse_view_projection;
    float4 camera;  // xyz = camera position, w = surface thickness (m)
    float4 depth;   // x, y = depth unpack (distance = x / (y - depth)), z = IBL scale, w = 2 reflections only
    float4 march;   // x = linear steps, y = bisection steps, z, w = roughness fade start, end
    float4 screen;  // x = edge fade (fraction of the screen)
};

[[vk::binding(0, 0)]] Texture2D<float4>           t_color;     // opaque scene, specular IBL left out
[[vk::binding(1, 0)]] Texture2D<float>            t_depth;     // Depth32Float
[[vk::binding(2, 0)]] Texture2D<float4>           t_surface;   // rgb = specular weight, a = roughness
[[vk::binding(3, 0)]] Texture2D<float2>           t_normal;    // octahedral normal
[[vk::binding(4, 0)]] Texture2D<float4>           t_prefilter; // specular IBL (equirect, mips by roughness)
[[vk::binding(5, 0)]] SamplerState                s_prefilter;
[[vk::binding(6, 0)]] ConstantBuffer<SsrParams>   params;

static const float MAX_REFLECTION_LOD = 4.0; // as pbr_opaque_fragment.hlsl
// Least cosine between the reflection and the geometric surface
static const float MIN_REFLECTION_ELEVATION = 0.05;

float2 ndc_to_uv(float2 ndc) {
    return ndc * float2(0.5, -0.5) + 0.5;
}

float linear_distance(float depth) {
    return params.depth.x / (params.depth.y - depth);
}

float load_distance(int2 pixel, int2 size) {
    return linear_distance(t_depth.Load(int3(clamp(pixel, int2(0, 0), size - 1), 0)));
}

float3 world_position(int2 pixel, int2 size) {
    pixel = clamp(pixel, int2(0, 0), size - 1);
    float2 ndc   = (float2(pixel) + 0.5) / float2(size) * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float4 world = mul(params.inverse_view_projection, float4(ndc, t_depth.Load(int3(pixel, 0)), 1.0));
    return world.xyz / world.w;
}

// Surface normal from the depth buffer (as ssao_fragment.hlsl): per axis the neighbour on the
// side with the smaller depth step, so silhouettes don't tilt it; faces the viewer
float3 geometric_normal(int2 pixel, int2 size, float3 position, float3 view) {
    float3 left  = world_position(pixel - int2(1, 0), size) - position;
    float3 right = world_position(pixel + int2(1, 0), size) - position;
    float3 up    = world_position(pixel - int2(0, 1), size) - position;
    float3 down  = world_position(pixel + int2(0, 1), size) - position;
    float3 dx = dot(left, left) < dot(right, right) ? -left : right;
    float3 dy = dot(up, up) < dot(down, down) ? -up : down;
    float3 normal = normalize(cross(dx, dy));
    return dot(normal, view) < 0.0 ? -normal : normal;
}

// Depth-buffer surface vs a point on the ray: `distance` < 0 once the ray is behind it.
// Bevy tests against both the point-sampled and the bilinear depth (stacked bricks vs a
// shrink-wrapped surface) and claims a crossing only below both.
struct RayDistance {
    float distance;
    bool  valid;
    float penetration;
};

RayDistance evaluate(float3 ray_ndc, int2 size) {
    float2 corner    = ndc_to_uv(ray_ndc.xy) * float2(size) - 0.5;
    int2   base      = int2(floor(corner));
    float2 weight    = corner - float2(base);
    float  d00       = load_distance(base, size);
    float  d10       = load_distance(base + int2(1, 0), size);
    float  d01       = load_distance(base + int2(0, 1), size);
    float  d11       = load_distance(base + int2(1, 1), size);
    float  bilinear  = lerp(lerp(d00, d10, weight.x), lerp(d01, d11, weight.x), weight.y);
    // The texel the point lies in is the bilinear tap it is closest to
    float  nearest   = weight.y < 0.5 ? (weight.x < 0.5 ? d00 : d10) : (weight.x < 0.5 ? d01 : d11);
    float  ray       = linear_distance(ray_ndc.z);
    RayDistance result;
    result.distance    = max(nearest, bilinear) * (1.0 + 1e-5) - ray;
    result.penetration = ray - min(nearest, bilinear);
    result.valid       = result.penetration < params.camera.w;
    return result;
}

// Ray behind the depth-buffer surface (thickness not checked yet)
bool behind(RayDistance sample) {
    return sample.distance < 0.0;
}

// Linear march from `start_t`; at each step that goes from in front of to behind the surface,
// bisection and one secant step (Bevy's hybrid root finder) refine the crossing, which is a hit
// when the refined point lies within the surface thickness. Otherwise the ray passed behind an
// object and the march goes on. Judging thickness on the refined point, not on the linear
// samples, keeps hits from depending on where the coarse steps happen to land (meters apart on
// far rays), which sliced far reflections into bands.
bool find_hit(float3 start, float3 end, float start_t, uint linear_steps, int2 size, out float hit_t, out RayDistance hit) {
    float3 direction = end - start;
    float  previous_t = start_t;
    RayDistance previous = evaluate(start + direction * start_t, size);
    hit_t = start_t;
    hit   = previous;
    for (uint step = 1; step <= linear_steps; step++) {
        float       t = lerp(start_t, 1.0, float(step) / float(linear_steps));
        RayDistance d = evaluate(start + direction * t, size);
        if (behind(d) && !behind(previous)) {
            float       min_t = previous_t;
            float       max_t = t;
            RayDistance min_d = previous;
            RayDistance max_d = d;
            for (uint i = 0; i < uint(params.march.y); i++) {
                float       mid_t = (min_t + max_t) * 0.5;
                RayDistance mid   = evaluate(start + direction * mid_t, size);
                if (behind(mid)) {
                    max_t = mid_t;
                    max_d = mid;
                } else {
                    min_t = mid_t;
                    min_d = mid;
                }
            }
            // One secant step, kept only if it is a valid point closer to the surface than both
            // bisection ends
            float       secant_t = lerp(min_t, max_t, min_d.distance / (min_d.distance - max_d.distance));
            RayDistance secant_d = evaluate(start + direction * secant_t, size);
            if (secant_d.valid && abs(secant_d.distance) < min(min_d.distance, -max_d.distance)) {
                max_t = secant_t;
                max_d = secant_d;
            }
            if (max_d.valid) {
                hit_t = max_t;
                hit   = max_d;
                return true;
            }
        }
        previous_t = t;
        previous   = d;
    }
    return false;
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    uint width, height;
    t_color.GetDimensions(width, height);
    int2   size  = int2(width, height);
    int2   pixel = clamp(int2(uv * float2(size)), int2(0, 0), size - 1);
    float4 color = t_color.Load(int3(pixel, 0));
    float4 surface = t_surface.Load(int3(pixel, 0));
    float3 weight  = surface.rgb;
    if (all(weight <= 0.0)) {
        return params.depth.w > 1.5 ? float4(0.0, 0.0, 0.0, color.a) : color;
    }
    float roughness = surface.a;
    float raw_depth = t_depth.Load(int3(pixel, 0));

    // Surface position and mirror direction
    float2 ndc      = (float2(pixel) + 0.5) / float2(size) * float2(2.0, -2.0) + float2(-1.0, 1.0);
    float3 position = world_position(pixel, size);
    float3 normal   = oct_decode(t_normal.Load(int3(pixel, 0)));
    float3 view     = normalize(params.camera.xyz - position);
    float3 reflection = reflect(-view, normal);
    // A normal-mapped normal can mirror the ray into the surface it starts on; keep it above the
    // geometric surface (from the depth buffer), at least grazing
    float3 geometric = geometric_normal(pixel, size, position, view);
    float  above     = dot(reflection, geometric);
    if (above < MIN_REFLECTION_ELEVATION) {
        reflection = normalize(reflection + (MIN_REFLECTION_ELEVATION - above) * geometric);
    }

    // Ray from the surface to where it leaves the view frustum (Bevy depth_ray_march_to_cs_dir_impl):
    // P + view_distance · R in clip space, divided, then stretched to the far edge of [-1, 1]² × [0, 1]
    float3 start   = float3(ndc, raw_depth);
    float4 end_cs  = float4(start, 1.0) + mul(params.view_projection, float4(reflection, 0.0));
    end_cs        /= (end_cs.w >= 0.0 ? 1.0 : -1.0) * max(1e-10, abs(end_cs.w));
    float3 delta   = end_cs.xyz - start;
    delta          = select(abs(delta) < 1e-8, 1e-8, delta);
    float3 far_edge = float3(delta.x >= 0.0 ? 1.0 : -1.0, delta.y >= 0.0 ? 1.0 : -1.0, delta.z >= 0.0 ? 1.0 : 0.0);
    float3 to_edge = (far_edge - start) / delta;
    float3 end     = start + delta * min(min(to_edge.x, to_edge.y), to_edge.z);

    // Steps capped by the ray's length in pixels (Bevy: at least one pixel per step); the march
    // starts one pixel out, so the surface cannot reflect its own texel
    float  length_px = length((ndc_to_uv(end.xy) - ndc_to_uv(start.xy)) * float2(size));
    uint   steps     = uint(max(2.0, min(params.march.x, floor(length_px))));
    float  start_t   = min(1.0 / max(length_px, 1.0), 1.0);

    float       hit_t;
    RayDistance hit;
    float       confidence = 0.0;
    float3      reflected  = float3(0.0, 0.0, 0.0);
    if (find_hit(start, end, start_t, steps, size, hit_t, hit)) {
        float2 hit_uv = ndc_to_uv(lerp(start.xy, end.xy, hit_t));
        float2 border = min(hit_uv, 1.0 - hit_uv);
        confidence    = smoothstep(0.0, max(params.screen.x, 1e-4), min(border.x, border.y));
        reflected     = t_color.Load(int3(clamp(int2(hit_uv * float2(size)), int2(0, 0), size - 1), 0)).rgb;
    }
    confidence *= 1.0 - smoothstep(params.march.z, params.march.w, roughness);

    if (params.depth.w > 1.5) {
        return float4(weight * reflected * confidence, color.a);
    }
    float3 environment = t_prefilter.SampleLevel(s_prefilter, dir_to_equirect_uv(reflection), roughness * MAX_REFLECTION_LOD).rgb * params.depth.z;
    return float4(color.rgb + weight * lerp(environment, reflected, confidence), color.a);
}
