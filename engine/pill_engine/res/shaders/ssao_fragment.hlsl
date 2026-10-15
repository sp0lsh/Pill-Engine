// Half-resolution horizon-based ambient occlusion (method: Activision GTAO [Jimenez et al. 2016]
// as in Intel XeGTAO's main pass; one slice per pixel, Morton-ordered 4×4 slice directions and
// line offsets, depth mips picked by the projected sample distance and a median depth chain, after
// Sebastian Aaltonen's mobile GTAO). Reads the view-space depth chain written by
// ssao_depth_downsample_fragment.hlsl; positions and normals come from depth since this forward
// renderer has no normal G-buffer.
//
// Outputs (ao, viewZ) — the downstream bilateral filter gets its guide channel in the same fetch.

// Mirrors SsaoParams in pass_ssao.rs
struct SsaoParams {
    float tan_half_x;     // tan(fov/2) * aspect
    float tan_half_y;     // tan(fov/2)
    float radius;         // world units
    float intensity;      // scales the occlusion
    float power;          // exponent on the visibility
    float _pad0;
    float _pad1;
    float _pad2;
};

// View-space depth at ½, ¼, ⅛ and 1/16 resolution
[[vk::binding(0, 0)]] Texture2D<float> t_depth0;
[[vk::binding(1, 0)]] Texture2D<float> t_depth1;
[[vk::binding(2, 0)]] Texture2D<float> t_depth2;
[[vk::binding(3, 0)]] Texture2D<float> t_depth3;
[[vk::binding(4, 0)]] ConstantBuffer<SsaoParams> params;

static const float PI      = 3.14159265;
static const float HALF_PI = 1.57079633;
static const int   STEPS   = 4;            // per side of the slice
static const float MAX_MIP = 3.0;
// Calibrated against ray-traced references (XeGTAO's defaults)
static const float RADIUS_MULTIPLIER         = 1.457;
static const float SAMPLE_DISTRIBUTION_POWER = 2.0;   // small crevices weigh more
static const float MIP_SAMPLING_OFFSET       = 3.3;
// Fraction of the radius over which samples fade out: XeGTAO's 0.615, ×1.5 against flicker far
// away (Aaltonen)
static const float FALLOFF_RANGE = 0.615 * 1.5;
// Nearest sample distance (pixels): closer ones only see the centre texel
static const float PIXEL_TOO_CLOSE = 1.3;

float level_depth(int level, int2 texel) {
    uint width, height;
    if (level == 0) {
        t_depth0.GetDimensions(width, height);
        return t_depth0.Load(int3(clamp(texel, int2(0, 0), int2(width, height) - 1), 0));
    }
    if (level == 1) {
        t_depth1.GetDimensions(width, height);
        return t_depth1.Load(int3(clamp(texel, int2(0, 0), int2(width, height) - 1), 0));
    }
    if (level == 2) {
        t_depth2.GetDimensions(width, height);
        return t_depth2.Load(int3(clamp(texel, int2(0, 0), int2(width, height) - 1), 0));
    }
    t_depth3.GetDimensions(width, height);
    return t_depth3.Load(int3(clamp(texel, int2(0, 0), int2(width, height) - 1), 0));
}

int2 level_size(int level) {
    uint width, height;
    if (level == 0) {
        t_depth0.GetDimensions(width, height);
    } else if (level == 1) {
        t_depth1.GetDimensions(width, height);
    } else if (level == 2) {
        t_depth2.GetDimensions(width, height);
    } else {
        t_depth3.GetDimensions(width, height);
    }
    return int2(width, height);
}

// Depth at `uv` from mip level `mip`, blended between the two nearest levels so the level change
// with distance is gradual
float depth_at(float2 uv, float mip) {
    int   low  = (int)floor(mip);
    int   high = min(low + 1, (int)MAX_MIP);
    float z0   = level_depth(low, (int2)(uv * (float2)level_size(low)));
    float z1   = level_depth(high, (int2)(uv * (float2)level_size(high)));
    return lerp(z0, z1, mip - (float)low);
}

// View space with z forward (as XeGTAO); `uv` in [0, 1], y down
float3 view_position(float2 uv, float z) {
    return float3((uv.x * 2.0 - 1.0) * params.tan_half_x * z, (1.0 - uv.y * 2.0) * params.tan_half_y * z, z);
}

// acos for [-1, 1] → [0, π] (Lagarde's GCN approximation, as in XeGTAO)
float fast_acos(float x) {
    float a      = abs(x);
    float result = (-0.156583 * a + HALF_PI) * sqrt(1.0 - a);
    return x >= 0.0 ? result : PI - result;
}

// 4×4 Morton (Z-order) index: every 2×2 and 4×4 neighbourhood gets an even mix of directions and
// offsets, so the 4×4 bilateral blur averages a complete set
float morton_4x4(int2 p) {
    int x = p.x & 3;
    int y = p.y & 3;
    return (float)((x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2));
}

[shader("fragment")]
float4 fs_main(float2 uv : TEXCOORD0) : SV_Target {
    int2   size  = level_size(0);
    int2   coord = clamp((int2)(uv * (float2)size), int2(0, 0), size - int2(1, 1));
    float2 texel = 1.0 / (float2)size;
    float2 centre_uv = ((float2)coord + 0.5) * texel;

    float z = level_depth(0, coord);
    // Far plane / sky: nothing to occlude
    if (z <= 0.0 || z > 1e4) {
        return float4(1.0, z, 0.0, 0.0);
    }

    // Normal from a 5-tap cross, the closer neighbour per axis: a one-sided difference across a
    // silhouette tilts the normal into the surface and haloes every edge
    float3 P  = view_position(centre_uv, z);
    float3 Pl = view_position(centre_uv - float2(texel.x, 0.0), level_depth(0, coord - int2(1, 0)));
    float3 Pr = view_position(centre_uv + float2(texel.x, 0.0), level_depth(0, coord + int2(1, 0)));
    float3 Pu = view_position(centre_uv - float2(0.0, texel.y), level_depth(0, coord - int2(0, 1)));
    float3 Pd = view_position(centre_uv + float2(0.0, texel.y), level_depth(0, coord + int2(0, 1)));
    float3 dx = abs(Pr.z - z) < abs(Pl.z - z) ? Pr - P : P - Pl;
    float3 dy = abs(Pd.z - z) < abs(Pu.z - z) ? Pd - P : P - Pu;
    float3 V  = normalize(-P);
    float3 N  = normalize(cross(dx, dy));
    if (dot(N, V) < 0.0) {
        N = -N;
    }
    // Centre pulled slightly toward the camera against depth imprecision
    P *= 0.99999;

    float effect_radius = params.radius * RADIUS_MULTIPLIER;
    // View-space width of one half-res pixel at this depth → radius in pixels
    float screen_radius = effect_radius * (float)size.x / (z * 2.0 * params.tan_half_x);
    if (screen_radius < PIXEL_TOO_CLOSE) {
        return float4(1.0, z, 0.0, 0.0);
    }
    float visibility = saturate((10.0 - screen_radius) / 100.0) * 0.5;

    // Slice direction and line offset from the pixel's place in the 4×4 cell (R1 for the offset)
    float morton       = morton_4x4(coord);
    float slice_noise  = (morton + 0.5) / 16.0;
    float sample_noise = frac(0.5 + morton * 0.6180339887);

    float  phi       = slice_noise * PI;
    float  cos_phi   = cos(phi);
    float  sin_phi   = sin(phi);
    float2 omega     = float2(cos_phi, -sin_phi) * screen_radius;   // pixels, y down
    float3 direction = float3(cos_phi, sin_phi, 0.0);
    float3 ortho     = direction - dot(direction, V) * V;
    float3 axis      = normalize(cross(ortho, V));
    float3 projected = N - axis * dot(N, axis);
    float  projected_length = max(length(projected), 1e-4);
    float  sign_norm = dot(ortho, projected) >= 0.0 ? 1.0 : -1.0;
    float  cos_norm  = saturate(dot(projected, V) / projected_length);
    float  n         = sign_norm * fast_acos(cos_norm);

    float low_cos0 = cos(n + HALF_PI);
    float low_cos1 = cos(n - HALF_PI);
    float horizon_cos0 = low_cos0;
    float horizon_cos1 = low_cos1;

    float falloff_range = FALLOFF_RANGE * effect_radius;
    float falloff_mul   = -1.0 / falloff_range;
    float falloff_add   = (effect_radius - falloff_range) / falloff_range + 1.0;
    float min_s         = PIXEL_TOO_CLOSE / screen_radius;

    [unroll]
    for (int step = 0; step < STEPS; step++) {
        float step_noise = frac(sample_noise + (float)(step * STEPS) * 0.6180339887);
        float s = ((float)step + step_noise) / (float)STEPS;
        s = pow(s, SAMPLE_DISTRIBUTION_POWER) + min_s;

        float2 offset = s * omega;
        // ½·log2 of the squared length: no square root
        float  mip = clamp(0.5 * log2(max(dot(offset, offset), 1e-6)) - MIP_SAMPLING_OFFSET, 0.0, MAX_MIP);
        float2 offset_uv = round(offset) * texel;

        float2 uv0 = centre_uv + offset_uv;
        float2 uv1 = centre_uv - offset_uv;
        float3 delta0 = view_position(uv0, depth_at(uv0, mip)) - P;
        float3 delta1 = view_position(uv1, depth_at(uv1, mip)) - P;
        float  distance0 = length(delta0);
        float  distance1 = length(delta1);
        float  weight0 = saturate(distance0 * falloff_mul + falloff_add);
        float  weight1 = saturate(distance1 * falloff_mul + falloff_add);
        float  cos0 = lerp(low_cos0, dot(delta0, V) / max(distance0, 1e-6), weight0);
        float  cos1 = lerp(low_cos1, dot(delta1, V) / max(distance1, 1e-6), weight1);
        horizon_cos0 = max(horizon_cos0, cos0);
        horizon_cos1 = max(horizon_cos1, cos1);
    }

    // Arc integral of the cosine-weighted visible span between the two horizons
    projected_length = lerp(projected_length, 1.0, 0.05);
    float h0 = -fast_acos(horizon_cos1);
    float h1 = fast_acos(horizon_cos0);
    float arc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
    float arc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
    visibility += projected_length * (arc0 + arc1);

    visibility = max(0.03, pow(saturate(visibility), max(params.power, 1e-4)));
    visibility = saturate(1.0 - (1.0 - visibility) * params.intensity);
    return float4(visibility, z, 0.0, 0.0);
}
