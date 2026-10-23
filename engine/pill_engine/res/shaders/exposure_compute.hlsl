// Auto exposure (eye adaptation): one workgroup meters the HDR image the tonemap pass reads on a
// 64×64 grid, averages log2 luminance (clamped to the metering range), eases the adapted value
// toward it over time (faster toward bright than toward dark, like an eye) and writes the
// exposure multiplier the tonemap pass applies. No atomics and no readback: the state stays in a
// storage buffer across frames.

[[vk::binding(0, 0)]] Texture2D hdr_texture;

// From RenderStateComponent::exposure and ::auto_exposure
struct ExposureParams {
    float exposure_ev;            // manual exposure, also applied without auto exposure
    float compensation_ev;        // added to the metered exposure
    float min_ev;                 // metering range of log2 luminance
    float max_ev;
    float adapt_to_bright_speed;  // 1/s, scene got brighter
    float adapt_to_dark_speed;    // 1/s, scene got darker
    float key;                    // luminance the metered average is mapped to
    float delta_seconds;
    uint  enabled;
    uint  snap;                   // 1: jump to the metered value (first frame, camera cut)
    float _padding0, _padding1;
};
[[vk::binding(1, 0)]] ConstantBuffer<ExposureParams> params;
// [0] adapted mean log2 luminance, [1] exposure multiplier (read by the tonemap pass)
[[vk::binding(2, 0)]] RWStructuredBuffer<float> exposure_state;

static const uint THREADS = 256;
static const uint GRID = 64;
groupshared float partial_sums[THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void cs_main(uint thread : SV_GroupIndex) {
    // Off: no metering, just the manual exposure (`params` is uniform, so all threads leave here)
    if (params.enabled == 0) {
        if (thread == 0) {
            exposure_state[1] = exp2(params.exposure_ev);
        }
        return;
    }
    uint width, height;
    hdr_texture.GetDimensions(width, height);
    float sum = 0.0;
    for (uint cell_index = thread; cell_index < GRID * GRID; cell_index += THREADS) {
        float2 cell = float2(cell_index % GRID, cell_index / GRID);
        uint2 pixel = uint2((cell + 0.5) / float(GRID) * float2(width, height));
        float3 color = hdr_texture.Load(int3(pixel, 0)).rgb;
        float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
        sum += clamp(log2(max(luminance, 1e-6)), params.min_ev, params.max_ev);
    }
    partial_sums[thread] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            partial_sums[thread] += partial_sums[thread + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread != 0) {
        return;
    }
    float metered = partial_sums[0] / float(GRID * GRID);
    float previous = exposure_state[0];
    float adapted = metered;
    if (params.snap == 0) {
        float speed = metered > previous ? params.adapt_to_bright_speed : params.adapt_to_dark_speed;
        adapted = previous + (metered - previous) * (1.0 - exp(-params.delta_seconds * speed));
    }
    exposure_state[0] = adapted;
    exposure_state[1] = exp2(params.exposure_ev + params.compensation_ev) * params.key / exp2(adapted);
}
