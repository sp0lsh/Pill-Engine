// Stream compaction (bgfx cs_gdr_stream_compaction): an exclusive prefix sum of the predicates in
// one workgroup (256 threads × 8 instances, up to 2048) gives each visible instance its slot;
// thread 0 then writes each prop's indirect draw (5 uint: index count, instance count, first
// index, base vertex, first instance) and base slot, and clears the counts.

#include "include/gdr_common.hlsl"

struct Compact {
    uint instance_count;
    uint prop_count;
    uint index_count;
    uint padding;
};
[[vk::binding(0, 0)]] ConstantBuffer<Compact> config;
[[vk::binding(1, 0)]] StructuredBuffer<Instance> instances;
[[vk::binding(2, 0)]] StructuredBuffer<uint> predicates;
[[vk::binding(3, 0)]] RWStructuredBuffer<Atomic<uint>> counts;
[[vk::binding(4, 0)]] RWStructuredBuffer<Instance> culled;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> draws;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> bases;

static const uint THREADS = 256;
static const uint PER_THREAD = 8;
groupshared uint sums[THREADS];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void cs_main(uint t : SV_GroupIndex) {
    uint local_prefix[PER_THREAD];
    uint total = 0;
    for (uint k = 0; k < PER_THREAD; k++) {
        uint i = t * PER_THREAD + k;
        local_prefix[k] = total;
        if (i < config.instance_count) {
            total += predicates[i];
        }
    }
    sums[t] = total;
    GroupMemoryBarrierWithGroupSync();
    // Inclusive scan of the per-thread totals (Hillis–Steele)
    for (uint offset = 1; offset < THREADS; offset *= 2) {
        uint add = t >= offset ? sums[t - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        sums[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint base = sums[t] - total;
    for (uint j = 0; j < PER_THREAD; j++) {
        uint i = t * PER_THREAD + j;
        if (i < config.instance_count && predicates[i] != 0) {
            culled[base + local_prefix[j]] = instances[i];
        }
    }
    if (t == 0) {
        uint first = 0;
        for (uint prop = 0; prop < config.prop_count; prop++) {
            uint count = counts[prop].load();
            draws[prop * 5 + 0] = config.index_count;
            draws[prop * 5 + 1] = count;
            draws[prop * 5 + 2] = 0;
            draws[prop * 5 + 3] = 0;
            draws[prop * 5 + 4] = 0;
            bases[prop] = first;
            first += count;
            counts[prop].store(0);
        }
    }
}
