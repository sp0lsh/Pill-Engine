#ifndef OCTAHEDRAL_HLSL
#define OCTAHEDRAL_HLSL
// Octahedral mapping of a unit direction to [-1, 1]² (DDGI probe texels, SSR normal target)
float2 oct_encode(float3 direction) {
    float3 n = direction / (abs(direction.x) + abs(direction.y) + abs(direction.z));
    float2 folded = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.z >= 0.0 ? n.xy : folded;
}

float3 oct_decode(float2 encoded) {
    float3 n = float3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    float  t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

#endif
