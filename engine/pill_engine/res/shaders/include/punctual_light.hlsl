#ifndef PUNCTUAL_LIGHT_HLSL
#define PUNCTUAL_LIGHT_HLSL
// Point/spot light record (pass_lights.rs)
struct PunctualLight {
    float4 position_falloff;  // xyz = world position, w = 1 / range²
    float4 direction;         // xyz = direction the spot shines in
    float4 color_intensity;   // rgb = color, w = intensity
    // x, y: angle term saturate(cos · x + y) (point light: 0, 1); z: first shadow layer (−1 = no
    // shadow); w: 1 = point light (6 cube-face layers from z)
    float4 spot_scale_offset;
};

// Filament's attenuation toward `world_position`: inverse square with a smooth window to zero at
// the range, times the spot angle term. `incoming` = unit vector from the surface to the light.
float punctual_attenuation(PunctualLight light, float3 world_position, out float3 incoming, out float distance_square) {
    float3 to_light = light.position_falloff.xyz - world_position;
    distance_square = dot(to_light, to_light);
    float window    = saturate(1.0 - pow(distance_square * light.position_falloff.w, 2.0));
    incoming        = to_light * rsqrt(max(distance_square, 1e-8));
    float angle     = saturate(dot(light.direction.xyz, -incoming) * light.spot_scale_offset.x + light.spot_scale_offset.y);
    return window * window / max(distance_square, 1e-4) * angle * angle;
}
#endif
