#ifndef BINDINGS_HLSL
#define BINDINGS_HLSL
// Binding properties the shader code can't express, read from slangc's reflection by the Rust
// binding generator (pill_assets shader_bindings.rs). Defaults: float textures are filterable,
// samplers filtering.

// Float texture read with Load or a non-filtering sampler (e.g. R32Float targets)
[__AttributeUsage(_AttributeTargets.Var)]
struct UnfilterableAttribute {};
// Depth texture (sampled with SampleCmp or Load)
[__AttributeUsage(_AttributeTargets.Var)]
struct DepthAttribute {};
// Comparison sampler (SamplerComparisonState reflects as a plain sampler)
[__AttributeUsage(_AttributeTargets.Var)]
struct ComparisonAttribute {};
// Non-filtering sampler (point sampling of unfilterable textures)
[__AttributeUsage(_AttributeTargets.Var)]
struct NonFilteringAttribute {};
// Uniform buffer bound with a dynamic offset
[__AttributeUsage(_AttributeTargets.Var)]
struct DynamicOffsetAttribute {};
#endif
