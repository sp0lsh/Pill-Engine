// PBR static fragment shader. Edit here — pill_assets regenerates the .wgsl.
// Full GGX microfacet BRDF with 3 directional lights and optional IBL fallback.

struct Camera {
    float4                position;
    column_major float4x4 viewProjection;
};
[[vk::binding(0, 0)]] ConstantBuffer<Camera> UCamera;

// IBL resources in globals bind group (set 0, bindings 1-6).
[[vk::binding(1, 0)]] Texture2D    texIrradiance;
[[vk::binding(2, 0)]] SamplerState smpIrradiance;
[[vk::binding(3, 0)]] Texture2D    texPrefilter;
[[vk::binding(4, 0)]] SamplerState smpPrefilter;
[[vk::binding(5, 0)]] Texture2D    texBrdfLut;
[[vk::binding(6, 0)]] SamplerState smpBrdfLut;

// PBR material textures (set 1) — bindings match DEFAULT_LIT_SHADER layout (0-3).
[[vk::binding(0, 1)]] Texture2D    texBaseColor;
[[vk::binding(1, 1)]] SamplerState smpBaseColor;
[[vk::binding(2, 1)]] Texture2D    texNormal;
[[vk::binding(3, 1)]] SamplerState smpNormal;

// PBR params UBO (set 2) — 32 bytes: baseColorFactor (vec3 + pad), roughnessFactor (f32 + pad + vec2 pad).
struct MaterialParams {
    float3 baseColorFactor;
    float  _pad0;
    float  roughnessFactor;
    float  _pad1;
    float2 _pad2;
};
[[vk::binding(0, 2)]] ConstantBuffer<MaterialParams> UMaterial;

static const float  PI         = 3.14159265359;

// Old renderer used a point light at (-10,10,-10); the resulting direction from light
// to scene (z≈12) is approximately (0.38,-0.38,0.84) — behind and above the camera.
static const float3 LIGHT_DIR0 = float3( 0.38, -0.38,  0.84); // key: behind-camera, upper-left
static const float3 LIGHT_DIR1 = float3(-0.50,  0.50, -0.71); // rim: front upper-right
static const float3 LIGHT_DIR2 = float3( 0.00, -1.00,  0.00); // bounce: from below
static const float4 LIGHT_COL0 = float4(1.0, 0.90, 0.80,  8.0); // warm key
static const float4 LIGHT_COL1 = float4(0.5, 0.55, 1.00,  2.5); // cool rim
static const float4 LIGHT_COL2 = float4(0.8, 0.80, 0.80,  0.8); // neutral bounce

float DistributionGGX(float3 N, float3 H, float roughness) {
    // Add epsilon to avoid singularities at very low roughness.
    float a     = max(roughness * roughness, 0.0025);
    float a2    = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom + 1e-7);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    // Heitz's k for direct lighting approximation.
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotL, roughness) * GeometrySchlickGGX(NdotV, roughness);
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (float3(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

float2 dir_to_equirect_uv(float3 dir) {
    float3 d = normalize(dir);
    float  u = 0.5 + atan2(d.x, -d.z) / (2.0 * PI);
    float  v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
    return float2(frac(u), clamp(v, 0.0, 1.0));
}

float3 accumulateDirLight(
    float3 N, float3 V, float3 F0,
    float3 albedo, float roughness, float metallic,
    float3 lightDir, float4 lightColor
) {
    // lightDir is direction from light to surface; incoming L is opposite.
    float3 L        = normalize(-lightDir);
    float3 H        = normalize(V + L);
    float3 radiance = lightColor.w * lightColor.xyz;
    float  NDF      = DistributionGGX(N, H, roughness);
    float  G        = GeometrySmith(N, V, L, roughness);
    float3 F        = fresnelSchlick(max(dot(H, V), 0.0), F0);
    float3 kD       = (float3(1.0, 1.0, 1.0) - F) * (1.0 - metallic);
    float  denom    = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    float3 specular = (NDF * G * F) / float3(denom, denom, denom);
    return (kD * (albedo / PI) + specular) * radiance * max(dot(N, L), 0.0);
}

[shader("fragment")]
float4 fs_main(
    [[vk::location(0)]] float2 uv      : TEXCOORD0,
    [[vk::location(1)]] float3 WorldPos: TEXCOORD1,
    [[vk::location(2)]] float3 NormalIn: TEXCOORD2
) : SV_TARGET {
    float3 albedo    = texBaseColor.Sample(smpBaseColor, uv).rgb * UMaterial.baseColorFactor;
    // roughnessFactor maps to specularity slot (offset 16 in current material layout).
    // metallic is not stored yet — default to 0.0 (dielectric).
    float  roughness = clamp(UMaterial.roughnessFactor, 0.045, 0.99);
    float  metallic  = 0.0;
    // TODO: Support normal mapping (tangent space) and AO texture.
    float3 N  = normalize(NormalIn);
    float3 V  = normalize(UCamera.position.xyz - WorldPos);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, float3(metallic, metallic, metallic));
    float3 Lo = float3(0.0, 0.0, 0.0);
    Lo += accumulateDirLight(N, V, F0, albedo, roughness, metallic, LIGHT_DIR0, LIGHT_COL0);
    Lo += accumulateDirLight(N, V, F0, albedo, roughness, metallic, LIGHT_DIR1, LIGHT_COL1);
    Lo += accumulateDirLight(N, V, F0, albedo, roughness, metallic, LIGHT_DIR2, LIGHT_COL2);
    // Diffuse IBL ambient.
    float3 kS             = fresnelSchlick(max(dot(N, V), 0.0), F0);
    float3 kD             = (float3(1.0, 1.0, 1.0) - kS) * (1.0 - metallic);
    float3 irradiance     = texIrradiance.Sample(smpIrradiance, dir_to_equirect_uv(N)).rgb;
    float3 ambientDiffuse = kD * irradiance * albedo;
    // Specular IBL.
    float3 R                  = reflect(-V, N);
    float  MAX_REFLECTION_LOD = 4.0;
    float3 prefilteredColor   = texPrefilter.SampleLevel(smpPrefilter, dir_to_equirect_uv(R), roughness * MAX_REFLECTION_LOD).rgb;
    float2 envBRDF            = texBrdfLut.Sample(smpBrdfLut, float2(max(dot(N, V), 0.0), roughness)).rg;
    float3 F                  = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 specularIBL        = prefilteredColor * (F * envBRDF.x + envBRDF.y);
    float3 ambient            = float3(0.04, 0.04, 0.04) * albedo;
    float3 color              = Lo + ambientDiffuse + specularIBL + ambient;
    return float4(color, 1.0);
}
