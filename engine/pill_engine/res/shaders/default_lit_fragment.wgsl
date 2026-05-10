@binding(0) @group(3) var diffuse_texture_0 : texture_2d<f32>;

@binding(1) @group(3) var diffuse_sampler_0 : sampler;

@binding(2) @group(3) var normal_texture_0 : texture_2d<f32>;

@binding(3) @group(3) var normal_sampler_0 : sampler;

struct _MatrixStorage_float4x4_ColMajorstd140_0
{
    @align(16) data_0 : array<vec4<f32>, i32(4)>,
};

struct CameraParams_std140_0
{
    @align(16) camera_position_0 : vec3<f32>,
    @align(16) camera_view_projection_0 : _MatrixStorage_float4x4_ColMajorstd140_0,
};

@binding(0) @group(1) var<uniform> camera_0 : CameraParams_std140_0;
struct MaterialParams_std140_0
{
    @align(16) tint_0 : vec3<f32>,
    @align(4) _pad_0 : f32,
    @align(16) spec_0 : f32,
};

@binding(0) @group(2) var<uniform> material_0 : MaterialParams_std140_0;
struct EngineParams_std140_0
{
    @align(16) fog_color_0 : vec3<f32>,
    @align(4) fog_density_0 : f32,
    @align(16) delta_time_0 : f32,
};

@binding(0) @group(0) var<uniform> engine_0 : EngineParams_std140_0;
struct pixelOutput_0
{
    @location(0) output_0 : vec4<f32>,
};

struct pixelInput_0
{
    @location(0) in_vertex_position_0 : vec3<f32>,
    @location(1) in_vertex_texture_coords_0 : vec2<f32>,
    @location(2) in_TBN_tangent_0 : vec3<f32>,
    @location(3) in_TBN_bitangent_0 : vec3<f32>,
    @location(4) in_TBN_normal_0 : vec3<f32>,
    @location(5) in_world_position_0 : vec3<f32>,
};

@fragment
fn fs_main( _S1 : pixelInput_0) -> pixelOutput_0
{
    const light_position_0 : vec3<f32> = vec3<f32>(-4.0f, 12.0f, -10.0f);
    const light_color_0 : vec3<f32> = vec3<f32>(0.55000001192092896f, 0.62000000476837158f, 1.0f);
    var object_color_0 : vec4<f32> = (textureSample((diffuse_texture_0), (diffuse_sampler_0), (_S1.in_vertex_texture_coords_0)));
    var object_normal_0 : vec4<f32> = (textureSample((normal_texture_0), (normal_sampler_0), (_S1.in_vertex_texture_coords_0)));
    var TBN_matrix_0 : mat3x3<f32> = mat3x3<f32>(_S1.in_TBN_tangent_0, _S1.in_TBN_bitangent_0, _S1.in_TBN_normal_0);
    var normal_tangent_0 : vec3<f32> = normalize(object_normal_0.xyz * vec3<f32>(2.0f) - vec3<f32>(1.0f));
    var normal_0 : vec3<f32> = normalize((((normal_tangent_0) * (TBN_matrix_0))));
    var ambient_light_factor_0 : vec3<f32> = light_color_0 * vec3<f32>(0.30000001192092896f);
    var light_direction_0 : vec3<f32> = normalize(light_position_0);
    var _S2 : f32 = max(dot(normal_0, light_direction_0), 0.0f);
    var diffuse_light_factor_0 : vec3<f32> = light_color_0 * vec3<f32>(_S2);
    var view_direction_0 : vec3<f32> = normalize(camera_0.camera_position_0 - _S1.in_world_position_0);
    var half_direction_0 : vec3<f32> = normalize(view_direction_0 + light_direction_0);
    var specular_light_strength_0 : f32 = pow(max(dot(normal_0, half_direction_0), 0.0f), 64.0f) * material_0.spec_0 * 4.0f;
    var _S3 : vec3<f32> = vec3<f32>(specular_light_strength_0);
    var final_color_0 : vec3<f32> = (ambient_light_factor_0 + diffuse_light_factor_0) * object_color_0.xyz * material_0.tint_0 + _S3;
    var fog_dist_0 : f32 = length(camera_0.camera_position_0 - _S1.in_world_position_0);
    var fog_factor_0 : f32 = clamp(1.0f - exp(- engine_0.fog_density_0 * engine_0.fog_density_0 * fog_dist_0 * fog_dist_0), 0.0f, 1.0f);
    var final_color_1 : vec3<f32> = mix(final_color_0, engine_0.fog_color_0, vec3<f32>(fog_factor_0));
    var _S4 : pixelOutput_0 = pixelOutput_0( vec4<f32>(final_color_1, 1.0f) );
    return _S4;
}

