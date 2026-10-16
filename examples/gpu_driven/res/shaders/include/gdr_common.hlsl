// Instance and bounds layouts of the GPU-driven example (scene::Instance, scene::Bounds)

struct Instance {
    column_major float4x4 world;
    uint prop;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct Bounds {
    float4 min_corner;
    float4 max_corner;
};
