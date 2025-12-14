use glam::Mat4;

// Reference: Dennis Gustafsson DoF blog + bgfx example 45-bokeh.

/// Right-handed perspective projection with NDC z in [0..1] (wgpu/Metal/Vulkan/D3D).
pub fn perspective_rh_zo(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let tan_half_fovy = (0.5 * fov_y_radians).tan();
    let ys = 1.0 / tan_half_fovy;
    let xs = ys / aspect;

    let zz = z_far / (z_near - z_far);
    let wz = (z_far * z_near) / (z_near - z_far);

    Mat4::from_cols_array(&[
        xs, 0.0, 0.0, 0.0, //
        0.0, ys, 0.0, 0.0, //
        0.0, 0.0, zz, -1.0, //
        0.0, 0.0, wz, 0.0, //
    ])
}

/// Right-handed perspective projection with NDC z in [-1..1] (OpenGL).
pub fn perspective_rh_no(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let tan_half_fovy = (0.5 * fov_y_radians).tan();
    let ys = 1.0 / tan_half_fovy;
    let xs = ys / aspect;

    let zz = (z_far + z_near) / (z_near - z_far);
    let wz = (2.0 * z_far * z_near) / (z_near - z_far);

    Mat4::from_cols_array(&[
        xs, 0.0, 0.0, 0.0, //
        0.0, ys, 0.0, 0.0, //
        0.0, 0.0, zz, -1.0, //
        0.0, 0.0, wz, 0.0, //
    ])
}

/// Depth-unpack constants derived from the projection matrix (mul=-m[14], add=m[10], with sign correction).
pub fn depth_unpack_consts_from_proj(proj: Mat4) -> (f32, f32) {
    let p = proj.to_cols_array(); // column-major float[16]
    let depth_mul = -p[3 * 4 + 2];
    let mut depth_add = p[2 * 4 + 2];
    if depth_mul * depth_add < 0.0 {
        depth_add = -depth_add;
    }
    (depth_mul, depth_add)
}
