use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use pill_assets::{Pipeline, Rule};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("res");
    // Track the output so a missing file forces a re-run
    println!(
        "cargo:rerun-if-changed={}",
        root.join("textures/generated/pill.png").display()
    );
    let pipeline = Pipeline {
        root,
        rules: vec![Box::new(PillProcTexture { resolution: 1024 })],
    };
    let stats = pipeline.run().expect("game asset pipeline failed");
    println!("cargo:rerun-if-changed=build.rs");
    for p in &stats.discovered {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}

// ── Rule ─────────────────────────────────────────────────────────────────────

struct PillProcTexture {
    resolution: u32,
}

impl Rule for PillProcTexture {
    fn name(&self) -> &'static str {
        "pill_proc_texture"
    }

    fn input_glob(&self) -> &'static str {
        "models/pill.obj"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        // input: .../res/models/pill.obj → .../res/textures/generated/pill.png
        let res_dir = input.parent().unwrap().parent().unwrap();
        res_dir.join("textures/generated/pill.png")
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let (models, _) = tobj::load_obj(
            input,
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )
        .with_context(|| format!("failed to load {input:?}"))?;

        let gray = image::Rgba([128u8, 128, 128, 255]);
        let mut img = image::RgbaImage::from_pixel(self.resolution, self.resolution, gray);

        for model in &models {
            let mesh = &model.mesh;
            if mesh.texcoords.is_empty() {
                bail!("{input:?}: model '{}' has no UV coordinates", model.name);
            }
            let n_tris = mesh.indices.len() / 3;
            for t in 0..n_tris {
                let i = [
                    mesh.indices[t * 3] as usize,
                    mesh.indices[t * 3 + 1] as usize,
                    mesh.indices[t * 3 + 2] as usize,
                ];
                let uv = [
                    [mesh.texcoords[i[0] * 2], mesh.texcoords[i[0] * 2 + 1]],
                    [mesh.texcoords[i[1] * 2], mesh.texcoords[i[1] * 2 + 1]],
                    [mesh.texcoords[i[2] * 2], mesh.texcoords[i[2] * 2 + 1]],
                ];
                let pos = [
                    [
                        mesh.positions[i[0] * 3],
                        mesh.positions[i[0] * 3 + 1],
                        mesh.positions[i[0] * 3 + 2],
                    ],
                    [
                        mesh.positions[i[1] * 3],
                        mesh.positions[i[1] * 3 + 1],
                        mesh.positions[i[1] * 3 + 2],
                    ],
                    [
                        mesh.positions[i[2] * 3],
                        mesh.positions[i[2] * 3 + 1],
                        mesh.positions[i[2] * 3 + 2],
                    ],
                ];
                rasterize_triangle(&mut img, self.resolution, uv, pos);
            }
        }

        img.save(output)
            .with_context(|| format!("failed to save {output:?}"))?;
        Ok(())
    }
}

// ── Rasterizer ────────────────────────────────────────────────────────────────

fn rasterize_triangle(
    img: &mut image::RgbaImage,
    res: u32,
    uv: [[f32; 2]; 3],
    pos: [[f32; 3]; 3],
) {
    let rf = res as f32;
    // V flipped: UV origin is bottom-left, image origin is top-left
    let to_px = |u: [f32; 2]| -> [f32; 2] { [u[0] * rf, (1.0 - u[1]) * rf] };
    let p = [to_px(uv[0]), to_px(uv[1]), to_px(uv[2])];

    let min_x = (p[0][0].min(p[1][0]).min(p[2][0]).floor().max(0.0) as u32).min(res - 1);
    let max_x = (p[0][0].max(p[1][0]).max(p[2][0]).ceil() as u32).min(res - 1);
    let min_y = (p[0][1].min(p[1][1]).min(p[2][1]).floor().max(0.0) as u32).min(res - 1);
    let max_y = (p[0][1].max(p[1][1]).max(p[2][1]).ceil() as u32).min(res - 1);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = [x as f32 + 0.5, y as f32 + 0.5];
            if let Some(bary) = barycentric(px, p[0], p[1], p[2]) {
                let obj = [
                    bary[0] * pos[0][0] + bary[1] * pos[1][0] + bary[2] * pos[2][0],
                    bary[0] * pos[0][1] + bary[1] * pos[1][1] + bary[2] * pos[2][1],
                    bary[0] * pos[0][2] + bary[1] * pos[1][2] + bary[2] * pos[2][2],
                ];
                let v = fbm(obj, 5);
                let c = (v.clamp(0.0, 1.0) * 255.0) as u8;
                img.put_pixel(x, y, image::Rgba([c, c, c, 255]));
            }
        }
    }
}

fn barycentric(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
    // Signed area of triangle formed by two edges meeting at p1, toward p2, tested at q
    let edge = |p1: [f32; 2], p2: [f32; 2], q: [f32; 2]| -> f32 {
        (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
    };
    let denom = edge(a, b, c);
    if denom.abs() < f32::EPSILON {
        return None;
    }
    let wa = edge(b, c, p) / denom;
    let wb = edge(c, a, p) / denom;
    let wc = edge(a, b, p) / denom;
    if wa >= 0.0 && wb >= 0.0 && wc >= 0.0 {
        Some([wa, wb, wc])
    } else {
        None
    }
}

// ── Noise ─────────────────────────────────────────────────────────────────────

fn fbm(p: [f32; 3], octaves: u32) -> f32 {
    let mut v = 0.0f32;
    let mut amp = 0.5f32;
    let mut q = p;
    for _ in 0..octaves {
        v += amp * value_noise3(q);
        q = [q[0] * 2.02, q[1] * 2.02, q[2] * 2.02];
        amp *= 0.5;
    }
    v
}

fn value_noise3(p: [f32; 3]) -> f32 {
    let ix = p[0].floor() as i32;
    let iy = p[1].floor() as i32;
    let iz = p[2].floor() as i32;
    let fx = p[0] - ix as f32;
    let fy = p[1] - iy as f32;
    let fz = p[2] - iz as f32;
    // Smoothstep
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let hash = |x: i32, y: i32, z: i32| -> f32 {
        let n = x
            .wrapping_mul(1619)
            .wrapping_add(y.wrapping_mul(31337))
            .wrapping_add(z.wrapping_mul(1013904223));
        let n = n.wrapping_mul(n.wrapping_mul(n).wrapping_mul(60493));
        (n as f32) / (i32::MAX as f32) * 0.5 + 0.5
    };
    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

    lerp(
        lerp(
            lerp(hash(ix, iy, iz), hash(ix + 1, iy, iz), ux),
            lerp(hash(ix, iy + 1, iz), hash(ix + 1, iy + 1, iz), ux),
            uy,
        ),
        lerp(
            lerp(hash(ix, iy, iz + 1), hash(ix + 1, iy, iz + 1), ux),
            lerp(hash(ix, iy + 1, iz + 1), hash(ix + 1, iy + 1, iz + 1), ux),
            uy,
        ),
        uz,
    )
}
