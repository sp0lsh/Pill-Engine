// Cargo build-script entry. Rules + runner live in `pill_assets`; the launcher
// invokes the same Pipeline via `-a assets`.

use std::env;
use std::path::PathBuf;

use pill_assets::{default_rules, Pipeline};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("res");
    let pipeline = Pipeline {
        root: root.clone(),
        rules: default_rules(),
    };

    // HlslToWgsl dates its outputs to their newest source, so writing them doesn't re-trigger
    // this script through the watched shaders directory
    let stats = pipeline.run().expect("asset pipeline failed");
    write_shader_bindings(&root.join("shaders"));

    println!("cargo:rerun-if-changed=build.rs");
    // Header-include directory: tracked recursively so changes to shared
    // structs (common.hlsl) re-trigger the build. The glob doesn't pick these
    // up (HlslToWgsl matches top-level shaders/*.hlsl only).
    println!(
        "cargo:rerun-if-changed={}",
        root.join("shaders/include").display()
    );
    // The directory itself, so a newly added shader re-runs the pipeline
    println!("cargo:rerun-if-changed={}", root.join("shaders").display());
    for input in &stats.discovered {
        println!("cargo:rerun-if-changed={}", input.display());
    }
}

/// `OUT_DIR/shader_bindings.rs` from every shader's reflection JSON (graphics/shader_bindings.rs)
fn write_shader_bindings(shaders: &std::path::Path) {
    let mut reflections: Vec<(String, String)> = std::fs::read_dir(shaders)
        .expect("read shaders directory")
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let stem = name.strip_suffix(pill_assets::rules::hlsl_to_wgsl::REFLECTION_SUFFIX)?;
            let json = std::fs::read_to_string(entry.path()).ok()?;
            Some((stem.to_owned(), json))
        })
        .collect();
    reflections.sort();
    let source =
        pill_assets::shader_bindings::generate(&reflections).expect("shader binding generation");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shader_bindings.rs");
    std::fs::write(out, source).expect("write shader_bindings.rs");
}
