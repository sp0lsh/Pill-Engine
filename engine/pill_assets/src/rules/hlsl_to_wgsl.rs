use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

use anyhow::{bail, Context, Result};

use crate::Rule;

/// HLSL → WGSL via the `slangc` CLI. Stage and entry-point inferred from
/// filename suffix:
///   `*_vertex.hlsl`   → `-stage vertex   -entry vs_main`
///   `*_fragment.hlsl` → `-stage fragment -entry fs_main`
///   `*_compute.hlsl`  → `-stage compute  -entry cs_main`
///
/// Other suffixes fail fast with a clear message; add a new naming convention
/// (or a richer rule) before introducing mesh stages.
///
/// Variants: each leading `// @variant <suffix> <DEFINE>[=value]...` line of a source compiles it
/// once more with those `-D` defines into `<name>_<suffix>_<stage>.wgsl`. The base output's
/// `.wgsl.d` depfile lists the `#include`s, so editing one rebuilds every output.
pub struct HlslToWgsl;

/// Marker of a variant line in a source's leading comment block.
const VARIANT_PREFIX: &str = "// @variant ";

impl Rule for HlslToWgsl {
    fn name(&self) -> &'static str {
        "hlsl_to_wgsl"
    }

    fn input_glob(&self) -> &'static str {
        // Top-level only; `shaders/include/*.hlsl` are header files (#include'd).
        "shaders/*.hlsl"
    }

    fn output_for(&self, input: &Path) -> PathBuf {
        input.with_extension("wgsl")
    }

    fn extra_outputs(&self, input: &Path) -> Vec<PathBuf> {
        read_variants(input)
            .map(|variants| {
                variants
                    .iter()
                    .map(|variant| variant.output.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn dependencies(&self, _input: &Path, output: &Path) -> Vec<PathBuf> {
        let depfile = depfile_of(output);
        match std::fs::read_to_string(&depfile) {
            Ok(text) => parse_depfile(&text),
            // Not written yet: the missing dependency makes the output stale
            Err(_) => vec![depfile],
        }
    }

    fn build(&self, input: &Path, output: &Path) -> Result<()> {
        let (entry, stage) = infer_stage(input)?;
        let depfile = depfile_of(output);

        compile(input, output, entry, stage, &[], Some(&depfile))?;
        let variants = read_variants(input)?;
        for variant in &variants {
            compile(input, &variant.output, entry, stage, &variant.defines, None)?;
        }

        // Every output gets the newest source's mtime: the engine build script watches the
        // shaders directory, and outputs dated by this run would trigger one more rebuild. The
        // up-to-date check (outputs >= sources) still holds.
        let sources = std::iter::once(input.to_path_buf())
            .chain(self.dependencies(input, output))
            .collect::<Vec<_>>();
        let newest = sources
            .iter()
            .filter_map(|source| source.metadata().and_then(|m| m.modified()).ok())
            .max()
            .unwrap_or_else(SystemTime::now);
        let outputs = [output.to_path_buf(), depfile]
            .into_iter()
            .chain(variants.into_iter().map(|variant| variant.output));
        for path in outputs {
            if let Ok(file) = std::fs::File::options().write(true).open(&path) {
                let _ = file.set_modified(newest);
            }
        }
        Ok(())
    }
}

/// One `// @variant` line: its output path and `-D` defines.
#[derive(Debug, PartialEq)]
struct Variant {
    output: PathBuf,
    defines: Vec<String>,
}

/// `<output>.d`, the depfile slangc writes for the base output.
fn depfile_of(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_owned();
    path.push(".d");
    PathBuf::from(path)
}

/// Make-style `target: dep dep ...`; the dependencies past the target.
fn parse_depfile(text: &str) -> Vec<PathBuf> {
    text.split_once(": ")
        .map(|(_, dependencies)| {
            dependencies
                .split_whitespace()
                .filter(|token| *token != "\\")
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default()
}

fn read_variants(input: &Path) -> Result<Vec<Variant>> {
    let source =
        std::fs::read_to_string(input).with_context(|| format!("read shader {input:?}"))?;
    parse_variants(input, &source)
}

/// Variants of the leading `//` comment lines of `source`.
fn parse_variants(input: &Path, source: &str) -> Result<Vec<Variant>> {
    let mut variants = Vec::new();
    for line in source.lines().take_while(|line| line.starts_with("//")) {
        let Some(rest) = line.strip_prefix(VARIANT_PREFIX) else {
            continue;
        };
        let mut tokens = rest.split_whitespace();
        let suffix = tokens
            .next()
            .with_context(|| format!("{input:?}: `// @variant` without a suffix"))?;
        let defines: Vec<String> = tokens.map(str::to_owned).collect();
        if defines.is_empty() {
            bail!("{input:?}: variant `{suffix}` has no defines");
        }
        variants.push(Variant {
            output: variant_output(input, suffix)?,
            defines,
        });
    }
    Ok(variants)
}

/// `name_fragment.hlsl` + `f16` → `name_f16_fragment.wgsl`.
fn variant_output(input: &Path, suffix: &str) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .with_context(|| format!("can't read filename stem of {input:?}"))?;
    let (name, stage) = stem
        .rsplit_once('_')
        .with_context(|| format!("{input:?}: no stage suffix"))?;
    Ok(input.with_file_name(format!("{name}_{suffix}_{stage}.wgsl")))
}

fn compile(
    input: &Path,
    output: &Path,
    entry: &str,
    stage: &str,
    defines: &[String],
    depfile: Option<&Path>,
) -> Result<()> {
    let mut command = Command::new("slangc");
    command.args([
        input.to_str().context("input path is not UTF-8")?,
        "-target",
        "wgsl",
        "-entry",
        entry,
        "-stage",
        stage,
        // -O0 keeps the emitted WGSL close to the source; no -g: debug info only
        // preserves local variable names and costs ~15% of the shader text that every
        // game embeds (struct, binding and function names stay)
        "-O0",
        "-o",
        output.to_str().context("output path is not UTF-8")?,
    ]);
    for define in defines {
        command.arg(format!("-D{define}"));
    }
    if let Some(depfile) = depfile {
        command.arg("-depfile");
        command.arg(depfile);
    }

    let out = match command.output() {
        Ok(out) => out,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
            "slangc not found on PATH. Install Slang from https://github.com/shader-slang/slang/releases and add slangc to PATH."
        ),
        Err(e) => return Err(e).context("failed to spawn slangc"),
    };

    if !out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!(
            "slangc exited {:?} for {input:?} {defines:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            out.status.code()
        );
    }

    let wgsl = std::fs::read_to_string(output).with_context(|| format!("read {output:?}"))?;
    let stripped = strip_subgroup_enables(&wgsl);
    if stripped.len() != wgsl.len() {
        std::fs::write(output, stripped).with_context(|| format!("write {output:?}"))?;
    }
    Ok(())
}

/// naga (wgpu 25) rejects the `subgroups` / `subgroups_f16` enable-extensions but accepts the
/// subgroup builtins once the device has `Features::SUBGROUP`.
fn strip_subgroup_enables(wgsl: &str) -> String {
    wgsl.lines()
        .filter(|line| !matches!(line.trim(), "enable subgroups;" | "enable subgroups_f16;"))
        .map(|line| format!("{line}\n"))
        .collect()
}

fn infer_stage(input: &Path) -> Result<(&'static str, &'static str)> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .with_context(|| format!("can't read filename stem of {input:?}"))?;

    if stem.ends_with("_vertex") {
        Ok(("vs_main", "vertex"))
    } else if stem.ends_with("_fragment") {
        Ok(("fs_main", "fragment"))
    } else if stem.ends_with("_compute") {
        Ok(("cs_main", "compute"))
    } else {
        bail!(
            "{input:?}: can't infer shader stage from filename. Expected suffix `_vertex`, `_fragment` or `_compute`."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_directive_f16_names_output_and_defines() {
        let input = Path::new("/s/pbr_opaque_fragment.hlsl");
        let source = "// @variant f16 PILL_F16\n// body\n#include \"x.hlsl\"\n";
        assert_eq!(
            parse_variants(input, source).unwrap(),
            vec![Variant {
                output: PathBuf::from("/s/pbr_opaque_f16_fragment.wgsl"),
                defines: vec!["PILL_F16".to_owned()],
            }]
        );
    }

    #[test]
    fn no_directive_no_variants() {
        let input = Path::new("/s/a_vertex.hlsl");
        assert!(parse_variants(input, "// plain\n#include \"x\"\n")
            .unwrap()
            .is_empty());
    }

    #[test]
    fn directive_after_code_ignored() {
        let input = Path::new("/s/a_vertex.hlsl");
        let source = "#include \"x\"\n// @variant f16 PILL_F16\n";
        assert!(parse_variants(input, source).unwrap().is_empty());
    }

    #[test]
    fn variant_without_defines_fails() {
        let input = Path::new("/s/a_vertex.hlsl");
        assert!(parse_variants(input, "// @variant f16\n").is_err());
    }

    #[test]
    fn subgroup_enables_stripped_other_enables_kept() {
        let wgsl = "enable f16;\n\nenable subgroups;\nenable subgroups_f16;\nfn f() {}\n";
        assert_eq!(strip_subgroup_enables(wgsl), "enable f16;\n\nfn f() {}\n");
    }

    #[test]
    fn depfile_lists_dependencies_after_target() {
        let text = "/o/a.wgsl: /s/a.hlsl /s/include/b.hlsl\n";
        assert_eq!(
            parse_depfile(text),
            vec![
                PathBuf::from("/s/a.hlsl"),
                PathBuf::from("/s/include/b.hlsl"),
            ]
        );
    }
}
