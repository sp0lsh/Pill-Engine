#![allow(non_snake_case, dead_code)]

use anyhow::*;
use clap::{App, AppSettings, Arg};
use config::Config;
use fs_extra::dir::CopyOptions;
use path_absolutize::Absolutize;
use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, SystemTime},
};

// - Cargo commands

enum Location {
    EngineProjectRoot, // Main engine project directory (containing creates, examples, etc)
    EngineCrates,
    PillEngineCrate,
    PillCoreCrate,
    PillStandaloneCrate,
    PillLauncherCrate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompileMode {
    Debug,
    Release,
    HotReload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BuildTarget {
    Standalone,
    Wasm,
}

// --- Platform helpers -------------------------------------------------------

#[cfg(target_os = "windows")]
const EXEC_SUFFIX: &str = ".exe";
#[cfg(not(target_os = "windows"))]
const EXEC_SUFFIX: &str = ""; // Linux, macOS, etc. – no extension

#[cfg(target_os = "windows")]
const DYLIB_PREFIX: &str = ""; //  pill_game.dll
#[cfg(not(target_os = "windows"))]
const DYLIB_PREFIX: &str = "lib"; //  libpill_game.so / .dylib

#[cfg(target_os = "windows")]
const DYLIB_SUFFIX: &str = ".dll";
#[cfg(target_os = "linux")]
const DYLIB_SUFFIX: &str = ".so";
#[cfg(target_os = "macos")]
const DYLIB_SUFFIX: &str = ".dylib";

fn dylib(name: &str) -> String {
    format!("{DYLIB_PREFIX}{name}{DYLIB_SUFFIX}")
}

fn target_dir_for(mode: &CompileMode) -> &'static str {
    match mode {
        CompileMode::Release => "release",
        _ => "debug",
    }
}

// Returns absolute paths
fn get_path(location: Location) -> PathBuf {
    let main_engine_directory = env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
        .join("..")
        .join("..")
        .join("..")
        .join("..")
        .absolutize()
        .unwrap()
        .to_path_buf();

    match location {
        Location::EngineProjectRoot => main_engine_directory,
        Location::EngineCrates => main_engine_directory.join("engine"),
        Location::PillEngineCrate => main_engine_directory.join("engine").join("pill_engine"),
        Location::PillCoreCrate => main_engine_directory.join("engine").join("pill_core"),
        Location::PillStandaloneCrate => {
            main_engine_directory.join("engine").join("pill_standalone")
        }
        Location::PillLauncherCrate => main_engine_directory.join("engine").join("pill_launcher"),
    }
}

fn modify_file<A: FnMut(String) -> String>(
    input_path: &PathBuf,
    output_path: &PathBuf,
    mut action: A,
) -> Result<()> {
    // Open files from path
    let input_file = File::open(input_path).unwrap();

    // Read lines from input file
    let lines = BufReader::new(input_file)
        .lines()
        .map(|v| v.unwrap())
        .collect::<Vec<String>>();

    // Create new file (overwrite if input and output paths are the same)
    let mut output_file = File::create(output_path).unwrap();

    // Write files to output file
    for line in lines {
        writeln!(output_file, "{}", action(line)).unwrap();
    }

    Ok(())
}

fn parse_file_lines<A: FnMut(String)>(input_path: &PathBuf, mut action: A) -> Result<()> {
    // Open files from path
    let input_file = File::open(input_path).unwrap();

    // Read lines from input file
    let lines = BufReader::new(input_file)
        .lines()
        .map(|v| v.unwrap())
        .collect::<Vec<String>>();

    // Write files to output file
    for line in lines {
        action(line);
    }

    Ok(())
}

// --- Utilities ---

fn get_game_build_path(
    game_project_directory_path: &Path,
    output_directory_path: &PathBuf,
) -> Result<PathBuf> {
    let game_project_build_path = if output_directory_path.as_os_str() == "." {
        game_project_directory_path
            .join("build")
            .join("dev")
            .absolutize()
            .context("Failed to absolutize directory path")?
            .to_path_buf()
    } else {
        output_directory_path.absolutize()?.to_path_buf()
    };

    Ok(game_project_build_path)
}

fn get_game_title(game_project_directory_path: &Path) -> Result<String> {
    // Get game title
    let config_path = game_project_directory_path.join("res").join("config.ini");
    let mut config = Config::default();
    config
        .merge(config::File::with_name(config_path.to_str().unwrap()))
        .context("Failed to find config.ini file in game project \"res\" folder")?;
    let game_title = config
        .get_str("TITLE")
        .context("Failed to get game config.ini")?
        .replace(' ', "");

    Ok(game_title)
}

fn check_if_game_project_validity(game_project_directory_path: &Path) -> Result<()> {
    if !game_project_directory_path.join("Cargo.toml").exists() {
        return Err(Error::msg("Missing Cargo.toml file in game project folder"));
    }
    if !game_project_directory_path.join("res").exists() {
        return Err(Error::msg("Missing \"res\" folder in game project folder"));
    }
    if !game_project_directory_path.join("src").exists() {
        return Err(Error::msg("Missing \"src\" folder in game project folder"));
    }
    if !game_project_directory_path
        .join("res")
        .join("config.ini")
        .exists()
    {
        return Err(Error::msg(
            "Missing \"config.ini\" file in game project folder",
        ));
    }

    Ok(())
}

fn remove_files_starting_with(directory_path: &PathBuf, file_name_prefix: &str) -> Result<()> {
    if !directory_path.exists() || !directory_path.is_dir() {
        return Ok(()); // Skip non-existent or non-dir
    }

    for entry in fs::read_dir(directory_path).context("Failed to read directory")? {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();
        if path.is_file() {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.starts_with(file_name_prefix) {
                    fs::remove_file(&path)
                        .with_context(|| format!("Failed to remove file: {}", path.display()))?;
                }
            }
        }
    }

    Ok(())
}

// Render all *.puml under <crate>/docs/uml into <crate>/docs/uml_out as SVGs
fn render_puml_for_crate(crate_dir: &Path) -> Result<()> {
    let in_dir = crate_dir.join("docs").join("uml");
    let out_dir = crate_dir.join("docs").join("uml_out");

    if !in_dir.exists() {
        return Ok(()); // Skip non-existent
    }
    fs::create_dir_all(&out_dir)?;

    // Collect input files
    let mut inputs = Vec::new();
    for entry in fs::read_dir(&in_dir)
        .with_context(|| format!("Failed to read directory: {}", in_dir.display()))?
    {
        let path = entry?.path();
        println!("Checking file: {}", path.display());
        if path.extension() == Some(OsStr::new("puml")) {
            inputs.push(path);
        }
    }

    println!(
        "Found {} PlantUML files to render in {}",
        inputs.len(),
        in_dir.display()
    );

    if inputs.is_empty() {
        return Ok(());
    }

    let have_cli = Command::new("plantuml")
        .arg("-version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("PlantUML version"))
        .unwrap_or(false);

    // Prefer "plantuml" CLI tool if available
    if !have_cli {
        bail!("Please install plantuml!");
    }

    for puml in &inputs {
        let svg_path = out_dir
            .join(puml.file_stem().unwrap())
            .with_extension("svg");

        let mut child = Command::new("plantuml")
            .arg("-tsvg")
            .arg("-pipe")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .context("Spawn plantuml -pipe")?;

        {
            let mut stdin = child.stdin.take().unwrap();
            let bytes =
                fs::read(puml).with_context(|| format!("Read PUML file {}", puml.display()))?;
            stdin.write_all(&bytes)?;
        }

        let out = child.wait_with_output().context("Wait plantuml")?;
        if !out.status.success() {
            bail!("plantuml failed with code {}", out.status);
        }
        fs::write(&svg_path, &out.stdout)
            .with_context(|| format!("Write SVG file {}", svg_path.display()))?;
    }

    // TODO: either distribute plantuml or download it automatically
    Ok(())
}

fn prepare_workspace_for_game(
    game_project_directory_path: &Path,
    compile_mode: &CompileMode,
) -> Result<PathBuf> {
    // Check if it is valid game project directory
    check_if_game_project_validity(game_project_directory_path)
        .context("Game project is invalid")?;

    // Compilation has to be done together on pill_standalone and pill_game together in the same context.
    // For that compilation through Cargo workspace is required.
    // Otherwise, typeids of types like "Mesh" will not match what will make all generic (templated) functions work improperly
    let engine_workspace_directory_path = get_path(Location::EngineCrates);

    let workspace_manifest_path = engine_workspace_directory_path.join("Cargo.toml");
    if !workspace_manifest_path.exists() {
        return Err(Error::msg("Cannot find engine workspace manifest file"));
    }

    // If game project has changed changed then previous compilation artifacts have to be removed
    let compilation_artifacts_folder_path = get_path(Location::EngineCrates)
        .join("target")
        .join(target_dir_for(compile_mode));
    let engine_workspace_manifest_game_project_directory_path = format!("    \"{}\", ### Game project crate (This will be changed by Pill Launcher on build to allow proper compilation of game project)", game_project_directory_path.to_str().unwrap().replace('\\', "/"));
    let mut game_project_directory_already_linked = false;
    parse_file_lines(&workspace_manifest_path, |line: String| {
        if line.contains(
            engine_workspace_manifest_game_project_directory_path
                .clone()
                .as_str(),
        ) {
            game_project_directory_already_linked = true;
        }
    })?;

    if !game_project_directory_already_linked {
        // Remove previous compilation artifacts
        let artifact_prefix = if cfg!(target_os = "windows") {
            "pill_game"
        } else {
            "libpill_game"
        };
        remove_files_starting_with(&compilation_artifacts_folder_path, artifact_prefix)?;
        remove_files_starting_with(
            &compilation_artifacts_folder_path.join("deps"),
            artifact_prefix,
        )?;
    }

    // Update workspace manifest file to include game project crate
    modify_file(
        &workspace_manifest_path,
        &workspace_manifest_path,
        |line: String| -> String {
            if line.contains("### Game project crate") {
                return engine_workspace_manifest_game_project_directory_path.clone();
            }
            line
        },
    )?;

    // Update workspace path in game project manifest
    modify_file(
        &game_project_directory_path.join("Cargo.toml"),
        &game_project_directory_path.join("Cargo.toml"),
        |line: String| -> String {
            if line.contains("workspace") {
                return format!(
                    "workspace = \"{}\"",
                    get_path(Location::EngineCrates)
                        .to_str()
                        .unwrap()
                        .replace("\\", "/")
                );
            }
            line
        },
    )?;

    Ok(engine_workspace_directory_path)
}

// --- Actions ---

fn create_game_project(
    game_project_parent_directory_path: &PathBuf, 
    game_name: &String
) -> Result<()> {
    const TEMPLATE_NAME: &str = "pill_default";

    let game_project_directory_path = game_project_parent_directory_path.join(game_name);
    if game_project_directory_path.exists() {
        return Err(Error::msg(format!(
            "Game project directory {} already exists",
            game_project_directory_path.display()
        )));
    }

    let game_resource_directory_path = game_project_directory_path.join("res");

    println!(
        "Creating new game project {} in directory {}",
        game_name,
        game_project_directory_path.display()
    );

    // Get templates (assuming that they are stored in res folder of pill_launcher crate)
    let template_game_project_directory_path = get_path(Location::PillLauncherCrate)
        .join("res")
        .join("templates");

    // Copy template
    println!("Copying project template...");

    fs_extra::dir::copy(
        template_game_project_directory_path.join(TEMPLATE_NAME),
        game_project_parent_directory_path,
        &CopyOptions::new().overwrite(true),
    )
    .context("Cannot copy template directory")?;

    // Rename project directory
    fs::rename(TEMPLATE_NAME, game_name)?;

    // Setup config file
    println!("Setting up config file...");
    modify_file(
        &game_resource_directory_path.join("config.ini"),
        &game_resource_directory_path.join("config.ini"),
        |line: String| -> String {
            if line.starts_with("TITLE") {
                return format!("TITLE={}", game_name);
            }
            if line.starts_with("WINDOW_TITLE") {
                return format!("WINDOW_TITLE={}", game_name);
            }
            line
        },
    )?;

    // Setup cargo.toml file
    println!("Setting up manifest file...");
    modify_file(
        &game_project_directory_path.join("Cargo.toml"),
        &game_project_directory_path.join("Cargo.toml"),
        |line: String| -> String {
            if line.contains("pill_engine") {
                return format!(
                    "pill_engine = {{ path = \"{}\", features = [\"game\"] }}",
                    get_path(Location::PillEngineCrate)
                        .to_str()
                        .unwrap()
                        .replace("\\", "/")
                );
            }
            line
        },
    )?;

    modify_file(
        &game_project_directory_path.join("Cargo.toml"),
        &game_project_directory_path.join("Cargo.toml"),
        |line: String| -> String {
            if line.contains("workspace") {
                return format!(
                    "workspace = \"{}\"",
                    get_path(Location::EngineCrates)
                        .to_str()
                        .unwrap()
                        .replace("\\", "/")
                );
            }
            line
        },
    )?;

    // Success
    println!("Game project creation completed!");

    Ok(())
}

fn run_game_project(
    game_project_directory_path: &PathBuf,
    output_directory_path: &PathBuf,
    compile_mode: &CompileMode,
    game_args: &[String],
) -> Result<()> {
    // Build game project
    build_game_project(
        game_project_directory_path,
        output_directory_path,
        compile_mode,
    )?;

    // Run game project
    println!(
        "Running game project from {}...",
        output_directory_path.display()
    );
    let game_title =
        get_game_title(game_project_directory_path).context("Failed to get game title")?;
    let standalone_executable_path =
        output_directory_path.join(format!("{game_title}{EXEC_SUFFIX}"));

    // Run exe (capture potential IO error here)
    let status = Command::new(&standalone_executable_path)
        .current_dir(output_directory_path)
        .args(game_args)
        .status()
        .context(format!(
            "Failed to launch game project executable: {}",
            standalone_executable_path.display()
        ))?;

    if !status.success() {
        // Game ran and exited with an error — don't say "failed to run" - just return Ok
        eprintln!(
            "Game exited with error code: {}",
            status.code().map_or("unknown".into(), |c| c.to_string())
        );
    }

    Ok(())
}

fn build_game_project(
    game_project_directory_path: &Path,
    output_directory_path: &Path,
    compile_mode: &CompileMode,
) -> Result<()> {
    println!(
        "Building game project from {}...",
        game_project_directory_path.display()
    );

    let engine_workspace_directory_path =
        prepare_workspace_for_game(game_project_directory_path, compile_mode)?;

    // Build standalone executable along with game dynamic library
    let mut arguments = vec!["build", "-p", "pill_game", "-p", "pill_standalone"];
    if *compile_mode == CompileMode::Release {
        arguments.push("--release");
    }
    Command::new("cargo")
        .args(&arguments)
        .current_dir(&engine_workspace_directory_path)
        .status()
        .context("failed to run cargo build")?
        .success()
        .then_some(())
        .ok_or_else(|| Error::msg("build failed"))?;

    // Create build directory if does not exist
    fs::create_dir_all(output_directory_path.join("data").as_path())
        .context("Failed to create build output directories")?;

    let compilation_artifacts_folder_path = get_path(Location::EngineCrates)
        .join("target")
        .join(target_dir_for(compile_mode));
    // Get game title
    let game_title =
        get_game_title(game_project_directory_path).context("Failed to get game title")?;

    if *compile_mode != CompileMode::HotReload {
        // Copy built standalone executable to build directory
        let standalone_output_path =
            compilation_artifacts_folder_path.join(format!("pill_standalone{EXEC_SUFFIX}"));
        if !standalone_output_path.exists() {
            return Err(Error::msg(
                "Standalone executable was not built successfully",
            ));
        }

        let destination_executable_path =
            output_directory_path.join(format!("{game_title}{EXEC_SUFFIX}"));
        fs::copy(&standalone_output_path, &destination_executable_path).with_context(|| {
            format!(
                "Can't copy standalone executable from {} to {}",
                standalone_output_path.display(),
                destination_executable_path.display()
            )
        })?;

        // ensure executable bit on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&destination_executable_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&destination_executable_path, perms)?;
        }
    }

    // Copy built dynamic library to build directory
    let game_library_output_path = compilation_artifacts_folder_path.join(dylib("pill_game"));
    if !game_library_output_path.exists() {
        return Err(Error::msg(format!(
            "Game dynamic library was not built successfully in {}",
            game_library_output_path.display()
        )));
    }

    let game_dynamic_library_name = if *compile_mode == CompileMode::HotReload {
        dylib("pill_game_hot_reloaded")
    } else {
        dylib("pill_game")
    };
    let output_game_library_path = output_directory_path
        .join("data")
        .join(game_dynamic_library_name);
    fs::copy(&game_library_output_path, &output_game_library_path).context(format!(
        "Can't copy game dynamic library from {} to {}",
        game_library_output_path.display(),
        output_game_library_path.display()
    ))?;

    // Success
    println!("Game built successfully!");

    Ok(())
}

// Builds the game for WASM + WebGPU, outputting to <game>/build/wasm/.
//
// Strategy: copy the wasm crate template (pill_launcher/res/templates/wasm/)
// into a scratch dir inside the game directory, rewrite path-deps to absolute
// paths (engine crates + the game at -p), then run wasm-pack. Keeps the engine
// directory pristine across multi-game use.
fn build_web_game_project(
    game_project_directory_path: &Path,
    compile_mode: &CompileMode,
) -> Result<()> {
    println!(
        "Building WASM/WebGPU target for game project at {}...",
        game_project_directory_path.display()
    );

    if *compile_mode == CompileMode::HotReload {
        println!("Note: hot-reload is not meaningful for WASM; using --dev mode.");
    }

    let wasm_template_dir = get_path(Location::PillLauncherCrate)
        .join("res")
        .join("templates")
        .join("wasm");

    let build_wasm_dir = game_project_directory_path.join("build").join("wasm");
    let scratch_dir = build_wasm_dir.join(".build");
    let scratch_pill_web_dir = scratch_dir.join("pill_web");
    let scratch_pkg_dir = scratch_dir.join("pkg");

    fs::create_dir_all(&scratch_pill_web_dir).with_context(|| {
        format!(
            "Failed to create scratch dir {}",
            scratch_pill_web_dir.display()
        )
    })?;

    fs::copy(
        wasm_template_dir.join("Cargo.toml"),
        scratch_pill_web_dir.join("Cargo.toml"),
    )
    .context("Failed to copy pill_web Cargo.toml to scratch")?;

    // Copy the engine workspace's Cargo.lock so the scratch build resolves to
    // the same crate versions as an in-place build of engine/pill_web. Without
    // this, cargo re-resolves and picks newer versions (wasm-bindgen in
    // particular), which on this branch breaks WebGPU rendering.
    let engine_lock = get_path(Location::EngineCrates).join("Cargo.lock");
    if engine_lock.exists() {
        fs::copy(&engine_lock, scratch_pill_web_dir.join("Cargo.lock"))
            .context("Failed to copy engine Cargo.lock into scratch")?;
    }

    let scratch_src_dir = scratch_pill_web_dir.join("src");
    if scratch_src_dir.exists() {
        fs::remove_dir_all(&scratch_src_dir).context("Failed to clean scratch src/")?;
    }
    fs_extra::dir::copy(
        wasm_template_dir.join("src"),
        &scratch_pill_web_dir,
        &CopyOptions::new().overwrite(true),
    )
    .context("Failed to copy pill_web src/ to scratch")?;

    let engine_crates_dir = get_path(Location::EngineCrates);
    let pill_engine_path = engine_crates_dir
        .join("pill_engine")
        .to_string_lossy()
        .replace("\\", "/");
    let pill_renderer_path = engine_crates_dir
        .join("pill_renderer")
        .to_string_lossy()
        .replace("\\", "/");
    let pill_core_path = engine_crates_dir
        .join("pill_core")
        .to_string_lossy()
        .replace("\\", "/");
    let pill_game_path = game_project_directory_path
        .to_string_lossy()
        .replace("\\", "/");

    let scratch_manifest = scratch_pill_web_dir.join("Cargo.toml");
    modify_file(
        &scratch_manifest,
        &scratch_manifest,
        |line: String| -> String {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pill_engine ") || trimmed.starts_with("pill_engine=") {
                return format!(
                    "pill_engine = {{ path = \"{}\", features = [\"game\", \"internal\"] }}",
                    pill_engine_path
                );
            }
            if trimmed.starts_with("pill_renderer ") || trimmed.starts_with("pill_renderer=") {
                return format!("pill_renderer = {{ path = \"{}\" }}", pill_renderer_path);
            }
            if trimmed.starts_with("pill_core ") || trimmed.starts_with("pill_core=") {
                return format!("pill_core = {{ path = \"{}\" }}", pill_core_path);
            }
            line
        },
    )?;

    // Append:
    //  - pill_game dep pointing at the -p game directory (not in the committed
    //    Cargo.toml; injected here so pill_web is only buildable via launcher).
    //  - [workspace] + resolver = "2" so cargo doesn't walk up into the game's
    //    parent workspace and matches engine/Cargo.toml's feature unification
    //    (wgpu backend selection is sensitive to resolver version).
    {
        let mut f = fs::OpenOptions::new()
            .append(true)
            .open(&scratch_manifest)
            .context("Failed to open scratch Cargo.toml for append")?;
        writeln!(f, "\npill_game = {{ path = \"{}\" }}", pill_game_path)
            .context("Failed to append pill_game dep to scratch Cargo.toml")?;
        writeln!(f, "\n[workspace]\nresolver = \"2\"")
            .context("Failed to append [workspace] to scratch Cargo.toml")?;
    }

    let scratch_pkg_str = scratch_pkg_dir.to_string_lossy().to_string();
    let mut args: Vec<String> = vec![
        "build".into(),
        "--target".into(),
        "web".into(),
        "--out-dir".into(),
        scratch_pkg_str,
    ];
    match compile_mode {
        CompileMode::Release => {}
        CompileMode::Debug | CompileMode::HotReload => args.push("--dev".into()),
    }

    println!(
        "Running wasm-pack in scratch crate {}...",
        scratch_pill_web_dir.display()
    );

    // Prefer rustup's toolchain over Homebrew's rustc — Homebrew doesn't ship
    // the wasm32-unknown-unknown target.
    let mut cmd = Command::new("wasm-pack");
    cmd.args(&args).current_dir(&scratch_pill_web_dir);
    if let Some(home) = env::var_os("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo").join("bin");
        let existing = env::var_os("PATH").unwrap_or_default();
        let mut parts: Vec<PathBuf> = vec![cargo_bin];
        parts.extend(env::split_paths(&existing).filter(|p| p != Path::new("/opt/homebrew/bin")));
        if let std::result::Result::Ok(joined) = env::join_paths(parts) {
            cmd.env("PATH", joined);
        }
    }
    let status = cmd.status().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::msg("wasm-pack not found on PATH. Install it with: cargo install wasm-pack")
        } else {
            Error::new(e).context("Failed to execute wasm-pack")
        }
    })?;

    if !status.success() {
        bail!("wasm-pack build failed (exit {:?})", status.code());
    }

    fs::create_dir_all(&build_wasm_dir)
        .with_context(|| format!("Failed to create {}", build_wasm_dir.display()))?;

    for file in ["pill_web.js", "pill_web_bg.wasm"] {
        let src = scratch_pkg_dir.join(file);
        let dst = build_wasm_dir.join(file);
        fs::copy(&src, &dst)
            .with_context(|| format!("Failed to copy {} to {}", src.display(), dst.display()))?;
    }

    // Copy the engine's web template dir (index.html, pill_logo.png, any future
    // assets) into the build output. This is the default chrome.
    let template_web_dir = get_path(Location::PillLauncherCrate)
        .join("res")
        .join("templates")
        .join("web");
    for entry in fs::read_dir(&template_web_dir)
        .with_context(|| format!("Failed to read template dir {}", template_web_dir.display()))?
    {
        let entry = entry?;
        // path().metadata() follows symlinks — pill_logo.png is a symlink into
        // media/logo/ and must be treated as the file it points at.
        if entry.path().metadata()?.is_file() {
            let dst = build_wasm_dir.join(entry.file_name());
            fs::copy(entry.path(), &dst).with_context(|| {
                format!(
                    "Failed to copy {} to {}",
                    entry.path().display(),
                    dst.display()
                )
            })?;
        }
    }

    // Overlay any per-game customizations from <game>/web/ on top. Each file
    // individually overrides the engine default (e.g. drop in a custom
    // index.html, swap the logo, add a favicon).
    let user_web_dir = game_project_directory_path.join("web");
    if user_web_dir.is_dir() {
        for entry in fs::read_dir(&user_web_dir)
            .with_context(|| format!("Failed to read {}", user_web_dir.display()))?
        {
            let entry = entry?;
            if entry.path().metadata()?.is_file() {
                let dst = build_wasm_dir.join(entry.file_name());
                fs::copy(entry.path(), &dst).with_context(|| {
                    format!(
                        "Failed to overlay {} onto {}",
                        entry.path().display(),
                        dst.display()
                    )
                })?;
            }
        }
    }

    // Size report — only meaningful on release builds (debug wasm is dominated
    // by debuginfo symbols).
    if *compile_mode == CompileMode::Release {
        let preopt_wasm = scratch_pill_web_dir
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("release")
            .join("pill_web.wasm");
        print_wasm_size_report(&build_wasm_dir, &preopt_wasm);
    }

    println!();
    println!("Done! Serve with:");
    println!(
        "  PillLauncher -a run -t wasm -p {}",
        game_project_directory_path.display()
    );
    println!(
        "  (or any static server pointed at {})",
        build_wasm_dir.display()
    );

    Ok(())
}

// Prints a compact post-build size report: final (post-wasm-opt) vs pre-opt
// sizes, per-crate breakdown, and top symbols. No-ops with a hint if `twiggy`
// is not installed.
fn print_wasm_size_report(build_wasm_dir: &Path, preopt_wasm: &Path) {
    let preopt_size = match fs::metadata(preopt_wasm).map(|m| m.len()) {
        core::result::Result::Ok(v) => v,
        core::result::Result::Err(_) => return,
    };
    let final_wasm = build_wasm_dir.join("pill_web_bg.wasm");
    let final_size = fs::metadata(&final_wasm).ok().map(|m| m.len());

    println!();
    match final_size {
        Some(f) => println!(
            "wasm size: final {} | pre-opt {}",
            fmt_bytes(f),
            fmt_bytes(preopt_size)
        ),
        None => println!("wasm size: pre-opt {}", fmt_bytes(preopt_size)),
    }

    let output = match Command::new("twiggy")
        .args(["top", "-n", "15000"])
        .arg(preopt_wasm)
        .stderr(Stdio::null())
        .output()
    {
        core::result::Result::Ok(o) => o,
        core::result::Result::Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("(install twiggy for per-crate breakdown: cargo install twiggy)");
            return;
        }
        core::result::Result::Err(_) => return,
    };
    if !output.status.success() {
        return;
    }
    let stdout = match String::from_utf8(output.stdout) {
        core::result::Result::Ok(s) => s,
        core::result::Result::Err(_) => return,
    };

    // Parse twiggy text output. Each data row:
    //   "   <bytes> ┊ <pct>% ┊ <item name>"
    let mut items: Vec<(u64, String)> = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Shallow")
            || trimmed.starts_with('─')
            || trimmed.starts_with("Σ")
            || (trimmed.contains("and ") && trimmed.contains("more"))
        {
            continue;
        }
        let parts: Vec<&str> = trimmed.split('┊').collect();
        if parts.len() < 3 {
            continue;
        }
        let bytes: u64 = match parts[0].trim().parse() {
            core::result::Result::Ok(v) => v,
            core::result::Result::Err(_) => continue,
        };
        if bytes == 0 {
            continue;
        }
        items.push((bytes, parts[2].trim().to_string()));
    }

    if items.is_empty() {
        return;
    }
    // Percentages are against the actual binary size, not the sum of twiggy's
    // shallow bytes (which double-counts — e.g. symbols referencing .rodata).
    let total = preopt_size;

    // Aggregate by crate (mirrors the awk heuristics in the old wasm-size.sh).
    let mut by_crate: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    for (bytes, name) in &items {
        *by_crate.entry(classify_crate(name)).or_insert(0) += *bytes;
    }
    let mut groups: Vec<(String, u64)> = by_crate.into_iter().collect();
    groups.sort_by(|a, b| b.1.cmp(&a.1));

    println!();
    println!("Crate breakdown (% of pre-opt {}):", fmt_bytes(total));
    println!("  {:<18} {:>10} {:>7}", "crate", "size", "%");
    for (crate_name, bytes) in groups.iter().take(15) {
        let pct = 100.0 * *bytes as f64 / total as f64;
        println!(
            "  {:<18} {:>10} {:>6.1}%",
            crate_name,
            fmt_bytes(*bytes),
            pct
        );
    }

    println!();
    println!("Top 10 symbols:");
    for (bytes, name) in items.iter().take(10) {
        let pct = 100.0 * *bytes as f64 / total as f64;
        let display_name = if name.chars().count() > 72 {
            let truncated: String = name.chars().take(69).collect();
            format!("{truncated}...")
        } else {
            name.clone()
        };
        println!(
            "  {:>10} {:>5.1}%  {}",
            fmt_bytes(*bytes),
            pct,
            display_name
        );
    }
}

fn fmt_bytes(n: u64) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    const KB: f64 = 1024.0;
    let f = n as f64;
    if f >= MB {
        format!("{:.2} MB", f / MB)
    } else if f >= KB {
        format!("{:.1} KB", f / KB)
    } else {
        format!("{n} B")
    }
}

// Classify a twiggy item name into a coarse crate bucket. Heuristics ported
// verbatim from wasm-size.sh's awk block — extract the leading identifier,
// normalize known crate families into groups.
fn classify_crate(name: &str) -> String {
    if name.contains(".rodata") || name.contains("data segment") {
        return "[rodata]".into();
    }
    if name.contains("function names") {
        return "[debug:names]".into();
    }
    if name.contains("__wasm_bindgen") {
        return "[wasm-bindgen]".into();
    }
    if name.contains("custom section") {
        return "[custom]".into();
    }
    if name.starts_with("elem[")
        || name.starts_with("type[")
        || name.starts_with("import ")
        || name.starts_with("table[")
    {
        return "[wasm-meta]".into();
    }

    // Peel leading `<` and any `&` / `&mut ` so `<&mut Foo as Bar>::baz` resolves to `Foo`.
    let rest = name.strip_prefix('<').unwrap_or(name);
    let rest = rest
        .strip_prefix("&mut ")
        .or_else(|| rest.strip_prefix('&'))
        .unwrap_or(rest);
    let ident: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();

    if ident.is_empty() {
        return "[other]".into();
    }
    match ident.as_str() {
        "core" | "alloc" | "std" | "compiler_builtins" | "rustc_demangle" | "dlmalloc"
        | "str" | "bool" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64"
        | "f32" | "f64" | "char" | "usize" | "isize" | "T" => "[rust-std]".into(),
        "jpeg_decoder" | "png" | "tiff" | "gif" | "weezl" | "miniz_oxide" | "color_quant"
        | "qoi" | "exr" => "image".into(),
        "epaint" | "emath" | "egui_wgpu" | "egui_winit" => "egui".into(),
        "codespan_reporting" | "codespan" | "pp_rs" | "spirv" => "naga".into(),
        "wgpu_hal" | "wgpu_core" | "wgpu_types" => "wgpu".into(),
        "js_sys" => "web_sys".into(),
        _ => ident,
    }
}

// Serves <game>/build/wasm/ on localhost:8080 with live reload.
// Ensures the build is up to date first, then runs a tiny_http server that:
//   - serves static files from build/wasm/
//   - injects a long-poll client into HTML responses
//   - watches build/wasm/ mtimes and triggers a browser reload on any change
fn run_web_game_project(
    game_project_directory_path: &Path,
    compile_mode: &CompileMode,
) -> Result<()> {
    build_web_game_project(game_project_directory_path, compile_mode)?;

    let build_wasm_dir = game_project_directory_path.join("build").join("wasm");
    let addr = "127.0.0.1:8080";

    // Broadcast set for long-poll subscribers. Each pending /__reload request
    // parks a Sender here; when any file in build_wasm_dir changes, the watcher
    // drains the vec and signals every subscriber.
    let subscribers: Arc<Mutex<Vec<mpsc::Sender<()>>>> = Arc::new(Mutex::new(Vec::new()));

    // File-change watcher thread — polls mtimes every 500ms.
    {
        let subscribers = Arc::clone(&subscribers);
        let watch_dir = build_wasm_dir.clone();
        let mut last = latest_mtime(&watch_dir);
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(500));
            let cur = latest_mtime(&watch_dir);
            if cur > last && cur.is_some() {
                last = cur;
                let mut subs = subscribers.lock().unwrap();
                for tx in subs.drain(..) {
                    let _ = tx.send(());
                }
            }
        });
    }

    let server = tiny_http::Server::http(addr).map_err(|e| Error::msg(e.to_string()))?;
    println!();
    println!("Serving {} at http://{}", build_wasm_dir.display(), addr);
    println!("Live reload enabled — the page will refresh on wasm rebuilds.");
    println!("Ctrl+C to stop.");

    for request in server.incoming_requests() {
        let subscribers = Arc::clone(&subscribers);
        let build_wasm_dir = build_wasm_dir.clone();
        thread::spawn(move || {
            if let Err(e) = handle_http_request(request, &build_wasm_dir, subscribers) {
                eprintln!("http request error: {:#}", e);
            }
        });
    }

    Ok(())
}

fn latest_mtime(dir: &Path) -> Option<SystemTime> {
    fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).filter_map(|e| {
        let name = e.file_name();
        let name_str = name.to_string_lossy();
        // Skip dotfiles and .build scratch dir
        if name_str.starts_with('.') {
            return None;
        }
        let md = e.metadata().ok()?;
        if !md.is_file() {
            return None;
        }
        md.modified().ok()
    }).max()
}

fn handle_http_request(
    request: tiny_http::Request,
    build_wasm_dir: &Path,
    subscribers: Arc<Mutex<Vec<mpsc::Sender<()>>>>,
) -> Result<()> {
    let url_path = request.url().split('?').next().unwrap_or("/").to_string();

    // Live-reload long-poll endpoint.
    if url_path == "/__reload" {
        let (tx, rx) = mpsc::channel();
        subscribers.lock().unwrap().push(tx);
        // Block up to 30s waiting for a file-change signal.
        let signaled = rx.recv_timeout(Duration::from_secs(30)).is_ok();
        let status = if signaled { 200 } else { 204 };
        let resp = tiny_http::Response::from_string("").with_status_code(status);
        request.respond(resp)?;
        return Ok(());
    }

    // Map URL path to a file under build_wasm_dir.
    let rel = url_path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    // Reject path traversal.
    if rel.split('/').any(|seg| seg == "..") {
        let resp = tiny_http::Response::from_string("bad path").with_status_code(400);
        request.respond(resp)?;
        return Ok(());
    }
    let path = build_wasm_dir.join(rel);
    if !path.is_file() {
        let resp = tiny_http::Response::from_string("not found").with_status_code(404);
        request.respond(resp)?;
        return Ok(());
    }

    let content_type = match path.extension().and_then(|s| s.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("png") => "image/png",
        Some("svg") => "image/svg+xml",
        Some("css") => "text/css; charset=utf-8",
        Some("ico") => "image/x-icon",
        Some("json") => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    };
    let ct_header = tiny_http::Header::from_bytes("Content-Type", content_type)
        .map_err(|_| Error::msg("invalid content-type header"))?;

    // Inject live-reload client into HTML responses.
    if content_type.starts_with("text/html") {
        let mut html = fs::read_to_string(&path)?;
        let inject = concat!(
            "<script>(async function reloadLoop(){for(;;){try{",
            "const r=await fetch('/__reload?v='+Date.now(),{cache:'no-store'});",
            "if(r.status===200){location.reload();return;}",
            "}catch(_){await new Promise(r=>setTimeout(r,500));}}})();</script>"
        );
        if let Some(idx) = html.rfind("</body>") {
            html.insert_str(idx, inject);
        } else {
            html.push_str(inject);
        }
        let resp = tiny_http::Response::from_string(html).with_header(ct_header);
        request.respond(resp)?;
        return Ok(());
    }

    let file = File::open(&path)?;
    let resp = tiny_http::Response::from_file(file).with_header(ct_header);
    request.respond(resp)?;
    Ok(())
}

// Runs "cargo doc" command for engine
fn generate_docs(output_directory_path: &PathBuf) -> Result<()> {
    // Set empty project as dependency
    let empty_example_game_path = get_path(Location::EngineProjectRoot)
        .join("examples")
        .join("Empty");
    if !empty_example_game_path.exists() {
        return Err(Error::msg(
            "Cannot find Empty project in examples directory",
        ));
    }

    // Update engine project dependency in game's cargo.toml
    modify_file(
        &empty_example_game_path.join("Cargo.toml"),
        &empty_example_game_path.join("Cargo.toml"),
        |line: String| -> String {
            if line.contains("pill_engine") {
                return format!(
                    "pill_engine = {{path = \"{}\", features = [\"game\"]}}",
                    get_path(Location::PillEngineCrate)
                        .to_str()
                        .unwrap()
                        .replace("\\", "/")
                );
            }
            line
        },
    )?;

    // Update game project dependency in standalone's cargo.toml
    modify_file(
        &get_path(Location::PillStandaloneCrate).join("Cargo.toml"),
        &get_path(Location::PillStandaloneCrate).join("Cargo.toml"),
        |line: String| -> String {
            if line.contains("pill_game") {
                return format!(
                    "pill_game = {{path = \"{}\"}}",
                    empty_example_game_path.to_str().unwrap().replace("\\", "/")
                );
            }
            line
        },
    )?;

    let output_path = if output_directory_path.as_os_str() == "." {
        env::current_dir().context("Failed to get current directory")?
    } else {
        output_directory_path
            .absolutize()
            .context("Failed to absolutize output path")?
            .to_path_buf()
    };

    let docs_path = output_path.join("docs");

    if docs_path.exists() {
        fs::remove_dir_all(&docs_path)
            .with_context(|| format!("Cannot clear output directory: {}", docs_path.display()))?;
    }

    let output_game_dev_path = docs_path.join("game_dev");
    let output_engine_dev_path = docs_path.join("engine_dev");

    // Prepare output directories
    fs::create_dir_all(&docs_path)?;
    fs::create_dir_all(&output_game_dev_path)?;
    fs::create_dir_all(&output_engine_dev_path)?;

    let engine_crate_manifest_path = get_path(Location::PillEngineCrate).join("Cargo.toml");
    let full_engine_manifest_path = empty_example_game_path.join("Cargo.toml");

    // Pre-render all PUML in the engine crate
    let pill_engine_dir = get_path(Location::PillEngineCrate);
    render_puml_for_crate(&pill_engine_dir)
        .context("Failed to render PlantUML diagrams for pill_engine")?;

    // Game dev docs
    let arguments = vec![
        "doc",
        "--no-deps",
        "--features",
        "game,internal",
        "--manifest-path",
        full_engine_manifest_path.to_str().unwrap(),
        "--target-dir",
        output_game_dev_path.to_str().unwrap(),
        "--release",
    ];
    let status = Command::new("cargo")
        .args(arguments)
        .status()
        .context("Failed to execute command for generating game dev docs")?;

    if !status.success() {
        bail!("Engine docs failed to generate (exit {:?})", status.code());
    }
    println!("Engine dev docs generated successfully!");

    // Engine dev docs
    // Generate pill_core before pill_engine and don't generate other dependencies
    let core_crate_manifest_path = get_path(Location::PillCoreCrate).join("Cargo.toml");
    let arguments = vec![
        "doc",
        "--no-deps",
        "--document-private-items",
        "--manifest-path",
        core_crate_manifest_path.to_str().unwrap(),
        "--target-dir",
        output_engine_dev_path.to_str().unwrap(),
        "--release",
    ];
    let status = Command::new("cargo")
        .args(arguments)
        .status()
        .context("Failed to execute command for generating core dev docs")?;

    // Success
    if status.success() {
        println!("Core dev docs generated successfully!");
    }

    let arguments = vec![
        "doc",
        "--no-deps",
        "--document-private-items",
        "--features",
        "all",
        "--manifest-path",
        engine_crate_manifest_path.to_str().unwrap(),
        "--target-dir",
        output_engine_dev_path.to_str().unwrap(),
        "--release",
    ];
    let status = Command::new("cargo")
        .args(arguments)
        .status()
        .context("Failed to execute command for generating engine dev docs")?;

    // Success
    if !status.success() {
        bail!(
            "Game dev docs failed to generate (exit {:?})",
            status.code()
        );
    }
    println!("Game dev docs generated successfully!");

    Ok(())
}

fn cargo_passthrough(
    game_project_directory_path: &Path,
    compile_mode: &CompileMode,
    cargo_args: &[String],
) -> Result<()> {
    if cargo_args.is_empty() {
        bail!("Must call cargo with at least one argument");
    }

    let engine_workspace_directory_path =
        prepare_workspace_for_game(game_project_directory_path, compile_mode)?;

    println!(
        "Running cargo {:?} in workspace {}...",
        cargo_args,
        engine_workspace_directory_path.display()
    );

    let status = Command::new("cargo")
        .args(cargo_args)
        .current_dir(engine_workspace_directory_path)
        .status()
        .context("Failed to run cargo passthrough")?;

    if !status.success() {
        bail!(
            "Cargo command failed: cargo {:?} (exit {:?})",
            cargo_args,
            status.code()
        );
    }

    Ok(())
}

fn run_app() -> Result<()> {
    let app = App::new("Pill Engine Launcher").about("Tool for managing Pill Engine game projects");

    // Definition of the options for the CLI
    let action_option = Arg::with_name("action")
        .short("a")
        .long("action")
        .takes_value(true)
        .possible_values(&["create", "run", "build", "docs", "cargo"])
        .required(true)
        .help("Specify action to perform: creating/running/building the game project or generating docs, alternatively run any cargo command on the project");

    let name_option = Arg::with_name("name")
        .short("n")
        .long("name")
        .takes_value(true)
        .required_if("action", "create")
        .help("Specify name of new game project");

    let path_option = Arg::with_name("path")
        .short("p")
        .long("path")
        .takes_value(true)
        .default_value(".")
        .required(false)
        .help("Specify the path for game project creating/running/building");

    let output_path_option = Arg::with_name("output-path")
        .short("o")
        .long("output-path")
        .takes_value(true)
        .default_value(".")
        .required(false)
        .help("Specify action output directory");

    let compile_mode_option = Arg::with_name("compile-mode")
        .short("c")
        .long("compile-mode")
        .takes_value(true)
        .help("Specify compile mode")
        .possible_values(&["debug", "release", "hot-reload"])
        .default_value("debug")
        .required(false);

    let target_option = Arg::with_name("target")
        .short("t")
        .long("target")
        .takes_value(true)
        .possible_values(&["standalone", "wasm"])
        .default_value("standalone")
        .required(false)
        .help("Build/run target: native standalone executable or WASM+WebGPU for the browser");

    let game_args = Arg::with_name("game-args")
        .help("Arguments passed through to cargo/game (use `--` to separate them)")
        .multiple(true)
        .last(true)
        .allow_hyphen_values(true);

    // Addition of the options to the CLI
    let app = app
        .arg(action_option)
        .arg(name_option)
        .arg(path_option)
        .arg(output_path_option)
        .arg(compile_mode_option)
        .arg(target_option)
        .arg(game_args)
        .setting(AppSettings::TrailingVarArg);

    // Extraction of the arguments
    let matches = app.get_matches();

    // Arguments
    let passthrough_args: Vec<String> = matches
        .values_of("game-args")
        .map(|vals| vals.map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let action_argument = matches
        .value_of("action")
        .expect("Action has to be specified");
    let directory_path_argument = matches.value_of("path");
    let game_name_argument = matches.value_of("name");
    let output_directory_path_argument = matches.value_of("output-path");
    let compile_mode_argument = matches.value_of("compile-mode").unwrap_or("debug");

    let compile_mode: CompileMode = match compile_mode_argument {
        "release" => CompileMode::Release,
        "hot-reload" => CompileMode::HotReload,
        _ => CompileMode::Debug,
    };

    let target: BuildTarget = match matches.value_of("target").unwrap_or("standalone") {
        "wasm" => BuildTarget::Wasm,
        _ => BuildTarget::Standalone,
    };

    match action_argument {
        "create" => {
            let game_parent_directory_path = PathBuf::from(directory_path_argument.expect("Game project parent directory path has to be specified using --path flag. For example: --path <PROJECT_DIR>"))
                .absolutize().context("Failed to absolutize game project parent directory path")?
                .to_path_buf();
            let game_name = String::from(game_name_argument.expect("Game name has to be specified using --name flag. For example: --name <MY_GAME_NAME>"));

            create_game_project(&game_parent_directory_path, &game_name)
                .context("Failed to create new game project")?;
        }
        "run" => {
            let game_project_directory_path = PathBuf::from(directory_path_argument.expect("Game project directory path has to be specified using --path flag. For example: --path <GAME_PROJECT_DIR>"))
                .absolutize().context("Failed to absolutize game project directory path")?
                .to_path_buf();

            match target {
                BuildTarget::Standalone => {
                    let mut output_directory_path = PathBuf::from(output_directory_path_argument.expect("Output directory path has to be specified using --output-path flag. For example: --output-path <OUTPUT_DIR>"));
                    output_directory_path =
                        get_game_build_path(&game_project_directory_path, &output_directory_path)
                            .unwrap();
                    run_game_project(
                        &game_project_directory_path,
                        &output_directory_path,
                        &compile_mode,
                        &passthrough_args,
                    )
                    .context("Failed to run game project")?;
                }
                BuildTarget::Wasm => {
                    run_web_game_project(&game_project_directory_path, &compile_mode)
                        .context("Failed to run game project for wasm")?;
                }
            }
        }
        "build" => {
            let game_project_directory_path = PathBuf::from(directory_path_argument.expect("Game project directory path has to be specified using --path flag. For example: --path <GAME_PROJECT_DIR>"))
                .absolutize().context("Failed to absolutize game project directory path")?
                .to_path_buf();

            match target {
                BuildTarget::Standalone => {
                    let mut output_directory_path = PathBuf::from(output_directory_path_argument.expect("Output directory path has to be specified using --output-path flag. For example: --output-path <OUTPUT_DIR>"));
                    output_directory_path =
                        get_game_build_path(&game_project_directory_path, &output_directory_path)?;
                    build_game_project(
                        &game_project_directory_path,
                        &output_directory_path,
                        &compile_mode,
                    )
                    .context("Failed to build game project")?;
                }
                BuildTarget::Wasm => {
                    if matches.occurrences_of("output-path") > 0 {
                        println!(
                            "Note: `-o/--output-path` is ignored with `-t wasm`; output is fixed at <game>/build/wasm/"
                        );
                    }
                    build_web_game_project(&game_project_directory_path, &compile_mode)
                        .context("Failed to build game project for wasm")?;
                }
            }
        }
        "docs" => {
            let output_directory_path = PathBuf::from(output_directory_path_argument.expect("Output directory path has to be specified using --output-path flag. For example: --output-path <OUTPUT_DIR>"))
                .absolutize().context("Failed to absolutize output directory path")?
                .to_path_buf();

            generate_docs(&output_directory_path).context("Failed to generate docs")?;
        }
        "cargo" => {
            let game_project_directory_path = PathBuf::from(
                directory_path_argument
                    .expect("Game project must be specified when running cargo commands."),
            )
            .absolutize()
            .context("Failed to absolutize game project directory path")?
            .to_path_buf();

            cargo_passthrough(
                &game_project_directory_path,
                &compile_mode,
                &passthrough_args,
            )
            .context("Cargo passthrough failed")?;
        }
        _ => {
            println!("Undefined action");
        }
    };
    Ok(())
}

fn main() {
    if let Err(e) = run_app() {
        eprintln!("{:#}", e);
        std::process::exit(1);
    }
}
