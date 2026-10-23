//! Fetches a LibreQuake `.map`, the texture PNGs it uses and Quake's palette from the LibreQuake
//! repository at a pinned commit (art BSD-3-Clause) and builds the level with the engine's Quake
//! map loader. All parsing, geometry, materials and lights are in the engine
//! (`pill_engine`'s `quake_map_loading`); this module is file lookup + I/O.

use std::collections::HashMap;

use pill_engine::game::{
    quake_map::{self, QuakeCollision, QuakeLevel},
    quake_mdl::{self, QuakeModel},
};

use crate::fetch;

/// LibreQuake at commit dd4d5bf: every path in [`FILES`] is relative to it.
const LIBRE_QUAKE_BASE: &str =
    "https://raw.githubusercontent.com/lavenderdotpet/LibreQuake/dd4d5bffcabbcf9ceca32a1d83829109928c3f08/";

/// The repository's `.map` sources and texture PNGs at that commit, one path per line (the
/// texture folders are the WADs' sources: `texture-wads/<wad>/<texture>.png`).
const FILES: &str = include_str!("libre_quake_files.txt");

const PALETTE_PATH: &str = "lq1/gfx/palette.lmp";
/// The weapons' view models and their source skins.
const AXE_MODEL_PATH: &str = "lq1/progs/v_axe.mdl";
const AXE_SKIN_PATH: &str = "lq1/progs/src/textures/v_axe.mdl.0.png";
const SHOTGUN_MODEL_PATH: &str = "lq1/progs/v_shot.mdl";
const SHOTGUN_SKIN_PATH: &str = "lq1/progs/src/textures/v_shot.mdl.0.png";

/// What the loader hands to the game: the level, its collision and the player's weapons.
pub struct LoadedLevel {
    pub level: QuakeLevel,
    pub collision: QuakeCollision,
    /// `None` if a model doesn't parse (that weapon slot is then empty); a failed download fails
    /// the whole load.
    pub axe: Option<QuakeModel>,
    pub shotgun: Option<QuakeModel>,
}

const DEFAULT_MAP: &str = "lqdm1";

/// Quake light units → `LightComponent::intensity` (see `quake_map::QuakeLight`).
const LIGHT_SCALE: f32 = 4.0;

/// The map to load: `EKAUQ_MAP` natively, `?map=` in the browser, else `lqdm1`.
pub fn requested_map() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    let requested = std::env::var("EKAUQ_MAP").ok();
    #[cfg(target_arch = "wasm32")]
    let requested = web_sys::window()
        .and_then(|window| window.location().search().ok())
        .and_then(|query| {
            query
                .trim_start_matches('?')
                .split('&')
                .find_map(|pair| pair.strip_prefix("map=").map(str::to_string))
        });
    requested
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| DEFAULT_MAP.to_string())
}

/// (file stem, repository path) of every map.
fn map_paths() -> impl Iterator<Item = (&'static str, &'static str)> {
    FILES
        .lines()
        .filter(|path| path.ends_with(".map"))
        .filter_map(|path| Some((path.rsplit('/').next()?.strip_suffix(".map")?, path)))
}

/// Texture name (lowercase file stem) → (WAD folder, repository path) of every PNG with it.
fn texture_files() -> HashMap<String, Vec<(&'static str, &'static str)>> {
    let mut files: HashMap<String, Vec<(&'static str, &'static str)>> = HashMap::new();
    for path in FILES
        .lines()
        .filter(|path| path.starts_with("texture-wads/"))
    {
        let mut parts = path.split('/').skip(1);
        let (Some(wad), Some(file)) = (parts.next(), parts.next()) else {
            continue;
        };
        if let Some(stem) = file.strip_suffix(".png") {
            files
                .entry(stem.to_ascii_lowercase())
                .or_default()
                .push((wad, path));
        }
    }
    files
}

/// The PNG for a map texture name. The repository spells `*` as `star_` and `+` as `+` or
/// `plus_`, and appends `_fbr` to textures with fullbright texels; among several matches, the
/// WAD listed first in the map's `wad` key wins.
fn resolve(
    name: &str,
    wads: &[String],
    files: &HashMap<String, Vec<(&'static str, &'static str)>>,
) -> Option<&'static str> {
    let spellings: Vec<String> = match (name.strip_prefix('*'), name.strip_prefix('+')) {
        (Some(rest), _) => vec![format!("star_{rest}"), name.to_string()],
        (_, Some(rest)) => vec![name.to_string(), format!("plus_{rest}")],
        _ => vec![name.to_string()],
    };
    spellings
        .iter()
        .flat_map(|spelling| [spelling.clone(), format!("{spelling}_fbr")])
        .filter_map(|stem| files.get(&stem))
        .flatten()
        .min_by_key(|(wad, _)| {
            wads.iter()
                .position(|listed| listed.eq_ignore_ascii_case(wad))
                .unwrap_or(usize::MAX)
        })
        .map(|(_, path)| *path)
}

fn url(path: &str) -> String {
    format!("{LIBRE_QUAKE_BASE}{path}")
}

/// Fetches map `name` (a file stem, e.g. `lqdm1`), its textures, the palette and the weapons;
/// builds the level, its collision and the weapon models.
pub async fn load(name: &str) -> Result<LoadedLevel, String> {
    let (_, map_path) = map_paths()
        .find(|(stem, _)| stem.eq_ignore_ascii_case(name))
        .ok_or_else(|| {
            let names: Vec<&str> = map_paths().map(|(stem, _)| stem).collect();
            format!("no map `{name}`; maps: {}", names.join(" "))
        })?;
    let mut fetched = fetch::get_all(&[
        url(map_path),
        url(PALETTE_PATH),
        url(AXE_MODEL_PATH),
        url(AXE_SKIN_PATH),
        url(SHOTGUN_MODEL_PATH),
        url(SHOTGUN_SKIN_PATH),
    ])
    .await?
    .into_iter();
    let text = String::from_utf8_lossy(&fetched.next().unwrap_or_default()).into_owned();
    let palette = fetched.next().unwrap_or_default();
    let axe_model = fetched.next().unwrap_or_default();
    let axe_skin = fetched.next().unwrap_or_default();
    let shotgun_model = fetched.next().unwrap_or_default();
    let shotgun_skin = fetched.next().unwrap_or_default();
    let map = quake_map::parse_map(&text).map_err(|e| format!("{map_path}: {e}"))?;

    // Every drawn texture and the frames of the animated ones, resolved to repository files
    let files = texture_files();
    let wads = map.wads();
    let mut names = map.texture_names();
    let frames: Vec<String> = names
        .iter()
        .flat_map(|name| quake_map::animation_frame_names(name))
        .collect();
    names.extend(frames);
    names.sort();
    names.dedup();
    let resolved: Vec<(String, &str)> = names
        .into_iter()
        .filter_map(|name| {
            let path = resolve(&name, &wads, &files)?;
            Some((name, path))
        })
        .collect();
    let urls: Vec<String> = resolved.iter().map(|(_, path)| url(path)).collect();
    log::info!("ekauq3D: {map_path}: fetching {} textures", urls.len());
    let images: HashMap<String, Vec<u8>> = resolved
        .into_iter()
        .map(|(name, _)| name)
        .zip(fetch::get_all(&urls).await?)
        .collect();

    let level = quake_map::build_quake_level(name, &map, &images, Some(&palette), LIGHT_SCALE)
        .map_err(|e| format!("{map_path}: {e}"))?;
    let stats = &level.stats;
    log::info!(
        "ekauq3D: {name}: {} entities, {} brushes, {} faces ({} hidden), {} triangles in {} meshes, \
         {} materials, {} lights, {} spawns",
        stats.entities,
        stats.brushes,
        stats.faces,
        stats.hidden_faces,
        stats.triangles,
        level.scene.meshes.len(),
        level.scene.materials.len(),
        level.lights.len(),
        level.spawns.len(),
    );
    if !level.missing_textures.is_empty() {
        log::warn!(
            "ekauq3D: textures not in the repository (checker): {}",
            level.missing_textures.join(" ")
        );
    }
    let collision = QuakeCollision::from_map(&map);
    let model = |name: &str, path: &str, bytes: &[u8], skin: &[u8]| {
        quake_mdl::build_quake_model(name, bytes, Some(skin), Some(&palette))
            .inspect_err(|error| log::warn!("ekauq3D: {path}: {error}"))
            .ok()
    };
    Ok(LoadedLevel {
        level,
        collision,
        axe: model("v_axe", AXE_MODEL_PATH, &axe_model, &axe_skin),
        shotgun: model("v_shot", SHOTGUN_MODEL_PATH, &shotgun_model, &shotgun_skin),
    })
}
