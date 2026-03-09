<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="media/logo/pill_logo_horizontal_white.png">
    <img src="media/logo/pill_logo_horizontal_white.png">
  </picture>
</p>

Data-driven game engine written in Rust

## Design Goals
- Clean and simple
- Blazing fast
- Highly extensible

## Features
- Archetype-based Entity Component System 
- 3D graphics (Blinn-Phong shading model, instancing)
- Scenes
- Input handling (keyboard, mouse, gamepad)
- Sound playing (mono, spatial)
- Resource system (mesh, texture, shader, material, sound)
- Material system and custom shader loading
- Custom systems, components and resources support
- Error chaining
- Launcher tool
- Game project hot-reloading
- Engine code hot-reloading
- Serialization/deserialization of scene 🚧
- Shader code hot-reloading 🚧
- Lights 🚧
- Skybox 🚧
- Post-processing 🚧
- Networking 🚧
- Physics 🚧
- Configurable logging 🚧
- Editor 🚧

## Getting Started (Native)
1. Install Rust  
https://www.rust-lang.org/tools/install
2. Download and unpack this repository
3. Build Pill Launcher  
`cargo build --release --manifest-path <ENGINE_PATH>\Pill-Engine\engine\pill_launcher\Cargo.toml`
4. Add Pill Launcher to PATH (optional)  
On Windows: follow [these steps](https://superuser.com/questions/1861276/how-to-set-a-folder-to-the-path-environment-variable-in-windows-11) add `<ENGINE_PATH>\Pill-Engine\engine\pill_launcher\target\release`
On Linux: `echo 'export PATH="$PATH:<ENGINE_PATH>/Pill-Engine/engine/pill_launcher/target/release"' >> ~/.bashrc && source ~/.bashrc`  
and restart terminal
5. Create new game project  
`PillLauncher.exe -a create -n Hello-Pill`
6. Run it!  
`PillLauncher.exe -a run -p ./Hello-Pill`

## Getting Started (WebAssembly)

Build games for the web using WebGPU. Requires a WebGPU-compatible browser (Chrome 113+, Edge 113+, Firefox Nightly).

### Prerequisites
```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

### Build
```bash
PillLauncher -a build -t wasm -p /path/to/your/game
```

Example:
```bash
PillLauncher -a build -t wasm -p /Users/mk/dev/demo/pill_demo_upstream
cd /Users/mk/dev/demo/pill_demo_upstream/build/wasm && python3 -m http.server 8080
```

To customize the HTML shell for a specific game, commit a `<game>/web/index.html` — the launcher prefers it over the engine default at `engine/pill_launcher/res/templates/web/index.html`.

**Note:** Audio, networking, and gamepad input are disabled in the web build. The WASM entry point lives at [engine/pill_launcher/res/templates/wasm/](engine/pill_launcher/res/templates/wasm/) — the launcher instantiates this template per game.

Ref: https://rustwasm.github.io/docs/wasm-pack/

Check [demo](examples/Floating-Pills "demo")!

## Documentation
[Repository](https://github.com/MattSzymonski/Pill-Engine-Docs "Repository")

- For game developers - [Docs](https://raw.githack.com/MattSzymonski/Pill-Engine-Docs/main/docs/game_dev/doc/pill_engine/game/index.html "Docs")  
- For engine developers - [Docs](https://raw.githack.com/MattSzymonski/Pill-Engine-Docs/main/docs/engine_dev/doc/pill_engine/index.html "Docs")  

## Showcase
<p align="center">
  <img src="examples/Floating-Pills/media/floating_pills_1.gif" img width=100%>
</p>

<p align="center">
  <img src="media/logo/pill_pile.png" img width=100%>
</p>