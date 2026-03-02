#!/bin/bash
set -e

# Use rustup toolchain, not Homebrew
export PATH="$HOME/.cargo/bin:${PATH/\/opt\/homebrew\/bin:/}"

ENGINE_DIR="$(cd "$(dirname "$0")" && pwd)"
GAME_DIR="${1:-.}"

if [ ! -d "$GAME_DIR" ]; then
    echo "Error: Game directory '$GAME_DIR' not found"
    exit 1
fi

GAME_DIR="$(cd "$GAME_DIR" && pwd)"

echo "Building WASM for: $GAME_DIR"
echo "Using engine: $ENGINE_DIR"

cd "$ENGINE_DIR/engine/pill_web"
wasm-pack build --target web --out-dir pkg

mkdir -p "$GAME_DIR/web"
cp pkg/pill_web.js "$GAME_DIR/web/"
cp pkg/pill_web_bg.wasm "$GAME_DIR/web/"
cp index.html "$GAME_DIR/web/"

echo ""
echo "Done! Run:"
echo "  cd $GAME_DIR/web && python3 -m http.server 8080"
echo "  Open http://localhost:8080"
