#!/bin/bash
set -e

# WASM Size Report
# Analyzes cargo's pre-wasm-opt output (has symbol names) to show crate breakdown
# Usage: ./wasm-size.sh [game_directory]

GAME_DIR="${1:-}"
ENGINE_DIR="$(cd "$(dirname "$0")" && pwd)"
WASM="$ENGINE_DIR/engine/target/wasm32-unknown-unknown/release/pill_web.wasm"

# Build only if binary doesn't exist
if [ ! -f "$WASM" ]; then
    echo "Building WASM..."
    cd "$ENGINE_DIR/engine/pill_web"
    RUSTFLAGS='--cfg getrandom_backend="wasm_js"' cargo build --target wasm32-unknown-unknown --release 2>/dev/null
fi

if [ ! -f "$WASM" ]; then
    echo "Error: $WASM not found"
    exit 1
fi

TOTAL=$(stat -f%z "$WASM" 2>/dev/null || stat -c%s "$WASM")
MB=$(echo "scale=2; $TOTAL/1024/1024" | bc)

# Show final size if available
if [ -n "$GAME_DIR" ] && [ -f "$GAME_DIR/web/pill_web_bg.wasm" ]; then
    FINAL=$(stat -f%z "$GAME_DIR/web/pill_web_bg.wasm" 2>/dev/null || stat -c%s "$GAME_DIR/web/pill_web_bg.wasm")
    FINAL_MB=$(echo "scale=2; $FINAL/1024/1024" | bc)
    echo "Final (wasm-opt): ${FINAL_MB}MB | Analyzed (pre-opt): ${MB}MB"
    echo ""
fi

echo "══════════════════════════════════════════════════════════════════"
echo " CRATE BREAKDOWN (${MB}MB pre-wasm-opt)"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Part 1: Crate-level breakdown
twiggy top "$WASM" -n 15000 2>/dev/null | tail -n +3 | sed 's/┊/|/g' | awk -F'|' -v total="$TOTAL" '
function get_crate(name) {
    gsub(/^[ \t]+|[ \t]+$/, "", name)
    if (name ~ /more\.$/ || name ~ /Total Rows/) return ""
    
    # Special WASM sections
    if (name ~ /\.rodata|data segment/) return "[rodata]"
    if (name ~ /function names/) return "[debug:names]"
    if (name ~ /__wasm_bindgen/) return "[wasm-bindgen]"
    if (name ~ /custom section/) return "[custom]"
    if (name ~ /^elem\[|^type\[|^import |^table\[/) return "[wasm-meta]"
    
    # Handle <Type as Trait> - get first identifier after <
    if (substr(name, 1, 1) == "<") {
        rest = substr(name, 2)
        if (substr(rest, 1, 1) == "&") rest = (substr(rest, 1, 4) == "&mut") ? substr(rest, 6) : substr(rest, 2)
        c = ""; for (i=1; i<=length(rest); i++) { ch=substr(rest,i,1); if (ch ~ /[a-zA-Z0-9_]/) c=c ch; else break }
        if (c ~ /^(str|bool|u8|u16|u32|u64|i8|i16|i32|i64|f32|f64|char|usize|isize|T)$/) return "[rust-std]"
        if (c != "") return c
    }
    
    # Standard crate::path format
    c = ""; for (i=1; i<=length(name); i++) { ch=substr(name,i,1); if (ch ~ /[a-zA-Z0-9_]/) c=c ch; else break }
    if (c == "") return "[other]"
    
    # Normalize known crates
    if (c ~ /^(core|alloc|std|compiler_builtins|rustc_demangle|dlmalloc)$/) return "[rust-std]"
    if (c ~ /^(jpeg_decoder|png|tiff|gif|weezl|miniz_oxide|color_quant|qoi|exr)$/) return "image"
    if (c ~ /^(epaint|emath|egui_wgpu|egui_winit)$/) return "egui"
    if (c ~ /^(codespan_reporting|codespan|pp_rs)$/) return "naga"
    if (c ~ /^(wgpu_hal|wgpu_core|wgpu_types)$/) return "wgpu"
    if (c == "js_sys") return "web_sys"
    if (c == "spirv") return "naga"
    
    return c
}
NF >= 3 {
    gsub(/[^0-9]/, "", $1); bytes = int($1); if (bytes == 0) next
    crate = get_crate($3); if (crate == "") next
    sizes[crate] += bytes; count[crate]++
}
END {
    for (c in sizes) printf "%012d %s\n", sizes[c], c
}' | sort -rn | head -20 | awk -v total="$TOTAL" '
BEGIN { printf "%-18s %10s %7s\n", "CRATE", "SIZE KB", "%"; printf "%-18s %10s %7s\n", "------------------", "----------", "-------" }
{ kb=$1/1024; pct=$1/total*100; printf "%-18s %10.1f %6.1f%%\n", $2, kb, pct; sum+=$1 }
END { printf "%-18s %10s %7s\n", "------------------", "----------", "-------"; printf "%-18s %10.1f %6.1f%%\n", "TOTAL", total/1024, 100 }'

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " TOP 20 LARGEST SYMBOLS"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# Part 2: Top symbols
twiggy top "$WASM" -n 25 2>/dev/null | tail -n +3 | head -22 | sed 's/┊/|/g' | awk -F'|' -v total="$TOTAL" '
BEGIN { printf "%10s %6s  %-50s\n", "SIZE KB", "%", "SYMBOL"; printf "%10s %6s  %-50s\n", "----------", "------", "--------------------------------------------------" }
/[0-9]/ {
    gsub(/[^0-9]/, "", $1); bytes = int($1); if (bytes == 0) next
    name = $3; gsub(/^[ \t]+|[ \t]+$/, "", name)
    if (name ~ /more\.$/ || name ~ /Total Rows/) next
    # Truncate long names
    if (length(name) > 60) name = substr(name, 1, 57) "..."
    printf "%10.1f %5.1f%%  %s\n", bytes/1024, bytes/total*100, name
}'
