# ekauq3D

## Quick commands (from the repo root)

```
# listen server: you play and host on UDP 27500; bots fill the match; the session is recorded
EKAUQ_HOST=1 cargo run --release --manifest-path engine/pill_launcher/Cargo.toml -- -a run -p examples/ekauq3D -o examples/ekauq3D/build/release -c release

# join it (use the same EKAUQ_MAP as the host, default lqdm1)
EKAUQ_CONNECT=<host-ip> cargo run --release --manifest-path engine/pill_launcher/Cargo.toml -- -a run -p examples/ekauq3D -o examples/ekauq3D/build/release -c release

# dedicated server, no window (run from engine/); the match pauses while nobody is connected
EKAUQ_HOST=27500 EKAUQ_BOTS=3 cargo run --release -p pill_game --example ekauq_server

# replay a recorded session and check it (run from engine/)
cargo run --release -p pill_game --example ekauq_tune -- replay ../examples/ekauq3D/build/sessions/<file>.ekr
```
