# Bistro (native only)

Amazon Lumberyard Bistro (Exterior), lit by day: sun with shadows, sky image-based
lighting, diffuse GI (DDGI probes), screen-space reflections, SSAO, bloom, depth of field, motion
blur, Lottes tonemapping, FXAA.

```
./engine/pill_launcher/target/release/PillLauncher -a run -c release -p examples/bistro/
```

The first run downloads ~600 MB (the glTF, its Git LFS buffer, KTX2 textures, the HDR sky) into
`build/fetch_cache/` and cooks every KTX2 (UASTC) texture to BC7 with mips into `build/cooked/`
(about 20 s on an M-series Mac). Later runs read the cooked files and load in about 2 s. Delete
either folder to refetch or recook.

Not a web example: the assets are too large and the BC7 cooker (Intel ISPC kernels) is native.

## Controls

- 1–4: viewpoints
- Hold left mouse: look; WASD, E/Q, Shift: move, up/down, faster
- K / G / H: sun shadows / diffuse GI / indirect light only
- R / T: screen-space reflections / reflections only
- P: wet street
- O / B / F / M: SSAO / bloom / depth of field / motion blur

The "Bistro" panel (folded, top left) has the sun's direction and intensity and the same toggles.

## Attribution

- Amazon Lumberyard Bistro, Open Research Content Archive (ORCA),
  <http://developer.nvidia.com/orca/amazon-lumberyard-bistro>, CC BY 4.0.
- glTF + KTX2 conversion: <https://github.com/qian-o/GLTF-Assets/tree/main/Bistro>, CC BY 4.0.
- `san_giuseppe_bridge_4k.hdr`, HDRI Haven, CC0.
