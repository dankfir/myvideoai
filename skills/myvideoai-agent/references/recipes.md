# Operation Recipes

Copy/adapt these tool calls. All go through the MCP server (`tools/call` with `name` + `arguments`). Paths are relative to `<repo-root>` unless absolute.

## Inspect first

```
probe_video   {"video": "myvideoai/samples/sample_lowres.mp4"}
list_plugins  {}
list_backends {}
```

Read the probe: note `width`, `height`, `fps`, `frames`, `has_audio`. Expected output size = `width × scale`, `height × scale` (rounded to even). Expected frames = `frames × interp`.

## Enhance (the main job)

Basic 2× upscale with Real-ESRGAN:
```
enhance_video {
  "input": "myvideoai/samples/sample_lowres.mp4",
  "output": "myvideoai/output/my_fix.mp4",
  "backend": "esrgan",
  "scale": 2,
  "crf": 18,
  "codec": "libx264"
}
```

Choose a backend:
- `esrgan` — photoreal / general detail. `upscale_model` = `x4plus` (default) or `x2plus`.
- `cugan` — anime / clean lines. Faster. `cugan_tile_mode` 0–4.
- `model` — drop-in ONNX/torch via a registered model plugin (see model-onboarding.md).

### Interpolation — smoother / higher fps
```
enhance_video { "input": "...", "interp": 2, ... }   // 2× frames
enhance_video { "input": "...", "interp": 4, ... }   // 4× frames
```
Uses RIFE v4.6. 60fps source + `interp:2` → ~119 output frames. Verify with the probe summary.

### Face restoration — old / blurry faces
```
enhance_video { "input": "...", "face": true, "face_weight": 0.5, ... }
```
GFPGAN v1.4. Runs after upscale, before interpolation. `face_weight` 0–1 blends restored vs original (0.5 is balanced). Best on faces that occupy a reasonable pixel area; tiny distant faces won't improve much.

### HDR & stabilization
```
enhance_video { "input": "...", "hdr": true, ... }        // SDR → HDR10 (bt2020/pq), 10-bit output
enhance_video { "input": "...", "stabilize": true, ... }  // vidstab two-pass deflicker/shake removal
```
Both can be combined. HDR output is `yuv420p10le` with `smpte2084` transfer + `bt2020` primaries — verify via `probe_video` (or ffprobe if you need the color tags; `probe_video` reports dimensions/frames). Stabilization enables `builtin.stabilize` (params shakiness/smoothing configurable via `plugin_params`).

### Extra frame plugins
```
enhance_video {
  "input": "...",
  "plugins": ["examples.unsharp"],
  "plugin_params": {"examples.unsharp": {"amount": 0.8}}
}
```
Plugin params override manifest defaults; last wins. Only plugins whose `enabled` is true or whose id is in `plugins` run.

## Preview before committing

```
preview_frame {
  "input": "myvideoai/samples/sample_lowres.mp4",
  "time": 0.5,
  "backend": "esrgan",
  "scale": 2,
  "face": false
}
```
Returns paths to `mcp_preview_*_orig.png`, `*_enh.png`, `*_compare.png` (left = original, right = enhanced). It uses the **same** model chain as the full pipeline, so what you see matches the render. Use this to tune `backend`/`scale`/`face` before a long encode.

If the user asks "show me before/after", present `*_compare.png` to them.

## Drop-in model backend preview

When `backend: "model"`, the tool auto-enables `builtin.onnx_superres`. To use a specific registered model:
```
preview_frame {
  "input": "...", "backend": "model",
  "plugins": ["my_model_id"],
  "model_backend": "auto"
}
```
`model_backend`: `auto` | `cuda` | `directml` | `cpu`.

## Reporting results

Always include:
- output path
- dimensions + frame count from the tool's probe summary
- which features were on (backend, scale, interp, face, hdr, stabilize)

Example good report: *"Enhanced → `output/my_fix.mp4`: 640×360, 119 frames @ 60fps (2× interpolation), ESRGAN 2×, audio preserved. Preview compare at `output/mcp_preview_..._compare.png`."*
