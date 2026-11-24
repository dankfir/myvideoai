# Troubleshooting

Work through these in order before declaring "it doesn't work". Evidence, not guesses.

## Verify the basics
1. Env vars set? Both `PYTHONPATH` entries + `KMP_DUPLICATE_LIB_OK=TRUE`, cwd = `<repo-root>`.
2. `probe_video` the input — does the file exist and decode? Garbage-in → garbage-out.
3. `list_backends` — is anything detected? If only `cpu`, GPU deps may be missing (still works, just slow).

## MCP server won't start / client can't parse responses
- **stdout must be pristine.** The server redirects `sys.stdout` to stderr and keeps a private `_REAL_STDOUT` for JSON-RPC only. If you fork/modify it, never `print()` to the real stdout — that corrupts the line-delimited JSON stream and strict clients throw `JSONDecodeError: Expecting value: line 1 column 2`.
- Launch with `PYTHONIOENCODING=utf-8` if you see encoding errors on Chinese paths.
- Run `python -m myvideoai.tests.test_mcp` — 15 checks covering handshake, tools, resources, preview, plugin scaffold + validation, traversal guard, error paths. All must pass.

## Enhancement output wrong size / frame count
- Output dimensions = input × `scale`, rounded to even. If source is 320×180 and scale 2 → 640×360.
- Frames = input frames × `interp`. 60 frames + interp 2 → ~119 (off-by-one is normal due to endpoint handling).
- If HDR is on, expect `yuv420p10le` + bt2020/pq tags. If stabilization is on, expect two-pass (first pass writes a `.trf` in the work dir).

## "Model not found" / shape mismatch on ONNX model
- See model-onboarding.md → dynamic axes. Fixed-size export is the cause 9/10 times.
- Check `model_file` basename actually exists in `myvideoai/models/weights/`.
- Wrong `input_name`/`output_name` → `InferenceSession` error; inspect the ONNX graph to get exact names.

## Plugin not taking effect
- Is `enabled: true` in its manifest, OR its id in `enhance_video.plugins`? Neither → it doesn't run.
- `validate_plugin` first — import errors (bad relative import, missing class) surface here.
- Frame hook changing shape → rejected silently by design; use a model hook.
- Multiple upscale providers enabled at once (`provides: "upscale"`) → pipeline raises a clear ValueError. Disable all but one.
- Builtin id collision: `builtin.esrgan`/`builtin.cugan` are auto-injected based on `backend`. Passing both as `plugins` triggers the duplicate-provider guard.

## Device confusion
- torch model using `ctx.device` instead of `ctx.torch_device` → wrong device when `--device cpu` + `--model-backend auto`. Symptom: unexpectedly slow or OOM. Use `ctx.torch_device` in torch plugins.
- ONNX on `cuda` actually runs on DirectML (by design, see model-onboarding.md). Not a bug.

## Dignified failure
If a genuine blocker persists after the checklist, report: what you verified (with tool output), what you ruled out, the narrowed-down cause, and the concrete next step. Never claim success without a probe summary / VALID result / test output. Never weaken tests to make them pass.
