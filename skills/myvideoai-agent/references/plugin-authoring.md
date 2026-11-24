# Authoring Plugins

A plugin is a directory with `plugin.json` (data) + `plugin.py` (code). Three kinds, one contract each. The app discovers them from `myvideoai/plugins/` (builtin) and `myvideoai/models/plugins/` (user). `write_plugin` scaffolds into the user dir; you edit the TODOs, then `validate_plugin`.

## Workflow

1. `write_plugin` with `id`, `kind`, and optional `description`/`params`/`provides`/`backends`.
   - `id` must match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$` — no path separators (the tool rejects `../../evil`).
2. Read the returned `dir`, open `plugin.py`, replace the TODO body with real logic.
3. `validate_plugin({"id": "..."})` → must return `VALID ... manifest+entry: OK (importable)`.
4. Use it: pass the id to `enhance_video.plugins`, or set `enabled: true` in the manifest.

## The three hooks (from `plugins/spec.py`)

### frame hook — transform one BGR frame, SAME shape
```
class Plugin:
    def setup(self, ctx): ...          # ctx.params is YOUR merged params
    def process(self, frame): ...      # np.ndarray BGR uint8 -> np.ndarray BGR uint8
    def teardown(self): ...
```
- `process` is required. Output shape/dtype **must** equal input — the registry enforces this.
- Good for: sharpen, denoise, color grade, watermark, overlays.
- Runs once per frame, setup once per job. Use `setup` to load models/state, not `process`.
- Keep it stateless across frames unless you intentionally track state (e.g. temporal filter).

### model hook — super-res a frame, MAY change size
```
class Plugin:
    def setup(self, ctx): ...
    def process(self, frame, outscale: float): ...   # -> larger BGR frame
    def teardown(self): ...
```
- This is the hook for super-resolution / generative models. It may upscale (change shape) — `outscale` is the target multiplier.
- `provides: "upscale"` tells the pipeline "I supply the upscale step" (competes with esrgan/cugan; only one upscale provider may be enabled at a time).
- **Device split (critical):** torch models use `ctx.torch_device` (follows `--device`); ONNX/DirectML use `ctx.ort_provider` + `ctx.device`. Using the wrong one causes silent CPU fallback or provider warnings.
- Weights live in `ctx.weights_dir` (`myvideoai/models/weights/`). Fetch with `from myvideoai.core.weights import ensure_weights`.

### video hook — whole-file ffmpeg transforms
```
class Plugin:
    def setup(self, ctx): ...
    def process(self, in_path: str, out_path: str, ctx) -> str: ...
    def teardown(self): ...
```
- For operations on the encoded file (HDR tone-map, stabilization, transcode) — the two builtin video plugins (`hdr`, `stabilize`) are the reference implementations.
- Chain: multiple video plugins relay via intermediate files in `ctx.work_dir`.

## PluginContext fields you may use

| Field | Meaning |
| --- | --- |
| `ctx.params` | merged params for THIS plugin (manifest defaults + CLI/GUI overrides + `plugin_params`) |
| `ctx.work_dir` | scratch dir for intermediate files |
| `ctx.device` | backend-resolved device (ONNX/DirectML semantics) |
| `ctx.torch_device` | torch device — use for torch models |
| `ctx.ort_provider` | resolved onnxruntime provider, e.g. `DmlExecutionProvider` |
| `ctx.ffmpeg` / `ctx.ffprobe` | binary paths |
| `ctx.weights_dir` | shared weights directory |
| `ctx.log(msg)` | emit a log line to the user |

Plugins never touch pipeline internals — only the context. Fail loud: a bad manifest names the plugin id.

## Params & ordering

- `params` in the manifest are defaults; override via `enhance_video.plugin_params = {"<plugin-id>": {"key": value}}`.
- `order` (lower first), `after`, `before` control execution order within a kind.
- `requires` lists plugin ids that must be present (validation fails loudly otherwise).

## Known traps

- Path inside an ffmpeg filter value is fragile on Windows: a `F:\` drive colon is parsed as a filter-option separator, backslashes as escapes. The builtin video plugins solve this by using a **bare filename** for the `.trf` file, running with `cwd=work_dir`, and passing absolute paths only as `-i` inputs. Copy that pattern.
- Relative imports like `from ...core.weights` fail because the registry loads plugins via `spec_from_file_location` (no package context). Use absolute imports: `from myvideoai.core.weights import ensure_weights`.
- A frame plugin that changes shape will be rejected. If you need size change, use a **model** plugin.
