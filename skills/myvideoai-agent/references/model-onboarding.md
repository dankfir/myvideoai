# Onboarding a Drop-in Model

The model-plugin design goal: **add a new super-res model with zero core-code changes**. You do it through `register_model`, which creates a model plugin reusing the built-in generic ONNX adapter (`builtin.onnx_superres`).

## When to use which path

- **ONNX super-res model** → `register_model` (uses the data-driven generic adapter).
- **Torch / new architecture (RRDBNet etc.)** → `write_plugin(kind="model")` and load the model in `setup()` — see plugin-authoring.md. The `examples.torch_sr` plugin is a working reference: both ONNX and torch models share the same ModelPlugin hook.
- **Native esrgan / cugan** → just `enhance_video` with `backend: "esrgan"` / `"cugan"`. They are themselves model plugins now (builtin.esrgan / builtin.cugan), fully equal to third-party ones.

## Register an ONNX model (the common case)

```
register_model {
  "id": "my_sr_model",
  "model_file": "my_model.onnx",
  "source_path": "F:\\path\\to\\my_model.onnx",   // copies into models/weights/
  "native_scale": 4,
  "tile": 256,
  "input_name": "input",
  "output_name": "output",
  "norm": "0to1",
  "layout": "nchw",
  "backends": ["cuda", "directml", "cpu"]
}
```

Or, if the weights are downloadable, give `model_url` instead of `source_path` — the adapter's `ensure_weights` fetches it with mirror fallback into `models/weights/` (atomic `.part` → rename).

Then verify:
```
preview_frame {
  "input": "myvideoai/samples/sample_lowres.mp4",
  "backend": "model",
  "plugins": ["my_sr_model"],
  "model_backend": "auto"
}
```
`validate_plugin({"id":"my_sr_model"})` → VALID. Full render: `enhance_video` with the same `backend`/`plugins`.

## Parameter semantics

| Param | Meaning |
| --- | --- |
| `native_scale` | The model's fixed upscale factor (e.g. 4 for x4). Output is resized to your requested `scale`. |
| `tile` | Internal tile size for inference; 0 = no tiling. Lower if you OOM, raise for speed. |
| `input_name` / `output_name` | ONNX graph I/O names — inspect with `onnx.checker` or netron if unsure. |
| `norm` | `0to1` (float [0,1], default) or `0to255`. Adapter handles BGR→RGB NCHW. |
| `layout` | `nchw` (default) or `nhwc`. |

## The #1 trap: dynamic axes

An ONNX model exported with a **fixed** input size (e.g. dummy 32×32) will reject real frames:
`InvalidArgument: Got:180 Expected:32`. Export with dynamic H/W axes:
```python
torch.onnx.export(model, dummy, "my_model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {2: "h", 3: "w"},
                                "output": {2: "h", 3: "w"}})
```
If you can't re-export, set `tile` to the model's fixed size and accept tiled-only operation — but prefer re-export.

## Backend honesty

- `cuda` choice with an ONNX model: the vendored `onnxruntime-directml` build has no CUDAExecutionProvider, so the adapter **honestly maps** `cuda` to `DmlExecutionProvider` (no fake CUDA claim). If you truly need ONNX CUDAExecutionProvider, that requires `onnxruntime-gpu`, which is deliberately NOT installed (DLL conflict with DirectML).
- `directml` gives cross-GPU (AMD/Intel/NVIDIA) — this is how the app covers the "cross-GPU" gap vs competitors.
- `cpu` always works, slowest.

## Dogfooding evidence

Both built-in super-res models were migrated to the model-plugin path. A golden regression (`--backend cugan --scale 2 --face --interp 2 --stabilize --hdr --crf 20`) produced md5 `f92d24fb4d8bf827b2688a395e69b814` before and after migration — byte-identical. The protocol is general: ONNX and torch models both run through the same `ModelPlugin` hook.
