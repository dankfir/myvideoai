# 设计文档：模型可插拔架构（Model-as-Plugin）

日期：2026-11-18
状态：✅ 已实现并验证（approach + 后端均经用户拍板；三后端 drop-in 端到端实测通过；内置 esrgan/cugan 已迁入 model 插件路径，md5 逐帧一致）
方法论：⬛ Musk（The Algorithm：质疑 → 删除 → 简化 → 加速 → 自动化）

## 1. Intent / Why

竞品对比的结论：与成熟商业产品的结构性差距在"自研模型层 + 生态层"。自研模型层短期补不了，
但**接入层**可以做成非对称优势——别人出好模型（ONNX / 新架构），我们 5 分钟接进来，
用"模型可插拔生态"绕开"自研模型数量"的劣势。

这是把上一轮的插件系统从"效果插件（HDR/防抖/锐化）"升级为"**模型本身也是插件**"。

## 2. 现状与硬约束

探查代码发现两个必须先解决的约束：

1. **frame hook 挂不了模型**：`plugins/spec.py` 的 FramePlugin 契约 + `registry.py` 的
   `frame_chain.apply()` 强制输出 shape == 输入 shape。而超分模型恰恰**改变尺寸**，硬塞会
   被 `PluginError` 拒绝。
2. **upscaler 是硬编码 if/else**：`core/pipeline.py` 第 119-128 行按 `upscale_backend` 在
   cugan / esrgan 间二选一，加任何新模型都得改核心代码——这正是要打破的"模型不自由"。

3. **onnxruntime 未安装**：环境只有 torch + cv2。要走 ONNX 后端必须新装依赖。

## 3. Goals / Non-goals / Anti-goals

**Goals**
- 新增 `kind="model"` 插件类型，输出尺寸可不同于输入（支持任意放大倍数）
- 三后端自动探测：CUDA(torch) / DirectML(onnxruntime) / CPU，按可用性择优，可手动覆盖
- drop-in 新模型：一个 ONNX 超分模型丢进目录 + 一个 manifest 即可接入，零核心代码改动
- 现有 esrgan/cugan 包装成内置 model 插件（dogfooding：内置与第三方同权）
- 一石二鸟：DirectML 后端顺带补上"CUDA-only、跨不了显卡"的竞品短板

**Non-goals**
- 不自研任何新模型（本轮只做接入层）
- 不做模型训练 / 微调
- 不接任何闭源商业模型（授权红线，见项目原则）

**Anti-goals（做了就算失败）**
- 为接一个模型要改 pipeline / config 核心代码 → 说明抽象失败
- 后端探测在开发机上能跑、换台 AMD 机就崩 → 探测必须优雅降级
- 装 onnxruntime 破坏已验证的 torch 环境 → 必须隔离在 vendored

## 4. Approaches（用户已选 C）

| 方案 | 做法 | 取舍 | 结论 |
| --- | --- | --- | --- |
| A 纯 PyTorch | 新模型只写 PyTorch 加载代码 | 零依赖，但绑死 N 卡、新模型必须 PyTorch | 否 |
| B ONNX+CUDA | onnxruntime-gpu | N 卡快，但版本匹配繁琐、仍跨不了显卡 | 否 |
| **C 全都要自动探测** | CUDA/DirectML/CPU 按可用性择优 | 最灵活、一石二鸟补跨显卡；实现/测试成本最高 | **用户选定** |

## 5. 架构设计

### 5.1 第三种 hook：`kind="model"`

```python
class ModelPlugin(Protocol):
    """model-hook: 改变尺寸的图像到图像变换（超分/修复）。
    与 frame hook 的区别：输出尺寸可任意，且绑定一个推理后端。
    """
    backends: tuple[str, ...]        # 支持哪些后端 ("cuda","directml","cpu")
    def setup(self, ctx: PluginContext) -> None: ...   # 按 ctx.backend 加载模型
    def process(self, frame: np.ndarray, outscale: float) -> np.ndarray: ...
    def teardown(self) -> None: ...
```

- manifest 增加可选字段 `backends`，缺省 inferred
- `PluginContext` 增加 `backend: str` 字段（探测/指定的实际后端）
- registry 增加 `_model_order`，`model_chain(ctx_factory) -> (apply, close)`；
  `apply(frame, outscale)` 与 frame chain 分开调度

### 5.2 后端探测层 `core/model_backend.py`

```python
def detect() -> list[str]:
    """按可用性从优到劣返回 ['cuda'|'directml'|'cpu', ...]，全部失败也返回 ['cpu']。"""
    # 1. torch.cuda.is_available() -> cuda
    # 2. import onnxruntime; 'DmlExecutionProvider' in get_available_providers() -> directml
    # 3. 永远兜底 cpu
def pick(preferred="auto") -> str:
    """preferred='auto' 取 detect()[0]；否则校验后返回，不可用则降级 + 日志。"""
```

关键：每个探测都用 try/except 包住，**任何后端 import 失败都不影响其他后端**。
torch 与 onnxruntime 可共存（各自加载 DLL，互不干扰；已有的 KMP_DUPLICATE_LIB_OK 兜底）。

### 5.3 通用 ONNX 超分适配器（drop-in 的核心）

`plugins/builtin/onnx_superres/` —— 一个**数据驱动**的 model 插件：
- `plugin.json` 用 params 声明：`model_file`、`input_name`、`output_name`、
  `native_scale`、`norm`（0-1 / 0-255）、`layout`（nchw/nhwc）
- `onnx_superres.py` 是**通用推理壳**：按 manifest params 建 session、预处理、分块、后处理
- 接新模型 = 复制目录 + 改 plugin.json 的 `model_file` / `native_scale`，**零代码**

后端选择：`ort.InferenceSession(path, providers=[...])`，providers 由 `ctx.backend` 决定：
- `cuda` → `CUDAExecutionProvider`（若 onnxruntime-gpu）
- `directml` → `DmlExecutionProvider`（onnxruntime-directml，通吃各显卡）
- `cpu` → `CPUExecutionProvider`

### 5.4 权重管理 `core/weights.py`

抽出通用 `ensure_weights(urls, filename) -> Path`：
- URL 列表逐镜像回退（hf-mirror 优先，复用现有经验）
- 断点 + 大小校验
- 落 `models/weights/`
- manifest 可声明 `model_url`，插件首次 setup 时自动下载

### 5.5 内置模型 dogfooding

现有 esrgan / cugan 包装成 `kind="model"` 的内置插件（保留原 Python 类作实现）：
- 证明内置与第三方走**同一条 registry 路径**
- pipeline 里 --backend esrgan/cugan 解析为"启用对应 model 插件"

### 5.6 调度顺序（pipeline.run 改造后）

```
probe → 几何 → 后端探测 → 建 registry
  → model 插件 setup(ctx.backend)  [一次]
  → 逐帧: upscale(首选 model 插件，回退内置) → face → frame 插件 → interp → encode
  → video 插件链 → 输出
```

向后兼容：`do_upscale=True` 且未指定 model 插件时，仍走原 esrgan/cugan 路径（现包装成插件）。

### 5.7 CLI / GUI

- `--model-plugin ID`：启用某个 model 插件作为超分器（可多次，形成链）
- `--backend auto|cuda|directml|cpu`：后端偏好（默认 auto）
- GUI：后端下拉框（含 `model (drop-in)`）+ model 插件多选（复用已有插件列表）+ 选中 model 后端时显示 ONNX 文件选择器/原生倍数/推理 tile

## 6. 依赖与约束

- onnxruntime 必须装进 `<repo-root>/.vendored/py_pkgs/`（pip 不可用，
  用自写 `install_wheel.py` urllib+zipfile 离线装）
- DirectML 版 wheel 是否支持 cp313 需先验证；若不支持则降级为"torch+CPU ONNX"两腿，
  并在文档披露（不伪造三腿）
- 每次 ops_bash 必设 PYTHONPATH 双路径 + KMP_DUPLICATE_LIB_OK=TRUE

## 7. 验证计划（evidence before assertion）

1. `model_backend.detect()` 在开发机返回含 cuda；手动指定不存在后端能优雅降级
2. onnxruntime 安装后 `get_available_providers()` 含 DmlExecutionProvider（若 wheel 可得）
3. drop-in 测试：造一个最小 ONNX 超分模型（2x 最近邻/反卷积），只改 plugin.json 接入，
   pipeline 输出尺寸正确 → 证明零核心改动
4. 内置 esrgan/cugan 走 model 插件路径，端到端输出与改造前一致
5. 全量回归：单测 + import + 原三合一场景不回退

## 8. What's left to the builder（实现顺序见 todo）

依赖可得性探明 → 后端层 → 协议/registry → 权重工具 → ONNX 适配器 → 内置包装 →
pipeline/CLI/GUI 接入 → 端到端 + 回归。

## 9. 实现落地记录（2026-11-18，对照验证计划逐条闭环）

**结论：三后端全部落地并实测，drop-in 零核心改动验证通过。** 唯一与设计文档初稿不同的
是 CLI 参数命名（见下），以实际代码为准。

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| 后端探测 detect() | ✅ | `['cuda','directml','cpu']`；pick('auto')=cuda；pick(无效)=cpu 带降级 note |
| onnxruntime DirectML | ✅ | 1.24.4，`providers=['DmlExecutionProvider','CPUExecutionProvider']`（cp313 wheel 可得） |
| drop-in 零核心改动 | ✅ | 造 `test_sr_2x.onnx`(1.7KB)，只改 plugin.json params，320×180→640×360 端到端通，89 it/s |
| 三后端端到端 | ✅ | auto(DML)/directml/cpu 均输出 640×360/60帧，日志证明各自 provider 生效 |
| 内置 esrgan/cugan 迁入 model 路径 | ✅ | 三条 backend 全走 model_chain；同参数重跑与改造前 golden **md5 完全相同** `f92d24fb…` |
| 全量回归 | ✅ | test_plugins 5/5、import 16/16、plugin discover 7 个；双 upscale provider 显式拒绝 |

**实际 CLI（与设计初稿命名不同，以代码为准）**
- `--backend esrgan|cugan|model`：`model` 启用提供 upscale 的 model 插件
- `--model-backend auto|cuda|directml|cpu`：模型插件后端偏好（默认 auto）
- `--plugin-param builtin.onnx_superres.model_file=X.onnx` 指定 drop-in 模型文件
- GUI：后端下拉新增 "model (drop-in)" + 模型后端下拉 + ONNX 文件选择器（选中即 stage 进 weights 目录，自动填 model_file）+ 原生倍数/推理 tile 输入；时间轴预览同步走 model chain

**新增文件**
- `core/model_backend.py`：BackendChoice + detect()/pick()/describe()
- `core/weights.py`：ensure_weights() 镜像回退下载器
- `plugins/builtin/onnx_superres/`：通用 ONNX 超分适配器（数据驱动，零代码接新模型）
- `plugins/builtin/esrgan/`、`plugins/builtin/cugan/`：内置超分包装成 model 插件（与第三方同权）
- `core/device.py`：`auto_tile()` 抽离，pipeline 与插件共享，避免循环导入
- `tests/export_test_onnx.py`：造测试用 2x ONNX 模型（fixture）
- `plugins/spec.py` 加 ModelPlugin/kind=model；`registry.py` 加 model_chain()

**已知边界（诚实披露）**
1. **ONNX 输入尺寸**：多数 ONNX 超分模型导出时固化输入 H/W。接新模型前需确认模型用
   `dynamic_axes` 导出，否则真实帧尺寸会被拒（本次测试模型首版就踩了这个坑，加 dynamic_axes 重导出后解决）。
2. **`cuda` 后端对 ONNX 的语义**：本机装的是 onnxruntime-**directml** 版（无 CUDAExecutionProvider），
   故 `detect()` 把 cuda choice 的 `ort_provider` 如实设为 DmlExecutionProvider——选 `cuda` 跑
   ONNX 模型时实际走 DirectML，不造假。要真·ONNX-CUDA 需另装 onnxruntime-gpu（与 directml 版冲突，故意不装）。
3. **内置 esrgan/cugan 已迁入 model 插件路径（原"保留原路径"边界已消除）**：
   `plugins/builtin/esrgan` 与 `plugins/builtin/cugan` 是标准 ModelPlugin，pipeline 中按
   `upscale_backend` 硬编码构造 upscaler 的分支已删除；`--backend esrgan|cugan|model` 三条路
   统一走 `registry.model_chain()`。逐帧一致性用同参数重跑对比改造前 golden，**md5 完全相同**
   (`f92d24fb4d8bf827b2688a395e69b814`)。torch 插件用 `ctx.torch_device`（跟随 `cfg.device`）
   而非 ONNX 的 `ctx.device`，避免 `--device cpu` 时后端语义串味；两个 upscale provider 同时
   启用会被 `_build_registry` 显式拒绝。`examples/torch_sr` 仍是"包装任意新 torch 模型"的范本。
4. **GUI 的 model 后端模型文件选择已补齐**：选 `model (drop-in)` 时显示 ONNX 文件选择器
   （选中的 `.onnx` 自动 stage 进 `models/weights/` 以普通文件名引用）+ 原生倍数 + 推理 tile，
   `_make_cfg` 写入 `plugin_params['builtin.onnx_superres']`。时间轴预览同步改为走
   `registry.model_chain()`（原为 esrgan/cugan 硬编码 fallback），所见即所得。验证：headless
   GUI→cfg→端到端 drop-in 320×180→640×360 通；预览同配置 (180,320,3)→(360,640,3)；内置
   golden md5 不变。CLI `--plugin-param` 仍可用且优先级最高。
