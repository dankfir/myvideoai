# myvideoai — 开源自主 AI 画质修复工具

基于**开源模型**自研的视频画质增强软件：超分辨率放大 + 帧率补帧 + 人脸修复 + 重编码。
完全自主可控，无商业授权负担。

> 设计原则：只用 BSD/MIT/Apache 许可的开源模型
> （Real-ESRGAN、Real-CUGAN、RIFE、GFPGAN）自建管线，自己写调度/IO/GUI。

---

## 能力

| 功能 | 模型 | 说明 |
| --- | --- | --- |
| **超分辨率（写实）** | Real-ESRGAN x4plus / x4plus_anime / x2plus | 2x/4x 放大，重建高频细节、去压缩噪点 |
| **超分辨率（动漫）** | Real-CUGAN 4x (UpCunet4x) | 专为动画/插画优化，线条锐利、无伪影 |
| **人脸修复** | GFPGAN v1.4 | 检测+修复人脸，恢复五官/纹理，可与超分叠加 |
| **帧率补帧** | RIFE v4.6 (IFNet_HDv3) | 2x/4x 插帧，让 30fps → 60/120fps，动作更顺 |
| **批量处理** | CLI `--batch` / GUI 队列 | 多视频排队顺序处理，逐个 try/except 不中断 |
| **时间轴预览** | GUI 拖动时间轴 | 任意时间点抽帧→增强→左右对比，先看效果再出片 |
| **SDR→HDR10** | ffmpeg libplacebo 逆色调映射 | 输出 10-bit BT.2020 PQ 真 HDR10 元数据 |
| **视频防抖** | ffmpeg vidstab（两遍） | 检测相机抖动→平滑变换，消除手持晃动 |
| **插件系统** | 自建 | 5 分钟写一个新效果，内置功能与第三方插件同权 |
| **模型可插拔** | ONNX / PyTorch drop-in | 任意超分模型零核心改动接入；三后端自动探测，DirectML 跨显卡；内置 esrgan/cugan 同为 model 插件、完全同权 |
| **MCP 接口** | 零依赖 stdio server | 全部功能暴露为 9 个 MCP tool + 3 个 resource，agent 可全程操作；配套 skill 文件指导操作 |
| **重编码** | ffmpeg (libx264 / *_nvenc) | 保留音轨，可调 CRF/预设/像素格式 |
| **GPU 加速** | CUDA (RTX 4060 实测) | FP16 半精度 + 自动 tile 分块控显存 |

## 快速开始

### 命令行（CLI）

```bash
# Windows 一键脚本
myvideoai\myvideoai.bat -i 输入.mp4 -o 输出.mp4 --scale 2 --interp 2

# 或（已 source run_env.sh / 设好 PYTHONPATH）
python -m myvideoai.cli -i 输入.mp4 -o 输出.mp4 --scale 2 --interp 2

# 动漫超分 + 人脸修复
python -m myvideoai.cli -i 输入.mp4 --backend cugan --face --scale 2

# 批量：多个视频排队（输出自动加 _enhanced）
python -m myvideoai.cli --batch -i a.mp4 b.mp4 c.mp4 --scale 2 --interp 2

# 超分 + 防抖 + SDR→HDR10（内置功能即插件，一条命令全开）
python -m myvideoai.cli -i in.mp4 --backend cugan --scale 2 --stabilize --hdr

# 插件：列出 / 启用第三方 / 覆盖插件参数
python -m myvideoai.cli --list-plugins
python -m myvideoai.cli -i in.mp4 --plugins examples.unsharp --plugin-param examples.unsharp.amount=0.8

# 接入 drop-in ONNX 模型（零核心改动）
python -m myvideoai.cli -i in.mp4 --backend model --scale 2 \
  --plugin-param builtin.onnx_superres.model_file=my_model.onnx --model-backend auto
```

### MCP（agent 操作入口）

把软件的全部能力暴露为 MCP tool，任何支持 MCP 的 agent（如 Claude Desktop / Cline 等）都能直接操作：

```bash
# 启动 stdio MCP server（零新增依赖，无需 pip）
python -m myvideoai.mcp_server
# 9 个 tool：get_info / probe_video / list_backends / list_plugins /
#           enhance_video / preview_frame / write_plugin / register_model / validate_plugin
# 3 个 resource：myvideoai://docs/{plugin-spec,model-plugin-design,readme}
```

客户端配置示例（cwd 必须是仓库根目录，即包含 `myvideoai/` 的父目录）：

```json
{
  "mcpServers": {
    "myvideoai": {
      "command": "python",
      "args": ["-m", "myvideoai.mcp_server"],
      "cwd": "<repo-root>",
      "env": {
        "PYTHONPATH": "<repo-root>\\.vendored\\torch_pkgs;<repo-root>\\.vendored\\py_pkgs",
        "KMP_DUPLICATE_LIB_OK": "TRUE"
      }
    }
  }
}
```

配套 **`skills/myvideoai-agent/SKILL.md`** 是专门教 agent 操作本软件的 skill：视频增强、时间轴预览、插件编写（frame/model/video 三类）、ONNX 模型接入、后端选择、故障排查全覆盖。协议测试：`python -m myvideoai.tests.test_mcp`（15 项全绿）。

**常用参数**

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `-i / --input` | 输入视频（可多个，配合 `--batch`） | 必填 |
| `-o / --output` | 输出视频（批量时可省略，自动生成） | 自动 |
| `--backend` | 超分后端：esrgan / cugan / **model**（drop-in 插件） | esrgan |
| `--model-backend` | 模型插件后端：auto / cuda / directml / cpu | auto |
| `--scale` | 放大倍数 | 2 |
| `--model` | ESRGAN 模型：x4plus / x4plus_anime / x2plus | x4plus |
| `--cugan-tile` | CUGAN tile 模式 0~4（越大越省显存） | 0 |
| `--face` | 开启 GFPGAN 人脸修复 | 关 |
| `--face-weight` | 人脸修复融合权重 0~1 | 0.5 |
| `--interp` | 补帧倍数（1=不补，2/4=翻倍/四倍） | 1 |
| `--tile` | ESRGAN 分块大小（0=自动按显存） | 0 |
| `--codec` | 编码器：libx264 / h264_nvenc / hevc_nvenc | libx264 |
| `--crf` | 画质（越小越好，18~23 常用） | 18 |
| `--preset` | x264 预设：fast/medium/slow | medium |
| `--no-half` | 关闭 FP16（调试用） | 关 |
| `--batch` | 批量模式（`-i` 可跟多个文件） | 关 |
| `--hdr` | SDR→HDR10（libplacebo 逆色调映射，10bit BT.2020 PQ） | 关 |
| `--stabilize` | 视频防抖（vidstab 两遍） | 关 |
| `--plugins` | 启用第三方插件 id（可多个，空格分隔） | 无 |
| `--plugin-param PLUGIN.K=V` | 覆盖插件参数（值自动转 int/float/bool） | 无 |
| `--list-plugins` | 列出所有已发现插件（含 3 类 hook） | — |

完整参数：`python -m myvideoai.cli --help`

### 图形界面（GUI）

```bash
myvideoai\myvideoai-gui.bat
```

- **批量队列**：「添加视频…」可多选，队列逐条处理，进度条 + 日志实时更新
- **时间轴预览**：选视频后拖动时间轴 → 点「预览此帧」→ 左原图/右增强图对比
- 后端（写实/动漫）、放大倍数、补帧、GFPGAN 人脸开关、编码器全部可配
- 插件勾选：HDR/防抖复选框 + 自动发现的第三方插件多选（列表选中即启用）

## 插件系统

把超分、补帧、人脸修复、HDR、防抖统一抽象成**插件**——内置功能和第三方插件同权，
用一套 5 分钟就能上手的规范，让生态可自由扩展。完整规范见
[`docs/PLUGIN_SPEC.md`](docs/PLUGIN_SPEC.md)。

- **零继承接入**：插件只需实现 `setup(ctx)` + `process(...)` 两个方法（鸭子类型 Protocol），
  不用继承任何基类
- **三类 hook**：`frame`（逐帧、不改尺寸，如锐化）、`video`（整段视频，如 HDR/防抖）、
  **`model`（改变尺寸的超分/修复模型）**
- **声明式清单**：每个插件一个 `plugin.json`（id/kind/order/after/before/params/requires），
  注册表自动做依赖校验 + 拓扑排序
- **参数外置**：manifest 默认值可被 `--plugin-param PLUGIN.K=V` 或 GUI 覆盖，无需改代码
- **两处发现目录**：`plugins/`（内置示例）和 `models/plugins/`（用户自建，升级不覆盖）

### 模型可插拔（Model-as-Plugin）

把"模型本身"也做成插件——**别人出好模型（ONNX / 新 PyTorch 架构），5 分钟接进来**，
用接入层生态绕开自研模型数量的劣势。一个 ONNX 超分模型丢进目录 + 改几行 manifest 参数，
**零核心代码改动**即可接入；三后端自动探测（CUDA / DirectML / CPU），DirectML 让 AMD/Intel
显卡也能跑，顺带补上"CUDA-only"短板。

```bash
# 用 drop-in ONNX 模型超分（自动选最优后端）
python -m myvideoai.cli -i in.mp4 --backend model --scale 2 \
  --plugin-param builtin.onnx_superres.model_file=你的模型.onnx \
  --plugin-param builtin.onnx_superres.native_scale=4

# 指定后端：auto / cuda / directml / cpu
python -m myvideoai.cli -i in.mp4 --backend model --model-backend directml
```

内置的 esrgan/cugan 也是标准 model 插件——`--backend esrgan|cugan|model` 三条路统一走
`registry.model_chain()`，与第三方 drop-in 模型完全同权，无特权路径。实测同参数重跑与
改造前 golden 的 **md5 完全相同**（逐帧二进制一致）。

新增模型架构：复制 `plugins/builtin/onnx_superres/` 或 `plugins/examples/torch_sr/`，
改 `plugin.json` 的 `entry`/`model_url`/`native_scale` 即可。协议对 ONNX 和 torch 一视同仁
（已实测两种异构模型走同一 hook）。完整设计 + 已知边界见
[`docs/MODEL_PLUGIN_DESIGN.md`](docs/MODEL_PLUGIN_DESIGN.md)。

插件用声明式清单（`plugin.json`）描述身份与能力，通过一个轻量 Context 注入运行环境，
三种 hook（frame/video/model）按顺序接力。详细规范见 PLUGIN_SPEC.md。

## 实测效果（RTX 4060 Laptop 8GB）

| 场景 | 速度 | 说明 |
| --- | --- | --- |
| Real-ESRGAN 320×180→640×360 | ~3.1 it/s | 写实超分 2x |
| Real-CUGAN 320×180→640×360 | ~8.9 it/s | 动漫超分 2x，更快 |
| ESRGAN + RIFE 补帧 2x | ~3.9 it/s | 60→119 帧 |
| GFPGAN 人脸修复 | 512×512 单帧 <1s | 五官/纹理明显恢复 |
| 时间轴预览（抽帧+增强） | 单帧 <1s | 两后端均验证尺寸正确 |
| CUGAN 超分+防抖+HDR10 | ~25 it/s | video 插件链 stabilize→hdr，输出真 HDR10 元数据 |
| 内置模型插件化迁移 | — | esrgan/cugan/model 三路同链；重跑与改造前 golden md5 完全相同 |
| frame 插件（unsharp） | ~25 it/s | 逐帧钩子生效，参数可命令行覆盖 |
| **drop-in ONNX 模型** | ~89 it/s | 三后端 auto(DML)/directml/cpu 均 320×180→640×360，零核心改动 |
| **torch 模型同协议接入** | 64×64→128×128 | examples.torch_sr 验证 ONNX/torch 异构模型共享一套 hook |
| **MCP 协议测试** | 15/15 PASS | 握手/tools+resources/预览/插件脚手架+校验/路径穿越防护/错误降级全绿 |
| **MCP 端到端增强** | 640×360→1280×720 | 经 stdio MCP `enhance_video` 实跑，协议通道干净、probe 证据齐全 |

- 画质锐度（Laplacian 方差）对比双三次插值：**14~93 倍**提升（取决于素材高频内容）
- 文字/网格线在双三次插值下糊成一团，Real-ESRGAN 可清晰重建
- GFPGAN 对比图：`output/face_astronaut_cmp.png`、`output/face_chelsea_cmp.png`
- 超分对比图：`output/compare_3way.png`、`output/real_compare.png`
- 端到端样片：`output/test_cugan_2x.mp4`、`output/test_2x_2fps.mp4`、`output/final_audio_2x.mp4`（含音轨）
- MCP 增强样片：`output/mcp_enhance_e2e.mp4`（经 stdio MCP 实跑）、`output/skill_smoke.mp4`（skill 冒烟）

## 项目结构

```
myvideoai/
├── core/
│   ├── config.py          # EnhanceConfig、权重 URL、路径、后端注册
│   ├── video_io.py        # ffprobe 探测 / 流式读帧 / 流式编码（保留音轨）
│   ├── upscaler.py        # Real-ESRGAN 封装 + 权重多镜像回退下载
│   ├── cugan_arch.py      # Real-CUGAN UpCunet2x/3x/4x + RealWaifuUpScaler 架构
│   ├── cugan_upscaler.py  # CUGAN 超分封装
│   ├── face_enhancer.py   # GFPGAN 人脸修复（绝对权重路径，不改进程 CWD）
│   ├── interpolator.py    # RIFE v4.6 IFNet 架构 + 推理
│   ├── preview.py         # 时间轴抽帧 + 单帧增强（无头可测）
│   ├── model_backend.py   # 三后端探测 CUDA/DirectML/CPU（auto 择优 + 优雅降级）
│   ├── device.py          # auto_tile() 显存探测（pipeline 与 model 插件共享）
│   ├── weights.py         # 通用权重镜像回退下载器（drop-in 模型用）
│   └── pipeline.py        # 端到端调度：探测→model_chain 超分→人脸→补帧→编码→插件链
├── plugins/               # 自建插件系统
│   ├── spec.py            # PluginManifest / PluginContext / 三种 hook 协议
│   ├── registry.py        # discover + 依赖排序 + frame/model/video 三链接力
│   ├── builtin/hdr/       # 内置：SDR→HDR10（video hook）
│   ├── builtin/stabilize/ # 内置：视频防抖（video hook）
│   ├── builtin/onnx_superres/ # 内置：通用 ONNX 超分适配器（model hook，drop-in）
│   ├── builtin/esrgan/    # 内置：Real-ESRGAN 包装成 model hook
│   ├── builtin/cugan/     # 内置：Real-CUGAN 包装成 model hook
│   ├── examples/torch_sr/ # 示例：torch 模型同协议接入（model hook）
│   └── examples/unsharp/  # 示例：OpenCV 锐化（frame hook，5 分钟写一个）
├── mcp_server.py          # 零依赖 stdio MCP server（9 tools + 3 resources）
├── skills/myvideoai-agent/ # 教 agent 操作本软件的 skill（SKILL.md + 4 references）
├── docs/PLUGIN_SPEC.md    # 插件规范全文（PR/FAQ + 契约 + 概念对照）
├── docs/MODEL_PLUGIN_DESIGN.md # 模型可插拔设计 + 实现落地 + 已知边界
├── gui/app.py             # PySide6 GUI（批量队列 + 时间轴预览 + 插件勾选）
├── cli.py                 # CLI 入口（批量 + 插件）
├── tests/make_sample.py   # 生成测试视频
├── tests/export_test_onnx.py # 造测试用 2x ONNX 模型（fixture）
├── tests/test_plugins.py  # 插件系统单测（含三后端 drop-in）
├── tests/test_mcp.py      # MCP 协议集成测试（spawn server + JSON-RPC 15 项）
├── download_assets.py     # 权重下载器（统一走 core/weights.py 的 ensure_weights）
├── myvideoai.bat          # CLI 启动脚本（Windows）
├── myvideoai-gui.bat      # GUI 启动脚本（Windows）
└── run_env.sh             # 环境变量 helper
```

## 依赖环境

标准安装直接 `pip install -r myvideoai/requirements.txt`（torch/torchvision 需按你的
CUDA 版本单独安装，见该文件注释）。若在离线/受限环境中运行，可把依赖**离线 vendored** 到
`<repo-root>/.vendored/`：

- `.vendored/torch_pkgs/` — torch 2.8.0+cu128 + torchvision 0.23.0+cu128
- `.vendored/py_pkgs/` — numpy/cv2/scipy/scikit-image/basicsr/realesrgan/facexlib/gfpgan 等
- `.vendored/py_pkgs/sitecustomize.py` — 自动兼容补丁（新 torchvision 移除的
  `functional_tensor`、torch.cuda.amp 的 custom_fwd/custom_bwd）

`run_env.sh` / `*.bat` 已把这两条路径设进 `PYTHONPATH`，并设 `KMP_DUPLICATE_LIB_OK=TRUE`
规避 OMP 重复运行库告警。手动运行时：

```bash
export PYTHONPATH="<repo-root>/.vendored/torch_pkgs:<repo-root>/.vendored/py_pkgs"  # Windows 用 ";" 分隔
export KMP_DUPLICATE_LIB_OK=TRUE
```

注意：包名导入须用完整路径 `python -m myvideoai.cli`（从 `<repo-root>` 运行），
不要从项目根直接 `import gui.app`（相对导入会越界）。

## 模型权重

首次运行自动下载到 `myvideoai/models/weights/`（走 hf-mirror.com，国内可访问）：

| 文件 | 大小 | 用途 |
| --- | --- | --- |
| `x4plus.pth` | 64MB | Real-ESRGAN 通用超分 |
| `cugan_4x.pth` | 5.4MB | Real-CUGAN 动漫超分 |
| `rife46.pth` | 20MB | RIFE v4.6 补帧（flownet.pkl） |
| `GFPGANv1.4.pth` + detection + parsing | 348+104+81MB | GFPGAN 人脸修复三件套 |

> GFPGAN 三件套与其它权重同放在 `myvideoai/models/weights/`，代码用绝对路径加载，不会修改进程工作目录。

## 许可

本项目代码以 **Apache-2.0** 发布（见 `LICENSE`）。所用开源模型与依赖的归属、
许可证与再分发边界见 `THIRD_PARTY_NOTICES.md`（含各许可证全文）：Real-ESRGAN (BSD-3)、
Real-CUGAN (MIT)、RIFE (MIT)、GFPGAN (Apache-2.0)、facexlib (MIT)、
ONNX Runtime (MIT)、OpenCV (MIT/Apache-2.0)、PySide6/Qt6 (LGPL-3.0)。

仓库**不含任何模型权重**，首次运行时按 `myvideoai/models/weights/` 自动下载。
样例媒体由 `tests/make_sample.py` 用 FFmpeg lavfi 合成，不含第三方素材。
