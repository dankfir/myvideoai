# myvideoai Plugin Spec — PR/FAQ + 对接规范

> Working Backwards: 先从"插件作者/用户"倒推体验，再定契约。设计取两条主线之交集：声明式 manifest（含扩展点与生命周期）与可组合的滤镜链（用户可写）。

## PR/FAQ（虚构发布稿，先写再设计）

**myvideoai 0.4 — "插件化画质引擎"**
- 一句话：把 HDR、防抖、超分、人脸、补帧做成一个个可插拔的"画质滤镜"，第三方 5 分钟就能写一个新效果。
- 目标用户：① 想加自己模型/滤镜的研究者；② 想要一站式"效果面板"但不想付授权成本的用户。
- 关键差异：同类的封闭式商业软件模型不开放，video2x 模型开放但缺少扩展规范；myvideoai 用一套极简契约让**内置功能和第三方插件走同一条路**（dogfooding）。
- FAQ：
  - Q: 写插件要装什么？A: 什么都不装，一个文件夹 + 一个 JSON + 一个 .py。
  - Q: 会不会像浏览器插件那样有兼容性地狱？A: 契约只有 3 个方法 + 1 个 Context，无继承、无框架依赖。
  - Q: 内置 HDR 是不是特殊待遇？A: 不是，它就是 `plugins/builtin/hdr/`，和你写的完全一样。

## 1. 设计哲学（三板斧）

1. **声明式**：`plugin.json` 只描述"我是谁"，不含可执行逻辑。
2. **两种 hook**：`frame`（逐帧 numpy 变换）/ `video`（整片 ffmpeg 滤镜图）。覆盖 99% 画质效果。
3. **链式组合**：按 `order` + `after`/`before` 排序，前一个的输出喂给后一个。

## 2. 目录结构

```
myvideoai/plugins/                  # 内置插件（随软件发布）
  builtin/
    hdr/        plugin.json + hdr.py
    stabilize/  plugin.json + stabilize.py
  examples/
    unsharp/    plugin.json + unsharp.py      # 默认 enabled=false
myvideoai/models/plugins/           # 用户第三方插件（drop-in 目录）
  <your_plugin>/  plugin.json + <module>.py
```

## 3. Manifest（plugin.json）字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| id | str | ✅ | 全局唯一，如 `builtin.hdr` |
| name | str | ✅ | 显示名 |
| version | str | | 默认 `1.0.0` |
| kind | `frame`\|`video` | | 默认 `frame` |
| description | str | | 一句话说明 |
| author / license | str | | 归属 |
| entry | str | ✅ | `module.py::ClassName` |
| enabled | bool | | 默认 true；示例插件设 false |
| order | int | | 越小越先执行，默认 100 |
| after / before | list[str] | | 相对其他插件的排序约束 |
| params | dict | | 默认参数，CLI/GUI 可覆盖 |
| requires | list[str] | | 依赖的插件 id（缺失则报错） |

未知字段会**直接报错**（fail loud），不会静默忽略。

## 4. Hook 契约

两种 hook 都是"鸭子类型"——不继承任何基类，实现约定方法即可。

### FramePlugin（`kind: frame`）
逐帧处理，跑在超分/人脸之后、补帧之前。

```python
class MyPlugin:
    def setup(self, ctx): ...                    # 可选，链建立时调用 1 次
    def process(self, frame: np.ndarray) -> np.ndarray: ...  # 必填，BGR 帧进、BGR 帧出
    def teardown(self): ...                      # 可选
```
约束：输出 shape/dtype 必须与输入一致，否则 registry 报错。

### VideoPlugin（`kind: video`）
整片后处理，主编码完成后按顺序接力。适合需要两遍分析或 ffmpeg 滤镜图的效果（防抖、HDR、去水印、转码）。

```python
class MyPlugin:
    def setup(self, ctx): ...
    def process(self, in_path: str, out_path: str, ctx) -> str: ...  # 必填
    def teardown(self): ...
```
约束：必须写出有效视频到 `out_path` 并返回路径。临时文件放 `ctx.work_dir`。

### PluginContext（注入的能力面）
插件只能看到这些，接触不到 pipeline 内部：
`work_dir` / `device` / `half` / `ffmpeg` / `ffprobe` / `params`（合并后的参数）/ `emit(msg)`（日志回调）。

## 5. 执行顺序

```
probe → upscale(esrgan/cugan) → face(GFPGAN) → [frame plugins 链] → interpolate(RIFE)
      → encode(临时文件) → [video plugins 链, 如 stabilize→hdr] → 最终输出
```

video 插件链的后一个吃前一个的输出。`after`/`before`/`order` 决定顺序（本例 stabilize.order=100 < hdr.order=200）。

## 6. CLI / GUI 用法

```bash
python -m myvideoai.cli --list-plugins                          # 列出所有插件
python -m myvideoai.cli -i in.mp4 --stabilize --hdr             # 启用内置插件
python -m myvideoai.cli -i in.mp4 --plugins examples.unsharp \  # 启用第三方
       --plugin-param examples.unsharp.amount=0.8
python -m myvideoai.cli -i in.mp4 --plugin-param builtin.hdr.crf=22
```
`--stabilize`/`--hdr` 本质只是把 `builtin.stabilize`/`builtin.hdr` 加入启用列表——**内置功能与第三方完全同权**。

## 7. 5 分钟写一个插件

1. 在 `myvideoai/models/plugins/` 下建目录 `my_effect/`
2. 写 `plugin.json`（必填 id/name/entry，kind 选 frame 或 video）
3. 写 `my_effect.py`，实现 `process`
4. `python -m myvideoai.cli --list-plugins` 看到它 → `--plugins my_effect` 启用

## 8. 已知边界（诚实披露）

- 无沙箱隔离：插件是被 `exec_module` 的本地 Python，信任模型与"装个 Python 包"相同，不加载来源不明代码。
- 无热重载：新增插件需重启进程/重新运行（CLI 天然每次启动；GUI 需重开）。
- ffmpeg 滤镜路径坑：Windows 盘符冒号 `:` 是 filter 参数分隔符，绝对路径在 filter 值里不可靠——内置插件用"裸文件名 + cwd 切到 work_dir"规避（见 stabilize.py）。第三方 video 插件若往 `-vf` 里塞 Windows 路径，须自行规避。
