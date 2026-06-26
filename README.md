# AutoCut MVP

中文视频自动化粗剪 MVP。

当前目标不是一开始就接剪映或 Premiere，而是先把核心链路跑通：

```text
输入视频 -> 音频/字幕识别 -> 生成句子索引 -> 生成内容块 -> 给 LLM 做剪辑决策 -> 输出时间线 -> FFmpeg 粗剪
```

## 当前能力

- 检查本地 FFmpeg/ffprobe。
- 获取视频基础信息。
- 从视频提取 WAV 音频。
- 使用 FunASR 做中文 ASR 转写。
- 使用 OCR 识别硬字幕，适合影视剧、带字幕素材。
- 将 transcript JSON 导出为 SRT。
- 将 transcript 转成 sentence index。
- 将 sentence index 聚合成 topic blocks。
- 将 topic blocks 压缩成 LLM 低 token 输入。
- 将 LLM edit plan 转成可执行 timeline JSON。
- 使用 FFmpeg 根据 timeline 粗剪输出视频。

## 目录结构

```text
JSP/
  src/autocut/        # 核心代码
  tests/              # 自动化测试
  examples/           # 可提交的样例 JSON
  inputs/             # 原始视频，不提交
  runs/               # 单条视频的一次处理任务，不提交
  outputs/            # 最终导出结果，不提交
  tools/ffmpeg/       # 本地 FFmpeg，不提交
  pyproject.toml
  uv.lock
  README.md
```

`runs/` 建议按任务命名，例如：

```text
runs/
  monkey_king/
    transcript.json
    sentence_index.json
    topic_blocks.json
    llm_input.md
```

## 安装

当前可以通过 `python -m uv` 使用 uv：

```powershell
python -m uv sync
```

安装 ASR 依赖：

```powershell
python -m uv sync --extra asr
```

安装 OCR 依赖：

```powershell
python -m uv sync --extra ocr
```

可选安装 PP-OCRv6 后端，用于和默认 RapidOCR/PP-OCRv4 ONNX 做速度、准确率对比：

```powershell
python -m uv sync --extra ocr --extra ppocrv6
```

配置 LLM。当前支持 OpenAI-compatible Chat Completions 和 Responses 接口，模型和 API key 通过 `.env`、环境变量或命令参数传入：

```powershell
$env:AUTOCUT_LLM_BASE_URL="https://llmapi.debinxiang.top"
$env:AUTOCUT_LLM_MODEL="gpt-5.5"
$env:AUTOCUT_LLM_WIRE_API="responses"
$env:AUTOCUT_LLM_REASONING_EFFORT="xhigh"
$env:AUTOCUT_LLM_DISABLE_RESPONSE_STORAGE="true"
$env:AUTOCUT_LLM_API_KEY="你的 API key"
```

也可以复制 `.env.example` 为 `.env`，再填入自己的 key。

项目支持本地 FFmpeg。当前推荐路径：

```text
tools/ffmpeg/bin/ffmpeg.exe
tools/ffmpeg/bin/ffprobe.exe
```

CLI 会自动查找 `tools/**/ffmpeg.exe` 和 `tools/**/ffprobe.exe`。

## 基础命令

检查环境：

```powershell
python -m uv run autocut doctor
```

查看视频信息：

```powershell
python -m uv run autocut probe "inputs/01 - 猴王初问世.flv"
```

## 一键流水线

推荐先用 `run` 命令跑通一条视频：

```powershell
python -m uv run autocut run "inputs/01 - 猴王初问世.flv" --job monkey_king --mode ocr
```

它会自动生成：

```text
runs/monkey_king/transcript.json
runs/monkey_king/sentence_index.json
runs/monkey_king/topic_blocks.json
runs/monkey_king/llm_input.md
outputs/monkey_king.subtitles.srt
```

下一步自动调用 LLM 生成 `edit_plan.json`：

```powershell
python -m uv run autocut llm-plan --job monkey_king
```

如果 `runs/monkey_king/edit_plan.json` 已经存在，可以再次运行：

```powershell
python -m uv run autocut run "inputs/01 - 猴王初问世.flv" --job monkey_king --mode ocr --render
```

这会继续生成：

```text
runs/monkey_king/timeline.json
runs/monkey_king/cut_report.md
outputs/monkey_king.rough_cut.srt
outputs/monkey_king.rough_cut.mp4
```

ASR 路线也可以使用同一个入口：

```powershell
python -m uv run autocut run "inputs/01 - 猴王初问世.flv" --job monkey_king_asr --mode asr
```

影视剧、综艺、带硬字幕素材优先用 `--mode ocr`；口播、播客、采访类素材优先用 `--mode asr`。

## 手动拆解流程

提取音频：

```powershell
python -m uv run autocut extract-audio "inputs/01 - 猴王初问世.flv" --output runs/monkey_king/audio.wav
```

FunASR 转写：

```powershell
python -m uv run autocut transcribe "inputs/01 - 猴王初问世.flv" --output runs/monkey_king/transcript.json --srt-output outputs/monkey_king.asr.srt
```

OCR 硬字幕识别：

```powershell
python -m uv run autocut ocr-subtitles-adaptive "inputs/01 - 猴王初问世.flv" --output outputs/monkey_king.subtitles.srt --transcript-output runs/monkey_king/transcript.json
```

试用 PP-OCRv6 识别后端：

```powershell
python -m uv run autocut ocr-subtitles-adaptive "inputs/01 - 猴王初问世.flv" --ocr-engine ppocrv6 --ppocrv6-model-size small --workers 4 --output outputs/monkey_king.ppocrv6.subtitles.srt --transcript-output runs/monkey_king/ppocrv6.transcript.json
```

当前推荐 PP-OCRv6 `small`。`medium` 在部分抖动场景更稳，但 CPU 推理明显更慢；`tiny` 更快但误识别和边缘噪声更多。PP-OCRv6 在 Windows CPU 上避免使用完整检测路径，AutoCut 会先用 OpenCV 裁出硬字幕区域，再走 PP-OCRv6 纯识别。

构建 LLM 输入：

```powershell
python -m uv run autocut build-index runs/monkey_king/transcript.json --output runs/monkey_king/sentence_index.json
python -m uv run autocut build-blocks runs/monkey_king/sentence_index.json --output runs/monkey_king/topic_blocks.json
python -m uv run autocut compact runs/monkey_king/topic_blocks.json --output runs/monkey_king/llm_input.md
```

自动调用 LLM 生成 edit plan：

```powershell
python -m uv run autocut llm-plan --job monkey_king
```

LLM 会返回类似：

```json
{
  "edit_decisions": [
    {
      "block_id": "B0001",
      "decision": "keep",
      "reason": "核心情节"
    }
  ]
}
```

再把 LLM 输出保存为 `runs/monkey_king/edit_plan.json`，转换成 timeline：

```powershell
python -m uv run autocut apply-plan --job monkey_king
```

这一步会在渲染前生成剪辑报告：

```text
runs/monkey_king/timeline.json
runs/monkey_king/cut_report.md
outputs/monkey_king.rough_cut.srt
```

`outputs/monkey_king.rough_cut.srt` 是粗剪预览字幕，时间已经重映射到粗剪视频，从 `00:00:00` 开始。

你可以先检查 `cut_report.md`，确认保留/删除逻辑没问题，再根据 timeline 粗剪：

```powershell
python -m uv run autocut apply-plan --job monkey_king --render --video "inputs/01 - 猴王初问世.flv"
```

也可以单独重建报告：

```powershell
python -m uv run autocut report --job monkey_king
```

也可以单独重建粗剪预览字幕：

```powershell
python -m uv run autocut preview-srt --job monkey_king
```

也可以直接用底层 render 命令：

```powershell
python -m uv run autocut render "inputs/01 - 猴王初问世.flv" runs/monkey_king/timeline.json --output outputs/monkey_king.rough_cut.mp4
```

## 数据格式

### Transcript

```json
{
  "source": "inputs/sample.mp4",
  "language": "zh",
  "segments": [
    {
      "id": "seg_001",
      "start": 0.0,
      "end": 2.4,
      "text": "今天我们先做一个自动化剪辑的最小版本。",
      "speaker": "speaker_001",
      "chars": []
    }
  ]
}
```

### Timeline

```json
{
  "segments": [
    {
      "id": "seg_001",
      "start": 0.0,
      "end": 5.0,
      "decision": "keep",
      "text": "示例保留片段",
      "reason": "核心信息",
      "confidence": 0.9
    }
  ]
}
```

## 测试

```powershell
python -m uv run pytest
```

## MVP 当前状态

当前 MVP 已完成一条真实样例的端到端验证，样例视频为：

```text
inputs/01 - 猴王初问世.flv
```

已跑通的完整链路：

```text
原视频
-> OCR 硬字幕识别
-> transcript.json
-> sentence_index.json
-> topic_blocks.json
-> llm_input.md
-> edit_plan.json
-> timeline.json
-> cut_report.md
-> rough_cut.srt
-> rough_cut.mp4
```

当前样例产物：

```text
runs/monkey_king/transcript.json
runs/monkey_king/sentence_index.json
runs/monkey_king/topic_blocks.json
runs/monkey_king/llm_input.md
runs/monkey_king/edit_plan.json
runs/monkey_king/timeline.json
runs/monkey_king/cut_report.md
outputs/monkey_king.subtitles.srt
outputs/monkey_king.rough_cut.srt
outputs/monkey_king.rough_cut.mp4
```

当前样例结果：

```text
Topic blocks: 55
Decisions: 55
Keep segments: 39
Delete segments: 16
Estimated rough-cut duration: 17m 15s
Rendered rough-cut duration: about 17m 17s
Rough-cut size: about 239.5 MB
```

PP-OCRv6 small 全片 OCR 试验结果：

```text
Command:
python -m uv run autocut ocr-subtitles-adaptive "inputs/01 - 猴王初问世.flv" --ocr-engine ppocrv6 --ppocrv6-model-size small --workers 4 --ocr-threads 1 --output outputs/ppocrv6_small_full.srt --transcript-output runs/ppocrv6_test/ppocrv6_small_full.transcript.json --work-dir runs/ppocrv6_test/adaptive_ppocrv6_small_full

Elapsed: about 60.6s
Detected ranges: 104
Detect frames: 2668
Refine frames: 2001
OCR calls: 1206
Skipped frames: 378
Reused frames: 417
Subtitle segments: 324
```

短片段对比结论：

```text
5-minute sample range: 248s-548s
RapidOCR: about 30.8s, 28 segments
PP-OCRv6 small: about 9.0s, 24 segments
PP-OCRv6 medium: about 49.9s, 24 segments
```

PP-OCRv6 后端目前不会为每个 worker 各自加载 Paddle 模型；它使用单模型识别 runner，`workers` 仍保留给非 PP-OCRv6 路径和前置帧处理参数。`--ocr-batch-size` 可调，但当前 Windows CPU 测试中默认 `1` 最稳。

当前工作流支持两种 LLM 使用方式：

```text
方式一：autocut llm-plan --job <job> 自动调用 OpenAI-compatible API
方式二：把 llm_input.md 复制到对话框，让 AI 返回 edit_plan.json，再运行 apply-plan
```

当前阶段结论：

```text
MVP 主流程已经跑通。下一阶段不再是验证能否剪，而是提升剪辑质量、产品体验和工具集成。
```

已知限制：

- 当前主要依赖字幕文本理解，尚未使用画面、镜头、人物、音乐等多模态信息。
- OCR 对影视剧硬字幕可用，但复杂画面、字体变化、字幕遮挡仍可能影响识别。
- PP-OCRv6 small 已比默认 RapidOCR 更快，但仍会出现连续帧近似文本抖动，例如个别字在相邻帧间跳变。
- 片尾歌曲、演员表和字幕混杂时仍可能产生噪声，需要额外的片尾过滤或人工校对。
- 剪辑决策质量取决于 LLM 或人工给出的 `edit_plan.json`。
- 当前 FFmpeg 输出是粗剪成片，不是剪映、Premiere、FCPXML 等可编辑工程。
- 目标时长控制、人工审核 UI、批量处理暂未实现。

## 后续方向

- 剪辑质量：目标时长控制、节奏判断、过长片段提示、章节完整性检查。
- OCR 质量：连续帧投票/聚类、片尾歌曲和演员表过滤、按素材类型自动选择 RapidOCR 或 PP-OCRv6。
- 多模态：镜头变化、人物出场、动作场面、片头片尾、音乐/音效高潮。
- 产品体验：Web UI 或桌面端，用于查看报告、修改 edit plan、预览粗剪。
- 工程输出：OpenTimelineIO、FCPXML、Premiere XML、剪映草稿。
- 批量处理：多视频导入、素材索引、搜索字幕、批量生成短视频版本。
