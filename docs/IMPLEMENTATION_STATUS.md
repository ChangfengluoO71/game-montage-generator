# AI 初剪工作流实施状态

更新时间：2026-09-06

## 技术结论

- 内置 Battlefield 6 使用现成的 V6 `skull_row` calibrated profile，并按输入分辨率选择 profile。
- 自定义游戏使用 normalized ROI 的 `template_match` 适配；workflow rule 的 `detector_type` 负责路由。
- UI 继续使用 PySide6，扫描与渲染由 `QThread` worker 执行。
- 先采用最简 v1 FFmpeg 分段、concat、音频混合管线；v2 管线留到真实反馈后再评估。
- RAW 目录只读，产物写入 `work/` 和 `output/`。

## 里程碑

- M1 契约打通：桌面配置 JSON 与 `MontageWorkflow` 双向适配，保留多规则、ROI、阈值、样本、profile、metadata；桌面保存的 Apex 配置可直接 `import_json`。
- M2 后台扫描：QThread 进度、日志、错误、source ledger、events、diagnostics、result JSON 已接入；V6 与 template_match 均有路由和回归测试。
- M3 项目生成：事件构建 `TimelineClip`，做源边界、同源合并、时间线连续性校验，导出 `game-montage-project-v1`。
- M4 试产渲染：使用真实 RAW 事件生成项目并输出 MP4，游戏音频与独立音乐均接入。
- M5 UI 首版：生成按钮运行时置灰，显示实时进度、日志和结果卡片，保存音乐/选中 workflow，支持打开输出目录。

## 真实验收证据

证据目录：`D:\91\集锦\work\desktop-production-evidence\trial-20260906`

- `progress.json`：扫描、项目生成、分段渲染的真实进度。
- `result-summary.json`：4 个 V6 事件、2 个 timeline clips 及结果路径。
- `ffprobe.json`：输出为 1920×1200、60 fps 的 H.264 视频和 48 kHz AAC 音频，时长约 10.84 秒。
- `decode.txt`：FFmpeg 完整解码到 null 成功。
- `validation.json`：输出文件约 50 MB，项目时长约 10.83 秒，RAW 前后均为 47 个文件且清单一致。
- `ui-button-status.json` 与 `ui-button-result.png`：PySide6 按钮真实触发，进度 100、4 events、输出按钮可用。

## 验证

- `python -m pytest -q`：268 passed，2 warnings。
- 目标模块 `python -m py_compile`：通过。
- `QT_QPA_PLATFORM=offscreen`：可创建 4 页 `MontageLab`，按钮真实运行完成。
