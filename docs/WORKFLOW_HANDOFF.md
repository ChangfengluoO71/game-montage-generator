# Montage Lab 项目交接（Handoff）与下一步工作流接入

> 本文件面向接手的 AI：请先完整阅读本文件与文末列出的关键源文件，再作技术评估与方案决策。
> 交接时间背景：桌面应用（PySide6）已能创建「游戏 → 多规则」配置并持久化显示，但"开始生成 AI 初剪"仍是占位；引擎侧已有大量检测/时间线/渲染代码可复用，尚未经 UI 串联成端到端流程。

---

## 1. 项目目标

本地优先的「游戏集锦生成器」：

1. 用户为某个游戏创建一条或多条高光检测规则（如 Apex：击杀、击倒；均可归属同一游戏）。
2. 用户选择视频素材文件夹（只读扫描，绝不改写）与独立背景音乐。
3. 设置剪辑规则（事件前后保留秒数、合并阈值、桥接长度、淡黑秒数）。
4. 点击「开始生成 AI 初剪」→ 扫描素材 → 规则检测 → 生成候选事件 → 构建可编辑项目（timeline）→ 渲染试生产视频。

用户明确的规则：机瞄（准星）旁出现**白色/红色骷髅击杀反馈**等高可信信息即可确认击杀。RAW 素材目录 `D:\91\集锦\raw` 必须只读。

## 2. 代码形态现状（重要）

- **不再是静态网页**。目前唯一推荐形态是原生 Qt 桌面应用：
  - 入口：`python -m montage.desktop_app`（依赖 PySide6，本机已装）
  - 文件：`montage/desktop_app.py`（约 314 行）
- `frontend/` 目录是早期静态 HTML 原型（`index.html` + `app.js` + `styles.css` + `profile-wizard.html`），曾出现多次交互失灵；**不建议继续维护**，除非下一个 AI 明确选择用 Electron/PyWebView 重做 UI 并给出理由。
- 引擎侧是成熟 Python 模块（见 §4），全部可复用。

## 3. 配置模型（已确定为产品事实）

```
游戏（display_name，game_id 由名称派生，如 Apex Legends → apex-legends）
└── 规则列表 rules[]
    ├── 击杀 (marker/positive/negative 样本 + normalized ROI + threshold)
    ├── 击倒
    └── ...
```

- UI 已移除「检测方式」选择（template_match / skull_row 是内部实现，不暴露给用户）。
- 用户再次创建同名游戏时，新规则**追加**到同名 workflow JSON（实现在 `ProfileWizard.save_profile`）。
- 持久化位置：`QSettings` 数据目录下 `profiles/<game_id>-workflow.json`（Windows：`%APPDATA%/MontageLab/GameMontageGenerator/profiles/`）。

当前桌面端输出 schema（`montage/desktop_app.py` 的 `CustomProfile.workflow()`）：

```json
{
  "schema": "game-montage-workflow-v1",
  "game": { "id": "apex-legends", "display_name": "Apex Legends" },
  "detectors": { "rules": [ { "id": "...", "label": "击杀", "type": "template_match",
                              "search_roi": [0.3,0.55,0.48,0.74], "threshold": 0.65,
                              "templates": [...], "positive_samples": [...], "negative_samples": [...] } ] },
  "metadata": { ... }
}
```

### ⚠️ 关键契约债务（已用代码验证）

引擎侧 `montage/workflow.py` 的 `MontageWorkflow.from_dict` 期望：

```json
{ "schema": "game-montage-workflow-v1", "game_id": "...", "display_name": "...",
  "detector": { "detector_type": "...", "event_label": "...", "roi": {...}, "templates": [...],
                "positive_samples": [...], "negative_samples": [...], "thresholds": {...} },
  "edit_rules": {...}, "audio_output": {...}, ... }
```

直接 `MontageWorkflow.import_json()` 解析桌面端文件会抛 `KeyError: 'game_id'`（已验证）。**下一步第一步应定义统一的「UI 配置格式 → 引擎 DetectorConfig/编辑规则」适配层**，并补一个双向转换 + 测试，再谈扫描。注意内置示例 `examples/battlefield6_workflow.json` 也是引擎格式，可作为兼容基准。

## 4. 引擎侧已有多条可复用管线

### 4.1 游戏无关通用工作流契约（`montage/workflow.py`）
- `EditRules`（事件前/后秒、合并、桥接、淡黑）— 桌面应用步骤 3 的参数即映射到此。
- `DetectorConfig`（detector_type/event_label/normalized ROI/templates/positive/negative/thresholds）。
- `MontageWorkflow`（schema `game-montage-workflow-v1`，import/export JSON）。
- `TimelineClip` / `EditableProject`（schema `game-montage-project-v1`：clips + source_ledger + music_source + render_settings）。
- `extract_audio()`：MP4/MKV 提取第一条音轨，不写原始文件。
- `resolve_roi()`：normalized ROI → 像素坐标。

### 4.2 Battlefield 6「击杀真相」扫描（现成可用，最接近用户机瞄骷髅规则）
模块 `montage/kill_truth/`，核心：
- `profile.py`：`HudProfile`（四档分辨率 1920×1200、1680×1050、2560×1600、2560×1440 的校准配置，`config.yaml` 有 `v6_profile_id` 等）。
- `scanner.py`：`scan_source(record, profile, toolchain, scan_config, cache_dir, raw_dir, use_cache) -> V6ScanResult`；`dry_run_source(...)` 可做快速 dry-run 验证；`V6ScanConfig(coarse_fps, dense_fps, panel_disappear_s, refinement_radius_s)`。
- `cli.py`：`run_v6_scan(config_path, dry_run=False)`、`run_v6_review`、`run_v6_verify`、`run_v6_calibrate`、入口 `run_v6_command`；输出事件索引（`_write_indexes`）与评审报告。

### 4.3 视频/音频/编辑基建
- `media_index.py`：`build_media_index`（ffprobe 探针、指纹）。
- `ffmpeg_renderer.py`、`v2_renderer.py`、`transitions.py`、`audio_mix.py`、`audio_analysis.py`、`music_analysis.py`、`payoff_detection.py`、`curation.py`、`ranking.py`、`timeline.py`、`beam_timeline.py`、`candidate.py`、`condense.py`、`dedupe.py`、`proxy.py`、`review.py`、`models.py`、`toolchain.py`（ffmpeg/ffprobe 自动发现）、`config.py`。
- 现有测试基线：`223 passed, 2 warnings`（v6 阶段版本），本次会话新增 `tests/test_desktop_app.py`（2 passed）覆盖多规则序列化。

## 5. 桌面应用现状（`montage/desktop_app.py`）

主窗口 4 步：选择游戏 → 素材与音乐 → 剪辑规则 → 准备生成。

- 步骤 1 现在会在启动与保存后刷新本机 profiles，显示「游戏 | N rules | 规则列表」。
- 步骤 2：视频文件夹（只读扫描，显示视频数）、独立音乐（含 MP4 提取提示）。
- 步骤 3：pre/post/merge/bridge/fade 五参数，含 bridge≤merge 校验。
- 步骤 4 目前：`generate` 按钮只弹「Worker 接口将在下一实现切片接入」气泡 —— **这正是“点击没反应”的原因，尚未接入真实执行**。
- `advance()` 中的 4 步校验已存在（步骤 2 需文件夹、步骤 3 数字/桥接校验、步骤 4 摘要）。
- 已知小问题：内置按钮与自定义按钮共用 `QButtonGroup`（exclusive），"选中的游戏配置"尚无显式状态存回配置；音乐选择未持久化；全局设置的语言/主题已生效（QSettings），素材目录绑定写全局 `video_folder`。

## 6. 历史教训（重要，避免重蹈）

1. **中文源码必须用 UTF-8 文件工具写入**：多次在 bash heredoc/PowerShell 控制台注入中文被转码成错误字节，造成 `SyntaxError: unterminated f-string`、按钮失灵。凡含中文的 Python/JS 源码，一律用 write/edit 工具或 Python `Path.write_text(encoding='utf-8')` 写入。
2. **对象模型改动后必须全量审计旧字段引用**：把 desktop_app 从「单规则」改为「多规则」（CustomProfile.rules）后，`go_next`/`_validate`/确认页仍引用 `profile.roi / profile.event_label / self.detector`，导致第 3 步「下一步」无反应。教训：改模型 → grep 全部旧字段 → 写回归测试（含 native 离屏导航测试：`QT_QPA_PLATFORM=offscreen` 下构造 `ProfileWizard`，走 `go_next()` ×3 断言 `stack.currentIndex()==3`）。
3. **改动后必须跑**：`python -m py_compile montage/desktop_app.py` + `python -m pytest -q tests/test_desktop_app.py` + offscreen 启动 `MontageLab()` 冒烟。
4. 本地工作区是 git worktree：`D:\91\集锦\battlefield-montage\.worktrees\v1.1-editorial-optimization`，分支 `v1.1-editorial-optimization`，推送到 `origin main`（GitHub 有时 schannel TLS 握手失败，推送后必须 `git ls-remote origin refs/heads/main` 或 GitHub API 复核）。
5. **不要生成到 `D:\91\集锦\raw`**；工作产物写 `D:\91\集锦\work` 与 `D:\91\集锦\output`（config.yaml 已指向）。截图等临时文件不要提交。

## 7. 下一步：接入真实工作流 + 视频试生产（待接管的 AI 评估细化）

目标：把「开始生成 AI 初剪」接到真实执行，产出至少一个可播放的试产（MP4）。

建议拆分：

1. **M1 契约打通**：定义并实现「桌面配置 JSON ↔ 引擎 `MontageWorkflow`」双向适配（见 §3 债务），单测覆盖；让 `MontageWorkflow.import_json` 能解析桌面保存的文件。
2. **M2 扫描执行**：`开始生成 AI 初剪` → 后台线程运行：所选素材文件夹 → `build_media_index/media_index` → 对每个视频 `scan_source`（或复用 `run_v6_scan` 的索引产物）→ 汇总事件；UI 显示进度（QThread + 信号，勿阻塞主线程）、日志、错误；产出可复现的 JSON 结果（与 `docs/reusable-game-montage-workflow.md` 保持一致）。
3. **M3 项目生成**：事件 → `TimelineClip` 列表 → `EditableProject`（含 music_source、render_settings）→ 导出 `game-montage-project-v1` JSON。
4. **M4 试生产渲染**：用现有 `ffmpeg_renderer.py`/`v2_renderer.py` 管线渲染一个短试产 MP4 到 `output/`（可选先用 1–2 个片段、短时长），确认 ffmpeg/nvenc 可用（`toolchain.py`）。
5. **M5 UI 完成度**：生成按钮置灰/进度条/结果卡片（打开输出目录、查看日志）；步骤 1 选中的游戏显式绑定到本次项目；音乐持久化；导出 workflow 与 project 按钮。

**评估时请决策**：
- 检测后端优先走 V6 skull-row（Battlefield 6 现成校准配置，最贴近用户机瞄骷髅规则）还是新写的 template_match 通用规则（用户自定义 Apex 等）？建议两者都支持：内置游戏用现成 profile 管线，自定义游戏用 `DetectorConfig` 的模板匹配；保持「检测器按规则解析」的抽象。
- 渲染走 v1 通用管线还是 v2 复杂管线？试生产建议从最简 path（v1 ffmpeg_renderer + 直连 clip）开始，验证端到端后再接 v2。
- UI 技术栈：继续 PySide6（推荐，已可用）还是换 WebView/Electron？无强理由不换。
- 进度与并发：QThread worker（推荐）vs subprocess CLI；无论如何**不得阻塞 Qt 主线程**。

## 8. 验证清单（接管 AI 的“完成”标准）

- [ ] 适配层 + 双向转换单测通过；`MontageWorkflow.import_json` 能读桌面保存的 apex 配置。
- [ ] 原生离屏导航测试通过：`ProfileWizard` 1→4、`MontageLab` 启动无异常。
- [ ] `pytest` 全量通过（原 223 + 新增）。
- [ ] 用 1 个真实 RAW 视频（或临时小样）跑通：扫描 → 事件 → 项目 JSON → 输出 MP4，`ffprobe` 确认可解码。
- [ ] 「开始生成 AI 初剪」点击后有真实进度/日志/结果反馈，失败有明确错误气泡（不再是无反应/纯占位气泡）。
- [ ] 提交并推送到 GitHub，`git ls-remote origin refs/heads/main` 复核。

## 9. 关键文件索引

| 用途 | 路径 |
|---|---|
| 桌面应用 | `montage/desktop_app.py` |
| 通用工作流契约 | `montage/workflow.py` |
| V6 击杀扫描 | `montage/kill_truth/scanner.py`、`cli.py`、`profile.py` |
| 媒体索引 | `montage/media_index.py` |
| 渲染 | `montage/ffmpeg_renderer.py`、`montage/v2_renderer.py` |
| 配置 | `config.yaml`（raw/work/output 路径、v6 参数） |
| 内置示例 | `examples/battlefield6_workflow.json` |
| 前端原型（勿优先维护） | `frontend/` |
| 架构文档 | `docs/reusable-game-montage-workflow.md`、`docs/frontend-game-montage-generator-plan.md` |