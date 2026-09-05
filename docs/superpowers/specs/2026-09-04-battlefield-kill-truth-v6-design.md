# Battlefield Kill Truth V6 设计规格

状态：实施中；当前里程碑为校准集，尚未授权进入剪辑层。

## 目标

V6 是独立的事实层，只回答：

> 在某个 RAW 源中，玩家本人何时被 HUD 确认完成了一次击杀，以及哪些击杀属于同一段连续击杀。

主要事实来源是当前用户 HUD 中央偏左、准星下方的 Personal Kill Feedback Panel，尤其是其中的 skull row 状态变化。V5 的 ROI 像素变化、通用 motion、audio transient、reward ROI 和右上 kill-feed 只能作为 `ActivityEvidence` 辅助信息，不能直接产生 OwnKillEvent。

V6 不做高光排序、音乐分析、踩点、转场、Montage 或任何成片渲染。

## 不可变边界

- `D:\91\集锦\raw` 永久只读：只允许读取和解码，禁止删除、移动、重命名、覆盖、原地编码或修改 metadata。
- V6 生成物只写入 `D:\91\集锦\work\analysis\v6_kill_truth`；人工整理账本写入 `D:\91\集锦\work\analysis\curation`。
- 不读取旧候选、旧 `payoff_events`、旧 `verified_kill` 或 V1/V2/V5 candidates 来推导 V6 truth。旧结果最多用于 debug 对比。
- V6 使用同一次运行中锁定的实测可用 FFmpeg 8.x 与同发行版 ffprobe；路径、版本和 NVENC 实测结果写入运行日志与 V6 environment artifact。
- 在校准结果可信之前，不扫描全部 RAW；在质量 gate 未通过前，不生成任何 Preview 或 Montage。

## HUD profile

profile 按用户、游戏 HUD、分辨率和版本命名，例如 `cfl_bf6_1920x1200_v1`。profile 保存 normalized search ROI、row sub-ROI、模板目录、阈值、detector 版本和模板指纹。上传截图只能作为校准起点，不能直接把其像素坐标映射到视频。

当前粗搜索 seed 为 `[0.25, 0.42, 0.66, 0.75]`。校准必须从真实 1920×1200 RAW 抽取至少 5–10 个明显面板帧，再确定实际 normalized ROI。混合分辨率素材使用对应分辨率的 profile 或明确标记为未校准，不静默复用不匹配的绝对像素。

真实校准观察到两类相关结构：

1. 连杀 banner：包含“双杀/三杀”等文字和水平排列的 skull row，是连续击杀计数的主要结构证据。
2. 个人击杀卡片：包含单次/当前累计 skull、分数、敌人名和奖励文本，可作为 panel neighborhood 与 skull row 的辅助结构证据。

检测器不对整个动态面板做 template match，而是在 search ROI 内检测 skull、颜色、水平几何和时序一致性。

## Skull detector

`SkullDetector` 只在 profile ROI 内工作，支持：

- normal skull 与 headshot/orange skull 的多模板 bank；模板覆盖 fade-in、full opacity、fade-out、轻微比例漂移和压缩差异。
- grayscale/normalized correlation 与有限的约 ±5% scale search。
- 对候选 bbox 内计算 orange color evidence；ROI 外的橙色不参与 headshot 判断。
- `DetectedSkull` 包含 bbox、center、kind、template score、color score。
- `SkullRowState` 包含时间、panel presence、normal/headshot/count、row bbox、geometry score、结构 confidence 和原始 detections。

真正的 row 必须满足：Y 近似一致、X 水平排列、尺寸近似一致、间距合理、数量离散且位于正确子 ROI。geometry 不达标时只能输出无效/不确定状态，不能产生 OwnKillEvent。

## Temporal State Machine

状态机不使用“700ms 内 merge”来定义击杀：

```text
NO_PANEL
  -> skull row appears
PANEL_ACTIVE(count=N)
  -> same count: one persistent state
PANEL_ACTIVE(count=N)
  -> count increases: new OwnKillEvent
PANEL_ACTIVE(count=M)
  -> panel absent for configured duration: sequence end
```

`0→1` 产生一次事件，`1→2` 产生第二次事件，即使只间隔 200–300ms 也不能合并。重复帧不重复计数。粗扫发现 `2→4` 时必须请求 ±1s dense refinement；若恢复为 `2→3→4` 则产生两个事件，若 30/60fps 仍只有 `2→4` 则记录 `SIMULTANEOUS_MULTI_KILL` 与 `kill_count_delta=2`，不猜中间事件。

OwnKillEvent 的 truth confidence 只能主要来自 skull match、row geometry、panel structure、temporal continuity 和合法 count transition。audio/motion/transient 不能显著抬高 truth confidence；impact time 可以为空。

## 数据合同

OwnKillEvent 至少包含：

`event_id`, `source_id`, `source_path`, `sequence_id`, `type`, `confirmation_time`, `impact_time`, `sequence_index`, `skull_count_before`, `skull_count_after`, `kill_count_delta`, `kill_type`, `confidence`, `evidence`, `dense_refinement_used`。

其中 `confirmation_time` 是 HUD transition 的语义时间；`impact_time` 只在 confirmation 前 600ms 至后 100ms 的辅助信号足够可靠时填写，否则为 null。`kill_type` 不确定时为 `UNKNOWN_KILL`。

KillSequence 只由真实 OwnKillEvent 和 panel 生命周期构造。事件间固定时间 gap 只能作为 panel tracking 不确定时的 fallback。类型由真实数量决定：SINGLE、DOUBLE、TRIPLE、QUAD、MULTI_KILL_5_PLUS；不使用 OCR 连杀文字作为 truth。

## 扫描和缓存

- PASS A：10–15fps，仅解码/pipe Personal Kill ROI，寻找 panel、row、count transition 候选。
- PASS B：对 panel 出现、count transition、跳计数和 ambiguous state 周围以 30/60fps dense refinement；不把全段帧导出为 PNG。
- cache key 至少包含 source path/size/mtime、profile id/version、detector version、template bank fingerprint、threshold fingerprint、选定 FFmpeg 版本与必要解码参数。
- source index 可复用现有 ffprobe cache，但 V6 truth 必须独立存储和独立失效。

## 校准、审阅和质量 gate

第一轮只建立 Calibration Set，覆盖单杀、快速双杀/多杀、headshot 和 hard negative。生成 `v6_calibration_report.md`，明确 resolution、ROI、模板数量、阈值、扫描帧率、已测试事件、FP/FN、rapid recovery 和已知 failure modes。

完整扫描后才生成：

- `kill_event_index_v6.json`
- `kill_sequence_index_v6.json`
- `kill_review.html`
- per-source `skull_state_timeline.png`
- `v6_gold_set.json` 与验证指标

review 页面每个 positive 显示 source、timestamp、before/transition/after panel crop、full-frame context、count before/after、confidence、type，并提供 CONFIRM/FALSE_POSITIVE/UNKNOWN。还要抽查 negative，特别是 teammate kill-feed、assist、objective/reward、hit marker、爆炸、fade、暗/亮场景和 <500ms 连杀。

目标值只是 gate target，不得伪造：OwnKill precision 优先达到 98% 以上，recall 目标 95% 以上；同时单独报告 double/triple/quad 和 rapid interval <1s、<500ms 的 recovery。任何 truth 或 rapid recovery 不可信，都必须停在 V6。

## 输出和 STOP rule

V6 当前只允许输出事实索引、校准/验证报告、review/debug artifacts 和 curation ledger。禁止 `preview_60s_v6.mp4`、Fast Montage、Full Highlights。Calibration 不通过时停止；Calibration 通过后全量扫描，完成报告后仍停止，等待人工审核。
