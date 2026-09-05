# Battlefield Kill Truth V6 实施计划

> 执行方式：按本计划逐项实现并在每个阶段运行测试；先校准集，校准不可信时停止，不进入全量扫描或剪辑层。

## 1. 建立隔离合同与文档

- 新增 `montage/kill_truth/` 包，所有 truth 类型使用 `OwnKillEvent`、`KillSequence`、`ActivityEvidence` 等清晰命名；不从旧 Candidate/PayoffEvent 导入语义事实。
- 新增 V6 输出目录属性、profile/cache 配置和 CLI 合同；保留现有 V1/V5 文件与入口不变。
- 加入 RAW 路径保护、V6 output path 保护和 artifact schema 测试。
- 先运行全套 baseline tests，确认当前工作树已有 V5 变更未被破坏。

## 2. 先写测试：数据模型和状态机

- 测试 `OwnKillEvent`/`KillSequence` 序列化、nullable impact、类型映射和不允许把 old `verified_kill` 当 V6 字段的接口边界。
- 测试 `NO_PANEL→NO_KILL`、静态 skull row 不重复、`0→1`、`1→2`、`3→4`/300ms、panel disappearance 结束 sequence。
- 测试 coarse `2→4` 请求 dense refinement；dense `2→3→4` 生成两个事件；真实 `2→4` 生成 delta=2 simultaneous event。
- 测试 headshot 的模板+局部颜色证据、橙色背景不在 skull bbox 内时不判 headshot。
- 先运行这些测试得到预期 RED，再实现最小生产代码使其 GREEN。

## 3. HUD profile、模板 bank 和真实校准资产

- 实现 `profile.py`：normalized ROI、row sub-ROI、resolution profile id、thresholds、模板目录和指纹。
- 从真实 RAW 直接抽取 5–10 个明显 panel/skull 帧，建立 normal bank；从 09-01 真实 RAW 抽取橙色 headshot 实例，数量不足就明确标记 incomplete。
- 保存校准 manifest、panel crop、full-frame context 和 profile JSON 到 `work\analysis\v6_kill_truth\profiles`/`calibration`，不引用 V5 event 时间作为 truth。
- 用 profile/asset manifest 测试实际 resolution、normalized ROI 与 bank fingerprint 稳定性。

## 4. Skull detector 和 row geometry

- 实现 `skull_detector.py`：只在 Personal Kill search/row ROI 内使用有限 scale template matching；normal/headshot 分开计算模板、颜色与几何证据。
- 实现 `skull_row.py`：按 Y/X/尺寸/间距聚类 detections，过滤低 geometry score 和面板外误检。
- 先用合成小图测试单/双/不规则排列、橙色背景、尺寸漂移，再用真实校准帧做 smoke test。
- detector 输出只表示 `SkullRowState`；不在 detector 中命名或创建旧 `verified_kill`。

## 5. Temporal state machine 和 dense refinement

- 实现 `panel_state.py` 的显式状态、合法 transition、静态 state 去重、panel disappearance timeout 和 sequence boundary。
- 实现 `refinement.py`：维护 coarse request、dense window、2→N recovery、ambiguous state 标记；时间接近不作为 merge truth。
- 加入 rapid multi-kill regression tests，覆盖 `<1s` 和 `<500ms`，确保新事件不会被吞掉。

## 6. OwnKillEvent、sequence 和 impact auxiliary

- 实现 `events.py`/`sequence.py`，把合法 count transition 转为 OwnKillEvent，再按 panel 生命周期构造 KillSequence。
- 实现 `impact.py` 的可选窗口搜索；audio/motion/hit transient 只能写 auxiliary evidence，找不到可靠时写 null。
- 测试 sequence 类型、headshot count、span/density 和 simultaneous multi-kill schema。

## 7. 直接 RAW scanner 与 cache

- 实现 `scanner.py`：由 MediaRecord 直接读取 RAW，不读取候选、payoff 或旧 verified 结果；PASS A 用 FFmpeg8 pipe ROI，PASS B 对候选窗口 dense decode。
- 实现 V6 独立 cache：source fingerprint + profile/detector/template/threshold/toolchain fingerprint；命中时跳过 ffprobe/dense analysis 能跳过的 expensive step。
- 加入 source read-only assertions、FFmpeg/ffprobe absolute path logging、dry-run summary 和 mixed resolution profile selection。
- 生成 per-source raw state artifacts，确保可复核每个 count timeline。

## 8. Review、gold set、metrics、curation

- 实现 `review.py`：输出每个 event 的 before/transition/after panel crop、full-frame context、contact sheet 和 `kill_review.html`。
- 实现 `gold_set.py`/`metrics.py`：支持人工 manifest、±500ms matching、TP/FP/FN/precision/recall/F1、single/double/triple/quad 与 rapid recovery。
- 初始化 `work\analysis\curation\curation_ledger.json` 为 UNREVIEWED；manual reject 只影响未来 editorial layer，不改变 V6 truth。
- 负例抽样与报告必须和正例报告同等可见。

## 9. CLI 和 calibration gate

- 在 `main.py` 增加独立 `v6-calibrate`、`v6-scan --dry-run`、`v6-scan`、`v6-review`、`v6-verify`；V6 命令不调用旧 `all-v2-quality` 或任何 renderer。
- `v6-calibrate` 只执行 calibration set、写 profile/template/report；没有 calibration PASS 时 `v6-scan` 明确拒绝全量扫描。
- Calibration report 明确真实阈值、样本数量、FP/FN、rapid recovery 和 failure modes；不以降低阈值换取通过。
- 若 calibration 通过，才扫描全部 44 RAW 并写两个 V6 index；若不通过，保留 artifacts 并按 STOP rule 报告。

## 10. 验证和交付

- 每次代码阶段运行相关 pytest；最终运行全套 pytest、compileall、CLI dry-run/verify 和 RAW manifest 前后对比。
- 检查 V6 index/review/report 路径、schema、source count、sequence type、headshot、dense refinement、gold metrics 和主要 FP/FN。
- 明确报告使用的 HUD profile、FFmpeg/ffprobe/NVENC 实测路径；最终不生成任何 mp4，等待人工审核后才考虑 V7 editorial layer。
