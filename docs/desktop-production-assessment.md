# 桌面 AI 初剪接入：技术评估与验收记录

## 评估结论（2026-09-06）

已完整阅读 WORKFLOW_HANDOFF.md、desktop_app.py、workflow.py、kill_truth/scanner.py、kill_truth/cli.py、config.yaml 和 battlefield6_workflow.json。

采用现有 PySide6 桌面界面。内置 Battlefield 6 走 V6 skull-row，按视频分辨率选择已有校准 profile；自定义游戏走 normalized ROI 模板匹配适配。通用服务负责媒体探测、检测规则分派、事件落盘、可编辑项目构建和渲染，QThread 负责隔离耗时工作，Qt 信号负责界面反馈。

先使用 v1 FFmpeg 渲染路径：时间线片段直接转换为渲染输入，保留游戏音频和独立音乐。v2 排名、候选扩展及音乐节拍搜索不属于本次接入前置条件。相比直接调用 V6 CLI，服务级接入能明确绑定本次选中的游戏、源文件和输出目录，避免 CLI 旧媒体索引覆盖用户选择。相比重建 Web UI，延续 Qt 能复用现有原生向导和离屏测试。

## 已核实的差异和风险

1. 桌面多规则 schema 与引擎单 detector schema 不兼容，必须先适配，不能只取第一条规则。
2. 当前 save_profile 实际覆盖同名游戏；import_profile 实际只有提示。交接文档关于追加的描述与提交代码不一致。
3. Windows QSettings.fileName 返回注册表位置，不能拿它的 parent 当配置目录。实际用户 Apex 配置位于 `D:/HKEY_CURRENT_USER/Software/MontageLab/profiles/apex-workflow.json`。迁移应保留旧文件，并使用真正的应用数据目录。
4. V6 scan_source 暂无进度回调，失败可能返回 PARTIAL_ERROR；服务必须显式处理，不能把部分错误标成成功。
5. v1 当前要求音乐及游戏音轨，接入时需测试无音乐、无游戏音轨、短音乐和淡黑行为，避免静默截短。
6. RAW 中包含一个 7 字节的 task8-preflight.mp4；完整目录索引必须如实记录坏文件，不应把它当有效素材。试产选择单个正常源文件。
7. 已有校准报告 PASS 只建立了有限校准画面的检测表现，不等于全素材事件精确率已得到人工验证；本次 MP4 为可复核的试产初剪。

## 本机预检证据

- 起始提交：d842e40，分支 v1.1-editorial-optimization，工作树开始时无修改。
- Python：D:/miniconda/python.exe。
- discover_toolchain 实际选中 FFmpeg/ffprobe 8.0；h264_nvenc 与 hevc_nvenc 运行探测均为 true。环境同时存在 4.3.1，不能直接假定 PATH 首个 ffmpeg 就是合适版本。
- config.yaml 指定的 FLAC 音乐文件存在。
- work/analysis/v6_kill_truth/profiles 下存在 1920×1200、1680×1050、2560×1600、2560×1440 四档 profile。
- 优先试产候选：RAW 中 `战地风云™ 6 - 2026-07-05 22-53-34.mp4`，旧索引时长 17.461667 秒、1920×1200；最终验收须重新探测并 fresh scan，不能将旧缓存当新证据。

## 分阶段执行与证据

详细步骤见 [实施计划](superpowers/plans/2026-09-06-desktop-production.md)。实施由 GPT-5.6-luna 执行，主代理负责评估和验收。

| 阶段 | 交付 | 当前状态 |
| --- | --- | --- |
| M1 | 双向配置契约、实际 Apex 导入、多规则保存回归 | 执行中 |
| M2 | 后台扫描、事件和源索引、进度日志错误 | 待实施 |
| M3 | EditableProject、时间线、音乐与渲染配置 | 待实施 |
| M4 | 单真实视频 fresh scan → JSON → MP4 → 全片解码 | 待实施 |
| M5 | 按钮状态、结果卡、输出目录、持久化与截图 | 待实施 |

每阶段记录 py_compile、完整 pytest、offscreen 冒烟输出及提交号。最终记录输出 MP4、项目 JSON、ffprobe JSON、ffmpeg 解码输出、RAW 前后文件信息和 Qt 截图路径。截图及视频留在 work/output，不提交媒体文件。
