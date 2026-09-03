# Battlefield Montage V1

这是一个 Windows 本地运行、非破坏性的 Battlefield 高光自动剪辑流水线。V1 的唯一交付物是经过验证的 45–60 秒 `preview_60s.mp4`；人工观看确认前，完整 Fast Montage 与 Full Highlights 被 CLI 明确阻止。

## 运行环境

- Windows、Python 3.13+
- 已安装并可导入：NumPy、SciPy、librosa、SoundFile、Matplotlib、PyYAML、pytest
- NVIDIA GPU 可用时优先使用实测通过的 `h264_nvenc`；否则自动记录失败原因并回退 `libx264`
- `config.yaml` 已指向当前素材、音乐、`work` 和 `output` 目录

安装依赖：

```powershell
Set-Location 'D:\91\集锦\battlefield-montage'
D:\miniconda\python.exe -m pip install -r requirements.txt
```

## 推荐执行顺序

先生成全部分析、缓存和 EDL，不渲染：

```powershell
D:\miniconda\python.exe main.py all --dry-run
```

确认 `work\analysis\preview\preview_edit.json` 已生成后，仅渲染第一轮 Preview：

```powershell
D:\miniconda\python.exe main.py render-preview
D:\miniconda\python.exe main.py verify-preview
```

也可以用 `python main.py all` 一次完成分析和 Preview 渲染，但仍不会生成完整成片。`render-fast` 和 `render-full` 在人工批准 Preview 前固定返回非零状态并停止。

## 安全边界

- `D:\91\集锦\raw` 只作为输入读取；程序不会删除、移动、重命名、覆盖或修改其中任何文件。
- 代理、缓存、分析、审阅页、EDL 和日志全部写入 `D:\91\集锦\work`。
- 成片只写入 `D:\91\集锦\output`。
- 所有 FFmpeg 调用都使用参数数组和 `shell=False`，Unicode 文件名不会拼接进 shell 命令。
- 每次运行保存 RAW 的 path/size/mtime 清单；Preview 渲染前后会比较清单。

## 分析与编辑策略

短片（≤90 秒）保留人工选择先验，尽量保持连续精彩操作；只对明显无内容的头尾做保守处理。超过 300 秒的 3 个长录像才生成 proxy，并重点做音频活动、运动和视觉活动候选提取。候选会做指纹去重，同一 duplicate group 在一个 EDL 中最多使用一次，Fast 方向优先人工保存短片。

音乐分析不仅使用 BPM，还输出 beat、strong beat、bar/downbeat 尝试、onset strength、RMS/energy、section boundary、高低能量区和置信度。Preview 优先选包含 build-up、结构转折和高能段的音乐区间，而不是固定使用歌曲开头 60 秒。

编辑优先级是高光质量 > Gameplay continuity > 音乐结构 > beat sync > 转场效果。硬切占绝大多数；不会自动加入 glitch、RGB split、spin、zoom、shake、flash spam 或模板式 speed ramp。游戏原声保留在混音中，并在爆炸、击杀、炮击等高活动区间平滑压低音乐。

Preview 保持 1920×1200 和接近源帧率，不插帧、不 upscale、不强制裁成 16:9，也不提前做上传版构图决定。

## 主要产物

分析完成后可查看：

- `D:\91\集锦\work\analysis\environment.json`：本次固定使用的 FFmpeg/ffprobe 绝对路径、版本、NVENC 实测结果和 GPU 信息
- `D:\91\集锦\work\analysis\media_index.json` / `.csv`
- `D:\91\集锦\work\analysis\music\beat_map.json`
- `D:\91\集锦\work\analysis\music\music_structure.json`
- `D:\91\集锦\work\analysis\music\music_analysis.png`
- `D:\91\集锦\work\analysis\highlight_candidates.json` / `.csv`
- `D:\91\集锦\work\analysis\preview\preview_edit.json`
- `D:\91\集锦\work\analysis\preview\preview_timeline.txt`
- `D:\91\集锦\work\analysis\preview_report.md`
- `D:\91\集锦\work\logs\pipeline.log`
- `D:\91\集锦\output\preview_60s.mp4`

`work` 内 expensive step 按源文件绝对路径、size、mtime 和参数缓存。素材未变化时不会重复 ffprobe、长视频 proxy、视频分析、音乐分析或候选指纹计算。

## 开发验证

```powershell
D:\miniconda\python.exe -m pytest -q
D:\miniconda\python.exe -m compileall -q .
```
