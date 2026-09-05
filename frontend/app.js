const toast = document.getElementById('toast');
const showToast = (message) => { toast.textContent = message; toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 2400); };
document.querySelectorAll('.tool').forEach((button) => button.addEventListener('click', () => { document.querySelectorAll('.tool').forEach((item) => item.classList.remove('active')); button.classList.add('active'); showToast(`筛选：${button.textContent}`); }));
document.querySelectorAll('.keep').forEach((button) => button.addEventListener('click', () => { const pending = button.classList.contains('muted'); button.classList.toggle('muted', !pending); button.textContent = pending ? '✓ 保留' : '○ 待审核'; showToast(pending ? '片段已加入保留列表' : '片段已退回待审核'); }));
document.getElementById('generate').addEventListener('click', () => { const values = ['pre', 'post', 'merge', 'bridge'].map((id) => Number(document.getElementById(id).value)); if (values.some((value) => value < 0) || values[3] > values[2]) { showToast('请检查规则：桥接时长不能超过合并阈值'); return; } showToast('AI 初剪任务已排队 · 本地 worker 即将开始'); });
document.getElementById('exportProject').addEventListener('click', () => showToast('项目已导出为 editable project.json'));
document.getElementById('importProject').addEventListener('click', () => showToast('请选择 workflow 或 project JSON 文件'));
document.getElementById('newProfile').addEventListener('click', () => showToast('配置向导即将打开：上传标志、全屏正例和负例'));
document.getElementById('dropzone').addEventListener('click', () => showToast('素材导入接口已预留，本地 worker 接入后可直接扫描文件夹')));
