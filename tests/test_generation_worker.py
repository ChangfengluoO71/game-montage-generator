from __future__ import annotations

from pathlib import Path
from threading import Event

from PySide6.QtCore import QEventLoop, QTimer, QThread
from PySide6.QtWidgets import QApplication

from montage.generation import GenerationRequest
from montage.generation_worker import GenerationWorker
from montage.workflow import MontageWorkflow


def test_generation_worker_keeps_qt_event_loop_responsive(monkeypatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    request = GenerationRequest(MontageWorkflow("g", "G"), tmp_path, tmp_path / "work", tmp_path / "out")
    ticks: list[int] = []
    progress: list[int] = []
    finished: list[object] = []
    release = Event()
    progress_seen = Event()
    worker_threads: list[object] = []
    def fake_generation(request, progress=None, log=None):
        worker_threads.append(QThread.currentThread())
        if progress:
            progress(42, "background")
            progress_seen.set()
        release.wait(2)
        return object()
    monkeypatch.setattr("montage.generation_worker.run_generation", fake_generation)
    thread = QThread()
    worker = GenerationWorker(request)
    worker.moveToThread(thread)
    worker.progress.connect(lambda value, message: progress.append(value))
    worker.succeeded.connect(finished.append)
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)
    thread.finished.connect(lambda: loop.quit())
    loop = QEventLoop()
    heartbeat = QTimer()
    heartbeat.timeout.connect(lambda: ticks.append(2))
    heartbeat.timeout.connect(lambda: release.set() if progress_seen.is_set() else None)
    heartbeat.start(1)
    thread.start()
    loop.exec()
    heartbeat.stop()

    assert thread.isFinished()
    assert ticks
    assert progress == [42]
    assert worker_threads and worker_threads[0] != app.thread()
    thread.deleteLater()
    assert app is not None


def test_generation_worker_emits_failure(monkeypatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    request = GenerationRequest(MontageWorkflow("g", "G"), tmp_path, tmp_path / "work", tmp_path / "out")
    errors: list[str] = []
    monkeypatch.setattr("montage.generation_worker.run_generation", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad source")))
    thread = QThread()
    worker = GenerationWorker(request)
    worker.moveToThread(thread)
    worker.failed.connect(errors.append)
    worker.finished.connect(thread.quit)
    thread.started.connect(worker.run)
    loop = QEventLoop()
    thread.finished.connect(loop.quit)
    thread.start()
    loop.exec()

    assert errors == ["bad source"]
    assert app is not None
