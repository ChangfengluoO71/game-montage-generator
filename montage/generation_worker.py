"""PySide6 worker bridge for the Qt independent generation service."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .generation import GenerationRequest, run_generation


class GenerationWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, request: GenerationRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            result = run_generation(self.request, progress=self.progress.emit, log=self.log.emit)
        except Exception as exc:
            message = str(exc)
            failed_result = getattr(exc, "result", None)
            if failed_result is not None:
                message = f"{message} (diagnostics: {failed_result.run_dir})"
            self.failed.emit(message)
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()
