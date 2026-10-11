"""Qt relay between the GUI and the backend process."""

import multiprocessing as mp
import threading

from PySide6.QtCore import QObject, Signal, Slot

from .retrieval_backend import backend_process


class BackendWorker(QObject):
    succeeded = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str)
    reset = Signal()

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.process = None
        self.connection = None
        self.stopping = threading.Event()

    def start_backend(self):
        # CUDA requires a fresh interpreter, not a fork of the GUI process.
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=backend_process,
            args=(self.settings, child),
            name="VideoRetrievalBackend",
        )
        try:
            self.process.start()
        finally:
            child.close()

    def stop_backend(self):
        if self.process is not None and self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if not self.process.is_alive():
                self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = None
        self.connection = None

    @Slot(str, object)
    def execute(self, action, payload):
        try:
            if self.stopping.is_set():
                return
            if self.process is None:
                self.start_backend()
            self.connection.send((action, payload))
            while not self.stopping.is_set():
                if self.connection.poll(0.1):
                    kind, data = self.connection.recv()
                    if kind == "progress":
                        self.progress.emit(data)
                    elif kind == "succeeded":
                        self.succeeded.emit(action, data)
                        return
                    elif kind == "failed":
                        self.failed.emit(action, data)
                        return
                elif not self.process.is_alive():
                    raise RuntimeError(
                        "Backend exited unexpectedly. Reload the encoders."
                    )
        except Exception as exc:
            self.stop_backend()
            self.reset.emit()
            self.failed.emit(action, f"{type(exc).__name__}: {exc}")
        finally:
            if self.stopping.is_set():
                self.stop_backend()

    @Slot()
    def shutdown(self):
        self.stopping.set()
        self.stop_backend()
