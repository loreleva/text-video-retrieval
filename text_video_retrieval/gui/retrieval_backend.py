"""Backend process: owns the encoders and the database (no Qt)."""

import gc
import sys
from pathlib import Path

from .retrieval_core import make_thumbnail, rank_videos


class BackendEngine:
    def __init__(self, settings, send):
        self.settings = settings
        self.send = send
        self.databases = {}
        self.vision_encoder = None
        self.text_encoder = None

    def report(self, message):
        self.send(("progress", message))

    def get_database(self, dense=False):
        path = (
            self.settings.dense_database if dense else self.settings.database
        )
        path = str(Path(path).expanduser().resolve())
        if path not in self.databases:
            from ..video_database import VideoDatabase

            self.databases[path] = VideoDatabase(path)
        return self.databases[path]

    def unload_encoder(self, kind):
        """Drop this encoder and release unused PyTorch CUDA memory."""
        setattr(self, f"{kind}_encoder", None)
        try:
            gc.collect()
            # An encoder using PyTorch will already have imported it.
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_initialized():
                for device in range(torch.cuda.device_count()):
                    with torch.cuda.device(device):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
        except Exception as exc:
            return f"CUDA cache cleanup failed: {exc}"
        return None

    def execute(self, action, payload):
        try:
            if action == "vision":
                self.report(f"Starting vision encoder…")
                from ..vision_encoder import VisionEncoder

                self.vision_encoder = VisionEncoder(
                    self.settings.vision_config,
                    dense_sampling=bool(payload),
                )
                result = None
            elif action == "text":
                from ..text_encoder import TextEncoder
                self.report(f"Starting text encoder…")
                self.text_encoder = TextEncoder(self.settings.text_config)
                result = None
            elif action in ("unload_vision", "unload_text"):
                result = self.unload_encoder(action.removeprefix("unload_"))
            elif action == "index":
                if self.vision_encoder is None:
                    raise RuntimeError("Load the vision encoder first.")
                self.report(f"Indexing videos…")
                payload, dense = payload
                # dense_sampling is a property: no reload is needed.
                self.vision_encoder.dense_sampling = bool(dense)
                path = Path(payload).expanduser().resolve()
                if not (path.is_dir() or path.is_file()):
                    raise ValueError(f"Invalid video file or directory: {path}")
                paths, features = self.vision_encoder.encode_videos(path)
                if paths:
                    self.get_database(dense).add(paths, features)
                result = len(paths)
            elif action == "list_videos":
                result = self.get_database(bool(payload)).list_videos()
            elif action == "delete_entries":
                videos, directories, dense = payload
                database = self.get_database(dense)
                removed = database.delete_entries(videos, directories)
                result = (removed, database.list_videos())
            elif action == "search":
                if self.text_encoder is None:
                    raise RuntimeError("Load the text encoder first.")
                query, k, threshold, dense = payload
                self.report("Reading database embeddings…")
                paths, features = self.get_database(dense).load_embeddings()
                self.report(f"Read {len(paths)} database entries.")
                if not paths:
                    result = ([], 0, 0)
                else:
                    self.report("Encoding text query…")
                    text_feature = self.text_encoder.encode_text(query)
                    self.report("Text encoded. Ranking videos…")
                    matches, missing, below = rank_videos(
                        paths, features, text_feature, k, threshold
                    )
                    self.report("Creating previews…")
                    matches = [
                        (p, s, make_thumbnail(p)) for p, s in matches
                    ]
                    result = (matches, missing, below)
            else:
                raise ValueError(f"Unknown action: {action}")
            self.report(f"Finished {action} in backend.")
            self.send(("succeeded", result))
        except Exception as exc:
            self.send(("failed", f"{type(exc).__name__}: {exc}"))


def backend_process(settings, connection):
    """Spawn entry point: models never execute in a Qt worker thread."""
    engine = BackendEngine(settings, connection.send)
    try:
        while True:
            request = connection.recv()
            if request is None:
                break
            engine.execute(*request)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()
