"""Pure retrieval logic (no Qt): cosine ranking and thumbnails."""

from pathlib import Path

import numpy as np


def rank_videos(paths, embeddings, text_feature, k, threshold=0.3):
    """Return ([(path, score)], missing_files, below_threshold).

    Only videos with cosine similarity >= threshold are returned.
    """
    if embeddings is None or not paths:
        return [], 0, 0
    if k < 1:
        raise ValueError("k must be positive")
    videos = np.asarray(embeddings, dtype=np.float32)
    query = np.asarray(text_feature, dtype=np.float32)
    # Accept one query with shape (D,), (1, D) or (D, 1).
    if query.ndim == 2 and 1 in query.shape:
        query = query.reshape(-1)
    if query.ndim != 1:
        raise ValueError(
            "Text encoder must return one vector: (D,) or (1, D)."
        )
    if videos.ndim != 2 or len(paths) != len(videos):
        raise ValueError("Database embeddings must have shape (N, D).")
    if videos.shape[1] != query.size:
        raise ValueError("Text/video embedding dimensions do not match.")
    if not np.isfinite(query).all() or not np.isfinite(videos).all():
        raise ValueError("Embeddings contain NaN or infinity.")
    # Float64 norms avoid overflow while normalizing float32 features.
    norms = np.linalg.norm(videos.astype(np.float64), axis=1)
    query_norm = np.linalg.norm(query.astype(np.float64))
    if query_norm == 0 or np.any(norms == 0):
        raise ValueError(
            "Cannot calculate cosine similarity for zero vectors."
        )
    scores = (videos / norms[:, None]) @ (query / query_norm)
    existing = np.array(
        [i for i, p in enumerate(paths) if Path(p).is_file()],
        dtype=np.int64,
    )
    ordered = existing[np.argsort(-scores[existing], kind="stable")]
    above = ordered[scores[ordered] >= threshold]
    top = above[:k]
    return (
        [(paths[i], float(np.clip(scores[i], -1, 1))) for i in top],
        len(paths) - len(existing),
        len(existing) - len(above),
    )


def make_thumbnail(path, max_side=640):
    """Return JPEG bytes of a frame from the video, or None on failure."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 3)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        h, w = frame.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None
    except Exception:
        return None
    finally:
        cap.release()
