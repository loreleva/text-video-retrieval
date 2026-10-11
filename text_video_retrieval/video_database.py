"""SQLite storage for video paths and embeddings."""

from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
import sqlite3

import numpy as np


class VideoDatabase:
    """Store video paths and embeddings in a SQLite database."""
    
    def __init__(self, path: str):
        """Initialize the database, creating its storage if necessary.

        Args:
            path: Path to the SQLite database file.
        """
        self.database_path = Path(path)
        self.create_database()

    @contextmanager
    def connect(self):
        """Yield a connection and close it when the context exits.

        Commit on successful exit or roll back if an exception occurs.
        """
        conn = sqlite3.connect(self.database_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create_database(self) -> None:
        """Create missing parent directories, database, and videos table."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    directory TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (directory, filename)
                )
                """,
            )

    def add(
        self,
        path: str | Path | list[str | Path],
        embedding: np.ndarray,
    ) -> None:
        """Add videos and their embeddings to the database.

        Embeddings are updated if the video entry already exists.

        Args:
            path: One video path or a list of video paths.
            embedding: Video embeddings with shape (N, D),
                matching the path order, or (D,) for one video.
        """
        paths = [path] if isinstance(path, (str, Path)) else path
        paths = [Path(p).expanduser().resolve() for p in paths]

        embedding = embedding.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        if embedding.ndim != 2:
            raise ValueError("embedding must have shape (N, D) or (D,)")

        if len(paths) != len(embedding):
            raise ValueError("Each path must have exactly one embedding")

        for p in paths:
            if not p.is_file():
                raise RuntimeError(f"Invalid path: {p}")

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO videos (directory, filename, embedding)
                VALUES (?, ?, ?)
                ON CONFLICT (directory, filename)
                DO UPDATE SET embedding = excluded.embedding
                """,
                (
                    (str(p.parent), p.name, vector.tobytes())
                    for p, vector in zip(paths, embedding)
                ),
            )

    def list_videos(self) -> list[tuple[str, str]]:
        """List stored videos.

        Returns:
            (directory, filename) pairs sorted by directory
            and filename.
        """
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT directory, filename
                FROM videos
                ORDER BY directory, filename
                """
            ).fetchall()

    def delete_entries(
        self,
        videos: Sequence[tuple[str, str]] = (),
        directories: Sequence[str] = (),
    ) -> int:
        """Remove database entries.

        Remove single videos or all the videos inside
        the given directories from the database.

        Args:
            videos: Pairs (directory, filename) to remove.
            directories: Directories whose entries should be removed.
                Subdirectories are not included.

        Returns:
            The number of entries removed.
        """
        removed = 0
        with self.connect() as conn:
            if videos:
                removed += conn.executemany(
                    "DELETE FROM videos WHERE directory = ? AND filename = ?",
                    list(videos),
                ).rowcount
            if directories:
                removed += conn.executemany(
                    "DELETE FROM videos WHERE directory = ?",
                    [(d,) for d in directories],
                ).rowcount
        return removed

    def load_embeddings(self) -> tuple[list[str], np.ndarray | None]:
        """Load video paths and their corresponding float32 embeddings.

        Returns:
            A tuple containing paths sorted by directory and an
            embeddings array of shape (N, D), in matching order.
            Returns ([], None) if the database is empty.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT directory, filename, embedding
                FROM videos
                ORDER BY directory
                """
            ).fetchall()

        if not rows:
            return [], None

        paths = [
            str(Path(directory) / filename)
            for directory, filename, _ in rows
        ]
        embeddings = np.stack(
            [
                np.frombuffer(blob, dtype=np.float32)
                for _, _, blob in rows
            ]
        )
        return paths, embeddings

    def __str__(self) -> str:
        """Return stored video paths, one per line."""
        with self.connect() as conn:
            paths = [
                str(Path(rows[0]) / rows[1])
                for rows in conn.execute(
                    """
                    SELECT directory, filename
                    FROM videos
                    ORDER BY directory
                    """
                )
            ]
        return "\n".join(paths)