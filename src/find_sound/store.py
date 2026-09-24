"""On-disk index: SQLite in WAL mode, so searches can read while an indexer writes.

  content  analysis per unique file content (blake2b hash): duration, loudness, BPM, tags
  files    one row per path: size/mtime to detect changes, its content hash, description, kind
  vectors  embedding cache keyed by (key, model): "audio:<hash>", "text:<hash of description>",
           "label:<prompt>". Keeping the model in the key means switching models and back
           costs nothing, and duplicate files are embedded once.

The vector search itself is brute force in numpy (see search.py): exact, and a few ms for
100k files, so no ANN index is needed at sound-library scale.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import Analysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS content (
    hash TEXT PRIMARY KEY,
    duration REAL, sample_rate INTEGER, channels INTEGER, rms_db REAL, peak_db REAL,
    bpm REAL, bpm_confidence REAL, tags TEXT, analysis_rev INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    root TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    hash TEXT,
    description TEXT,
    text_key TEXT,
    kind TEXT,
    kind_source TEXT,
    error TEXT,
    updated_at REAL
);
CREATE INDEX IF NOT EXISTS files_hash ON files(hash);
CREATE TABLE IF NOT EXISTS vectors (
    key TEXT NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (key, model)
) WITHOUT ROWID;
"""


def text_key(description: str) -> str:
    return "text:" + hashlib.blake2b(description.encode(), digest_size=16).hexdigest()


def audio_key(content_hash: str) -> str:
    return "audio:" + content_hash


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(content)")}
        if "analysis_rev" not in cols:
            # Indexes built before analysis revisions existed hold revision-1 results.
            self.db.execute("ALTER TABLE content ADD COLUMN analysis_rev INTEGER NOT NULL DEFAULT 1")
            self.db.commit()

    def close(self) -> None:
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    # -- change tracking -------------------------------------------------------------------

    def paths_missing_vectors(self, audio_model: str, text_model: str) -> set[str]:
        """Indexed, error-free files that lack a vector for the configured models."""
        rows = self.db.execute(
            """SELECT f.path FROM files f
               LEFT JOIN vectors a ON a.key = 'audio:' || f.hash AND a.model = ?
               LEFT JOIN vectors t ON t.key = f.text_key AND t.model = ?
               WHERE f.error IS NULL AND (a.key IS NULL OR t.key IS NULL)""",
            (audio_model, text_model),
        )
        return {r[0] for r in rows}

    def version(self) -> int:
        row = self.db.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        return int(row[0]) if row else 0

    def bump_version(self) -> None:
        self.db.execute(
            "INSERT INTO meta VALUES ('version', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
        )
        self.db.commit()

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # -- content & vectors -----------------------------------------------------------------

    def get_content(self, content_hash: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM content WHERE hash=?", (content_hash,)).fetchone()

    def put_content(self, content_hash: str, a: Analysis) -> None:
        # numpy scalars would be stored as BLOBs (sqlite3 takes anything with a buffer).
        d = {k: v.item() if isinstance(v, np.generic) else v for k, v in asdict(a).items()}
        d["tags"] = json.dumps(d["tags"])
        cols = ["hash", *d]
        self.db.execute(
            f"INSERT OR REPLACE INTO content ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [content_hash, *d.values()],
        )

    def get_vector(self, key: str, model: str) -> np.ndarray | None:
        row = self.db.execute("SELECT vec FROM vectors WHERE key=? AND model=?", (key, model)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def has_vector(self, key: str, model: str) -> bool:
        return self.db.execute("SELECT 1 FROM vectors WHERE key=? AND model=?", (key, model)).fetchone() is not None

    def put_vector(self, key: str, model: str, vec: np.ndarray) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
            (key, model, np.asarray(vec, dtype=np.float32).tobytes()),
        )

    # -- files -----------------------------------------------------------------------------

    def put_file(self, path: str, root: str, size: int, mtime_ns: int, **cols) -> None:
        cols = {"hash": None, "description": None, "text_key": None, "kind": None,
                "kind_source": None, "error": None, **cols, "updated_at": time.time()}
        names = ["path", "root", "size", "mtime_ns", *cols]
        self.db.execute(
            f"INSERT INTO files ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
            f"ON CONFLICT(path) DO UPDATE SET {', '.join(f'{n}=excluded.{n}' for n in names[1:])}",
            [path, root, size, mtime_ns, *cols.values()],
        )

    def delete_paths(self, paths) -> None:
        self.db.executemany("DELETE FROM files WHERE path=?", [(p,) for p in paths])

    def get_file(self, file_id: int) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()

    def stats(self, audio_model: str, text_model: str) -> dict:
        q = lambda sql, *a: self.db.execute(sql, a).fetchone()[0]  # noqa: E731
        kinds = dict(self.db.execute("SELECT kind, COUNT(*) FROM files WHERE kind IS NOT NULL GROUP BY kind").fetchall())
        return {
            "files": q("SELECT COUNT(*) FROM files"),
            "unique_contents": q("SELECT COUNT(DISTINCT hash) FROM files WHERE error IS NULL"),
            "errors": q("SELECT COUNT(*) FROM files WHERE error IS NOT NULL"),
            "missing_vectors": len(self.paths_missing_vectors(audio_model, text_model)),
            "kinds": kinds,
            "vectors_cached": q("SELECT COUNT(*) FROM vectors"),
            "last_sync": self.get_meta("last_sync"),
            "progress": self.get_meta("progress"),
        }


class IndexLock:
    """One indexer per index file, across processes (e.g. `serve` and a manual `index`)."""

    def __init__(self, index_path: Path):
        self.path = Path(str(index_path) + ".lock")
        self._fd: int | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def holder(self) -> str:
        try:
            return self.path.read_text().strip() or "?"
        except OSError:
            return "?"

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
