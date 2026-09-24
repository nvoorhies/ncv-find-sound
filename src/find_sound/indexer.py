"""Keeps the index in step with the library directories.

A sync scans the library, drops files that disappeared, and runs changed or new files through:

    hash (thread) -> analyze + cut segments (process pool) -> embed audio (HTTP) -> store
                                                           -> describe -> embed text (batched)

Every step is cached: an unchanged file (same size and mtime) is skipped, and a changed or
renamed file with known content reuses its analysis and audio vector. So after the first run,
a sync costs one directory walk plus work proportional to what changed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial

import numpy as np

from .audio import ANALYSIS_REVISION, analyze_file, hash_file, init_worker
from .config import Config
from .describe import KIND_PROMPTS, classify_audio, describe, kind_from_path, kind_rules_version
from .embed_client import EmbeddingClient, EmbeddingError, normalize
from .store import IndexLock, Store, audio_key, text_key

log = logging.getLogger("find_sound")

MAX_CONSECUTIVE_EMBED_FAILURES = 5


class IndexBusy(RuntimeError):
    pass


class EndpointDown(RuntimeError):
    pass


@dataclass
class SyncReport:
    scanned: int = 0
    queued: int = 0
    done: int = 0
    removed: int = 0
    audio_embedded: int = 0
    audio_reused: int = 0
    text_embedded: int = 0
    failed: int = 0
    started: float = field(default_factory=time.time)
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"scanned {self.scanned}, updated {self.done}/{self.queued}, removed {self.removed}, "
            f"audio embedded {self.audio_embedded} (reused {self.audio_reused}), "
            f"text embedded {self.text_embedded}, failed {self.failed} in {self.seconds:.1f}s"
        )


def scan(roots, extensions, settle_seconds: float = 0.0) -> tuple[dict[str, tuple[str, int, int]], set[str]]:
    """Every audio file under the roots: ({path: (root, size, mtime_ns)}, unsettled paths).

    Files modified in the last `settle_seconds` may still be being copied; they are reported as
    unsettled and left for the next scan.
    """
    found, unsettled = {}, set()
    exts = tuple(extensions)
    settled = time.time_ns() - int(settle_seconds * 1e9)
    for root in roots:
        root = str(root)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            # Hidden dirs and macOS resource-fork junk (__MACOSX/._foo.wav) are not audio.
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__MACOSX")
            for name in filenames:
                if name.startswith(".") or not name.lower().endswith(exts):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                if st.st_mtime_ns > settled:
                    unsettled.add(path)
                elif st.st_size > 0:
                    found.setdefault(path, (root, st.st_size, st.st_mtime_ns))
    return found, unsettled


class Indexer:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        audio: EmbeddingClient,
        text: EmbeddingClient,
        executor: Executor | None = None,
        on_progress=None,
    ):
        self.cfg = cfg
        self.store = store
        self.audio = audio
        self.text = text
        self._executor = executor
        self._own_executor = executor is None
        self.on_progress = on_progress
        self.lock = IndexLock(cfg.index_path)
        self.audio_model = cfg.embedding.audio_model_key
        self.text_model = cfg.text_endpoint.text_model_key
        self.running = False
        self.last_report: SyncReport | None = None

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                self.cfg.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker
            )
        return self._executor

    def close(self) -> None:
        if self._own_executor and self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None

    # -- public ----------------------------------------------------------------------------

    async def sync(self, limit: int | None = None) -> SyncReport:
        if not self.cfg.library:
            raise ValueError("no library directories configured (set `library` in find-sound.toml)")
        if not self.lock.acquire():
            raise IndexBusy(f"another indexer (pid {self.lock.holder()}) is updating {self.cfg.index_path}")
        self.running = True
        try:
            return await self._sync(limit)
        finally:
            self.running = False
            self.lock.release()

    async def watch(
        self, interval: float | None = None, stop: asyncio.Event | None = None, wake: asyncio.Event | None = None
    ) -> None:
        """Sync now, then every `interval` seconds (or when `wake` is set) until `stop` is set."""
        interval = interval or self.cfg.scan_interval
        stop = stop or asyncio.Event()
        wake = wake or asyncio.Event()
        while not stop.is_set():
            wake.clear()
            try:
                report = await self.sync()
                if report.queued or report.removed:
                    log.info("sync: %s", report.summary())
            except IndexBusy as e:
                log.info("skipping sync: %s", e)
            except (EndpointDown, EmbeddingError) as e:
                log.warning("sync stopped, embedding endpoint unavailable: %s", e)
            except Exception:
                log.exception("sync failed")
            waits = [asyncio.create_task(stop.wait()), asyncio.create_task(wake.wait())]
            await asyncio.wait(waits, timeout=interval, return_when=asyncio.FIRST_COMPLETED)
            for t in waits:
                t.cancel()

    # -- internals -------------------------------------------------------------------------

    async def _sync(self, limit: int | None) -> SyncReport:
        report = self.last_report = SyncReport()
        self._report = report
        self._pending_text: dict[str, str] = {}
        self._inflight: dict[str, asyncio.Future] = {}
        self._embed_failures = 0
        self._last_checkpoint = time.monotonic()

        present, unsettled = await asyncio.to_thread(
            scan, self.cfg.library, self.cfg.extensions, self.cfg.settle_seconds
        )
        report.scanned = len(present)
        known = {
            r["path"]: r
            for r in self.store.db.execute("SELECT f.*, c.tags FROM files f LEFT JOIN content c ON c.hash = f.hash")
        }
        removed = [p for p in known if p not in present and p not in unsettled]
        if removed:
            self.store.delete_paths(removed)
            self.store.commit()
            report.removed = len(removed)

        missing = self.store.paths_missing_vectors(self.audio_model, self.text_model)
        # Analysis improved since this content was analysed. Only clips long enough for a tempo
        # change results; the rest just get their revision bumped.
        self.store.db.execute(
            "UPDATE content SET analysis_rev = ? WHERE analysis_rev < ? AND duration < ?",
            (ANALYSIS_REVISION, ANALYSIS_REVISION, self.cfg.bpm_min_seconds),
        )
        stale = {r[0] for r in self.store.db.execute("SELECT hash FROM content WHERE analysis_rev < ?", (ANALYSIS_REVISION,))}
        todo = []
        for path, (root, size, mtime) in sorted(present.items()):
            row = known.get(path)
            if row is None or row["size"] != size or row["mtime_ns"] != mtime:
                todo.append((path, root, size, mtime, None))
            elif row["error"] is None and (
                path in missing
                or row["hash"] in stale
                # Description rules changed since this file was indexed: refresh the text side.
                or row["description"] != describe(path, root, json.loads(row["tags"] or "{}"))
            ):
                todo.append((path, root, size, mtime, row["hash"]))
        if limit is not None:
            todo = todo[:limit]
        report.queued = len(todo)
        self._kinds_changed = False
        if todo or self.store.get_meta("kind_rules") != kind_rules_version():
            self._labels = await self._kind_label_vectors()
            self._reclassify_if_rules_changed()
        if not todo:
            self._finish(report)
            return report
        self._publish_progress()

        queue: asyncio.Queue = asyncio.Queue()
        for item in todo:
            queue.put_nowait(item)

        async def worker():
            while not queue.empty():
                await self._process(*queue.get_nowait())
                report.done += 1
                self._checkpoint()
                if self.on_progress:
                    self.on_progress(report)

        try:
            async with asyncio.TaskGroup() as tg:
                for _ in range(min(len(todo), self.cfg.workers * 2)):
                    tg.create_task(worker())
            await self._flush_text(force=True)
        except* EndpointDown as eg:
            raise eg.exceptions[0] from None
        finally:
            self.store.commit()
            self._finish(report)
        return report

    def _finish(self, report: SyncReport) -> None:
        report.seconds = time.time() - report.started
        self.store.set_meta("last_sync", {"finished_at": time.time(), **asdict(report)})
        self.store.set_meta("progress", None)
        self.store.commit()
        if report.done or report.removed or self._kinds_changed:
            self.store.bump_version()

    def _checkpoint(self) -> None:
        self.store.commit()
        # Bumping the version makes a running web server reload its matrix; don't do it per file.
        if time.monotonic() - self._last_checkpoint > 5:
            self._last_checkpoint = time.monotonic()
            self._publish_progress()
            self.store.bump_version()

    def _publish_progress(self) -> None:
        """Progress in the index itself, so `serve` can show a sync run by another process."""
        r = self._report
        self.store.set_meta("progress", {"pid": os.getpid(), "done": r.done, "queued": r.queued, "updated_at": time.time()})
        self.store.commit()

    async def _kind_label_vectors(self) -> dict[str, np.ndarray]:
        """Mean text embedding of each kind's prompts, in the audio model's space (cached)."""
        model = self.cfg.embedding.text_model_key
        prompts = [p for ps in KIND_PROMPTS.values() for p in ps]
        missing = [p for p in prompts if not self.store.has_vector("label:" + p, model)]
        if missing:
            try:
                vecs = await self.audio.embed_texts(missing)
            except EmbeddingError as e:
                # First request of a sync: failing here means the endpoint is down or misconfigured.
                raise EndpointDown(str(e)) from e
            for p, v in zip(missing, vecs):
                self.store.put_vector("label:" + p, model, v)
            self.store.commit()
        return {
            kind: normalize(np.mean([self.store.get_vector("label:" + p, model) for p in ps], axis=0))
            for kind, ps in KIND_PROMPTS.items()
        }

    def _kind(self, path: str, root: str, vec: np.ndarray | None, duration: float | None):
        if kind := kind_from_path(path, root):
            return kind, "path"
        if vec is None:
            return None, None
        return classify_audio(vec[None, :], np.array([duration if duration is not None else np.nan]), self._labels)[0], "model"

    def _reclassify_if_rules_changed(self) -> None:
        """Re-derive every file's kind from stored vectors when the rules or prompts change."""
        version = kind_rules_version()
        if self.store.get_meta("kind_rules") == version:
            return
        rows = self.store.db.execute(
            """SELECT f.id, f.path, f.root, f.kind, f.kind_source, c.duration, a.vec
               FROM files f JOIN content c ON c.hash = f.hash
               JOIN vectors a ON a.key = 'audio:' || f.hash AND a.model = ?
               WHERE f.error IS NULL""",
            (self.audio_model,),
        ).fetchall()
        if rows:
            vecs = np.stack([np.frombuffer(r["vec"], np.float32) for r in rows])
            durs = np.array([r["duration"] if r["duration"] is not None else np.nan for r in rows])
            model_kinds = classify_audio(vecs, durs, self._labels)
            updates = []
            for r, mk in zip(rows, model_kinds):
                pk = kind_from_path(r["path"], r["root"])
                kind, src = (pk, "path") if pk else (mk, "model")
                if (kind, src) != (r["kind"], r["kind_source"]):
                    updates.append((kind, src, r["id"]))
            self.store.db.executemany("UPDATE files SET kind=?, kind_source=? WHERE id=?", updates)
            log.info("kind rules changed: reclassified %d of %d files", len(updates), len(rows))
        self.store.set_meta("kind_rules", version)
        self.store.commit()
        self._kinds_changed = True

    async def _embed(self, coro):
        """Await an embedding call, turning a run of failures into EndpointDown."""
        try:
            out = await coro
        except EmbeddingError:
            self._embed_failures += 1
            if self._embed_failures >= MAX_CONSECUTIVE_EMBED_FAILURES:
                raise EndpointDown(f"{MAX_CONSECUTIVE_EMBED_FAILURES} embedding requests failed in a row") from None
            raise
        self._embed_failures = 0
        return out

    async def _process(self, path: str, root: str, size: int, mtime: int, h: str | None) -> None:
        report, store, ep = self._report, self.store, self.cfg.embedding
        loop = asyncio.get_running_loop()
        try:
            if h is None:
                h = await asyncio.to_thread(hash_file, path)
            content = store.get_content(h)
            akey = audio_key(h)
            # Another worker may be embedding identical content right now (duplicate packs).
            if (fut := self._inflight.get(h)) is not None:
                await asyncio.shield(fut)
            need_vec = not store.has_vector(akey, self.audio_model)
            need_analysis = content is None or content["analysis_rev"] < ANALYSIS_REVISION
            if need_analysis or need_vec:
                fut = self._inflight[h] = loop.create_future()
                try:
                    prepared = await loop.run_in_executor(
                        self.executor,
                        partial(
                            analyze_file, path,
                            sample_rate=ep.sample_rate, segment_seconds=ep.segment_seconds,
                            max_segments=ep.max_segments, bpm_min_seconds=self.cfg.bpm_min_seconds,
                            want_analysis=need_analysis, want_segments=need_vec,
                        ),
                    )
                    if prepared.analysis:
                        store.put_content(h, prepared.analysis)
                        content = store.get_content(h)
                    if need_vec:
                        vecs = await self._embed(self.audio.embed_audio(prepared.segments))
                        store.put_vector(akey, self.audio_model, normalize(vecs.mean(axis=0)))
                        report.audio_embedded += 1
                    else:
                        report.audio_reused += 1
                finally:
                    del self._inflight[h]
                    fut.set_result(None)
            else:
                report.audio_reused += 1
        except (EndpointDown, asyncio.CancelledError):
            raise
        except EmbeddingError as e:
            # Leave the file unrecorded so the next sync retries it.
            report.failed += 1
            log.warning("embedding failed for %s: %s", path, e)
            return
        except Exception as e:
            # Undecodable or unreadable: remember the error until the file changes.
            report.failed += 1
            log.warning("cannot index %s: %s: %s", path, type(e).__name__, e)
            store.put_file(path, root, size, mtime, hash=h, error=f"{type(e).__name__}: {e}"[:500])
            return

        tags = json.loads(content["tags"] or "{}")
        desc = describe(path, root, tags)
        tkey = text_key(desc)
        kind, kind_source = self._kind(path, root, store.get_vector(akey, self.audio_model), content["duration"])
        store.put_file(path, root, size, mtime, hash=h, description=desc, text_key=tkey,
                       kind=kind, kind_source=kind_source)
        if not store.has_vector(tkey, self.text_model):
            self._pending_text[tkey] = desc
            await self._flush_text()

    async def _flush_text(self, force: bool = False) -> None:
        if not self._pending_text or (not force and len(self._pending_text) < self.cfg.text_endpoint.text_batch_size):
            return
        batch, self._pending_text = self._pending_text, {}
        try:
            vecs = await self._embed(self.text.embed_texts(list(batch.values())))
        except EmbeddingError as e:
            # The files are recorded; with no text vector they are re-queued on the next sync.
            log.warning("text embedding failed for %d descriptions: %s", len(batch), e)
            return
        for key, vec in zip(batch, vecs):
            self.store.put_vector(key, self.text_model, vec)
        self._report.text_embedded += len(batch)


def open_clients(cfg: Config, transport=None) -> tuple[EmbeddingClient, EmbeddingClient]:
    """(audio/multimodal client, text client); the same object when no separate text model is set."""
    audio = EmbeddingClient(cfg.embedding, transport=transport)
    text = EmbeddingClient(cfg.text_embedding, transport=transport) if cfg.text_embedding else audio
    return audio, text


async def close_clients(audio: EmbeddingClient, text: EmbeddingClient) -> None:
    await audio.aclose()
    if text is not audio:
        await text.aclose()
