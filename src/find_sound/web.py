"""Web UI and JSON API. Searches return playable/downloadable files; a background task keeps
the index in step with the library.

    GET  /                        search page
    GET  /api/search?q=...&k=5    results (kind/bpm/dur/group params mirror the CLI flags)
    GET  /api/audio/{id}          the file itself (?download=1 for an attachment)
    GET  /api/status              index size, kinds, indexing progress
    POST /api/rescan              scan the library now instead of waiting for the interval
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import threading
import time
from contextlib import asynccontextmanager, suppress
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .config import Config
from .embed_client import EmbeddingError
from .indexer import Indexer, close_clients, open_clients
from .search import Filters, SearchIndex, parse_kinds, parse_range, run_search
from .server_control import ServerUnavailable, endpoints, ensure_running, is_local, port_open
from .store import Store

# Reload the in-memory matrix at most this often while an indexer is writing.
RELOAD_SECONDS = 3.0


class _State:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lock = threading.Lock()  # guards the read-side SQLite connection and `index`
        self.index: SearchIndex | None = None
        self.loaded_at = 0.0
        self.stop = asyncio.Event()
        self.wake = asyncio.Event()
        self.indexer: Indexer | None = None

    def current_index(self) -> SearchIndex:
        with self.lock:
            stale = self.index is None or (
                self.store.version() != self.index.version and time.monotonic() - self.loaded_at > RELOAD_SECONDS
            )
            if stale:
                self.index = SearchIndex(
                    self.store, self.cfg.embedding.audio_model_key, self.cfg.text_endpoint.text_model_key
                )
                self.loaded_at = time.monotonic()
            return self.index

    def file_row(self, file_id: int):
        with self.lock:
            return self.store.get_file(file_id)

    def stats(self) -> dict:
        with self.lock:
            return self.store.stats(self.cfg.embedding.audio_model_key, self.cfg.text_endpoint.text_model_key)

    def relative(self, path: str) -> str:
        for root in self.cfg.library:
            if path.startswith(str(root) + os.sep):
                return os.path.relpath(path, root)
        return path


def create_app(cfg: Config, background_index: bool = True, transport=None) -> FastAPI:
    state = _State(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.store = Store(cfg.index_path)
        state.audio, state.text = open_clients(cfg, transport=transport)
        task = None
        if background_index and cfg.library:
            state.indexer = Indexer(cfg, Store(cfg.index_path), state.audio, state.text)
            task = asyncio.create_task(state.indexer.watch(stop=state.stop, wake=state.wake))
        try:
            yield
        finally:
            state.stop.set()
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            if state.indexer:
                state.indexer.close()
                state.indexer.store.close()
            await close_clients(state.audio, state.text)
            state.store.close()

    app = FastAPI(title="find-sound", lifespan=lifespan)
    page = resources.files("find_sound").joinpath("static/index.html").read_text()

    @app.get("/", response_class=HTMLResponse)
    async def index_page():
        return page

    @app.get("/api/search")
    async def api_search(
        q: str = "", k: int | None = None, kind: str | None = None, bpm: str | None = None,
        dur: str | None = None, group: bool = True,
    ):
        try:
            extra = Filters(
                kinds=parse_kinds(kind) if kind else None,
                bpm=parse_range(bpm, tolerance=cfg.search.bpm_tolerance) if bpm else None,
                duration=parse_range(dur) if dur else None,
            )
            index = await asyncio.to_thread(state.current_index)
            await ensure_running(cfg)  # wakes the embedding server if it idled out (~20 s)
            t = time.perf_counter()
            results, filters = await run_search(
                index, state.audio, state.text, q, cfg.search, k=min(k or cfg.search.k, 50), extra=extra,
                group_variants=group,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except EmbeddingError as e:
            raise HTTPException(502, f"embedding endpoint: {e}") from e
        except ServerUnavailable as e:
            raise HTTPException(503, str(e)) from e

        def file_ref(path: str) -> dict:
            fid = index.id_by_path.get(path)
            return {"path": path, "name": Path(path).name, "rel": state.relative(path),
                    "url": f"/api/audio/{fid}", "download_url": f"/api/audio/{fid}?download=1"}

        return {
            "query": q,
            "filters": filters.describe(),
            "searched": len(index),
            "ms": round((time.perf_counter() - t) * 1000),
            "results": [
                {**r.to_dict(), **file_ref(r.path),
                 "variants": [file_ref(p) for p in r.variants], "duplicates": r.duplicates}
                for r in results
            ],
        }

    @app.get("/api/audio/{file_id}")
    async def audio(file_id: int, download: bool = False):
        row = await asyncio.to_thread(state.file_row, file_id)
        if row is None or not os.path.isfile(row["path"]):
            raise HTTPException(404, "no such file in the index")
        path = row["path"]
        media = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if download:
            return FileResponse(path, media_type=media, filename=Path(path).name)
        return FileResponse(path, media_type=media)

    @app.get("/api/status")
    async def status():
        s = await asyncio.to_thread(state.stats)
        idx = state.index
        # Written by whichever process is indexing (this one or a CLI `index`/`watch`); a
        # progress record nobody has touched for a minute belongs to a sync that died.
        progress = s.pop("progress")
        if progress and time.time() - progress.get("updated_at", 0) > 60:
            progress = None
        return {
            **s,
            "searchable": len(idx) if idx is not None else None,
            "library": [str(p) for p in cfg.library],
            "model": cfg.embedding.model,
            "indexing": progress is not None,
            "progress": progress,
            # "asleep": a local server that exited when idle; the next search starts it.
            "embedding_server": "running" if all(
                port_open(ep.base_url) for _, ep, _ in endpoints(cfg) if is_local(ep.base_url)
            ) else ("asleep" if cfg.autostart_server else "down"),
        }

    @app.post("/api/rescan", status_code=202)
    async def rescan():
        if state.indexer is None:
            raise HTTPException(409, "background indexing is off (started with --no-index or no library)")
        state.wake.set()
        return {"ok": True}

    return app
