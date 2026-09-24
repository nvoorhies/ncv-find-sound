"""`find-sound doctor [--fix]`: is everything a search needs in place?

Checks the config, the library directories, each embedding endpoint (a text request, plus a
real audio clip for the audio endpoint, which catches a wrong `audio_format`), and the index.
With --fix it starts the bundled embedding server in the background when a local endpoint is
down, then brings the index up to date.
"""

from __future__ import annotations

import time

from .config import Config
from .indexer import EndpointDown, IndexBusy, Indexer, close_clients, open_clients
from .server_control import ServerUnavailable, endpoints, ensure_running, is_local, log_path, port_open, probe
from .store import Store


def _say(ok: bool | None, what: str, detail: str = "") -> None:
    tag = {True: "[ok]", False: "[!!]", None: "[..]"}[ok]
    print(f"{tag} {what}" + (f": {detail}" if detail else ""), flush=True)


async def run(cfg: Config, fix: bool = False) -> int:
    problems = 0
    _say(True, "config", str(cfg.source) if cfg.source else "no config file; using defaults and FIND_SOUND_* env vars")

    if not cfg.library:
        _say(False, "library", "none configured (set `library` in the config or FIND_SOUND_LIBRARY)")
        problems += 1
    for root in cfg.library:
        ok = root.is_dir()
        problems += not ok
        _say(ok, "library", str(root) if ok else f"{root} does not exist")

    if fix:
        try:
            await ensure_running(cfg, on_start=lambda url, pid: _say(
                None, "embedding server", f"started for {url} (pid {pid}, log {log_path(cfg)}); loading models"))
        except ServerUnavailable as e:
            _say(False, "embedding server", str(e))
            problems += 1
    for label, ep, audio in endpoints(cfg):
        if not fix and cfg.autostart_server and is_local(ep.base_url) and not port_open(ep.base_url):
            idle = f"; exits after {cfg.server_idle_timeout / 60:.0f} min idle" if cfg.server_idle_timeout else ""
            _say(None, label, f"{ep.model} at {ep.base_url}: asleep, starts on demand (~20 s){idle}")
            continue
        err = await probe(ep, audio)
        problems += err is not None
        detail = f"{ep.model} at {ep.base_url}" + ("" if err is None else f"\n    {err}")
        if err and not fix and is_local(ep.base_url):
            detail += "\n    run `find-sound doctor --fix` to start the bundled server"
        _say(err is None, label, detail)

    store = Store(cfg.index_path)
    try:
        s = store.stats(cfg.embedding.audio_model_key, cfg.text_endpoint.text_model_key)
        last = s["last_sync"]
        age = time.time() - last["finished_at"] if last else None
        stale = age is None or age > cfg.scan_interval or s["missing_vectors"]
        if fix and stale and not problems:
            audio, text = open_clients(cfg)
            indexer = Indexer(cfg, store, audio, text)
            _say(None, "index", "updating")
            try:
                report = await indexer.sync()
                print(f"     {report.summary()}", flush=True)
            except IndexBusy as e:
                _say(True, "index", f"{e}; searches still work meanwhile")
            except EndpointDown as e:
                _say(False, "index", f"update failed: {e}")
                problems += 1
            finally:
                indexer.close()
                await close_clients(audio, text)
            s = store.stats(cfg.embedding.audio_model_key, cfg.text_endpoint.text_model_key)
            last = s["last_sync"]
            age = time.time() - last["finished_at"] if last else None
        searchable = s["files"] - s["errors"] - s["missing_vectors"]
        ok = searchable > 0
        problems += not ok
        when = "never synced" if age is None else f"last synced {age / 60:.0f} min ago"
        _say(ok, "index", f"{searchable} searchable sounds, {s['missing_vectors']} waiting for embeddings, "
             f"{s['errors']} unreadable; {when} ({cfg.index_path})")
        if s["progress"] and time.time() - s["progress"]["updated_at"] < 60:
            p = s["progress"]
            _say(None, "index", f"being updated by pid {p['pid']}: {p['done']}/{p['queued']}")
    finally:
        store.close()
    return 1 if problems else 0
