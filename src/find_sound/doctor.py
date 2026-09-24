"""`find-sound doctor [--fix]`: is everything a search needs in place?

Checks the config, the library directories, each embedding endpoint (a text request, plus a
real audio clip for the audio endpoint, which catches a wrong `audio_format`), and the index.
With --fix it starts the bundled embedding server in the background when a local endpoint is
down, then brings the index up to date.
"""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .audio import to_wav_bytes
from .config import Config, EndpointConfig
from .embed_client import EmbeddingClient, EmbeddingError
from .indexer import EndpointDown, IndexBusy, Indexer, close_clients, open_clients
from .store import Store

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
SERVER_START_TIMEOUT = 600  # first start may download models


def _say(ok: bool | None, what: str, detail: str = "") -> None:
    tag = {True: "[ok]", False: "[!!]", None: "[..]"}[ok]
    print(f"{tag} {what}" + (f": {detail}" if detail else ""), flush=True)


async def _probe(ep: EndpointConfig, audio: bool) -> str | None:
    """None if the endpoint embeds (text, and audio when asked), else the error."""
    quick = replace(ep, timeout=10.0)
    client = EmbeddingClient(quick, attempts=1)  # a health check, not a job: no backoff
    try:
        v = await client.embed_texts(["a dog barking"])
        if audio:
            t = np.arange(ep.sample_rate // 2) / ep.sample_rate
            await client.embed_audio([to_wav_bytes((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), ep.sample_rate)])
        return None if v.shape[0] == 1 else "empty response"
    except EmbeddingError as e:
        return str(e)
    finally:
        await client.aclose()


def _endpoints(cfg: Config) -> list[tuple[str, EndpointConfig, bool]]:
    eps = [("audio+query model", cfg.embedding, True)]
    if cfg.text_embedding:
        eps.append(("name/tag text model", cfg.text_embedding, False))
    return eps


def _start_server(cfg: Config, base_url: str) -> subprocess.Popen | str:
    """Launch find-sound-embed-server for every configured model on this base_url."""
    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        return "the server extra is not installed here (uv sync --extra server, or use bin/find-sound)"
    url = urlparse(base_url)
    cmd = [sys.executable, "-m", "find_sound.embed_server", "--host", url.hostname or "127.0.0.1",
           "--port", str(url.port or 80), "--clap-model", "none"]
    if cfg.embedding.base_url == base_url:
        cmd[-1] = cfg.embedding.model
    if cfg.text_embedding and cfg.text_embedding.base_url == base_url:
        cmd += ["--text-model", cfg.text_embedding.model]
    log_path = Path(cfg.index_path).parent / "embed-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)  # outlives this command
    proc.log_path = log_path  # type: ignore[attr-defined]
    return proc


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

    started: dict[str, subprocess.Popen] = {}
    for label, ep, audio in _endpoints(cfg):
        err = await _probe(ep, audio)
        if err and fix and urlparse(ep.base_url).hostname in LOCAL_HOSTS:
            if ep.base_url not in started:
                proc = _start_server(cfg, ep.base_url)
                if isinstance(proc, str):
                    _say(False, f"{label} ({ep.model})", f"down, and cannot start the bundled server: {proc}")
                    problems += 1
                    continue
                started[ep.base_url] = proc
                _say(None, f"{label}", f"started the embedding server (pid {proc.pid}, log {proc.log_path}); "
                     "waiting for it to load its models")
            proc = started[ep.base_url]
            deadline = time.monotonic() + SERVER_START_TIMEOUT
            while err and time.monotonic() < deadline and proc.poll() is None:
                await asyncio.sleep(3)
                err = await _probe(ep, audio)
            if err and proc.poll() is not None:
                tail = Path(proc.log_path).read_text(errors="replace").strip().splitlines()[-5:]
                err = f"the server exited ({proc.returncode}):\n    " + "\n    ".join(tail)
        ok = err is None
        problems += not ok
        detail = f"{ep.model} at {ep.base_url}" + ("" if ok else f"\n    {err}")
        if not ok and not fix and urlparse(ep.base_url).hostname in LOCAL_HOSTS:
            detail += "\n    run `find-sound doctor --fix` to start the bundled server"
        _say(ok, label, detail)

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
