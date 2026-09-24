"""Start the bundled embedding server when it's needed, and only then.

With `autostart_server` on, a scan that finds new files or a search that needs a query
embedding starts the server if a local endpoint is down. The server exits again after
`server_idle_timeout` idle seconds. Scans that find nothing never touch the GPU, and a model
nobody is using doesn't hold ~9 GB of VRAM that other jobs could use.

A file lock makes sure that when several processes find the server down at once (the web UI,
a periodic scan, a CLI search), exactly one of them starts it and the others wait for it.
"""

from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import socket
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

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
START_TIMEOUT = 600.0  # the first start may download models

_process_lock: asyncio.Lock | None = None


class ServerUnavailable(RuntimeError):
    pass


def is_local(base_url: str) -> bool:
    return urlparse(base_url).hostname in LOCAL_HOSTS


def port_open(base_url: str, timeout: float = 0.5) -> bool:
    """Is something listening at base_url? The bundled server binds only after loading its
    models, so for it an open port means ready."""
    url = urlparse(base_url)
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        with socket.create_connection((url.hostname or "127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def endpoints(cfg: Config) -> list[tuple[str, EndpointConfig, bool]]:
    """(label, endpoint, embeds audio) for every endpoint the config uses."""
    eps = [("audio+query model", cfg.embedding, True)]
    if cfg.text_embedding:
        eps.append(("name/tag text model", cfg.text_embedding, False))
    return eps


async def probe(ep: EndpointConfig, audio: bool) -> str | None:
    """None if the endpoint embeds text (and a real audio clip, when asked), else the error."""
    client = EmbeddingClient(replace(ep, timeout=10.0), attempts=1)  # a health check: no backoff
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


def log_path(cfg: Config) -> Path:
    return Path(cfg.index_path).parent / "embed-server.log"


def start_server(cfg: Config, base_url: str) -> subprocess.Popen:
    """Launch find-sound-embed-server, detached, for every configured model on base_url."""
    if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
        raise ServerUnavailable("the server extra is not installed here (uv sync --extra server, or use bin/find-sound)")
    url = urlparse(base_url)
    cmd = [sys.executable, "-m", "find_sound.embed_server", "--host", url.hostname or "127.0.0.1",
           "--port", str(url.port or 80), "--idle-timeout", str(cfg.server_idle_timeout),
           "--clap-model", cfg.embedding.model if cfg.embedding.base_url == base_url else "none"]
    if cfg.text_embedding and cfg.text_embedding.base_url == base_url:
        cmd += ["--text-model", cfg.text_embedding.model]
    path = log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)  # outlives the command that started it


def _log_tail(cfg: Config, lines: int = 5) -> str:
    try:
        return "\n    ".join(log_path(cfg).read_text(errors="replace").strip().splitlines()[-lines:])
    except OSError:
        return ""


async def ensure_running(cfg: Config, on_start=None) -> list[int]:
    """Make sure every local endpoint is up, starting the bundled server where needed.

    Returns the pids of servers this call started. Raises ServerUnavailable when an endpoint is
    down and can't be started. Remote endpoints are left alone: their errors surface normally.
    """
    global _process_lock
    _process_lock = _process_lock or asyncio.Lock()
    down = sorted({ep.base_url for _, ep, _ in endpoints(cfg) if is_local(ep.base_url) and not port_open(ep.base_url)})
    if not down:
        return []
    if not cfg.autostart_server:
        raise ServerUnavailable(f"{', '.join(down)} not running (autostart_server is off; run `find-sound doctor --fix`)")

    started = []
    async with _process_lock:
        lock_file = open(Path(cfg.index_path).parent / "embed-server.lock", "a")
        try:
            await asyncio.to_thread(fcntl.flock, lock_file, fcntl.LOCK_EX)
            for base_url in down:
                if port_open(base_url):  # another process started it while we waited
                    continue
                proc = start_server(cfg, base_url)
                started.append(proc.pid)
                if on_start:
                    on_start(base_url, proc.pid)
                deadline = time.monotonic() + START_TIMEOUT
                while not port_open(base_url):
                    if proc.poll() is not None:
                        raise ServerUnavailable(f"the embedding server exited ({proc.returncode}):\n    {_log_tail(cfg)}")
                    if time.monotonic() > deadline:
                        raise ServerUnavailable(f"the embedding server did not come up in {START_TIMEOUT:.0f} s; see {log_path(cfg)}")
                    await asyncio.sleep(1)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
    return started
