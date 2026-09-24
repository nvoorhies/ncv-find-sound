"""A fake multimodal embedding server and a tiny synthetic sound library.

The fake model maps a tone's dominant frequency (audio) or an "<n>hz" mention (text) to a bump
on a log-frequency axis, so "440hz" lands next to a 440 Hz sine. Other words hash into separate
dimensions, which keeps the text channel meaningful for name matching.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from find_sound import embed_client
from find_sound.config import Config, EndpointConfig
from find_sound.indexer import Indexer, open_clients
from find_sound.store import Store

DIM = 48
FREQ_DIMS = 24


def freq_vec(f: float) -> np.ndarray:
    v = np.zeros(DIM)
    pos = np.log2(max(f, 50) / 100) * 4  # 100 Hz -> 0, 3.2 kHz -> 20
    v[:FREQ_DIMS] = np.exp(-0.5 * ((np.arange(FREQ_DIMS) - pos) / 1.0) ** 2)
    return v


def word_vec(text: str) -> np.ndarray:
    v = np.zeros(DIM)
    for w in re.findall(r"[a-z]+", text.lower()):
        v[FREQ_DIMS + zlib.crc32(w.encode()) % (DIM - FREQ_DIMS)] += 1
    return v


def fake_text(text: str) -> np.ndarray:
    v = word_vec(text) * 0.3
    if m := re.search(r"(\d+)\s*hz", text.lower()):
        v += freq_vec(float(m.group(1)))
    return v + 1e-3


def fake_audio(wav: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(wav), dtype="float32")
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec[1:])) + 1]
    return freq_vec(f) + 1e-3


class FakeServer:
    def __init__(self):
        self.calls = {"text": 0, "audio": 0}
        self.bodies: list[dict] = []
        self.down = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if "messages" in body:
            part = body["messages"][0]["content"][0]
            self.calls["audio"] += 1
            vecs = [fake_audio(base64.b64decode(part["input_audio"]["data"]))]
        elif body.get("modality") == "audio":
            self.calls["audio"] += 1
            vecs = [fake_audio(base64.b64decode(u.split(",", 1)[1])) for u in body["input"]]
        else:
            self.calls["text"] += 1
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            vecs = [fake_text(t) for t in items]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)]})

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def tone(path: Path, freq: float, seconds: float = 0.5, sr: int = 22050, fmt: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), sr, format=fmt)
    return path


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(embed_client, "RETRY_DELAY", 0.0)


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def library(tmp_path: Path) -> Path:
    lib = tmp_path / "lib"
    tone(lib / "UI" / "Beep Low 01.wav", 220)
    tone(lib / "UI" / "Beep Low 02.wav", 225)       # a second take: variant of 01
    tone(lib / "UI" / "Beep High.wav", 3000)
    tone(lib / "Tones" / "wav" / "Tone A440.wav", 440)
    tone(lib / "Tones" / "ogg" / "Tone A440.ogg", 440, fmt="OGG")  # same sound, other encoding
    tone(lib / "Music Loops" / "Drone 90BPM.wav", 110, seconds=8.0)
    tone(lib / "__MACOSX" / "UI" / "._Beep High.wav", 3000)  # resource-fork junk: never indexed
    return lib


@pytest.fixture
def cfg(tmp_path: Path, library: Path) -> Config:
    return Config(
        library=(library,),
        index_path=tmp_path / "index.sqlite",
        embedding=EndpointConfig(base_url="http://fake", model="fake-clap", sample_rate=22050),
        analysis_workers=2,
        settle_seconds=0,
    )


@pytest.fixture
async def make_indexer(cfg: Config, server: FakeServer):
    made = []

    def make(c: Config = cfg) -> Indexer:
        audio, text = open_clients(c, transport=server.transport)
        ix = Indexer(c, Store(c.index_path), audio, text, executor=ThreadPoolExecutor(2))
        made.append(ix)
        return ix

    yield make
    for ix in made:
        await ix.audio.aclose()
        ix.store.close()
