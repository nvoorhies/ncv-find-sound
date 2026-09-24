"""The bundled server's routing and request handling, with stand-in models (no torch)."""

import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from find_sound.embed_server import create_app


class FakeClap:
    audio, device = True, "cpu"

    def embed_text(self, texts):
        return np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1))

    def embed_audio(self, clips):
        return np.tile([0.0, 1.0, 0.0, 0.0], (len(clips), 1))

    def decode(self, data):
        x, _ = sf.read(io.BytesIO(data), dtype="float32")
        return x


class FakeText:
    audio, device = False, "cpu"

    def embed_text(self, texts):
        return np.tile([0.0, 0.0, 3.0, 4.0], (len(texts), 1))


def wav_uri() -> str:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(4800, np.float32), 48000, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def client():
    return TestClient(create_app({"clap": FakeClap(), "qwen": FakeText()}))


def test_routes_by_model_and_normalizes():
    c = client()
    r = c.post("/v1/embeddings", json={"model": "qwen", "input": ["a", "b"]}).json()
    assert r["model"] == "qwen" and np.allclose(r["data"][1]["embedding"], [0, 0, 0.6, 0.8])
    r = c.post("/embeddings", json={"model": "clap", "input": "a"}).json()
    assert np.allclose(r["data"][0]["embedding"], [1, 0, 0, 0])
    assert c.post("/embeddings", json={"model": "nope", "input": "a"}).status_code == 404
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} == {"clap", "qwen"}


def test_dimensions_truncate_then_renormalize():
    r = client().post("/embeddings", json={"model": "qwen", "input": "a", "dimensions": 3}).json()
    assert np.allclose(r["data"][0]["embedding"], [0, 0, 1])


def test_audio_formats():
    c = client()
    r = c.post("/embeddings", json={"model": "clap", "input": [wav_uri(), wav_uri()], "modality": "audio"}).json()
    assert len(r["data"]) == 2 and np.allclose(r["data"][0]["embedding"], [0, 1, 0, 0])
    b64 = wav_uri().split(",", 1)[1]
    msg = {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]}
    r = c.post("/embeddings", json={"model": "clap", "messages": [msg]}).json()
    assert np.allclose(r["data"][0]["embedding"], [0, 1, 0, 0])


def test_text_model_rejects_audio_and_bad_input():
    c = client()
    assert c.post("/embeddings", json={"model": "qwen", "input": [wav_uri()], "modality": "audio"}).status_code == 400
    assert c.post("/embeddings", json={"model": "clap", "input": ["not a uri"], "modality": "audio"}).status_code == 400
    assert c.post("/embeddings", json={"model": "clap", "input": []}).status_code == 400
