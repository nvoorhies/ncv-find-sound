"""Reference embedding server: CLAP (audio + text) and optionally a text-only embedding model
behind one OpenAI-compatible /embeddings endpoint, routed by the request's `model` field.

Use it when no audio-capable embedding server is running already. It needs the `server` extra:

    uv run --extra server find-sound-embed-server --port 7997 \\
        --clap-model laion/larger_clap_general --text-model Qwen/Qwen3-Embedding-4B

Request shapes it accepts (all return the standard OpenAI embeddings response):

    {"input": "text" | ["text", ...]}                                    text (modality defaults to text)
    {"input": ["data:audio/wav;base64,...", ...], "modality": "audio"}   audio, Infinity style
    {"messages": [{"role": "user", "content": [part]}]}                  one item, vLLM chat style, where
        part is {"type": "text"}, {"type": "input_audio", "input_audio": {"data": b64}} or
        {"type": "audio_url", "audio_url": {"url": "data:..."}}

`dimensions` truncates text-model embeddings (Matryoshka-trained models such as Qwen3-Embedding
keep most of their quality at 512-1024 dims).
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import threading
import time

import numpy as np
import soundfile as sf
import soxr
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

log = logging.getLogger("find_sound.embed_server")


def _device(torch, device: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


class ClapEmbedder:
    """Audio and text in one space."""

    audio = True

    def __init__(self, model_id: str, device: str = "auto"):
        import torch
        from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel

        self.torch = torch
        self.device = _device(torch, device)
        self.model_id = model_id
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.features = AutoFeatureExtractor.from_pretrained(model_id)
        self.sample_rate = self.features.sampling_rate
        self._lock = threading.Lock()  # one forward pass on the GPU at a time

    @staticmethod
    def _pooled(out):
        # transformers>=5 returns a model output whose pooler_output is the projected embedding.
        return out if hasattr(out, "float") else out.pooler_output

    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with self._lock, self.torch.inference_mode():
            return self._pooled(self.model.get_text_features(**inputs)).float().cpu().numpy()

    def embed_audio(self, clips: list[np.ndarray]) -> np.ndarray:
        # Mel features on the CPU, outside the lock, so concurrent requests overlap.
        inputs = self.features(clips, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._lock, self.torch.inference_mode():
            return self._pooled(self.model.get_audio_features(**inputs)).float().cpu().numpy()

    def decode(self, data: bytes) -> np.ndarray:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        return x if sr == self.sample_rate else soxr.resample(x, sr, self.sample_rate).astype(np.float32)


class DecoderTextEmbedder:
    """Decoder-LM embedding models pooled on the final token (Qwen3-Embedding and relatives).

    The tokenizer appends the end-of-text token and pads on the left, so position -1 is the
    pooled token for every row. Instructions for queries are the client's job (query_template).
    """

    audio = False

    def __init__(self, model_id: str, device: str = "auto", max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = _device(torch, device)
        self.model_id = model_id
        self.max_length = max_length
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self._lock = threading.Lock()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with self._lock, self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        return hidden[:, -1].float().cpu().numpy()


def _data_uri_bytes(uri: str) -> bytes:
    if not uri.startswith("data:") or "," not in uri:
        raise HTTPException(400, "audio inputs must be base64 data URIs (data:audio/wav;base64,...)")
    header, payload = uri.split(",", 1)
    if not header.endswith(";base64"):
        raise HTTPException(400, "audio data URIs must be base64-encoded")
    return base64.b64decode(payload)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


class Activity:
    """Requests in flight and when the last one finished, for the idle watchdog."""

    def __init__(self):
        self._lock = threading.Lock()
        self.in_flight = 0
        self.last = time.monotonic()

    def begin(self) -> None:
        with self._lock:
            self.in_flight += 1

    def end(self) -> None:
        with self._lock:
            self.in_flight -= 1
            self.last = time.monotonic()

    def idle_for(self) -> float:
        with self._lock:
            return 0.0 if self.in_flight else time.monotonic() - self.last


def watch_idle(activity: Activity, server, timeout: float, poll: float = 10.0) -> None:
    """Ask the server to exit once nothing has been requested for `timeout` seconds."""
    while not getattr(server, "should_exit", False):
        time.sleep(min(poll, timeout))
        if activity.idle_for() >= timeout:
            log.info("idle for %.0f s; exiting to free the GPU", activity.idle_for())
            server.should_exit = True


def create_app(embedders: dict[str, object], activity: Activity | None = None) -> FastAPI:
    """`embedders` maps the served model name to a ClapEmbedder / DecoderTextEmbedder."""
    activity = activity or Activity()
    app = FastAPI(title="find-sound embedding server")
    default = next(iter(embedders))

    def pick(name: str | None):
        if name in embedders:
            return name, embedders[name]
        if len(embedders) == 1 or name in (None, "", "default/not-specified"):
            return default, embedders[default]
        raise HTTPException(404, f"model {name!r} is not served here; available: {', '.join(embedders)}")

    def respond(name: str, vecs: np.ndarray, dimensions: int | None = None) -> JSONResponse:
        if dimensions:
            vecs = vecs[:, :dimensions]
        vecs = _normalize(vecs)
        return JSONResponse({
            "object": "list",
            "model": name,
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })

    def needs_audio(name: str, emb) -> None:
        if not emb.audio:
            raise HTTPException(400, f"model {name!r} embeds text only")

    def embed(body: dict):
        name, emb = pick(body.get("model"))
        dims = body.get("dimensions") or None
        if "messages" in body:
            texts, clips = [], []
            for msg in body["messages"]:
                content = msg.get("content")
                for part in [{"type": "text", "text": content}] if isinstance(content, str) else content or []:
                    t = part.get("type")
                    if t == "text":
                        texts.append(part["text"])
                    elif t == "input_audio":
                        clips.append(base64.b64decode(part["input_audio"]["data"]))
                    elif t == "audio_url":
                        clips.append(_data_uri_bytes(part["audio_url"]["url"]))
                    else:
                        raise HTTPException(400, f"unsupported content part type {t!r}")
            if clips and texts:
                raise HTTPException(400, "text and audio cannot be mixed in one item")
            if clips:
                needs_audio(name, emb)
                vec = _normalize(emb.embed_audio([emb.decode(c) for c in clips])).mean(axis=0, keepdims=True)
            elif texts:
                vec = emb.embed_text([" ".join(texts)])
            else:
                raise HTTPException(400, "empty messages")
            return respond(name, vec, dims)

        items = body.get("input")
        if isinstance(items, str):
            items = [items]
        if not items or not isinstance(items, list):
            raise HTTPException(400, "`input` must be a string or a non-empty list")
        modality = body.get("modality", "text")
        if modality == "text":
            vecs = np.concatenate([emb.embed_text(items[i : i + 64]) for i in range(0, len(items), 64)])
            return respond(name, vecs, dims)
        if modality == "audio":
            needs_audio(name, emb)
            clips = [emb.decode(_data_uri_bytes(u)) for u in items]
            return respond(name, np.concatenate([emb.embed_audio(clips[i : i + 32]) for i in range(0, len(clips), 32)]), dims)
        raise HTTPException(400, f"unsupported modality {modality!r}")

    # Plain `def` endpoints run in FastAPI's thread pool, keeping the event loop free.
    @app.post("/embeddings")
    @app.post("/v1/embeddings")
    def embeddings(body: dict):
        t = time.perf_counter()
        activity.begin()
        try:
            return embed(body)
        except HTTPException:
            raise
        except (sf.LibsndfileError, ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        finally:
            activity.end()
            log.debug("embeddings request in %.1f ms", (time.perf_counter() - t) * 1000)

    @app.get("/models")
    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": n, "object": "model", "owned_by": "find-sound"} for n in embedders]}

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "models": {n: {"device": e.device, "audio": e.audio} for n, e in embedders.items()},
        }

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clap-model", "--model", dest="clap_model", default="laion/larger_clap_general",
                    help="Hugging Face CLAP model id for audio + text ('none' to skip)")
    ap.add_argument("--text-model", action="append", default=[],
                    help="text-only embedding model (e.g. Qwen/Qwen3-Embedding-4B); repeatable")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7997)
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto")
    ap.add_argument("--idle-timeout", type=float, default=0,
                    help="exit after this many seconds without requests, freeing the GPU (0 = never)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    embedders: dict[str, object] = {}
    if args.clap_model.lower() != "none":
        log.info("loading %s", args.clap_model)
        embedders[args.clap_model] = ClapEmbedder(args.clap_model, args.device)
    for m in args.text_model:
        log.info("loading %s", m)
        embedders[m] = DecoderTextEmbedder(m, args.device)
    if not embedders:
        ap.error("nothing to serve")
    activity = Activity()
    server = uvicorn.Server(uvicorn.Config(create_app(embedders, activity), host=args.host, port=args.port,
                                           log_level="warning"))
    if args.idle_timeout > 0:
        threading.Thread(target=watch_idle, args=(activity, server, args.idle_timeout), daemon=True).start()
    log.info("serving %s on %s:%d%s", ", ".join(embedders), args.host, args.port,
             f", exiting after {args.idle_timeout:.0f} s idle" if args.idle_timeout > 0 else "")
    server.run()


if __name__ == "__main__":
    main()
