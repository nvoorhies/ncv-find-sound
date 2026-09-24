"""Reference embedding server: a CLAP model behind an OpenAI-compatible /embeddings endpoint.

Use it when no audio-capable embedding server is running already. It needs the `server` extra:

    uv run --extra server find-sound-embed-server --model laion/larger_clap_general --port 7997

Request shapes it accepts (all return the standard OpenAI embeddings response):

    {"input": "text" | ["text", ...]}                                    text (modality defaults to text)
    {"input": ["data:audio/wav;base64,...", ...], "modality": "audio"}   audio, Infinity style
    {"messages": [{"role": "user", "content": [part]}]}                  one item, vLLM chat style, where
        part is {"type": "text"}, {"type": "input_audio", "input_audio": {"data": b64}} or
        {"type": "audio_url", "audio_url": {"url": "data:..."}}
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


class ClapEmbedder:
    def __init__(self, model_id: str, device: str = "auto"):
        import torch
        from transformers import AutoFeatureExtractor, AutoTokenizer, ClapModel

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
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


def _data_uri_bytes(uri: str) -> bytes:
    if not uri.startswith("data:") or "," not in uri:
        raise HTTPException(400, "audio inputs must be base64 data URIs (data:audio/wav;base64,...)")
    header, payload = uri.split(",", 1)
    if not header.endswith(";base64"):
        raise HTTPException(400, "audio data URIs must be base64-encoded")
    return base64.b64decode(payload)


def create_app(embedder: ClapEmbedder, served_name: str | None = None) -> FastAPI:
    app = FastAPI(title="find-sound CLAP embedding server")
    name = served_name or embedder.model_id

    def respond(vecs: np.ndarray, n_tokens: int = 0) -> JSONResponse:
        return JSONResponse({
            "object": "list",
            "model": name,
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
        })

    def embed(body: dict):
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
                raise HTTPException(400, "CLAP embeds text or audio, not both in one item")
            if clips:
                vec = embedder.embed_audio([embedder.decode(c) for c in clips]).mean(axis=0, keepdims=True)
            elif texts:
                vec = embedder.embed_text([" ".join(texts)])
            else:
                raise HTTPException(400, "empty messages")
            return respond(vec / np.linalg.norm(vec, axis=1, keepdims=True))

        items = body.get("input")
        if isinstance(items, str):
            items = [items]
        if not items or not isinstance(items, list):
            raise HTTPException(400, "`input` must be a string or a non-empty list")
        modality = body.get("modality", "text")
        if modality == "text":
            return respond(np.concatenate([embedder.embed_text(items[i : i + 64]) for i in range(0, len(items), 64)]))
        if modality == "audio":
            clips = [embedder.decode(_data_uri_bytes(u)) for u in items]
            return respond(np.concatenate([embedder.embed_audio(clips[i : i + 32]) for i in range(0, len(clips), 32)]))
        raise HTTPException(400, f"unsupported modality {modality!r}")

    # Plain `def` endpoints run in FastAPI's thread pool, keeping the event loop free.
    @app.post("/embeddings")
    @app.post("/v1/embeddings")
    def embeddings(body: dict):
        t = time.perf_counter()
        try:
            return embed(body)
        except HTTPException:
            raise
        except (sf.LibsndfileError, ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}") from e
        finally:
            log.debug("embeddings request in %.1f ms", (time.perf_counter() - t) * 1000)

    @app.get("/models")
    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "find-sound"}]}

    @app.get("/health")
    def health():
        return {"ok": True, "model": name, "device": embedder.device, "sample_rate": embedder.sample_rate}

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="laion/larger_clap_general", help="Hugging Face CLAP model id")
    ap.add_argument("--served-name", help="model name reported in responses (default: --model)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7997)
    ap.add_argument("--device", default="auto", help="cuda, cpu or auto")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    log.info("loading %s", args.model)
    embedder = ClapEmbedder(args.model, args.device)
    log.info("loaded on %s, sample rate %d", embedder.device, embedder.sample_rate)
    uvicorn.run(create_app(embedder, args.served_name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
