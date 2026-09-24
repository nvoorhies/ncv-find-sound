"""Async client for OpenAI-compatible /embeddings endpoints, with audio input.

Text goes out as a standard OpenAI embeddings request. Audio has no standard shape, so it is
sent in the format named by EndpointConfig.audio_format (see config.py).
"""

from __future__ import annotations

import asyncio
import base64

import httpx
import numpy as np

from .config import EndpointConfig

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
RETRY_DELAY = 1.0  # seconds before the first retry; doubles each attempt


class EmbeddingError(RuntimeError):
    pass


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class EmbeddingClient:
    def __init__(self, ep: EndpointConfig, transport: httpx.AsyncBaseTransport | None = None, attempts: int = 4):
        self.ep = ep
        self.attempts = attempts
        headers = {"Authorization": f"Bearer {ep.api_key}"} if ep.api_key else {}
        self._http = httpx.AsyncClient(timeout=ep.timeout, headers=headers, transport=transport)
        self._sem = asyncio.Semaphore(ep.concurrency)
        self.requests = 0  # for stats and tests

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    @property
    def url(self) -> str:
        return self.ep.base_url.rstrip("/") + "/embeddings"

    async def _post(self, body: dict, expect: int) -> np.ndarray:
        body = {"model": self.ep.model, "encoding_format": "float", **body}
        if self.ep.dimensions:
            body["dimensions"] = self.ep.dimensions
        delay = RETRY_DELAY
        for attempt in range(self.attempts):
            try:
                async with self._sem:
                    self.requests += 1
                    r = await self._http.post(self.url, json=body)
                if r.status_code not in RETRY_STATUS:
                    break
                err = f"HTTP {r.status_code}: {r.text[:300]}"
            except httpx.TransportError as e:
                err = f"{type(e).__name__}: {e}"
            if attempt == self.attempts - 1:
                raise EmbeddingError(f"{self.url}: {err}")
            await asyncio.sleep(delay)
            delay *= 2
        if r.status_code >= 400:
            raise EmbeddingError(f"{self.url}: HTTP {r.status_code}: {r.text[:500]}")
        try:
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
        except (ValueError, KeyError, TypeError) as e:
            raise EmbeddingError(f"{self.url}: unexpected response: {r.text[:300]}") from e
        if len(vecs) != expect:
            raise EmbeddingError(f"{self.url}: asked for {expect} embeddings, got {len(vecs)}")
        return normalize(vecs)

    async def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), np.float32)
        n = self.ep.text_batch_size
        batches = [texts[i : i + n] for i in range(0, len(texts), n)]
        out = await asyncio.gather(*(self._post({"input": b}, len(b)) for b in batches))
        return np.concatenate(out)

    async def embed_query(self, query: str) -> np.ndarray:
        return (await self.embed_texts([self.ep.query_template.format(query=query)]))[0]

    async def embed_audio(self, clips: list[bytes]) -> np.ndarray:
        """One vector per WAV-encoded clip."""
        if not clips:
            return np.zeros((0, 0), np.float32)
        b64 = [base64.b64encode(c).decode("ascii") for c in clips]
        fmt = self.ep.audio_format
        if fmt == "infinity":
            return await self._post({"input": [f"data:audio/wav;base64,{b}" for b in b64], "modality": "audio"}, len(clips))
        if fmt in ("messages", "messages-url"):
            # Chat-style embedding requests carry one conversation each, so one clip per request.
            def part(b):
                if fmt == "messages":
                    return {"type": "input_audio", "input_audio": {"data": b, "format": "wav"}}
                return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b}"}}

            out = await asyncio.gather(
                *(self._post({"messages": [{"role": "user", "content": [part(b)]}]}, 1) for b in b64)
            )
            return np.concatenate(out)
        raise EmbeddingError(f"unknown audio_format {fmt!r} (expected infinity, messages or messages-url)")
