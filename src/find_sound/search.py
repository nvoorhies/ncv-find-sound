"""Query parsing, metadata filters and ranking.

A query is free text plus optional filter tokens, so a single string carries everything
(useful from a skill or a URL):

    "tense synth music bpm:120-140"      tempo range (half/double time counts for estimated BPM)
    "footsteps on gravel dur:<1.5"       duration in seconds: <x, >x, a-b
    "kind:ambience forest night"         music | sfx | voice | ambience (comma-separate for several)
    "explosion in:foley"                 substring of the path (case-insensitive)

Ranking: cosine similarity of the query against the audio embedding and against the
path/tag description embedding, each z-scored over the candidates (the two channels live on
different scales), then mixed by the configured weights.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import SearchConfig
from .describe import KINDS, bpm_from_path, identity_key, variant_key
from .store import Store

_KIND_ALIASES = {"sound": "sfx", "sfx": "sfx", "fx": "sfx", "effect": "sfx", "music": "music", "song": "music",
                 "voice": "voice", "vo": "voice", "speech": "voice", "ambience": "ambience", "ambient": "ambience",
                 "amb": "ambience"}
_TOKEN = re.compile(r"(?<!\S)(bpm|dur|duration|len|kind|type|in|path):(\S+)", re.I)
_NUM = r"(\d+(?:\.\d+)?)\s*(ms|s|min|m|bpm)?"


@dataclass
class Filters:
    kinds: set[str] | None = None
    bpm: tuple[float, float] | None = None
    duration: tuple[float, float] | None = None
    path: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.kinds:
            bits.append("kind=" + ",".join(sorted(self.kinds)))
        if self.bpm:
            bits.append(f"bpm {self.bpm[0]:g}-{self.bpm[1]:g}")
        if self.duration:
            lo, hi = self.duration
            bits.append(f"duration {lo:g}-{'∞' if hi == float('inf') else f'{hi:g}'}s")
        bits += [f"path~{p}" for p in self.path]
        return ", ".join(bits)


def parse_range(spec: str, *, tolerance: float = 0.0) -> tuple[float, float]:
    """'120' (± tolerance), '110-130', '<2', '>=30', '500ms', '1.5m', '120bpm'."""
    s = spec.strip().lower()

    def val(num: str, unit: str | None) -> float:
        return float(num) * {"m": 60.0, "min": 60.0, "ms": 0.001}.get(unit or "", 1.0)

    if m := re.fullmatch(r"(<|>)=?" + _NUM, s):
        v = val(m.group(2), m.group(3))
        return (0.0, v) if m.group(1) == "<" else (v, float("inf"))
    if m := re.fullmatch(_NUM + r"\s*-\s*" + _NUM, s):
        a, b = val(m.group(1), m.group(2)), val(m.group(3), m.group(4))
        return (min(a, b), max(a, b))
    if m := re.fullmatch(_NUM, s):
        v = val(m.group(1), m.group(2))
        return (v * (1 - tolerance), v * (1 + tolerance))
    raise ValueError(f"cannot parse range {spec!r} (try 120, 110-130, <2, >30)")


def parse_kinds(spec: str) -> set[str]:
    kinds = set()
    for k in re.split(r"[,|+]", spec.lower()):
        if k and (kind := _KIND_ALIASES.get(k)) is None:
            raise ValueError(f"unknown kind {k!r} (one of {', '.join(KINDS)})")
        if k:
            kinds.add(kind)
    return kinds


def parse_query(query: str, cfg: SearchConfig | None = None) -> tuple[str, Filters]:
    """Split filter tokens out of a query. Returns (remaining text, filters)."""
    cfg = cfg or SearchConfig()
    f = Filters()
    for m in _TOKEN.finditer(query):
        key, val = m.group(1).lower(), m.group(2)
        if key == "bpm":
            f.bpm = parse_range(val, tolerance=cfg.bpm_tolerance)
        elif key in ("dur", "duration", "len"):
            f.duration = parse_range(val)
        elif key in ("kind", "type"):
            f.kinds = parse_kinds(val)
        else:
            f.path.append(val.lower())
    text = re.sub(r"\s+", " ", _TOKEN.sub(" ", query)).strip()
    return text, f


@dataclass
class Result:
    id: int
    path: str
    score: float
    audio_similarity: float
    text_similarity: float
    kind: str | None
    duration: float | None
    bpm: float | None
    bpm_source: str | None
    description: str
    variants: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        for k in ("score", "audio_similarity", "text_similarity"):
            d[k] = round(d[k], 4)
        return d


class SearchIndex:
    """All searchable files as numpy arrays, loaded from the store in one query."""

    def __init__(self, store: Store, audio_model: str, text_model: str):
        self.version = store.version()
        rows = store.db.execute(
            """SELECT f.id, f.path, f.hash, f.kind, f.description, c.duration, c.bpm, c.bpm_confidence,
                      a.vec AS avec, t.vec AS tvec
               FROM files f
               JOIN content c ON c.hash = f.hash
               JOIN vectors a ON a.key = 'audio:' || f.hash AND a.model = ?
               JOIN vectors t ON t.key = f.text_key AND t.model = ?
               WHERE f.error IS NULL
               ORDER BY f.path""",
            (audio_model, text_model),
        ).fetchall()
        n = len(rows)
        self.ids = np.array([r["id"] for r in rows], dtype=np.int64)
        self.paths = [r["path"] for r in rows]
        self.id_by_path = {r["path"]: r["id"] for r in rows}
        self.hashes = [r["hash"] for r in rows]
        self.kinds = np.array([r["kind"] or "" for r in rows], dtype=object)
        self.descriptions = [r["description"] or "" for r in rows]
        self.duration = np.array([r["duration"] if r["duration"] is not None else np.nan for r in rows], dtype=np.float64)
        # A tempo written in the name beats an estimate; estimates only count where there is a pulse.
        bpm, src = np.full(n, np.nan), np.array([None] * n, dtype=object)
        for i, r in enumerate(rows):
            if (named := bpm_from_path(r["path"])) is not None:
                bpm[i], src[i] = named, "name"
            elif r["bpm"] is not None and (r["bpm_confidence"] or 0) >= 0.1:
                bpm[i], src[i] = r["bpm"], "estimated"
        self.bpm, self.bpm_source = bpm, src
        self.groups = [variant_key(p) for p in self.paths]
        self.idents = [identity_key(p) for p in self.paths]
        self._by_group: dict[str, list[int]] = defaultdict(list)
        self._by_ident: dict[str, list[int]] = defaultdict(list)
        self._by_hash: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            self._by_group[self.groups[i]].append(i)
            self._by_ident[self.idents[i]].append(i)
            self._by_hash[self.hashes[i]].append(i)
        self.audio = np.stack([np.frombuffer(r["avec"], np.float32) for r in rows]) if n else np.zeros((0, 1), np.float32)
        self.text = np.stack([np.frombuffer(r["tvec"], np.float32) for r in rows]) if n else np.zeros((0, 1), np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def mask(self, f: Filters) -> np.ndarray:
        m = np.ones(len(self), dtype=bool)
        if f.kinds:
            m &= np.isin(self.kinds, list(f.kinds))
        if f.duration:
            lo, hi = f.duration
            m &= (self.duration >= lo) & (self.duration <= hi)
        if f.bpm:
            lo, hi = f.bpm
            est = self.bpm_source == "estimated"
            ok = (self.bpm >= lo) & (self.bpm <= hi)
            # Beat trackers often land on half or double the felt tempo.
            ok |= est & (((self.bpm * 2 >= lo) & (self.bpm * 2 <= hi)) | ((self.bpm / 2 >= lo) & (self.bpm / 2 <= hi)))
            if not f.kinds:
                # An estimated tempo on a machine loop or ambience is noise; a named one is not.
                ok &= (self.kinds == "music") | ~est
            m &= ok
        for sub in f.path:
            m &= np.array([sub in p.lower() for p in self.paths], dtype=bool)
        return m

    def rank(
        self,
        audio_q: np.ndarray | None,
        text_q: np.ndarray | None,
        filters: Filters,
        cfg: SearchConfig,
        k: int | None = None,
        group_variants: bool | None = None,
    ) -> list[Result]:
        k = k or cfg.k
        group_variants = cfg.group_variants if group_variants is None else group_variants
        idx = np.flatnonzero(self.mask(filters))
        if not idx.size:
            return []
        sa = self.audio[idx] @ audio_q if audio_q is not None else np.zeros(idx.size)
        st = self.text[idx] @ text_q if text_q is not None else np.zeros(idx.size)
        if audio_q is None and text_q is None:
            score = -self.duration[idx]  # filter-only query: arbitrary but stable order
        else:
            score = cfg.audio_weight * _z(sa) + cfg.text_weight * _z(st)

        keep = np.zeros(len(self), dtype=bool)
        keep[idx] = True
        taken = np.zeros(len(self), dtype=bool)
        results: list[Result] = []
        for j in np.argsort(-score, kind="stable"):
            if len(results) >= k:
                break
            i = idx[j]
            if taken[i]:
                continue
            family = self._same_sound(i)
            dupes = [x for x in family if keep[x] and x != i]
            variants = []
            if group_variants:
                # Single linkage: a sparse "Intensity 1" joins via the fuller version it resembles,
                # even if it is not close to the top hit itself.
                cluster = [i]
                pending = [x for x in self._by_group[self.groups[i]] if keep[x] and x not in family]
                grew = True
                while grew and pending:
                    grew = False
                    for x in list(pending):
                        if x in family:
                            pending.remove(x)
                        elif float(np.max(self.audio[cluster] @ self.audio[x])) >= cfg.variant_similarity:
                            cluster.append(x)
                            variants.append(x)
                            family |= self._same_sound(x)
                            pending.remove(x)
                            grew = True
            taken[list(family)] = True
            results.append(Result(
                id=int(self.ids[i]), path=self.paths[i], score=float(score[j]),
                audio_similarity=float(sa[j]), text_similarity=float(st[j]),
                kind=self.kinds[i] or None,
                duration=None if np.isnan(self.duration[i]) else round(float(self.duration[i]), 2),
                bpm=None if np.isnan(self.bpm[i]) else float(self.bpm[i]),
                bpm_source=self.bpm_source[i], description=self.descriptions[i],
                variants=[self.paths[x] for x in variants],
                duplicates=[self.paths[x] for x in dupes],
            ))
        return results

    def _same_sound(self, i: int) -> set[int]:
        """i plus its identical copies: same bytes, or same name in another format folder / pack copy."""
        return {i, *self._by_hash[self.hashes[i]], *self._by_ident[self.idents[i]]}


def _z(x: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x - x.mean()


async def run_search(
    index: SearchIndex,
    audio_client,
    text_client,
    query: str,
    cfg: SearchConfig,
    k: int | None = None,
    extra: Filters | None = None,
    group_variants: bool | None = None,
) -> tuple[list[Result], Filters]:
    """Embed the query (the only model call on the search path) and rank."""
    text, f = parse_query(query, cfg)
    if extra:
        f.kinds = extra.kinds or f.kinds
        f.bpm = extra.bpm or f.bpm
        f.duration = extra.duration or f.duration
        f.path += extra.path
    audio_q = text_q = None
    if text:
        if text_client is audio_client and text_client.ep.query_template == audio_client.ep.query_template:
            audio_q = text_q = await audio_client.embed_query(text)
        else:
            audio_q, text_q = await asyncio.gather(audio_client.embed_query(text), text_client.embed_query(text))
    return index.rank(audio_q, text_q, f, cfg, k=k, group_variants=group_variants), f


def results_json(results: list[Result], filters: Filters, query: str) -> str:
    return json.dumps({"query": query, "filters": filters.describe(), "results": [r.to_dict() for r in results]}, indent=2)
