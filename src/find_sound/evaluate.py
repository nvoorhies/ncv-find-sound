"""Retrieval evaluation with queries whose right answers are recognisable by path.

A cases file is TOML:

    [[case]]
    query = "blade clash"
    match = ["sword|blade|dagger|knife", "hit|clash|impact|parry|clang"]  # every regex must match the path

Each query is embedded once and then ranked at several audio/text weightings, reporting
precision@5 and MRR@10. Use it to compare embedding models or tune `audio_weight` on your own
library. Judging by path rewards descriptive names, so phrase queries in words the files don't
use, and read results for the audio-heavy weightings with that bias in mind.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import Config
from .indexer import close_clients, open_clients
from .search import SearchIndex, embed_query, parse_query
from .server_control import ensure_running
from .store import Store

DEFAULT_WEIGHTS = (1.0, 0.8, 0.65, 0.5, 0.35, 0.2, 0.0)


@dataclass
class Case:
    query: str
    match: list[re.Pattern]

    def relevant(self, path: str) -> bool:
        return all(rx.search(path) for rx in self.match)


def load_cases(path: str | Path) -> list[Case]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    cases = []
    for c in data.get("case", []):
        match = c["match"] if isinstance(c["match"], list) else [c["match"]]
        cases.append(Case(c["query"], [re.compile(m, re.I) for m in match]))
    if not cases:
        raise ValueError(f"no [[case]] entries in {path}")
    return cases


async def evaluate(cfg: Config, cases: list[Case], weights=DEFAULT_WEIGHTS, k: int = 5, depth: int = 10) -> dict:
    store = Store(cfg.index_path)
    index = SearchIndex(store, cfg.embedding.audio_model_key, cfg.text_endpoint.text_model_key)
    store.close()
    if not len(index):
        raise ValueError(f"the index at {cfg.index_path} has no vectors for the configured models; run `find-sound index`")
    await ensure_running(cfg)
    audio, text = open_clients(cfg)
    try:
        per_case = []
        for case in cases:
            q, filters = parse_query(case.query, cfg.search)
            aq, tq = await embed_query(audio, text, q)
            mask = index.mask(filters)
            available = sum(case.relevant(p) for p, m in zip(index.paths, mask) if m)
            runs = {}
            for w in weights:
                scfg = replace(cfg.search, audio_weight=w, text_weight=1 - w)
                hits = [case.relevant(r.path) for r in index.rank(aq, tq, filters, scfg, k=depth)]
                first = next((i + 1 for i, h in enumerate(hits) if h), None)
                runs[w] = {"p_at_k": sum(hits[:k]) / k, "rr": 1 / first if first else 0.0, "first": first}
            per_case.append({"query": case.query, "available": available, "runs": runs})
    finally:
        await close_clients(audio, text)

    summary = {
        w: {"p_at_k": float(np.mean([c["runs"][w]["p_at_k"] for c in per_case])),
            "mrr": float(np.mean([c["runs"][w]["rr"] for c in per_case]))}
        for w in weights
    }
    return {"k": k, "depth": depth, "summary": summary, "cases": per_case,
            "audio_model": cfg.embedding.audio_model_key, "text_model": cfg.text_endpoint.text_model_key}


def format_report(result: dict, default_weight: float, verbose: bool = False) -> str:
    k, depth = result["k"], result["depth"]
    lines = [f"audio: {result['audio_model']}", f"text:  {result['text_model']}", "",
             f"{'audio/text weight':>18}  {'P@' + str(k):>6}  {'MRR@' + str(depth):>7}"]
    for w, m in result["summary"].items():
        mark = "  <- configured" if abs(w - default_weight) < 1e-9 else ""
        lines.append(f"{w:>9.2f} / {1 - w:<6.2f}  {m['p_at_k']:>6.3f}  {m['mrr']:>7.3f}{mark}")
    thin = [c for c in result["cases"] if c["available"] < k]
    if thin:
        lines += ["", f"note: {len(thin)} case(s) have fewer than {k} relevant files in the index: "
                  + ", ".join(f"{c['query']!r} ({c['available']})" for c in thin)]
    if verbose:
        w = min(result["summary"], key=lambda x: abs(x - default_weight))
        lines += ["", f"per case at weight {w:.2f}:", f"{'P@' + str(k):>6}  {'first':>5}  {'avail':>5}  query"]
        for c in sorted(result["cases"], key=lambda c: c["runs"][w]["p_at_k"]):
            r = c["runs"][w]
            lines.append(f"{r['p_at_k']:>6.2f}  {r['first'] or '-':>5}  {c['available']:>5}  {c['query']}")
    return "\n".join(lines)
