"""find-sound: index a sound library with a multimodal embedding model, then search it by description."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .config import Config, load_config
from .embed_client import EmbeddingError
from .indexer import EndpointDown, IndexBusy, Indexer, SyncReport, close_clients, open_clients
from .search import Filters, SearchIndex, parse_kinds, parse_range, results_json, run_search
from .server_control import ServerUnavailable, ensure_running
from .store import Store

log = logging.getLogger("find_sound")


def _progress_printer():
    last = [0.0]
    t0 = time.monotonic()
    tty = sys.stderr.isatty()

    def show(r: SyncReport, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - last[0] < (0.25 if tty else 30):
            return
        last[0] = now
        rate = r.done / max(now - t0, 1e-6)
        eta = (r.queued - r.done) / rate if rate > 0 else 0
        line = (f"[{r.done}/{r.queued}] audio {r.audio_embedded} embedded, {r.audio_reused} reused · "
                f"text {r.text_embedded} · failed {r.failed} · {rate:.1f} files/s · eta {eta / 60:.0f} min")
        print(("\r\033[K" if tty else "") + line, end="\n" if final or not tty else "", file=sys.stderr, flush=True)

    return show


async def _index(cfg: Config, limit: int | None) -> int:
    store = Store(cfg.index_path)
    audio, text = open_clients(cfg)
    show = _progress_printer()
    indexer = Indexer(cfg, store, audio, text, on_progress=show)
    try:
        report = await indexer.sync(limit=limit)
        if report.queued:
            show(report, final=True)
        print(report.summary(), file=sys.stderr)
        return 1 if report.failed and report.failed == report.queued else 0
    finally:
        indexer.close()
        await close_clients(audio, text)
        store.close()


async def _watch(cfg: Config, interval: float | None) -> None:
    store = Store(cfg.index_path)
    audio, text = open_clients(cfg)
    indexer = Indexer(cfg, store, audio, text, on_progress=_progress_printer())
    log.info("watching %s every %ss", ", ".join(map(str, cfg.library)), interval or cfg.scan_interval)
    try:
        await indexer.watch(interval)
    finally:
        indexer.close()
        await close_clients(audio, text)
        store.close()


async def _search(cfg: Config, args) -> int:
    store = Store(cfg.index_path)
    index = SearchIndex(store, cfg.embedding.audio_model_key, cfg.text_endpoint.text_model_key)
    if not len(index):
        print(f"the index at {cfg.index_path} is empty; run `find-sound index` first", file=sys.stderr)
        return 1
    extra = Filters(
        kinds=parse_kinds(args.kind) if args.kind else None,
        bpm=parse_range(args.bpm, tolerance=cfg.search.bpm_tolerance) if args.bpm else None,
        duration=parse_range(args.dur) if args.dur else None,
        path=[s.lower() for s in args.within or []],
    )
    search_cfg = cfg.search
    if args.audio_weight is not None:
        search_cfg = replace(search_cfg, audio_weight=args.audio_weight, text_weight=1 - args.audio_weight)
    try:
        await ensure_running(cfg, on_start=lambda url, pid: print(
            f"starting the embedding server for {url} (pid {pid}); first search takes ~20 s", file=sys.stderr))
    except ServerUnavailable as e:
        print(f"find-sound: {e}", file=sys.stderr)
        return 2
    audio, text = open_clients(cfg)
    try:
        query = " ".join(args.query)
        results, filters = await run_search(
            index, audio, text, query, search_cfg, k=args.k, extra=extra,
            group_variants=False if args.no_group else None,
        )
    finally:
        await close_clients(audio, text)
        store.close()

    if args.copy_to:
        dest = Path(args.copy_to).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)
        for r in results:
            r.source, r.path = r.path, _copy_into(r.path, dest)
            if args.with_variants:
                r.copied_variants = [_copy_into(v, dest) for v in r.variants]

    if args.json:
        print(results_json(results, filters, query))
    elif args.long:
        if filters.describe():
            print(f"# filters: {filters.describe()}")
        for i, r in enumerate(results, 1):
            meta = [r.kind or "?", f"{r.duration:.1f}s" if r.duration is not None else "",
                    f"{r.bpm:g} bpm ({r.bpm_source})" if r.bpm else ""]
            print(f"{i}. {r.path}" + (f"  (from {r.source})" if r.source else ""))
            print(f"   {' · '.join(m for m in meta if m)} · score {r.score:.2f} "
                  f"(audio {r.audio_similarity:.3f}, text {r.text_similarity:.3f})")
            print(f"   {r.description}")
            if r.variants:
                print(f"   +{len(r.variants)} variant(s), e.g. {Path(r.variants[0]).name}")
    else:
        for r in results:
            print(r.path)
    if not results:
        print("no matches" + (f" for filters: {filters.describe()}" if filters.describe() else ""), file=sys.stderr)
        return 1
    return 0


def _copy_into(src: str, dest: Path) -> str:
    """Copy src into dest, keeping its name unless a different file already has it."""
    target = dest / Path(src).name
    n = 1
    while target.exists() and not target.samefile(src):
        target = dest / f"{Path(src).stem}-{n}{Path(src).suffix}"
        n += 1
    shutil.copy2(src, target)
    return str(target)


def _stats(cfg: Config, as_json: bool) -> None:
    store = Store(cfg.index_path)
    s = store.stats(cfg.embedding.audio_model_key, cfg.text_endpoint.text_model_key)
    store.close()
    s = {"index": str(cfg.index_path), "library": [str(p) for p in cfg.library], "model": cfg.embedding.model, **s}
    if as_json:
        print(json.dumps(s, indent=2))
        return
    for k, v in s.items():
        if k == "last_sync" and v:
            v = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v['finished_at']))}: " + SyncReport(
                **{f: v[f] for f in SyncReport.__dataclass_fields__ if f in v}).summary()
        print(f"{k:>16}: {v}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="find-sound", description=__doc__)
    ap.add_argument("-c", "--config", help="config file (default: $FIND_SOUND_CONFIG, ./find-sound.toml, ~/.config/find-sound/config.toml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="scan the library once and embed anything new or changed")
    p.add_argument("--limit", type=int, help="process at most N new/changed files (for trying things out)")

    p = sub.add_parser("watch", help="keep the index up to date, rescanning periodically")
    p.add_argument("--interval", type=float, help="seconds between scans (default: scan_interval)")

    p = sub.add_parser("search", help="find the sounds that best match a description")
    p.add_argument("query", nargs="+", help="description, optionally with filters: bpm:120-140 dur:<2 kind:sfx in:foley")
    p.add_argument("-k", type=int, help="number of results (default from config, 5)")
    p.add_argument("--kind", help="music, sfx, voice, ambience (comma-separated)")
    p.add_argument("--bpm", help="tempo: 120 (± tolerance), 110-130")
    p.add_argument("--dur", help="duration in seconds: <2, >30, 1-3, 500ms, 2m")
    p.add_argument("--in", dest="within", action="append", help="path must contain this (repeatable)")
    p.add_argument("--audio-weight", type=float, help="0..1 weight of audio vs. name/tag similarity")
    p.add_argument("--no-group", action="store_true", help="don't collapse numbered/lettered variants")
    p.add_argument("--copy-to", metavar="DIR", help="copy the results into DIR and print the copies' paths")
    p.add_argument("--with-variants", action="store_true", help="with --copy-to: copy each result's variants too")
    out = p.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="JSON with scores, metadata and variants")
    out.add_argument("-l", "--long", action="store_true", help="human-readable details")

    p = sub.add_parser("serve", help="web UI + JSON API, keeping the index up to date in the background")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-index", action="store_true", help="search only; don't scan or embed")

    p = sub.add_parser("service", help="run `serve` as a systemd user service (periodic indexing + web UI, survives reboots)")
    p.add_argument("action", choices=["install", "uninstall", "status"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)

    p = sub.add_parser("stats", help="index size, kinds, errors, last sync")
    p.add_argument("--json", action="store_true")

    sub.add_parser("config", help="print the effective configuration")

    p = sub.add_parser("doctor", help="check config, library, embedding endpoints and index")
    p.add_argument("--fix", action="store_true",
                   help="start the bundled embedding server if a local endpoint is down, then update the index")

    p = sub.add_parser("eval", help="measure retrieval quality on a cases file (see evals/)")
    p.add_argument("cases", help="TOML file of [[case]] query/match entries")
    p.add_argument("--weights", help="comma-separated audio weights to try (default 1,0.8,0.65,0.5,0.35,0.2,0)")
    p.add_argument("-v", "--per-case", action="store_true", help="show each query's result")
    p.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
        if args.cmd == "index":
            sys.exit(asyncio.run(_index(cfg, args.limit)))
        elif args.cmd == "watch":
            asyncio.run(_watch(cfg, args.interval))
        elif args.cmd == "search":
            sys.exit(asyncio.run(_search(cfg, args)))
        elif args.cmd == "serve":
            import uvicorn

            from .web import create_app

            uvicorn.run(create_app(cfg, background_index=not args.no_index), host=args.host, port=args.port,
                        log_level="warning")
        elif args.cmd == "stats":
            _stats(cfg, args.json)
        elif args.cmd == "service":
            from . import service

            sys.exit({"install": lambda: service.install(cfg, args.host, args.port),
                      "uninstall": service.uninstall, "status": service.status}[args.action]())
        elif args.cmd == "doctor":
            from .doctor import run as doctor

            sys.exit(asyncio.run(doctor(cfg, fix=args.fix)))
        elif args.cmd == "eval":
            from .evaluate import DEFAULT_WEIGHTS, evaluate, format_report, load_cases

            weights = tuple(float(w) for w in args.weights.split(",")) if args.weights else DEFAULT_WEIGHTS
            weights = tuple(sorted(set(weights) | {cfg.search.audio_weight}, reverse=True))
            result = asyncio.run(evaluate(cfg, load_cases(args.cases), weights))
            print(json.dumps(result, indent=2, default=str) if args.json
                  else format_report(result, cfg.search.audio_weight, args.per_case))
        elif args.cmd == "config":
            d = asdict(cfg)
            print(json.dumps(d, indent=2, default=str))
    except KeyboardInterrupt:
        sys.exit(130)
    except (IndexBusy, EndpointDown, EmbeddingError, ValueError, FileNotFoundError) as e:
        print(f"find-sound: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
