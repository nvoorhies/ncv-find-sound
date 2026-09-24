"""Settings, loaded from a TOML file with a few environment-variable overrides.

Lookup order for the file: --config, $FIND_SOUND_CONFIG, ./find-sound.toml,
~/.config/find-sound/config.toml. Every setting has a default, so a config file is optional
as long as FIND_SOUND_LIBRARY names the sound directory.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".oga", ".mp3", ".opus", ".m4a", ".aif", ".aiff")


@dataclass(frozen=True)
class EndpointConfig:
    """One OpenAI-compatible /embeddings endpoint and the model behind it."""

    base_url: str = "http://localhost:7997"
    model: str = "laion/larger_clap_general"
    # Name of the env var holding the API key; the key itself never goes in the file.
    api_key_env: str = "FIND_SOUND_API_KEY"
    # How audio is put on the wire. The OpenAI API has no audio embedding input, so servers differ:
    #   "infinity": {"input": ["data:audio/wav;base64,..."], "modality": "audio"}  (Infinity, the
    #               bundled find-sound-embed-server)
    #   "messages": {"messages": [{"role": "user", "content": [{"type": "input_audio", ...}]}]}
    #               (vLLM-style chat embeddings, one clip per request)
    audio_format: str = "infinity"
    # Applied to search queries only. Instruction-tuned embedders want e.g.
    # "Instruct: Retrieve sounds matching the description\nQuery: {query}".
    query_template: str = "{query}"
    # Clips are resampled to this rate before upload; CLAP models expect 48 kHz.
    sample_rate: int = 48000
    # Long files are embedded as up to max_segments windows of segment_seconds, averaged.
    segment_seconds: float = 10.0
    max_segments: int = 3
    concurrency: int = 8
    text_batch_size: int = 64
    timeout: float = 120.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None

    @property
    def audio_model_key(self) -> str:
        """Cache key for audio vectors: anything that changes the vector is part of it."""
        return (
            f"{self.model}|{self.audio_format}|sr{self.sample_rate}"
            f"|seg{self.segment_seconds:g}x{self.max_segments}"
        )

    @property
    def text_model_key(self) -> str:
        return f"{self.model}|text"


@dataclass(frozen=True)
class SearchConfig:
    k: int = 5
    # Weights of the two similarity channels after each is z-scored across the library:
    # query vs. the audio itself, and query vs. a description built from path + tags.
    audio_weight: float = 0.65
    text_weight: float = 0.35
    # Collapse "Laser 001", "Laser 002", ... into one result with variants, when their audio
    # embeddings are at least this similar (numbered files can also be unrelated sounds).
    group_variants: bool = True
    variant_similarity: float = 0.75
    # Relative tolerance for a single-value BPM filter ("bpm:120" matches 115.2-124.8).
    bpm_tolerance: float = 0.04


@dataclass(frozen=True)
class Config:
    library: tuple[Path, ...] = ()
    index_path: Path = Path("~/.cache/find-sound/index.sqlite").expanduser()
    extensions: tuple[str, ...] = AUDIO_EXTENSIONS
    scan_interval: float = 300.0
    # Files modified more recently than this are skipped until the next scan (still copying).
    settle_seconds: float = 2.0
    # Processes for decoding / analysis (BPM etc.). 0 = min(8, cpu count).
    analysis_workers: int = 0
    # Only estimate BPM for clips at least this long; short one-shots have no tempo.
    bpm_min_seconds: float = 6.0
    embedding: EndpointConfig = field(default_factory=EndpointConfig)
    # Separate model for the path/tag description channel. None = use `embedding`'s text side.
    text_embedding: EndpointConfig | None = None
    search: SearchConfig = field(default_factory=SearchConfig)
    source: Path | None = None

    @property
    def text_endpoint(self) -> EndpointConfig:
        return self.text_embedding or self.embedding

    @property
    def workers(self) -> int:
        return self.analysis_workers or min(8, os.cpu_count() or 2)


def _build(cls, data: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown setting(s) in {where}: {', '.join(sorted(unknown))}")
    return cls(**data)


def _paths(value) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        value = [value]
    return tuple(Path(os.path.expandvars(str(v))).expanduser().resolve() for v in value)


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    candidates = [
        explicit,
        os.environ.get("FIND_SOUND_CONFIG"),
        Path.cwd() / "find-sound.toml",
        Path("~/.config/find-sound/config.toml").expanduser(),
    ]
    for c in candidates:
        if c and Path(c).expanduser().is_file():
            return Path(c).expanduser().resolve()
    if explicit:
        raise FileNotFoundError(f"config file not found: {explicit}")
    return None


def load_config(path: str | Path | None = None) -> Config:
    source = find_config_file(path)
    data: dict = {}
    if source:
        with open(source, "rb") as f:
            data = tomllib.load(f)

    emb = _build(EndpointConfig, data.pop("embedding", {}), "[embedding]")
    text = data.pop("text_embedding", None)
    # [text_embedding] inherits from [embedding], so it only needs the fields that differ.
    text_emb = _build(EndpointConfig, {**_endpoint_dict(emb), **text}, "[text_embedding]") if text else None
    search = _build(SearchConfig, data.pop("search", {}), "[search]")

    if "library" in data:
        data["library"] = _paths(data["library"])
    if "index_path" in data:
        data["index_path"] = _paths(data["index_path"])[0]
    if "extensions" in data:
        data["extensions"] = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in data["extensions"])
    cfg = _build(Config, data, str(source or "config"))
    cfg = replace(cfg, embedding=emb, text_embedding=text_emb, search=search, source=source)

    # Environment overrides, handy for one-off runs and for the skill wrapper.
    if lib := os.environ.get("FIND_SOUND_LIBRARY"):
        cfg = replace(cfg, library=_paths(lib.split(os.pathsep)))
    if idx := os.environ.get("FIND_SOUND_INDEX"):
        cfg = replace(cfg, index_path=_paths(idx)[0])
    if url := os.environ.get("FIND_SOUND_BASE_URL"):
        cfg = replace(cfg, embedding=replace(cfg.embedding, base_url=url))
    if model := os.environ.get("FIND_SOUND_MODEL"):
        cfg = replace(cfg, embedding=replace(cfg.embedding, model=model))
    return cfg


def _endpoint_dict(e: EndpointConfig) -> dict:
    return {f.name: getattr(e, f.name) for f in fields(e)}
