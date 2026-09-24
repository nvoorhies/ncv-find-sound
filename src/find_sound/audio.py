"""Decoding, segment selection and per-file analysis (duration, loudness, tempo, tags).

Everything here is synchronous and CPU-bound; the indexer runs `analyze_file` in a process pool.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

BPM_SAMPLE_RATE = 22050
BPM_WINDOW_SECONDS = 60.0


def hash_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash, so renamed or duplicated files reuse their cached embedding."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class AudioReader:
    """Random access to mono float32 windows of a file.

    libsndfile handles wav/flac/ogg/mp3/aiff and can seek, so long files are never decoded whole.
    Anything else (m4a, opus in some containers, ...) goes through ffmpeg once, in full.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._pcm: np.ndarray | None = None
        try:
            info = sf.info(self.path)
            self.sample_rate, self.channels, self.frames = info.samplerate, info.channels, info.frames
        except (sf.LibsndfileError, RuntimeError):
            self._pcm, self.sample_rate, self.channels = _ffmpeg_decode(self.path)
            self.frames = len(self._pcm)
        if self.frames <= 0 or self.sample_rate <= 0:
            raise ValueError("no audio frames")

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate

    def window(self, start: float, length: float) -> np.ndarray:
        a = max(0, int(start * self.sample_rate))
        n = max(1, int(length * self.sample_rate))
        if self._pcm is not None:
            return self._pcm[a : a + n]
        with sf.SoundFile(self.path) as f:
            f.seek(min(a, max(0, self.frames - 1)))
            data = f.read(n, dtype="float32", always_2d=True)
        return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]


def _ffmpeg_decode(path: str) -> tuple[np.ndarray, int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "json", path],
        capture_output=True, check=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise ValueError("no audio stream")
    sr, ch = int(streams[0]["sample_rate"]), int(streams[0]["channels"])
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.float32).copy(), sr, ch


def segment_windows(duration: float, seconds: float, max_segments: int) -> list[tuple[float, float]]:
    """(start, length) windows spread evenly over the file; short files are one window."""
    if duration <= seconds or max_segments <= 1:
        start = max(0.0, (duration - seconds) / 2)
        return [(start, min(duration, seconds))]
    n = min(max_segments, math.ceil(duration / seconds))
    return [(max(0.0, (i + 0.5) * duration / n - seconds / 2), seconds) for i in range(n)]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    return x if sr_in == sr_out else soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def estimate_bpm(x: np.ndarray, sr: int) -> tuple[float, float] | None:
    """Tempo and a 0..1 pulse-clarity score (autocorrelation of the onset envelope at the beat lag).

    Clarity separates rhythmic material from ambiences/drones, which librosa will happily assign
    a tempo to anyway.
    """
    import librosa  # slow import (numba); only the analysis workers pay for it

    with warnings.catch_warnings():
        # Silent stretches make librosa's numba kernels warn about NaN casts; the result is still fine.
        warnings.simplefilter("ignore", RuntimeWarning)
        return _estimate_bpm(librosa, x, sr)


def _estimate_bpm(librosa, x: np.ndarray, sr: int) -> tuple[float, float] | None:
    y = resample(x, sr, BPM_SAMPLE_RATE)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=BPM_SAMPLE_RATE, hop_length=hop)
    if env.size < 16 or not np.any(env):
        return None
    tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=BPM_SAMPLE_RATE, hop_length=hop))[0])
    if not 30 <= tempo <= 300:
        return None
    ac = librosa.autocorrelate(env - env.mean())
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lag = 60.0 * BPM_SAMPLE_RATE / hop / tempo
    lo, hi = int(lag) - 1, int(lag) + 2
    clarity = float(np.clip(ac[max(1, lo) : hi].max(initial=0.0), 0.0, 1.0)) if lo < len(ac) else 0.0
    return round(tempo, 1), round(clarity, 3)


def read_tags(path: str) -> dict[str, str]:
    """Best-effort title/artist/album/genre from ID3, Vorbis comments, MP4 atoms."""
    try:
        import mutagen

        f = mutagen.File(path, easy=True)
    except Exception:
        return {}
    if not f or not f.tags:
        return {}
    tags = {}
    for key in ("title", "artist", "album", "genre"):
        try:
            v = f.tags.get(key)
        except Exception:
            v = None
        if v:
            tags[key] = str(v[0] if isinstance(v, list) else v)[:200]
    return tags


@dataclass
class Analysis:
    duration: float
    sample_rate: int
    channels: int
    rms_db: float
    peak_db: float
    bpm: float | None = None
    bpm_confidence: float | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class Prepared:
    analysis: Analysis | None
    segments: list[bytes]  # WAV-encoded clips at the embedding sample rate


def _db(v: float) -> float:
    return round(20 * math.log10(max(v, 1e-9)), 1)


def analyze_file(
    path: str,
    *,
    sample_rate: int,
    segment_seconds: float,
    max_segments: int,
    bpm_min_seconds: float,
    want_analysis: bool = True,
    want_segments: bool = True,
) -> Prepared:
    reader = AudioReader(path)
    windows = [reader.window(s, n) for s, n in segment_windows(reader.duration, segment_seconds, max_segments)]
    windows = [w for w in windows if w.size] or [reader.window(0, reader.duration)]

    analysis = None
    if want_analysis:
        # Loudness from the embedded windows only: cheap, and representative enough to rank by.
        cat = np.concatenate(windows)
        analysis = Analysis(
            duration=round(reader.duration, 3),
            sample_rate=reader.sample_rate,
            channels=reader.channels,
            rms_db=_db(float(np.sqrt(np.mean(np.square(cat, dtype=np.float64))))),
            peak_db=_db(float(np.max(np.abs(cat)))),
            tags=read_tags(path),
        )
        if reader.duration >= bpm_min_seconds:
            mid = reader.duration / 2
            clip = reader.window(max(0.0, mid - BPM_WINDOW_SECONDS / 2), BPM_WINDOW_SECONDS)
            if est := estimate_bpm(clip, reader.sample_rate):
                analysis.bpm, analysis.bpm_confidence = est

    segments = []
    if want_segments:
        segments = [to_wav_bytes(resample(w, reader.sample_rate, sample_rate), sample_rate) for w in windows]
    return Prepared(analysis, segments)
