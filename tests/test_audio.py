import numpy as np
import pytest

from find_sound.audio import estimate_bpm, segment_windows


def drum_loop(bpm: float, seconds: float = 30, sr: int = 22050) -> np.ndarray:
    rng = np.random.default_rng(0)
    y = np.zeros(int(seconds * sr))
    n = int(0.15 * sr)
    kick = np.exp(-np.arange(n) / (0.03 * sr)) * np.sin(2 * np.pi * 60 * np.arange(n) / sr)
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.005 * sr)) * 0.3
    beat, t, i = 60 / bpm, 0.0, 0
    while t < seconds - 0.2:
        s = int(t * sr)
        y[s : s + n] += kick if i % 2 == 0 else 0.6 * kick
        h = int((t + beat / 2) * sr)
        if h + len(hat) <= len(y):
            y[h : h + len(hat)] += hat
        t, i = t + beat, i + 1
    return (y + 0.01 * rng.standard_normal(len(y))).astype(np.float32)


@pytest.mark.parametrize("bpm", [97, 128.5, 141, 173])
def test_bpm_is_not_quantized(bpm):
    tempo, clarity = estimate_bpm(drum_loop(bpm), 22050)
    assert tempo == pytest.approx(bpm, abs=0.6)
    assert clarity > 0.3


def test_noise_has_no_clear_pulse():
    noise = np.random.default_rng(1).standard_normal(22050 * 20).astype(np.float32) * 0.1
    est = estimate_bpm(noise, 22050)
    assert est is None or est[1] < 0.1


def test_segment_windows():
    assert segment_windows(4.0, 10, 3) == [(0.0, 4.0)]
    wins = segment_windows(60.0, 10, 3)
    assert len(wins) == 3 and all(n == 10 for _, n in wins)
    assert wins[0][0] >= 0 and wins[-1][0] + 10 <= 60
