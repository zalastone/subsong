"""Тесты спектральных признаков и фильтров мусора."""
from __future__ import annotations

import numpy as np
import pytest

from subsong.config import FilterSpec
from subsong.features import (
    frame_rms_db,
    hz_to_mel,
    log_mel,
    mel_filterbank,
    mel_to_hz,
    n_frames_for,
    segment_stats,
)
from subsong.gate import (
    SegmentInput,
    collect_track_stats,
    is_repetition,
    judge_segment,
    music_verdict,
)


def test_mel_scale_roundtrip() -> None:
    hz = np.array([0.0, 700.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_mel_filterbank_shape_and_peak() -> None:
    fb = mel_filterbank(n_freqs=161, n_mels=64, sr=16000)
    assert fb.shape == (64, 161)
    assert np.all(fb >= 0)
    assert fb.max() == pytest.approx(1.0, abs=0.01)


def test_log_mel_shape_and_rate() -> None:
    sr = 16000
    t = np.arange(sr) / sr
    sig = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    feats = log_mel(sig)
    assert feats.shape == (64, n_frames_for(len(sig)))
    assert feats.dtype == np.float32
    # 100 кадров в секунду до субсэмплирования
    assert feats.shape[1] == pytest.approx(len(sig) / 160, rel=0.01)
    # энергия сосредоточена в низких мел-полосах (440 Гц)
    assert int(np.argmax(feats.mean(axis=1))) < 30


def test_log_mel_handles_short_input() -> None:
    assert log_mel(np.zeros(100, dtype=np.float32)).shape[1] == 0


def test_frame_rms_db_drops_for_silence_and_noise() -> None:
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    noise = (np.random.default_rng(0).standard_normal(sr) * 0.1).astype(np.float32)
    assert frame_rms_db(silence)[0] < -100
    assert -30 < float(np.median(frame_rms_db(noise))) < -15


def test_segment_stats_tonal_vs_noisy() -> None:
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = (np.random.default_rng(2).standard_normal(sr * 2) * 0.2).astype(np.float32)
    assert segment_stats(tone).flatness < segment_stats(noise).flatness
    assert segment_stats(tone).low_ratio > 0.5


def test_is_repetition_detects_loops() -> None:
    assert is_repetition(["а", "а", "а"], 3) is not None
    assert is_repetition(["раз", "два"] * 4, 3) is not None
    assert is_repetition(["раз", "два", "три"], 3) is None
    assert is_repetition(["а", "а", "а"], 0) is None


def test_collect_track_stats_uses_median() -> None:
    stats = [segment_stats(np.random.default_rng(i).standard_normal(4000).astype(np.float32)) for i in range(5)]
    ctx = collect_track_stats(stats)
    assert ctx.rms_db == pytest.approx(float(np.median([s.rms_db for s in stats])))


def test_music_verdict_relative_to_track() -> None:
    rng = np.random.default_rng(3)
    sr = 16000
    t = np.arange(sr) / sr
    # "вокал": широкая полоса + шумные согласные
    vocal = (
        0.3 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 3000 * t)
        + 0.02 * rng.standard_normal(t.size)
    ).astype(np.float32)
    # "инструментал": ровный тон без широкой полосы
    tone = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)
    ctx = collect_track_stats([segment_stats(vocal)] * 3 + [segment_stats(tone)])
    drop, reason = music_verdict(segment_stats(tone), ctx, level=1, duration=1.0)
    assert drop and reason
    assert not music_verdict(segment_stats(vocal), ctx, level=1, duration=1.0)[0]
    assert music_verdict(segment_stats(tone), ctx, level=0, duration=1.0) == (False, "")


def test_judge_segment_reasons(speech_like: np.ndarray) -> None:  # type: ignore[no-untyped-def]
    stats = segment_stats(speech_like)

    def make(**kwargs: object) -> SegmentInput:
        base: dict[str, object] = {
            "samples": speech_like,
            "text": "да",
            "words": ["да"],
            "mean_logprob": -0.1,
            "n_aligned_words": 1,
            "stats": stats,
        }
        base.update(kwargs)
        return SegmentInput(**base)  # type: ignore[arg-type]

    assert judge_segment(make(), FilterSpec()).keep

    low = judge_segment(make(mean_logprob=-9.0), FilterSpec(min_confidence=-4.0))
    assert not low.keep and "уверен" in low.reason

    loop = judge_segment(make(words=["да", "да", "да"]), FilterSpec(max_repeat_words=3))
    assert not loop.keep and "повтор" in loop.reason

    unaligned = judge_segment(make(n_aligned_words=0), FilterSpec())
    assert not unaligned.keep and "не лёг" in unaligned.reason

    quiet_samples = np.zeros(1600, dtype=np.float32)
    quiet = judge_segment(
        SegmentInput(
            samples=quiet_samples, text="да", words=["да"], mean_logprob=-0.1,
            n_aligned_words=1, stats=segment_stats(quiet_samples),
        ),
        FilterSpec(),
    )
    assert not quiet.keep