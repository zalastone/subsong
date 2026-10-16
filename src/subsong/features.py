"""Спектральные признаки: log-mel для CTC-выравнивания и статистики для music-гейта.

log_mel повторяет `torchaudio.transforms.MelSpectrogram` с параметрами GigaAM:
n_mels=64, win_length=320, hop_length=160, n_fft=320, mel_scale='htk', center=False,
после чего берётся натуральный логарифм с отсечкой снизу (SpecScaler в gigaam).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from subsong.config import SR


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    """Шкала HTK: mel = 2595 * log10(1 + f/700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq, dtype=np.float64) / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """Обратное преобразование шкалы HTK."""
    return 700.0 * (np.power(10.0, np.asarray(mels, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    n_freqs: int = 161,
    n_mels: int = 64,
    sr: int = SR,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    """Треугольный мел-фильтрбанк [n_mels, n_freqs] в стиле torchaudio (нормализация отключена)."""
    if f_max is None:
        f_max = sr / 2.0
    all_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    m_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[None, :-1]
    up = slopes[:, 2:] / f_diff[None, 1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    return cast(NDArray[np.float64], fb.T)  # (n_mels, n_freqs)


_FILTERBANK_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _get_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Кэширует фильтрбанк: пересчёт на каждый сегмент не нужен."""
    key = (n_mels, n_fft, sr)
    if key not in _FILTERBANK_CACHE:
        _FILTERBANK_CACHE[key] = mel_filterbank(n_freqs=n_fft // 2 + 1, n_mels=n_mels, sr=sr)
    return _FILTERBANK_CACHE[key]


def hann_window(n: int) -> np.ndarray:
    """Периодическое окно Ханна (совпадает с torch.hann_window(periodic=True))."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def frame_signal(x: np.ndarray, win: int, hop: int) -> NDArray[np.float32]:
    """Нарезает сигнал на перекрывающиеся фреймы [n_frames, win] (без центрирования)."""
    if x.size < win:
        return np.zeros((0, win), dtype=np.float32)
    n_frames = 1 + (x.size - win) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    view = np.lib.stride_tricks.as_strided(
        x[: (n_frames - 1) * hop + win], (n_frames, win), strides
    )
    return cast(NDArray[np.float32], np.array(view, dtype=np.float32, copy=False))


def log_mel(
    samples: np.ndarray,
    n_mels: int = 64,
    n_fft: int = 320,
    hop: int = 160,
    win: int = 320,
    sr: int = SR,
    log_floor: float = 1e-9,
) -> np.ndarray:
    """Считает log-mel spectrogram [n_mels, n_frames] (float32)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size < win:
        return np.zeros((n_mels, 0), dtype=np.float32)
    frames = frame_signal(x, win, hop).astype(np.float64) * hann_window(win)
    # rfft требует дополнения до n_fft, когда win < n_fft
    if win < n_fft:
        frames = np.pad(frames, ((0, 0), (0, n_fft - win)))
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (spec.real**2 + spec.imag**2).T  # (n_freqs, n_frames)
    mel = _get_filterbank(n_mels, n_fft, sr) @ power
    return np.asarray(np.log(np.maximum(mel, log_floor)), dtype=np.float32)


def n_frames_for(samples_len: int, win: int = 320, hop: int = 160) -> int:
    """Число фреймов log-mel до субсэмплирования (совпадает с gigaam.FeatureExtractor.out_len)."""
    if samples_len < win:
        return 0
    return (samples_len - win) // hop + 1


@dataclass(frozen=True)
class SegmentStats:
    """Акустическая сводка по сегменту: нужна для отсева инструментала."""

    rms_db: float
    peak_db: float
    flatness: float          # 0 = чистый тон, 1 - 1 = шум/белый шум
    low_ratio: float         # доля энергии ниже 250 Гц
    high_ratio: float        # доля энергии выше 4 кГц (свистящие, шипящие согласные)
    flux: float              # средняя спектральная флуктуация (удары, атаки)


def segment_stats(
    samples: np.ndarray,
    sr: int = SR,
    n_fft: int = 1024,
    hop: int = 512,
) -> SegmentStats:
    """Считает спектральные метрики сегмента (одна агрегированная строка)."""
    x = np.asarray(samples, dtype=np.float32)
    if x.size == 0:
        return SegmentStats(-120.0, -120.0, 0.0, 0.0, 0.0, 0.0)
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    peak = float(np.max(np.abs(x)))
    frames = frame_signal(x, n_fft, hop).astype(np.float64) * hann_window(n_fft)
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).T  # (n_freqs, n_frames)
    power = spec**2 + 1e-20
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = float(power.sum()) + 1e-20
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    low = float(power[freqs < 250.0].sum()) / total
    high = float(power[freqs > 4000.0].sum()) / total
    if power.shape[1] > 1:
        diff = np.diff(power, axis=1)
        flux = float(np.mean(np.sum(np.abs(diff), axis=0)) / total)
    else:
        flux = 0.0
    return SegmentStats(
        rms_db=20.0 * float(np.log10(max(rms, 1e-6))),
        peak_db=20.0 * float(np.log10(max(peak, 1e-6))),
        flatness=flatness,
        low_ratio=low,
        high_ratio=high,
        flux=flux,
    )


def frame_rms_db(samples: np.ndarray, sr: int = SR, hop_sec: float = 0.02, win_sec: float = 0.04) -> np.ndarray:
    """Кадровая громкость в dBFS: используется для подрезки тишины по краям реплик."""
    x = np.asarray(samples, dtype=np.float32)
    hop = max(1, int(sr * hop_sec))
    win = max(hop, int(sr * win_sec))
    if x.size < win:
        return np.array([20.0 * float(np.log10(max(float(np.sqrt(np.mean(np.square(x))) if x.size else 1e-6), 1e-6)))])
    frames = frame_signal(x, win, hop)
    rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-6))
