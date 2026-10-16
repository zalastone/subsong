"""Фильтры мусорных реплик: инструментал, галлюцинации ASR на музыке, слишком короткие.

Основной сигнал - средняя лог-вероятность пути выравнивания из CTC forced alignment:
на настоящем вокале она около нуля и выше, а на инструментальных переходах модель
выдаёт правдоподобный, но низкоуверенный текст. Дополнительно ловим зацикливание -
типичную картину галлюцинаций ASR на музыке.

Спектральные проверки (music-гейт) считаются относительно самой дорожки: у абсолютных
порогов нет смысла, потому что спектр у каждой записи свой.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from subsong.config import SR, FilterSpec
from subsong.features import SegmentStats, segment_stats


@dataclass(frozen=True)
class TrackStats:
    """Медианные признаки дорожки - база для относительных сравнений."""

    rms_db: float = -40.0
    flatness: float = 1e-6
    high_ratio: float = 0.0
    flux: float = 0.0


@dataclass
class SegmentVerdict:
    """Решение по сегменту: оставить или отбросить, с указанием причины."""

    keep: bool
    reason: str
    stats: SegmentStats | None = None


def collect_track_stats(items: list[SegmentStats]) -> TrackStats:
    """Считает медианы признаков по всем сегментам дорожки."""
    if not items:
        return TrackStats()
    return TrackStats(
        rms_db=float(np.median([s.rms_db for s in items])),
        flatness=max(float(np.median([s.flatness for s in items])), 1e-9),
        high_ratio=float(np.median([s.high_ratio for s in items])),
        flux=float(np.median([s.flux for s in items])),
    )


def is_repetition(words: list[str], max_repeat: int) -> str | None:
    """Ищет зацикленный текст: N одинаковых слов подряд (или повторяющийся блок)."""
    if max_repeat <= 0 or len(words) < max_repeat:
        return None
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= max_repeat:
                return f"повтор слова x{run} ({words[i]!r})"
        else:
            run = 1
    for size in range(1, max(2, len(words) // 2) + 1):
        block = words[:size]
        reps = 1
        while words[reps * size : (reps + 1) * size] == block:
            reps += 1
        if reps >= max_repeat and reps * size >= len(words) - size // 2:
            return f"повтор блока x{reps} ({' '.join(block)})"
    return None


def music_verdict(
    stats: SegmentStats, ctx: TrackStats, level: int, duration: float
) -> tuple[bool, str]:
    """Отсеивает отрезки, которые звучат как инструментал, а не как вокал.

    Критерии относительные: отрезок подозрителен, если он заметно «глуше» и статичнее
    медианы дорожки. Абсолютных порогов здесь нет намеренно - у каждой записи свой спектр.
    """
    if level <= 0:
        return False, ""
    if duration < 0.3:
        return True, "слишком короткий отрезок"

    dull_flat = stats.flatness < 0.25 * ctx.flatness
    dull_high = stats.high_ratio < 0.5 * max(ctx.high_ratio, 1e-9) and stats.high_ratio < 0.01
    quiet = stats.rms_db < ctx.rms_db - 10.0

    if level >= 2 and dull_flat and dull_high and quiet:
        return True, (
            f"похоже на инструментал (rms {stats.rms_db:.0f} dB против {ctx.rms_db:.0f}, "
            f"ВЧ {stats.high_ratio:.4f})"
        )
    if dull_flat and dull_high:
        return True, (
            f"мало согласных и широкой полосы (flat={stats.flatness:.5f}, ВЧ={stats.high_ratio:.4f})"
        )
    return False, ""


@dataclass
class SegmentInput:
    """Всё, что известно про сегмент к моменту проверки."""

    samples: np.ndarray
    text: str
    words: list[str]
    mean_logprob: float
    n_aligned_words: int = -1
    stats: SegmentStats = field(default_factory=lambda: segment_stats(np.zeros(1, dtype=np.float32)))


def judge_segment(
    data: SegmentInput,
    filters: FilterSpec,
    ctx: TrackStats | None = None,
) -> SegmentVerdict:
    """Решает, годится ли реплика: по энергии, повторам, уверенности и спектру."""
    dur = len(data.samples) / SR
    stats = data.stats
    if stats.rms_db < filters.min_rms_db:
        return SegmentVerdict(False, f"тихо ({stats.rms_db:.0f} dBFS)", stats)
    if filters.min_segment_sec > 0 and dur < filters.min_segment_sec:
        return SegmentVerdict(False, f"короткий отрезок ({dur:.2f}s)", stats)
    if not data.text.strip():
        return SegmentVerdict(False, "пустой текст", stats)
    if data.n_aligned_words == 0:
        return SegmentVerdict(False, f"текст не лёг на аудио: {data.text[:40]!r}", stats)
    rep = is_repetition(data.words, filters.max_repeat_words)
    if rep:
        return SegmentVerdict(False, rep, stats)
    if filters.min_confidence is not None and data.mean_logprob < filters.min_confidence:
        return SegmentVerdict(
            False,
            f"низкая уверенность ({data.mean_logprob:.2f} < {filters.min_confidence:.2f})",
            stats,
        )
    if ctx is not None:
        drop, reason = music_verdict(stats, ctx, filters.music_gate, dur)
        if drop:
            return SegmentVerdict(False, reason, stats)
    return SegmentVerdict(True, "", stats)
