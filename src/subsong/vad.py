"""Сегментация: Silero VAD из sherpa-onnx плюс склейка, разрезание и подрезка тишины."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subsong.config import SR, VadSpec
from subsong.features import frame_rms_db


@dataclass
class SpeechSegment:
    """Речевой отрезок: абсолютные тайминги и соответствующие сэмплы."""

    index: int
    start: float
    end: float
    samples: np.ndarray


class SileroVad:
    """Обёртка над VoiceActivityDetector из sherpa-onnx (модель silero_vad.onnx)."""

    def __init__(self, model_path: str, cfg: VadSpec) -> None:
        import sherpa_onnx

        model = sherpa_onnx.VadModelConfig()
        model.silero_vad.model = str(model_path)
        model.silero_vad.threshold = cfg.threshold
        model.silero_vad.min_silence_duration = cfg.min_silence
        model.silero_vad.min_speech_duration = cfg.min_speech
        model.silero_vad.max_speech_duration = cfg.max_speech
        model.sample_rate = SR
        model.provider = "cpu"
        if not model.validate():
            raise RuntimeError(f"Некорректная конфигурация VAD: {model}")
        self._vad = sherpa_onnx.VoiceActivityDetector(config=model, buffer_size_in_seconds=cfg.buffer_sec)
        self.window = int(model.silero_vad.window_size) or 512

    def segments(self, samples: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
        """Прогоняет аудио через VAD и возвращает список (start, end, samples)."""
        self._vad.reset()
        out: list[tuple[float, float, np.ndarray]] = []
        step = self.window
        for pos in range(0, len(samples), step):
            self._vad.accept_waveform(samples[pos : pos + step])
            while not self._vad.empty():
                seg = self._vad.front
                start_sample = int(seg.start)
                chunk = np.asarray(seg.samples, dtype=np.float32)
                self._vad.pop()
                start = start_sample / SR
                out.append((start, start + len(chunk) / SR, chunk))
        self._vad.flush()
        while not self._vad.empty():
            seg = self._vad.front
            start_sample = int(seg.start)
            chunk = np.asarray(seg.samples, dtype=np.float32)
            self._vad.pop()
            start = start_sample / SR
            out.append((start, start + len(chunk) / SR, chunk))
        return out

    def speech_seconds(self, samples: np.ndarray, min_segment_sec: float = 0.4) -> float:
        """Суммарная длительность речи - быстрая проверка «в потоке вообще что-то есть»."""
        total = 0.0
        for start, end, _chunk in self.segments(samples):
            if end - start >= min_segment_sec:
                total += end - start
        return total


def merge_segments(
    segments: list[SpeechSegment], gap: float, max_dur: float | None = None
) -> list[SpeechSegment]:
    """Склеивает соседние отрезки, если пауза между ними меньше gap секунд."""
    if not segments:
        return []
    merged: list[SpeechSegment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        pause = seg.start - prev.end
        too_long = max_dur is not None and (seg.end - prev.start) > max_dur
        if pause <= gap and not too_long:
            merged[-1] = SpeechSegment(
                index=prev.index,
                start=prev.start,
                end=seg.end,
                samples=np.concatenate([prev.samples, seg.samples]),
            )
        else:
            merged.append(seg)
    return merged


def trim_silence(samples: np.ndarray, start: float, keep_db: float = 30.0) -> tuple[float, float, np.ndarray]:
    """Подрезает тишину по краям: оставляет хвост громкости в пределах keep_db от пика."""
    if samples.size == 0:
        return start, start, samples
    hop = SR // 50
    rms = frame_rms_db(samples)
    peak = float(rms.max())
    active = np.flatnonzero(rms >= peak - keep_db)
    if active.size == 0:
        return start, start + len(samples) / SR, samples
    lo = max(0, int(active[0]) - 2)
    hi = min(len(rms), int(active[-1]) + 3)
    cut = samples[lo * hop : hi * hop]
    if cut.size == 0:
        return start, start + len(samples) / SR, samples
    return start + lo / 50.0, start + hi / 50.0, cut


def split_long(segment: SpeechSegment, max_dur: float) -> list[SpeechSegment]:
    """Режет слишком длинный отрезок в тихих местах, иначе - равными частями."""
    dur = segment.end - segment.start
    if max_dur <= 0 or dur <= max_dur:
        return [segment]
    n_parts = int(np.ceil(dur / max_dur))
    bounds = [dur * i / n_parts for i in range(n_parts + 1)]
    rms = frame_rms_db(segment.samples)
    # ищем самый тихий 10-центовый участок рядом с каждой границей
    hop_sec = 0.02
    window = max(1, int(0.1 / hop_sec))
    for i in range(1, n_parts):
        want = bounds[i]
        center = int(want / hop_sec)
        lo = max(1, center - window)
        hi = min(len(rms) - 1, center + window)
        if hi <= lo:
            continue
        bounds[i] = (lo + int(np.argmin(rms[lo:hi]))) * hop_sec
    for i in range(1, n_parts):
        bounds[i] = max(bounds[i], bounds[i - 1] + 0.5)
    out: list[SpeechSegment] = []
    for i in range(n_parts):
        a, b = bounds[i], bounds[i + 1]
        lo_s = int(a * SR)
        hi_s = int(b * SR)
        out.append(
            SpeechSegment(
                index=segment.index,
                start=segment.start + a,
                end=segment.start + b,
                samples=segment.samples[lo_s:hi_s],
            )
        )
    return out


def build_segments(
    samples: np.ndarray,
    vad: SileroVad,
    *,
    merge_gap: float = 0.25,
    max_segment_sec: float = 20.0,
    trim_db: float = 30.0,
    min_segment_sec: float = 0.0,
) -> list[SpeechSegment]:
    """Полный путь от аудио к списку пригодных для распознавания отрезков."""
    raw = vad.segments(samples)
    segments: list[SpeechSegment] = []
    for i, (start, _end, chunk) in enumerate(raw):
        t0, t1, cut = trim_silence(chunk, start, trim_db)
        if cut.size < SR * 0.05:
            continue
        segments.append(SpeechSegment(index=i, start=t0, end=t1, samples=cut))
    segments = merge_segments(segments, merge_gap)
    split: list[SpeechSegment] = []
    for seg in segments:
        split.extend(split_long(seg, max_segment_sec))
    for i, seg in enumerate(split):
        seg.index = i
    if min_segment_sec > 0:
        split = [s for s in split if (s.end - s.start) >= min_segment_sec]
    return split
