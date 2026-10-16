"""Пайплайн: аудио -> реплики субтитров.

Три режима работы:

* **ASR** - сегментация Silero VAD, распознавание каждого отрезка, отсев мусора;
* **текст песни с таймкодами (LRC)** - каждая строка выравнивается по своему окну,
  это самый точный режим для караоке;
* **простой текст песни** - строки распределяются по речевым сегментам и выравниваются;
  если слов в сегменте оказалось слишком много, текст перераспределяется и пробуем снова.

Общий финал: сборка слов в реплики по правилам стиля и запись в нужные форматы.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from subsong import audio as audio_io
from subsong import gate as gate_mod
from subsong import preprocess
from subsong.aligner import AlignedResult, CtcAligner, WordSpan
from subsong.config import SR, Config
from subsong.cues import Cue, TimedWord, distribute_words, finalize, group_cues
from subsong.features import SegmentStats, segment_stats
from subsong.lyrics import Lyrics, LyricLine
from subsong.recognizer import SherpaRecognizer
from subsong.vad import SpeechSegment, SileroVad, build_segments

Logger = Callable[[str], None]
# Насколько расширять окно строки песни за её таймкоды (запас на вдох и согласные)
LYRIC_PRE_ROLL = 0.25
LYRIC_POST_ROLL = 0.35


@dataclass
class SegmentReport:
    """Что произошло с одним отрезком - пригодится для --report и отладки."""

    index: int
    start: float
    end: float
    text: str
    confidence: float
    kept: bool
    reason: str


@dataclass
class Result:
    """Итог работы пайплайна."""

    cues: list[Cue] = field(default_factory=list)
    reports: list[SegmentReport] = field(default_factory=list)
    audio_duration: float = 0.0
    processing_seconds: float = 0.0
    words_per_sec: float = 0.0
    model: str = ""


# ------------------------------------------------------------------ подготовка аудио
def prepare_audio(
    source: Path,
    cfg: Config,
    log: Logger,
    start: float | None = None,
    end: float | None = None,
    vad: SileroVad | None = None,
) -> np.ndarray:
    """Декодирует аудио в 16 кГц моно, при необходимости пропустив через предобработку."""
    samples = audio_io.decode(source, sr=SR, start=start, end=end)
    if samples.size == 0:
        raise audio_io.AudioError(f"В файле {source.name} нет аудиодорожки")
    samples = audio_io.normalize(samples, cfg.runtime.normalize)

    if cfg.denoise.enabled and cfg.denoise.model_path is not None:
        cleaned, _ = preprocess.denoise_array(source, cfg.denoise.model_path, log)
        samples = cleaned

    if cfg.separate.enabled and cfg.separate.model_dir is not None:
        log("Выделяю вокал из минусовки (Spleeter/UVR)")
        src = source
        if cfg.runtime.temp_dir is not None:
            cfg.runtime.temp_dir.mkdir(parents=True, exist_ok=True)
            src = audio_io.write_wav(cfg.runtime.temp_dir / "mix.wav", samples, SR)
        vocals, sr_v = preprocess.separate_vocals_array(
            src, cfg.separate.model_dir, cfg.separate.stem, log=log
        )
        vocals = _match_sample_rate(vocals, sr_v, cfg)
        problem = _is_broken_stem(vocals, samples)
        if problem is None and vad is not None:
            # Главная проверка: в удачном разделении речи остаётся почти столько же,
            # сколько было в миксе. Если вокал "исчез" - модель не справилась с записью.
            mix_speech = vad.speech_seconds(samples)
            stem_speech = vad.speech_seconds(vocals)
            if mix_speech > 0.5 and stem_speech < 0.4 * mix_speech:
                problem = f"в выделенном потоке речи почти нет ({stem_speech:.1f} c против {mix_speech:.1f} c)"
        if problem:
            log(f"Выделение вокала не удалось: {problem} - продолжаю с исходным миксом")
        else:
            samples = vocals
    return samples


def _is_broken_stem(candidate: np.ndarray, reference: np.ndarray) -> str | None:
    """Проверяет осмысленность выделенного потока: возвращает причину, если он плохой."""

    def rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))

    cand, ref = rms(candidate), rms(reference)
    if ref <= 1e-9:
        return "исходное аудио пустое"
    ratio = cand / ref
    if ratio < 0.02:
        return f"поток почти пустой ({ratio:.3f} от исходника)"
    if ratio > 0.95:
        return f"поток не отделён от микса ({ratio:.2f} от исходника)"
    return None


def _match_sample_rate(samples: np.ndarray, src_sr: int, cfg: Config) -> np.ndarray:
    """Приводит обработанный звук к частоте пайплайна (16 кГц)."""
    if src_sr == SR or samples.size == 0:
        return samples.astype(np.float32)
    tmp = cfg.runtime.temp_dir / "processed.wav" if cfg.runtime.temp_dir else None
    if tmp is not None:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        audio_io.write_wav(tmp, samples, src_sr)
        return audio_io.decode(tmp, sr=SR)
    ratio = SR / float(src_sr)
    idx = np.clip((np.arange(int(len(samples) * ratio)) / ratio).astype(int), 0, len(samples) - 1)
    return np.asarray(samples[idx], dtype=np.float32)


# ------------------------------------------------------------------ распознавание
def _build_engines(cfg: Config, align_words: bool) -> tuple[CtcAligner | None, SherpaRecognizer | None]:
    """Создаёт движки распознавания: CTC-выравниватель или обёртку sherpa-onnx."""
    if align_words and cfg.alignable:
        aligner = CtcAligner(
            cfg.model_file("model"),
            cfg.model_file("tokens"),
            cfg.runtime.num_threads,
            cfg.runtime.provider,
            feature_dim=cfg.feature_dim,
        )
        return aligner, None
    return None, SherpaRecognizer(cfg)


def _to_timed_words(words: list[WordSpan], offset: float) -> list[TimedWord]:
    """Переводит слова выравнивания в абсолютные тайминги пайплайна."""
    return [TimedWord(w.text, offset + w.start, offset + w.end, w.logprob) for w in words]


def _transcribe_segment(
    aligner: CtcAligner | None,
    recognizer: SherpaRecognizer | None,
    samples: np.ndarray,
    offset: float,
    hint: str | None = None,
) -> tuple[str, list[TimedWord], float]:
    """Распознаёт один отрезок нужным движком: текст, слова, уверенность."""
    if aligner is not None:
        result: AlignedResult = aligner.transcribe(samples, text_hint=hint)
        return result.text, _to_timed_words(result.words, offset), result.mean_logprob
    if recognizer is not None:
        tr = recognizer.transcribe(samples, offset=offset)
        words = tr.words or distribute_words(tr.text, offset, offset + len(samples) / SR)
        return tr.text, words, 0.0
    raise RuntimeError("Не инициализирован ни один движок распознавания")


def _make_input(
    samples: np.ndarray,
    text: str,
    words: list[TimedWord],
    conf: float,
    stats: SegmentStats,
) -> gate_mod.SegmentInput:
    """Собирает описание сегмента для проверок."""
    plain = [w.text for w in words] or text.split()
    return gate_mod.SegmentInput(
        samples=samples,
        text=text,
        words=plain,
        mean_logprob=conf,
        n_aligned_words=len(words),
        stats=stats,
    )


# ------------------------------------------------------------------ режим ASR
def _run_asr(
    segments: list[SpeechSegment],
    cfg: Config,
    aligner: CtcAligner | None,
    recognizer: SherpaRecognizer | None,
    log: Logger,
) -> tuple[list[TimedWord], list[SegmentReport]]:
    """Распознаёт речевые сегменты и отбрасывает то, что не похоже на вокал."""
    prepared: list[tuple[SpeechSegment, str, list[TimedWord], float, SegmentStats]] = []
    for seg in segments:
        text, seg_words, conf = _transcribe_segment(aligner, recognizer, seg.samples, seg.start)
        prepared.append((seg, text, seg_words, conf, segment_stats(seg.samples)))

    ctx = gate_mod.collect_track_stats([item[4] for item in prepared])
    words: list[TimedWord] = []
    reports: list[SegmentReport] = []
    for seg, text, seg_words, conf, stats in prepared:
        verdict = gate_mod.judge_segment(_make_input(seg.samples, text, seg_words, conf, stats), cfg.filters, ctx)
        reports.append(
            SegmentReport(seg.index, seg.start, seg.end, text, conf, verdict.keep, verdict.reason)
        )
        if verdict.keep and seg_words:
            words.extend(seg_words)
        elif not verdict.keep:
            log(f"  [{seg.start:6.2f}-{seg.end:6.2f}] пропущено: {verdict.reason} ({text[:60]!r})")
    return words, reports


# ------------------------------------------------------------------ режим LRC
def _run_timed_lyrics(
    samples: np.ndarray,
    lyrics: Lyrics,
    cfg: Config,
    aligner: CtcAligner | None,
    recognizer: SherpaRecognizer | None,
    log: Logger,
) -> tuple[list[TimedWord], list[SegmentReport]]:
    """Выравнивает каждую строку песни по своему временному окну (точнее всего)."""
    timed = [l for l in lyrics.lines if l.start is not None and l.text.strip()]
    total = len(samples) / SR
    prepared: list[tuple[int, float, float, np.ndarray, str, list[TimedWord], float, SegmentStats]] = []
    for i, line in enumerate(timed):
        line_start = float(line.start or 0.0)
        line_end = float(line.end if line.end is not None else line_start + 3.0)
        # окно шире таймкода: вдох перед строкой и хвост после неё
        win_start = max(0.0, line_start - LYRIC_PRE_ROLL)
        win_end = min(total, line_end + LYRIC_POST_ROLL)
        if win_end <= win_start:
            continue
        chunk = samples[int(win_start * SR) : int(win_end * SR)]
        if chunk.size < SR * 0.15:
            continue
        text, line_words, conf = _transcribe_segment(
            aligner, recognizer, chunk, win_start, hint=line.text
        )
        prepared.append((i, win_start, win_end, chunk, text, line_words, conf, segment_stats(chunk)))

    ctx = gate_mod.collect_track_stats([item[7] for item in prepared])
    words: list[TimedWord] = []
    reports: list[SegmentReport] = []
    for i, win_start, win_end, chunk, text, line_words, conf, stats in prepared:
        verdict = gate_mod.judge_segment(
            _make_input(chunk, text, line_words, conf, stats), cfg.filters, ctx
        )
        reports.append(
            SegmentReport(i, win_start, win_end, text, conf, verdict.keep, verdict.reason)
        )
        if verdict.keep and line_words:
            line_words[0].hard_break = True
            words.extend(line_words)
        elif not verdict.keep:
            log(f"  [{win_start:6.2f}-{win_end:6.2f}] строка пропущена: {verdict.reason}")
    return words, reports


# ------------------------------------------------------------------ режим plain-текста
def _run_plain_lyrics(
    segments: list[SpeechSegment],
    lyrics: Lyrics,
    cfg: Config,
    aligner: CtcAligner | None,
    recognizer: SherpaRecognizer | None,
    log: Logger,
) -> tuple[list[TimedWord], list[SegmentReport]]:
    """Распределяет строки текста по речевым сегментам и выравнивает их.

    Текст раздаётся сегментам пропорционально накопленному времени речи, поэтому ранние
    сегменты не «съедают» весь текст. Если выравнивание неуверенное (значит, слов
    в сегменте было слишком много), пробуем ещё раз с меньшей порцией текста.
    """
    lines = [l for l in lyrics.lines if l.text.strip()]
    total_words = sum(len(l.text.split()) for l in lines)
    speech_time = sum(s.end - s.start for s in segments)
    if total_words == 0 or speech_time <= 0:
        return [], []
    log(
        f"Текста {total_words} слов на {speech_time:.1f} c речи "
        f"(~{total_words / speech_time:.2f} слов/с)"
    )

    plan, warning = _plan_text_by_segments(lines, segments, total_words, speech_time)
    if warning:
        log(f"Внимание: {warning}")

    prepared: list[tuple[TextPlan, str, list[TimedWord], float, SegmentStats]] = []
    for item in plan:
        best: tuple[str, list[TimedWord], float] | None = None
        for take in item.attempts:
            hint = " ".join(item.words[:take])
            text, seg_words, conf = _transcribe_segment(
                aligner, recognizer, item.segment.samples, item.segment.start, hint=hint
            )
            best = (text, seg_words, conf)
            # хватает ли уверенности - иначе пробуем отдать сегменту меньше текста
            if conf >= (cfg.filters.min_confidence or -4.0) or not seg_words:
                break
        assert best is not None
        text, seg_words, conf = best
        prepared.append((item, text, seg_words, conf, segment_stats(item.segment.samples)))

    ctx = gate_mod.collect_track_stats([p[4] for p in prepared])
    words: list[TimedWord] = []
    reports: list[SegmentReport] = []
    for item, text, seg_words, conf, stats in prepared:
        seg = item.segment
        verdict = gate_mod.judge_segment(_make_input(seg.samples, text, seg_words, conf, stats), cfg.filters, ctx)
        reports.append(
            SegmentReport(seg.index, seg.start, seg.end, text, conf, verdict.keep, verdict.reason)
        )
        if verdict.keep and seg_words:
            if item.starts_line:
                seg_words[0].hard_break = True
            words.extend(seg_words)
        elif not verdict.keep:
            log(f"  [{seg.start:6.2f}-{seg.end:6.2f}] пропущено: {verdict.reason} ({text[:60]!r})")
    return words, reports


@dataclass
class TextPlan:
    """Что и в каком объёме выравнивать в очередном сегменте."""

    segment: SpeechSegment
    words: list[str]
    attempts: list[int]  # сколько слов пробовать (от большего к меньшему)
    starts_line: bool  # начинается ли здесь новая строка песни


def _plan_text_by_segments(
    lines: list[LyricLine],
    segments: list[SpeechSegment],
    total_words: int,
    speech_time: float,
) -> tuple[list[TextPlan], str | None]:
    """Решает, какие слова достаются каждому сегменту, и какие объёмы попробовать."""
    plan: list[TextPlan] = []
    elapsed = 0.0
    used = 0
    line_idx = 0
    line_used = 0
    for seg in segments:
        elapsed += seg.end - seg.start
        target = int(round(total_words * elapsed / speech_time))
        if line_idx >= len(lines):
            break
        chunk: list[str] = []
        starts_line = False
        while line_idx < len(lines) and used + len(chunk) < target:
            starts_line = starts_line or not chunk
            line_words = lines[line_idx].text.split()
            room = target - used - len(chunk)
            take = max(0, min(len(line_words) - line_used, room))
            chunk.extend(line_words[line_used : line_used + take])
            line_used += take
            if line_used >= len(line_words):
                line_idx += 1
                line_used = 0
        used += len(chunk)
        if not chunk:
            continue
        attempts = sorted(
            {len(chunk), max(1, int(len(chunk) * 0.75)), max(1, int(len(chunk) * 0.5))}, reverse=True
        )
        plan.append(TextPlan(seg, chunk, attempts, starts_line))
    warning = None
    if line_idx < len(lines):
        warning = f"{len(lines) - line_idx} строк текста не нашли своего сегмента"
    return plan, warning


def _shift_lyrics(lyrics: Lyrics | None, offset: float) -> Lyrics | None:
    """Сдвигает таймкоды LRC на начало обрабатываемого фрагмента."""
    if lyrics is None or not lyrics.timed or offset <= 0:
        return lyrics
    shifted = [
        LyricLine(
            line.text,
            max(0.0, (line.start or 0.0) - offset),
            None if line.end is None else max(0.0, line.end - offset),
        )
        for line in lyrics.lines
    ]
    return Lyrics(shifted)


def run(
    source: Path,
    cfg: Config,
    *,
    log: Logger = lambda msg: None,
    lyrics: Lyrics | None = None,
    align_words: bool = True,
    start: float | None = None,
    end: float | None = None,
) -> Result:
    """Прогоняет весь пайплайн и возвращает реплики субтитров."""
    started = time.perf_counter()
    vad = SileroVad(str(cfg.vad_model), cfg.vad)
    samples = prepare_audio(source, cfg, log, start=start, end=end, vad=vad)
    duration = len(samples) / SR
    log(f"Аудио: {duration:.1f} c")
    if start:
        # тайминги считаются от 0 внутри фрагмента, поэтому сдвигаем LRC на начало
        lyrics = _shift_lyrics(lyrics, start)

    aligner, recognizer = _build_engines(cfg, align_words)

    if lyrics is not None and lyrics.timed:
        log(f"Режим: выравнивание текста песни по таймкодам ({len(lyrics.lines)} строк)")
        words, reports = _run_timed_lyrics(samples, lyrics, cfg, aligner, recognizer, log)
    else:
        segments = build_segments(
            samples,
            vad,
            merge_gap=cfg.vad.min_silence,
            max_segment_sec=cfg.vad.max_speech,
            trim_db=30.0,
            min_segment_sec=cfg.filters.min_segment_sec,
        )
        log(f"Сегментов речи: {len(segments)}")
        if lyrics is not None:
            words, reports = _run_plain_lyrics(segments, lyrics, cfg, aligner, recognizer, log)
        else:
            words, reports = _run_asr(segments, cfg, aligner, recognizer, log)

    cues = finalize(group_cues(words, cfg.style), cfg.style, duration)
    elapsed = time.perf_counter() - started
    return Result(
        cues=cues,
        reports=reports,
        audio_duration=duration,
        processing_seconds=elapsed,
        words_per_sec=round(len(words) / elapsed, 2) if elapsed > 0 else 0.0,
        model=cfg.asr_kind,
    )
