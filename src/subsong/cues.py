"""Сборка субтитров: распределение слов по времени и нарезка на реплики."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from subsong.config import CueStyle

VOWELS = set("аеёиоуыэюяaeiouy")
WORD_RE = re.compile(r"[^\s]+")
SENTENCE_END = ".!?…:;"
CLAUSE_END = ",;—-"


@dataclass
class TimedWord:
    """Слово с абсолютным временем в секундах."""

    text: str
    start: float
    end: float
    logprob: float = 0.0
    hard_break: bool = False  # перед словом начинается новая строка песни


@dataclass
class Cue:
    """Реплика субтитров: строки плюс слова для караоке-выделения."""

    start: float
    end: float
    lines: list[str] = field(default_factory=list)
    words: list[TimedWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Текст реплики с переносами строк."""
        return "\n".join(self.lines)

    @property
    def duration(self) -> float:
        """Длительность реплики в секундах."""
        return max(0.0, self.end - self.start)

    @property
    def cps(self) -> float:
        """Скорость чтения: символов в секунду."""
        chars = sum(len(line) for line in self.lines)
        return chars / self.duration if self.duration > 0 else float("inf")


def syllables(word: str) -> int:
    """Грубое число слогов: считаем гласные, одинаковые подряд схлопываем в одну.

    Так «поёт» даёт два слога (о и ё - разные гласные), а «коол» - одно.
    """
    count = 0
    prev = ""
    for ch in word.lower():
        if ch in VOWELS and ch != prev:
            count += 1
        prev = ch
    return max(1, count)


def distribute_words(text: str, start: float, end: float) -> list[TimedWord]:
    """Распределяет текст по интервалу пропорционально числу слогов в словах."""
    tokens = WORD_RE.findall(text)
    if not tokens:
        return []
    dur = max(0.05, end - start)
    weights = [syllables(t) + 0.25 * sum(1 for c in t if c in CLAUSE_END + SENTENCE_END) for t in tokens]
    total = sum(weights)
    out: list[TimedWord] = []
    cursor = start
    for i, token in enumerate(tokens):
        share = dur * weights[i] / total
        nxt = cursor + share if i < len(tokens) - 1 else end
        out.append(TimedWord(token, cursor, max(cursor + 0.05, nxt)))
        cursor = nxt
    return out


def split_sentences(words: list[TimedWord]) -> list[list[TimedWord]]:
    """Делит слова на фразы: по знакам конца предложения и по границам строк песни."""
    out: list[list[TimedWord]] = []
    cur: list[TimedWord] = []
    for w in words:
        if w.hard_break and cur:
            out.append(cur)
            cur = []
        cur.append(w)
        if w.text and w.text[-1] in SENTENCE_END:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def wrap_lines(words: list[str], max_chars: int, max_lines: int) -> list[str]:
    """Раскладывает слова по строкам: сначала пробуем балансированно, потом жадно."""
    text = " ".join(words)
    if len(text) <= max_chars or max_lines <= 1:
        return _greedy_wrap(words, max_chars)[:max_lines]
    for n_lines in range(1, max_lines + 1):
        target = math.ceil(len(text) / n_lines)
        lines = _greedy_wrap(words, max(target, max_chars // n_lines))
        if len(lines) <= max_lines and all(len(l) <= max_chars for l in lines):
            return lines
    lines = _greedy_wrap(words, max_chars)
    return lines[:max_lines] if lines else [text]


def _greedy_wrap(words: list[str], max_chars: int) -> list[str]:
    """Жадно переносит слова по строкам заданной ширины."""
    lines: list[str] = []
    cur: list[str] = []
    length = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur and length + add > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
            length = len(w)
        else:
            cur.append(w)
            length += add
    if cur:
        lines.append(" ".join(cur))
    return lines


def _fits(words: list[TimedWord], style: CueStyle) -> bool:
    """Проверяет, укладывается ли набор слов в ограничения реплики."""
    text = " ".join(w.text for w in words)
    if len(text) > style.max_chars * style.max_lines:
        return False
    if style.max_cps <= 0:
        return True
    duration = max(0.1, words[-1].end - words[0].start)
    return len(text) / duration <= style.max_cps


def group_cues(words: list[TimedWord], style: CueStyle) -> list[Cue]:
    """Разбивает поток слов на реплики по длине, длительности и скорости чтения."""
    if not words:
        return []
    cues: list[Cue] = []
    for sentence in split_sentences(words):
        # Короткая фраза целиком всегда лучше, чем мелькающие реплики по одному слову
        if _fits(sentence, style) and (sentence[-1].end - sentence[0].start) <= style.max_duration:
            cues.append(_make_cue(sentence, style))
            continue
        cur: list[TimedWord] = []
        for w in sentence:
            cand = cur + [w]
            too_long_text = len(" ".join(x.text for x in cand)) > style.max_chars * style.max_lines
            too_long_time = cand[-1].end - cand[0].start > style.max_duration
            if cur and (too_long_text or too_long_time or not _fits(cand, style)):
                # Очень короткую реплику не режем: лучше превысить скорость чтения,
                # чем получить вспышку из одного слова (и каскад сдвигов таймингов).
                too_short = len(cur) == 1 or (cur[-1].end - cur[0].start) < style.min_duration
                if too_long_text or not too_short:
                    cues.append(_make_cue(cur, style))
                    cur = [w]
                    continue
            cur = cand
        if cur:
            cues.append(_make_cue(cur, style))
    return cues


def _make_cue(words: list[TimedWord], style: CueStyle) -> Cue:
    """Собирает реплику из слов: тайминги из слов, текст - с переносом по строкам."""
    texts = [w.text for w in words]
    lines = wrap_lines(texts, style.max_chars, style.max_lines)
    if style.uppercase:
        lines = [line.upper() for line in lines]
    return Cue(start=words[0].start, end=words[-1].end, lines=lines, words=list(words))


def finalize(cues: list[Cue], style: CueStyle, total_duration: float | None = None) -> list[Cue]:
    """Применяет финальную политику таймингов: сдвиги, минимальная длительность, паузы."""
    if not cues:
        return []
    for cue in cues:
        cue.start = max(0.0, cue.start - style.lead_in)
        cue.end = cue.end + style.lead_out
    out: list[Cue] = []
    prev_end = 0.0
    for i, cue in enumerate(cues):
        if i > 0 and cue.start < prev_end + style.min_gap:
            cue.start = prev_end + style.min_gap
        if cue.end - cue.start < style.min_duration:
            cue.end = cue.start + style.min_duration
        if total_duration:
            cue.end = min(cue.end, total_duration)
        if cue.end - cue.start <= 0.0:
            continue  # реплику выдавило за пределы аудио - выбрасываем
        prev_end = cue.end
        out.append(cue)
    return [c for c in out if c.text.strip()]
