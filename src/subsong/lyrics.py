"""Разбор текста песни: простой текст или LRC с таймкодами.

Готовый текст песни даёт заметно лучший результат, чем распознавание: вместо того
чтобы гадать над вокалом поверх минусовки, мы выравниваем известные слова по аудио.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LRC_TIME = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]")
META_LINE = re.compile(r"^\[(ti|ar|al|by|offset|length|re|ve):", re.IGNORECASE)


@dataclass
class LyricLine:
    """Строка песни с необязательным временем начала (LRC)."""

    text: str
    start: float | None = None
    end: float | None = None


@dataclass
class Lyrics:
    """Разобранный текст песни."""

    lines: list[LyricLine]

    @property
    def timed(self) -> bool:
        """Есть ли в тексте явные таймкоды LRC."""
        return any(line.start is not None for line in self.lines)

    @property
    def plain_text(self) -> str:
        """Весь текст песни одной строкой."""
        return " ".join(line.text for line in self.lines if line.text)


def _lrc_seconds(minutes: int, seconds: int, fraction: str | None) -> float:
    """Переводит [mm:ss.xx] в секунды."""
    frac = float(f"0.{fraction}") if fraction else 0.0
    return minutes * 60.0 + seconds + frac


def parse_lyrics(text: str) -> Lyrics:
    """Разбирает lyrics: если в строках есть [mm:ss], это LRC, иначе обычный текст."""
    entries: list[LyricLine] = []
    offset = 0.0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if META_LINE.match(line):
            m = re.match(r"^\[offset:([+-]?\d+)\]", line, re.IGNORECASE)
            if m:
                offset = int(m.group(1)) / 1000.0
            continue
        stamps = LRC_TIME.findall(line)
        if stamps:
            content = LRC_TIME.sub("", line).strip()
            for minutes, seconds, fraction in stamps:
                start = _lrc_seconds(int(minutes), int(seconds), fraction) + offset
                entries.append(LyricLine(content, max(0.0, start)))
        else:
            entries.append(LyricLine(line))
    if not entries:
        return Lyrics(entries)
    if all(item.start is None for item in entries):
        return Lyrics(entries)
    entries.sort(key=lambda item: item.start if item.start is not None else 0.0)
    for i, item in enumerate(entries):
        if item.start is None:
            continue
        nxt = next((n.start for n in entries[i + 1 :] if n.start is not None), None)
        item.end = nxt
    return Lyrics(entries)


def load_lyrics(path: str) -> Lyrics:
    """Читает файл с текстом песни (utf-8, с BOM)."""
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Файл с текстом не найден: {p}")
    return parse_lyrics(p.read_text(encoding="utf-8-sig"))
