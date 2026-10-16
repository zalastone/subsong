"""Запись субтитров в SRT, VTT, караоке-ASS, JSON и простой текст."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from subsong.cues import Cue


@dataclass
class Meta:
    """Служебная информация о запуске, попадает в JSON и в шапку VTT."""

    source: str = ""
    model: str = ""
    duration: float = 0.0
    words_per_sec: float = 0.0
    audio_duration: float = 0.0
    extra: dict[str, object] = field(default_factory=dict)


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Форматирует время как HH:MM:SS,mmm (SRT) или HH:MM:SS.mmm (VTT)."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    hours, ms_total = divmod(ms_total, 3_600_000)
    minutes, ms_total = divmod(ms_total, 60_000)
    secs, ms = divmod(ms_total, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def write_srt(cues: list[Cue], path: Path) -> Path:
    """Пишет обычный SRT (без BOM, с CRLF - так надёжнее для проигрывателей)."""
    blocks = []
    for i, cue in enumerate(cues, start=1):
        blocks.append(
            f"{i}\n{format_timestamp(cue.start)} --> {format_timestamp(cue.end)}\n{cue.text}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks), encoding="utf-8")
    return path


def write_vtt(cues: list[Cue], path: Path, meta: Meta | None = None) -> Path:
    """Пишет WebVTT; при наличии meta добавляет заголовок с источником."""
    lines = ["WEBVTT"]
    if meta and meta.source:
        lines.append(f"NOTE Источник: {meta.source}")
        if meta.model:
            lines.append(f"NOTE Модель: {meta.model}")
    if meta and meta.duration:
        lines.append("")
        lines.append(format_timestamp(meta.duration, sep="."))
    for cue in cues:
        lines.append("")
        lines.append(f"{format_timestamp(cue.start, '.')} --> {format_timestamp(cue.end, '.')}")
        lines.append(cue.text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_ass(
    cues: list[Cue],
    path: Path,
    *,
    font: str = "Arial",
    font_size: int = 48,
    primary: str = "&H00FFFFFF",
    highlight: str = "&H0000D7FF",
    margin_v: int = 60,
    karaoke: bool = True,
) -> Path:
    """Пишет ASS с подсветкой по словам (теги `\\k`), как в караоке-файлах.

    `\\k` задаёт длительность слова в центрисекундах, поэтому проигрыватель
    подсвечивает слово по мере пения.
    """
    head = f"""[Script Info]
Title: subsong
ScriptType: v4.00+
WrapStyle: 2
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{primary},{highlight},&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for cue in cues:
        text = _ass_line(cue, karaoke)
        events.append(
            "Dialogue: 0,"
            f"{_ass_time(cue.start)},{_ass_time(cue.end)},"
            f"Default,,0,0,0,,{text}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + "\n".join(events) + "\n", encoding="utf-8")
    return path


def _ass_line(cue: Cue, karaoke: bool) -> str:
    """Собирает текст реплики ASS: либо обычный, либо с тегами караоке `\\k`."""
    if not karaoke or not cue.words:
        return cue.text.replace("\n", "\\N")
    parts: list[str] = []
    idx = 0
    for line in cue.text.split("\n"):
        pieces: list[str] = []
        for token in line.split():
            w = cue.words[idx] if idx < len(cue.words) else None
            idx += 1
            if w is None:
                pieces.append(token)
                continue
            dur_cs = max(1, int(round((w.end - w.start) * 100.0)))
            pieces.append(f"{{\\k{dur_cs}}}{token}")
        if pieces:
            parts.append("".join(pieces))
    return "\\N".join(parts)


def _ass_time(seconds: float) -> str:
    """Время в формате ASS: H:MM:SS.cc (сотые доли секунды)."""
    cs_total = int(round(max(0.0, seconds) * 100.0))
    hours, cs_total = divmod(cs_total, 360_000)
    minutes, cs_total = divmod(cs_total, 6_000)
    secs, cs = divmod(cs_total, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def write_json(cues: list[Cue], path: Path, meta: Meta | None = None) -> Path:
    """Пишет подробный JSON: реплики, слова и метаданные (для внешней обработки)."""
    data: dict[str, object] = {
        "cues": [
            {
                "start": round(c.start, 3),
                "end": round(c.end, 3),
                "text": c.text,
                "lines": c.lines,
                "cps": round(c.cps, 2),
                "words": [
                    {
                        "text": w.text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "logprob": round(w.logprob, 4),
                    }
                    for w in c.words
                ],
            }
            for c in cues
        ]
    }
    if meta:
        data["meta"] = {
            "source": meta.source,
            "model": meta.model,
            "audio_duration": round(meta.audio_duration, 3),
            "processing_seconds": round(meta.words_per_sec, 3),
            **dict(meta.extra),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_txt(cues: list[Cue], path: Path) -> Path:
    """Пишет таймкод + текст (удобно для вычитки и поиска ошибок)."""
    blocks = [
        f"[{format_timestamp(c.start, '.')} --> {format_timestamp(c.end, '.')}] {c.text.replace(chr(10), ' / ')}"
        for c in cues
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return path
