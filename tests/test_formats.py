"""Тесты записи субтитров в разные форматы."""
from __future__ import annotations

import json

import pytest

from subsong.cues import Cue, TimedWord
from subsong.formats import (
    Meta,
    format_timestamp,
    write_ass,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)


def sample_cues() -> list[Cue]:
    return [
        Cue(
            start=1.234,
            end=3.5,
            lines=["Первая строка", "вторая строка"],
            words=[
                TimedWord("Первая", 1.234, 1.9, -0.1),
                TimedWord("строка", 1.9, 2.4, -0.2),
                TimedWord("вторая", 2.5, 2.9, -0.3),
                TimedWord("строка", 2.9, 3.5, -0.4),
            ],
        ),
        Cue(start=4.0, end=5.25, lines=["Одна фраза"], words=[TimedWord("Одна", 4.0, 4.6)]),
    ]


def test_format_timestamp_srt() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(3661.234) == "01:01:01,234"
    assert format_timestamp(-5) == "00:00:00,000"


def test_format_timestamp_vtt_uses_dot() -> None:
    assert format_timestamp(1.5, sep=".") == "00:00:01.500"


def test_write_srt_structure(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = write_srt(sample_cues(), tmp_path / "out.srt")
    text = path.read_text(encoding="utf-8")
    blocks = text.strip().split("\n\n")
    assert blocks[0].startswith("1\n")
    assert "00:00:01,234 --> 00:00:03,500" in blocks[0]
    assert "Первая строка\nвторая строка" in blocks[0]
    assert blocks[1].startswith("2\n")
    assert len(blocks) == 2


def test_write_vtt_has_header_and_dots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    meta = Meta(source="song.mp4", model="nemo_ctc", duration=10.0)
    path = write_vtt(sample_cues(), tmp_path / "out.vtt", meta)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "Источник: song.mp4" in text
    assert "00:00:01.234 --> 00:00:03.500" in text


def test_write_ass_has_karaoke_tags(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = write_ass(sample_cues(), tmp_path / "out.ass")
    text = path.read_text(encoding="utf-8")
    assert "[Script Info]" in text and "[Events]" in text
    first = [l for l in text.splitlines() if l.startswith("Dialogue:")][0]
    assert "{\\k67}Первая" in first  # 0.67 c -> 67 центiсекунд
    assert "\\N" in first
    assert "0:00:01.23,0:00:03.50" in first


def test_write_ass_without_karaoke(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = write_ass(sample_cues(), tmp_path / "out.ass", karaoke=False)
    text = path.read_text(encoding="utf-8")
    assert "\\k" not in text
    assert "Первая строка\\Nвторая строка" in text


def test_write_json_contains_words(tmp_path) -> None:  # type: ignore[no-untyped-def]
    meta = Meta(source="song.mp4", model="nemo_ctc", audio_duration=6.0, words_per_sec=4.2)
    path = write_json(sample_cues(), tmp_path / "out.json", meta)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["cues"]) == 2
    assert data["cues"][0]["words"][0]["text"] == "Первая"
    assert data["cues"][0]["words"][0]["start"] == pytest.approx(1.234, abs=1e-3)
    assert data["meta"]["source"] == "song.mp4"


def test_write_txt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = write_txt(sample_cues(), tmp_path / "out.txt")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("[00:00:01.234 --> 00:00:03.500]")
    assert " / " in lines[0]