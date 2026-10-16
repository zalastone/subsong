"""Тесты разбора текста песни: обычный текст и LRC."""
from __future__ import annotations

from subsong.lyrics import parse_lyrics


def test_parse_plain_text() -> None:
    lyrics = parse_lyrics("Первая строка\n\nВторая строка\nТретья\n")
    assert [l.text for l in lyrics.lines] == ["Первая строка", "Вторая строка", "Третья"]
    assert not lyrics.timed
    assert lyrics.plain_text == "Первая строка Вторая строка Третья"


def test_parse_lrc_timestamps() -> None:
    lyrics = parse_lyrics(
        "[ti: песня]\n[ar: кто-то]\n"
        "[00:12.50]Первая строка\n"
        "[00:15.00]Вторая строка\n"
        "[00:18.25]Третья строка\n"
    )
    assert lyrics.timed
    assert [l.text for l in lyrics.lines] == ["Первая строка", "Вторая строка", "Третья строка"]
    assert lyrics.lines[0].start == 12.5
    # конец строки = начало следующей
    assert lyrics.lines[0].end == 15.0
    assert lyrics.lines[2].end is None


def test_parse_lrc_minutes_and_colons() -> None:
    lyrics = parse_lyrics("[01:02:03]Строка\n[62:04]Другая\n")
    assert lyrics.lines[0].start == 62.03
    assert lyrics.lines[1].start == 62 * 60 + 4.0


def test_parse_lrc_offset() -> None:
    lyrics = parse_lyrics("[offset:+500]\n[00:10.00]Строка\n")
    assert lyrics.lines[0].start == 10.5


def test_parse_lrc_multiple_timestamps_on_one_line() -> None:
    lyrics = parse_lyrics("[00:10.00][00:20.00]Припев\n")
    assert [l.start for l in lyrics.lines] == [10.0, 20.0]
    assert all(l.text == "Припев" for l in lyrics.lines)


def test_parse_empty_text_keeps_instrumental_gaps() -> None:
    lyrics = parse_lyrics("[00:05.00]\n[00:09.00]Текст\n")
    assert [l.text for l in lyrics.lines] == ["", "Текст"]


def test_load_lyrics_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "words.txt"
    path.write_text("\ufeffМы идём по свету\n", encoding="utf-8")
    lyrics = parse_lyrics(path.read_text(encoding="utf-8-sig"))
    assert lyrics.lines[0].text == "Мы идём по свету"