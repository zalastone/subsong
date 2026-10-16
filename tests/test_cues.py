"""Тесты сборки субтитров: переносы, разбивка на реплики, финальные тайминги."""
from __future__ import annotations

import pytest

from subsong.config import CueStyle
from subsong.cues import (
    TimedWord,
    distribute_words,
    finalize,
    group_cues,
    split_sentences,
    syllables,
    wrap_lines,
)


def words(*pairs: tuple[str, float, float]) -> list[TimedWord]:
    return [TimedWord(text, start, end) for text, start, end in pairs]


def test_syllables_counts_vowel_groups() -> None:
    assert syllables("мы") == 1
    assert syllables("поёт") == 2
    assert syllables("вместе") == 2
    assert syllables("песня") == 2
    assert syllables("ярмарка") == 3


def test_distribute_words_is_monotonic() -> None:
    out = distribute_words("мы идём по свету", 10.0, 12.0)
    assert [w.text for w in out] == ["мы", "идём", "по", "свету"]
    assert out[0].start == pytest.approx(10.0)
    assert out[-1].end == pytest.approx(12.0, abs=1e-6)
    for a, b in zip(out, out[1:]):
        assert a.end <= b.start + 1e-6
        assert b.start >= a.start


def test_wrap_lines_respects_width() -> None:
    lines = wrap_lines(["одна", "два", "три", "четыре", "пять"], max_chars=12, max_lines=2)
    assert all(len(l) <= 12 for l in lines)
    assert len(lines) <= 2
    assert " ".join(lines).split() == ["одна", "два", "три", "четыре", "пять"]


def test_split_sentences_respects_hard_break() -> None:
    flow = words(("первая", 0.0, 0.5), ("строка", 0.5, 1.0))
    flow[1].hard_break = True
    flow.extend(words(("вторая", 1.0, 1.4), ("строка.", 1.4, 2.0)))
    groups = split_sentences(flow)
    assert [[w.text for w in g] for g in groups] == [["первая"], ["строка", "вторая", "строка."]]


def test_group_cues_keeps_short_sentence_whole(style: CueStyle) -> None:
    flow = words(("да", 0.0, 0.2), ("вай", 0.2, 0.4))
    cues = group_cues(flow, style)
    assert len(cues) == 1
    assert cues[0].text == "да вай"


def test_group_cues_splits_long_sentence(style: CueStyle) -> None:
    long_sentence = "слово " * 30
    flow = distribute_words(long_sentence.strip(), 0.0, 12.0)
    cues = group_cues(flow, style)
    assert len(cues) > 1
    for cue in cues:
        assert len(cue.text) <= style.max_chars * style.max_lines


def test_group_cues_respects_max_duration() -> None:
    style = CueStyle(max_chars=80, max_lines=2, max_duration=2.0, min_duration=0.1, max_cps=0)
    flow = distribute_words("раз два три четыре пять шесть", 0.0, 10.0)
    cues = group_cues(flow, style)
    assert len(cues) >= 3


def test_finalize_forces_min_duration_and_gap() -> None:
    style = CueStyle(min_duration=1.0, min_gap=0.1, lead_out=0.0, lead_in=0.0, max_cps=0)
    flow = words(("раз", 0.0, 0.1), ("два", 0.2, 0.25), ("три", 0.3, 0.35))
    cues = finalize(group_cues(flow, style), style, total_duration=10.0)
    assert len(cues) == 1  # всё влезает в одну реплику
    assert cues[0].end - cues[0].start >= 1.0


def test_group_cues_splits_when_reading_is_too_fast() -> None:
    style = CueStyle(max_chars=42, max_lines=2, max_duration=7.0, min_duration=0.1, max_cps=20.0)
    flow = words(("раз", 0.0, 0.1), ("два", 0.2, 0.25), ("три", 0.3, 0.35))
    assert len(group_cues(flow, style)) > 1


def test_finalize_no_overlap_between_cues() -> None:
    style = CueStyle(max_chars=10, max_lines=1, max_duration=1.0, min_duration=1.0, max_cps=0)
    flow = distribute_words("один два три четыре пять шесть семь", 0.0, 3.5)
    cues = finalize(group_cues(flow, style), style)
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.end - 1e-6


def test_finalize_clips_to_audio_length() -> None:
    style = CueStyle(min_duration=0.5, min_gap=0.0, lead_out=2.0)
    flow = words(("да", 4.0, 4.2))
    cues = finalize(group_cues(flow, style), style, total_duration=5.0)
    assert cues[0].end <= 5.0


def test_finalize_drops_empty_cues() -> None:
    from subsong.cues import Cue

    style = CueStyle()
    cues = finalize([Cue(0.0, 1.0, ["  "], [])], style)
    assert cues == []


def test_cue_cps(style: CueStyle) -> None:
    flow = words(("раз два", 0.0, 1.0))
    cue = group_cues(flow, style)[0]
    assert cue.cps == pytest.approx(len("раз два") / 1.0)