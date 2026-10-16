"""Тесты выравнивания: токенизация, декодирование, группировка в слова."""
from __future__ import annotations

import numpy as np
import pytest

from subsong.aligner import (
    CtcAligner,
    TokenSpan,
    load_tokens,
    make_contiguous,
    tokens_to_words,
)

# ids из фикстуры fake_aligner
UNK, MARK, S, M, O, T, SOFT, N, MARK_M, MARK_T, COMMA, DOT, BLANK = range(13)


def test_greedy_decode_skips_blanks_and_repeats(fake_aligner: CtcAligner) -> None:
    ids = [BLANK, MARK_M, MARK_M, BLANK, MARK_T, T, BLANK]
    lp = np.full((len(ids), 16), -5.0, dtype=np.float32)
    for i, tid in enumerate(ids):
        lp[i, tid] = 0.0
    assert fake_aligner.greedy_ids(lp) == [MARK_M, MARK_T, T]


def test_greedy_decode_separates_repeats_by_blank(fake_aligner: CtcAligner) -> None:
    ids = [T, T, BLANK, T, BLANK]
    lp = np.full((len(ids), 16), -5.0, dtype=np.float32)
    for i, tid in enumerate(ids):
        lp[i, tid] = 0.0
    assert fake_aligner.greedy_ids(lp) == [T, T]


def test_detokenize_keeps_spaces(fake_aligner: CtcAligner) -> None:
    # ▁м + с + о + т => "мсот"; отдельный ▁ тоже даёт пробел
    assert fake_aligner.detokenize([MARK_M, S, O, T]) == "мсот"
    assert fake_aligner.detokenize([S, MARK, MARK_T]) == "с т"


def test_detokenize_collapses_whitespace(fake_aligner: CtcAligner) -> None:
    assert fake_aligner.detokenize([MARK, MARK_M, MARK, MARK, MARK_T]) == "м т"


def test_detokenize_skips_special_pieces(fake_aligner: CtcAligner) -> None:
    assert fake_aligner.detokenize([UNK, MARK_M, BLANK, MARK_T]) == "м т"


def test_encode_text_roundtrip(fake_aligner: CtcAligner) -> None:
    for text in ("м т", "с т", "мом"):
        ids = fake_aligner.encode_text(text)
        assert fake_aligner.detokenize(ids) == text


def test_encode_text_uses_standalone_mark(fake_aligner: CtcAligner) -> None:
    # для 'м' и 'т' есть пики с ▁, а 'с' кодируется как отдельный ▁ + 'с'
    assert fake_aligner.encode_text("м т") == [MARK_M, MARK_T]
    assert fake_aligner.encode_text("с т") == [MARK, S, MARK_T]


def test_tokens_to_words_groups_pieces(fake_aligner: CtcAligner) -> None:
    spans = [
        TokenSpan(MARK_M, "▁м", 0.00, 0.10, -0.1),
        TokenSpan(O, "о", 0.10, 0.20, -0.2),
        TokenSpan(MARK, "▁", 0.20, 0.24, -0.05),
        TokenSpan(MARK_T, "▁т", 0.24, 0.34, -0.3),
        TokenSpan(T, "т", 0.34, 0.40, -0.3),
    ]
    words = tokens_to_words(spans)
    assert [w.text for w in words] == ["мо", "тт"]
    assert words[0].start == pytest.approx(0.0)
    assert words[1].start == pytest.approx(0.24)
    assert words[1].logprob == pytest.approx(-0.3)


def test_tokens_to_words_keeps_punctuation(fake_aligner: CtcAligner) -> None:
    spans = [TokenSpan(MARK_M, "▁м", 0.0, 0.1, -0.1), TokenSpan(COMMA, ",", 0.1, 0.14, -0.1)]
    assert [w.text for w in tokens_to_words(spans)] == ["м,"]


def test_make_contiguous_removes_overlaps() -> None:
    spans = [
        TokenSpan(1, "a", 0.0, 0.1, -1.0),
        TokenSpan(2, "b", 0.05, 0.6, -1.0),
        TokenSpan(3, "c", 0.55, 0.7, -1.0),
    ]
    out = make_contiguous(spans, frame_sec=0.04)
    assert out[1].start == pytest.approx(0.1)
    assert out[2].start == pytest.approx(0.6)


def test_make_contiguous_keeps_real_gaps() -> None:
    spans = [TokenSpan(1, "a", 0.0, 0.1, -1.0), TokenSpan(2, "b", 0.5, 0.6, -1.0)]
    assert make_contiguous(spans, frame_sec=0.04)[1].start == pytest.approx(0.5)


def test_align_ids_finds_wall_clock_positions(fake_aligner: CtcAligner) -> None:
    # 10 кадров: два токена в кадрах 1-2 и 7-8, один - в кадрах 4-5
    schedule = [BLANK, MARK_M, MARK_M, BLANK, MARK_T, MARK_T, BLANK, S, S, BLANK]
    lp = np.full((len(schedule), 16), -4.0, dtype=np.float32)
    for t, tid in enumerate(schedule):
        lp[t, tid] = -0.01
    spans, blank_ratio, path_lp = fake_aligner.align_ids(lp, [MARK_M, MARK_T, S], frame_sec=0.04)
    assert len(spans) == 3
    assert spans[0].start == pytest.approx(0.04, abs=0.05)
    assert spans[1].start == pytest.approx(0.16, abs=0.05)
    assert spans[2].start == pytest.approx(0.28, abs=0.05)
    assert 0.0 < blank_ratio < 1.0
    assert path_lp < 0.0


def test_align_ids_orders_tokens_monotonically(fake_aligner: CtcAligner) -> None:
    rng = np.random.default_rng(0)
    lp = rng.normal(-2.0, 1.0, size=(40, 16)).astype(np.float32)
    spans, _ratio, _lp = fake_aligner.align_ids(lp, [M, O, T, M, O], frame_sec=0.04)
    if spans:
        starts = [s.start for s in spans]
        assert starts == sorted(starts)


def test_align_ids_rejects_impossible_input(fake_aligner: CtcAligner) -> None:
    lp = np.zeros((2, 16), dtype=np.float32)
    spans, blank_ratio, path_lp = fake_aligner.align_ids(lp, [M, O, T, M], frame_sec=0.04)
    assert spans == []
    assert blank_ratio == 1.0
    assert path_lp < -1.0


def test_load_tokens(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "tokens.txt"
    path.write_text("▁ 0\nа 1\n<blk> 2\nbroken-line\n", encoding="utf-8")
    assert load_tokens(path) == {0: "▁", 1: "а", 2: "<blk>"}