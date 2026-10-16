"""Общие фикстуры тестов."""
from __future__ import annotations

import numpy as np
import pytest

from subsong.aligner import CtcAligner
from subsong.config import CueStyle, FilterSpec


@pytest.fixture()
def fake_aligner() -> CtcAligner:
    """Выравниватель без ONNX: словарь как у GigaAM (буквы + отдельный токен `▁`)."""
    al = object.__new__(CtcAligner)
    tokens = {
        0: "<unk>",
        1: "▁",      # отдельный маркер начала слова (как в словаре GigaAM)
        2: "с",
        3: "м",
        4: "о",
        5: "т",
        6: "ь",
        7: "н",
        8: "▁м",
        9: "▁т",
        10: ",",
        11: ".",
        12: "<blk>",
    }
    al.tokens = tokens
    al.inverse = {p: i for i, p in tokens.items()}
    al.unk_id = 0
    al.blank_id = 12
    al.max_piece_len = 2
    al.session = None  # type: ignore[attr-defined]
    al.input_name = ""
    al.length_name = None
    al.feature_dim = 64
    al.n_fft = 320
    al.hop = 160
    al.win = 320
    return al


@pytest.fixture()
def style() -> CueStyle:
    return CueStyle(max_chars=42, max_lines=2, max_duration=7.0, min_duration=1.0, max_cps=20.0)


@pytest.fixture()
def filters() -> FilterSpec:
    return FilterSpec()


@pytest.fixture()
def speech_like() -> np.ndarray:
    """Простой сигнал с широкополосными "согласными" - по признакам похож на речь."""
    sr = 16000
    rng = np.random.default_rng(1)
    t = np.arange(sr * 2) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
    noise = 0.08 * rng.standard_normal(t.size)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return ((tone + noise) * env).astype(np.float32)