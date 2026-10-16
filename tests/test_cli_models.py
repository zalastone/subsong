"""Тесты реестра моделей, конфигурации и CLI."""
from __future__ import annotations

from pathlib import Path

import pytest

from subsong import models
from subsong.cli import build_parser, main
from subsong.config import Config, CueStyle, FilterSpec, load_config


def test_preset_dir_layout(tmp_path: Path) -> None:
    assert models.preset_dir("gigaam-v3-ru", tmp_path) == tmp_path / "asr" / "gigaam-v3-ru"
    assert models.preset_dir("silero-vad", tmp_path) == tmp_path / "vad" / "silero-vad"


def test_model_ready_checks_all_files(tmp_path: Path) -> None:
    spec = models.PRESETS["gigaam-v3-ru"]
    target = models.preset_dir("gigaam-v3-ru", tmp_path)
    assert not models.model_ready(spec, target)
    target.mkdir(parents=True)
    (target / "model.int8.onnx").write_bytes(b"x")
    assert not models.model_ready(spec, target)
    (target / "tokens.txt").write_text("a 0", encoding="utf-8")
    assert models.model_ready(spec, target)


def test_ensure_model_rejects_unknown(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        models.ensure_model("нет-такой", tmp_path)


def test_describe_mentions_path(tmp_path: Path) -> None:
    text = models.describe("gigaam-v3-ru", tmp_path)
    assert "gigaam-v3-ru" in text and "не установлена" in text


def test_detect_asr_kind(tmp_path: Path) -> None:
    (tmp_path / "model.int8.onnx").write_bytes(b"x")
    (tmp_path / "tokens.txt").write_text("a 0", encoding="utf-8")
    assert models.detect_asr_kind([tmp_path / "model.int8.onnx", tmp_path / "tokens.txt"]) == "nemo_ctc"
    for name in ("encoder.int8.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert models.detect_asr_kind([tmp_path / n for n in
                                   ("encoder.int8.onnx", "decoder.onnx", "joiner.onnx")]) == "transducer"
    assert models.detect_asr_kind([tmp_path / "unknown.bin"]) == "unknown"


def test_config_model_file(tmp_path: Path) -> None:
    cfg = Config(asr_dir=tmp_path, vad_model=tmp_path / "vad.onnx", asr_kind="nemo_ctc",
                 asr_files={"model": "model.int8.onnx"})
    assert cfg.model_file("model") == tmp_path / "model.int8.onnx"
    with pytest.raises(KeyError):
        cfg.model_file("tokens")


def test_defaults_are_sane() -> None:
    cfg = Config(asr_dir=Path("."), vad_model=Path("vad.onnx"), asr_kind="nemo_ctc")
    assert cfg.style.max_chars >= 20 and cfg.style.max_lines >= 1
    assert cfg.filters.min_confidence is not None and cfg.filters.min_confidence < 0
    assert cfg.feature_dim == 64


def test_yaml_overrides_nested(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text(
        "style:\n  max_chars: 30\n  uppercase: true\n"
        "filters:\n  min_confidence: -2.0\n"
        "runtime:\n  num_threads: 8\n",
        encoding="utf-8",
    )
    values = load_config(path)
    cfg = Config(asr_dir=Path("."), vad_model=Path("v"), asr_kind="nemo_ctc")
    from subsong.config import apply_overrides

    apply_overrides(cfg, values)
    assert cfg.style.max_chars == 30
    assert cfg.style.uppercase is True
    assert cfg.filters.min_confidence == -2.0
    assert cfg.runtime.num_threads == 8


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "нет.yaml")


def test_cli_parses_transcribe_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["transcribe", "song.mp4"])
    assert args.command == "transcribe"
    assert args.input == "song.mp4"
    assert args.formats == "srt,ass"
    assert args.asr_preset == "gigaam-v3-ru"
    assert args.music_gate is None


def test_cli_short_form_routes_to_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    calls: dict[str, object] = {}

    def fake_transcribe(args: object) -> int:
        calls["input"] = getattr(args, "input", None)
        return 0

    monkeypatch.setattr("subsong.cli.cmd_transcribe", fake_transcribe)
    assert main(["song.mp4"]) == 0
    assert calls["input"] == "song.mp4"


def test_cli_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["transcribe", str(tmp_path / "нет.mp4"), "--no-download"])


def test_cli_rejects_unknown_format(tmp_path: Path) -> None:
    from subsong.cli import _write_outputs
    from subsong.cues import Cue
    from subsong.pipeline import Result

    cfg = Config(asr_dir=Path("."), vad_model=Path("v"), asr_kind="nemo_ctc")
    parser = build_parser()
    args = parser.parse_args(
        ["transcribe", str(tmp_path / "song.mp4"), "--formats", "srt,xyz", "-d", str(tmp_path)]
    )
    result = Result(cues=[Cue(0.0, 1.0, ["да"], [])])
    with pytest.raises(SystemExit):
        _write_outputs(result, args, cfg, lambda msg: None)


def test_cli_writes_requested_formats(tmp_path: Path) -> None:
    from subsong.cli import _write_outputs
    from subsong.cues import Cue
    from subsong.pipeline import Result

    cfg = Config(asr_dir=Path("."), vad_model=Path("v"), asr_kind="nemo_ctc")
    parser = build_parser()
    args = parser.parse_args(
        ["transcribe", str(tmp_path / "song.mp4"), "--formats", "srt,ass,vtt,json,txt",
         "-d", str(tmp_path / "out")]
    )
    result = Result(cues=[Cue(0.0, 1.0, ["да"], [])], audio_duration=1.0)
    written = _write_outputs(result, args, cfg, lambda msg: None)
    assert sorted(p.suffix for p in written) == [".ass", ".json", ".srt", ".txt", ".vtt"]
    assert all(p.is_file() for p in written)


def test_filter_defaults_and_dataclass_copy() -> None:
    f = FilterSpec()
    assert f.music_gate == 0
    assert f.min_confidence == -4.0
    style = CueStyle()
    assert style.lead_out == 0.2