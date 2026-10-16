"""Командный интерфейс subsong: субтитры для песенных видео."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from subsong import __version__, audio as audio_io, models as models_mod
from subsong.config import (
    Config,
    CueStyle,
    DenoiseSpec,
    FilterSpec,
    RuntimeSpec,
    SeparateSpec,
    VadSpec,
    load_config,
)

DEFAULT_PRESET = "gigaam-v3-ru"
VAD_PRESET = "silero-vad"
SEPARATE_PRESET = "spleeter-2stems"
DENOISE_PRESET = "gtcrn"


def _default_models_dir() -> Path:
    """Каталог моделей: переменная окружения SUBSONG_MODELS или ./models."""
    import os

    return Path(os.environ.get("SUBSONG_MODELS", "models"))


def build_parser() -> argparse.ArgumentParser:
    """Собирает разбор аргументов командной строки."""
    ap = argparse.ArgumentParser(
        prog="subsong",
        description="Субтитры для песенных видео: VAD + sherpa-onnx ASR + выравнивание по словам.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  subsong song.mp4                        -> song.srt + song.ass\n"
            "  subsong song.mp4 --formats srt,ass,json  -> три файла сразу\n"
            "  subsong song.mp4 --lyrics words.txt     -> субтитры по готовому тексту\n"
            "  subsong models download gigaam-v3-ru     -> скачать модели\n"
        ),
    )
    ap.add_argument("--version", action="version", version=f"subsong {__version__}")
    sub = ap.add_subparsers(dest="command")

    tr = sub.add_parser("transcribe", help="распознать видео/аудио и собрать субтитры")
    _add_transcribe_args(tr)

    md = sub.add_parser("models", help="работа с моделями")
    mdsub = md.add_subparsers(dest="models_command", required=True)
    mdsub.add_parser("list", help="показать известные модели и их состояние")
    dl = mdsub.add_parser("download", help="скачать модель")
    dl.add_argument("names", nargs="*", help="пресеты (по умолчанию: gigaam-v3-ru и silero-vad)")
    dl.add_argument("--force", action="store_true", help="перекачать даже если модель уже есть")

    sub.add_parser("selfcheck", help="проверить, что модели и ffmpeg работают")
    return ap


def _add_transcribe_args(ap: argparse.ArgumentParser) -> None:
    """Описывает аргументы команды transcribe."""
    ap.add_argument("input", help="путь к видео или аудиофайлу (mp4/mkv/webm/mp3/wav/...)")
    ap.add_argument("-o", "--output", help="базовое имя выходных файлов (без расширения)")
    ap.add_argument("-d", "--out-dir", help="каталог для результатов (по умолчанию - рядом с исходником)")
    ap.add_argument(
        "--formats",
        default="srt,ass",
        help="форматы через запятую: srt, ass, vtt, json, txt (по умолчанию srt,ass)",
    )

    grp = ap.add_argument_group("модели")
    grp.add_argument("--models-dir", default=None, help=f"каталог моделей (по умолчанию {_default_models_dir()})")
    grp.add_argument("--asr-model", help="путь к каталогу ASR-модели (переопределяет пресет)")
    grp.add_argument("--asr-preset", default=DEFAULT_PRESET, help=f"пресет ASR (по умолчанию {DEFAULT_PRESET})")
    grp.add_argument("--vad-model", help="путь к silero_vad.onnx")
    grp.add_argument("--no-download", action="store_true", help="не скачивать модели автоматически")
    grp.add_argument("--num-threads", type=int, default=0, help="потоков ONNX Runtime")
    grp.add_argument("--provider", default=None, help="onnxruntime provider (cpu/cuda/...)")

    grp = ap.add_argument_group("стиль субтитров")
    grp.add_argument("--max-chars", type=int, help="максимум символов в строке (по умолчанию 42)")
    grp.add_argument("--max-lines", type=int, help="максимум строк в реплике (по умолчанию 2)")
    grp.add_argument("--max-duration", type=float, help="максимум секунд на реплику")
    grp.add_argument("--min-duration", type=float, help="минимум секунд на реплику")
    grp.add_argument("--max-cps", type=float, help="максимум символов в секунду")
    grp.add_argument("--min-gap", type=float, help="минимальная пауза между репликами, с")
    grp.add_argument("--lead-in", type=float, help="сдвиг начала реплики на N секунд раньше")
    grp.add_argument("--lead-out", type=float, help="удлинение реплики на N секунд")
    grp.add_argument("--uppercase", action="store_true", help="ВЕРХНИЙ РЕГИСТР (как в караоке)")

    grp = ap.add_argument_group("сегментация и фильтры")
    grp.add_argument("--vad-threshold", type=float, help="порог Silero VAD (0..1)")
    grp.add_argument("--vad-min-silence", type=float, help="минимальная пауза тишины, с")
    grp.add_argument("--vad-min-speech", type=float, help="минимальная длительность речи, с")
    grp.add_argument("--vad-max-speech", type=float, help="максимум секунд в одном отрезке")
    grp.add_argument("--min-confidence", type=float, help="отбрасывать реплики ниже этой уверенности")
    grp.add_argument("--max-repeat", type=int, help="отбрасывать повторы слов/блоков (0 - выключено)")
    grp.add_argument("--music-gate", type=int, choices=[0, 1, 2], help="жёсткость отсева инструментала")
    grp.add_argument("--min-segment-sec", type=float, help="игнорировать отрезки короче N секунд")
    grp.add_argument("--no-align", action="store_true", help="не выравнивать по словам (быстрее, тайминги приблизительные)")

    grp = ap.add_argument_group("предобработка")
    grp.add_argument("--separate", action="store_true", help="выделить вокал из минусовки (Spleeter/UVR)")
    grp.add_argument("--separate-model", help="каталог модели выделения вокала")
    grp.add_argument("--denoise", action="store_true", help="шумоподавление (GTCRN)")
    grp.add_argument("--denoise-model", help="путь к модели шумоподавления")
    grp.add_argument("--normalize", choices=["none", "peak", "rms"], default="none", help="нормализация громкости")
    grp.add_argument("--start", type=float, help="начало фрагмента, с")
    grp.add_argument("--end", type=float, help="конец фрагмента, с")

    grp = ap.add_argument_group("прочее")
    grp.add_argument("--lyrics", help="файл с текстом песни (txt или lrc)")
    grp.add_argument("--config", help="YAML-конфиг со значениями по умолчанию")
    grp.add_argument("--report", action="store_true", help="показать разбор по сегментам")
    grp.add_argument("--keep-temp", action="store_true", help="не удалять временные файлы")
    grp.add_argument("--quiet", action="store_true", help="только ошибки")


def _resolve_asr(args: argparse.Namespace, models_dir: Path, log: Any) -> dict[str, Any]:
    """Определяет каталог и тип ASR-модели: из --asr-model или из пресета (с докачкой)."""
    spec = models_mod.PRESETS.get(args.asr_preset)
    if args.asr_model:
        asr_dir = Path(args.asr_model)
        if not asr_dir.is_dir():
            raise SystemExit(f"Каталог модели не найден: {asr_dir}")
        files: dict[str, str] = {p.name: p.name for p in sorted(asr_dir.glob("*.onnx"))}
        kind = models_mod.detect_asr_kind(asr_dir / name for name in files.values())
        if (asr_dir / "tokens.txt").is_file():
            files["tokens"] = "tokens.txt"
        return {
            "dir": asr_dir,
            "kind": kind,
            "files": files,
            "alignable": kind == "nemo_ctc",
            "feature_dim": 64,
            "language": "",
        }

    if spec is None:
        available = ", ".join(sorted(models_mod.PRESETS))
        raise SystemExit(f"Неизвестный пресет '{args.asr_preset}'. Доступные: {available}")
    asr_dir = models_mod.preset_dir(args.asr_preset, models_dir)
    if not models_mod.model_ready(spec, asr_dir):
        if args.no_download:
            raise SystemExit(
                f"Модель '{args.asr_preset}' не найдена в {asr_dir}.\n"
                f"Скачайте её: subsong models download {args.asr_preset}"
            )
        log(f"Модель '{args.asr_preset}' не найдена, скачиваю (~200 МБ)...")
        models_mod.ensure_model(args.asr_preset, models_dir, log)
    return {
        "dir": asr_dir,
        "kind": "nemo_ctc" if spec.alignable else _guess_kind(asr_dir),
        "files": dict(spec.files),
        "alignable": spec.alignable,
        "feature_dim": spec.feature_dim,
        "language": spec.language,
    }


def _resolve_vad(args: argparse.Namespace, models_dir: Path, log: Any) -> Path:
    """Находит модель Silero VAD, при необходимости скачивает."""
    vad_model = (
        Path(args.vad_model)
        if args.vad_model
        else models_mod.preset_dir(VAD_PRESET, models_dir) / "silero_vad.onnx"
    )
    if not vad_model.is_file():
        if args.no_download:
            raise SystemExit(
                f"Модель VAD не найдена: {vad_model}\nСкачайте: subsong models download silero-vad"
            )
        log("Модель VAD не найдена, скачиваю...")
        models_mod.ensure_model(VAD_PRESET, models_dir, log)
    if not vad_model.is_file():
        raise SystemExit(f"Модель VAD не найдена: {vad_model}")
    return vad_model


def _config_from_args(args: argparse.Namespace) -> Config:
    """Собирает конфигурацию запуска из аргументов, файла конфига и значений по умолчанию."""
    models_dir = Path(args.models_dir) if args.models_dir else _default_models_dir()
    log = _make_logger(args.quiet)
    asr = _resolve_asr(args, models_dir, log)
    vad_model = _resolve_vad(args, models_dir, log)

    cfg = Config(
        asr_dir=asr["dir"],
        vad_model=vad_model,
        asr_kind=asr["kind"],
        asr_files=asr["files"],
        feature_dim=asr["feature_dim"],
        alignable=asr["alignable"],
        language=asr["language"],
    )
    cfg.runtime = RuntimeSpec(
        num_threads=args.num_threads or 2,
        provider=args.provider or "cpu",
        quiet=args.quiet,
        keep_temp=args.keep_temp,
        temp_dir=models_dir / ".cache",
        normalize=args.normalize,
    )
    cfg.separate = SeparateSpec(enabled=args.separate)
    if args.separate_model:
        cfg.separate.model_dir = Path(args.separate_model)
    elif args.separate:
        cfg.separate.model_dir = _ensure_optional(SEPARATE_PRESET, models_dir, log, args.no_download)
    cfg.denoise = DenoiseSpec(enabled=args.denoise)
    if args.denoise_model:
        cfg.denoise.model_path = Path(args.denoise_model)
    elif args.denoise:
        cfg.denoise.model_path = _ensure_optional(DENOISE_PRESET, models_dir, log, args.no_download, file=True)

    overrides: dict[str, Any] = load_config(Path(args.config)) if args.config else {}
    _apply_style_args(cfg, args)
    _merge_overrides(cfg, overrides)
    _apply_style_args(cfg, args)  # аргументы CLI важнее файла конфига
    return cfg


def _guess_kind(asr_dir: Path) -> str:
    """Определяет тип модели по файлам в каталоге пресета."""
    names = {p.name for p in asr_dir.glob("*.onnx")}
    if {"encoder.int8.onnx", "decoder.onnx", "joiner.onnx"} <= names or (
        "encoder.onnx" in names and "joiner.onnx" in names
    ):
        return "transducer"
    if "conv_frontend.onnx" in names:
        return "qwen3"
    return "nemo_ctc"


def _ensure_optional(
    preset: str, models_dir: Path, log: Any, no_download: bool, file: bool = False
) -> Path | None:
    """Готовит дополнительную модель (выделение вокала / шумоподавление).

    file=True означает, что нужен путь к конкретному файлу .onnx, а не каталог.
    """
    spec = models_mod.PRESETS[preset]
    where = models_mod.preset_dir(preset, models_dir)
    if not models_mod.model_ready(spec, where):
        if no_download:
            log(f"Модель '{preset}' не найдена, пропускаю (скачайте: subsong models download {preset})")
            return None
        log(f"Скачиваю '{preset}'...")
        where = models_mod.ensure_model(preset, models_dir, log)
    if not file:
        return where
    onnx = sorted(where.glob("*.onnx"))
    if not onnx:
        log(f"В каталоге {where} нет .onnx файла, пропускаю")
        return None
    return onnx[0]


def _apply_style_args(cfg: Config, args: argparse.Namespace) -> None:
    """Переносит стиль и параметры VAD из аргументов в конфигурацию."""
    style: CueStyle = cfg.style
    for name in (
        "max_chars", "max_lines", "max_duration", "min_duration",
        "max_cps", "min_gap", "lead_in", "lead_out",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(style, name, value)
    if args.uppercase:
        style.uppercase = True

    vad: VadSpec = cfg.vad
    if args.vad_threshold is not None:
        vad.threshold = args.vad_threshold
    if args.vad_min_silence is not None:
        vad.min_silence = args.vad_min_silence
    if args.vad_min_speech is not None:
        vad.min_speech = args.vad_min_speech
    if args.vad_max_speech is not None:
        vad.max_speech = args.vad_max_speech

    flt: FilterSpec = cfg.filters
    if args.min_confidence is not None:
        flt.min_confidence = args.min_confidence
    if args.max_repeat is not None:
        flt.max_repeat_words = args.max_repeat
    if args.music_gate is not None:
        flt.music_gate = args.music_gate
    if args.min_segment_sec is not None:
        flt.min_segment_sec = args.min_segment_sec


def _merge_overrides(cfg: Config, overrides: dict[str, Any]) -> None:
    """Накладывает значения из YAML-конфига на готовую конфигурацию."""
    from subsong.config import apply_overrides

    apply_overrides(cfg, overrides)


def _make_logger(quiet: bool):  # type: ignore[no-untyped-def]
    """Возвращает функцию логирования, молчащую при --quiet."""
    if quiet:
        return lambda msg: None

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    return log


def _write_outputs(result: Any, args: argparse.Namespace, cfg: Config, log: Any) -> list[Path]:
    """Пишет запрошенные форматы и возвращает список созданных файлов."""
    from subsong import formats

    source = Path(args.input)
    base = Path(args.output) if args.output else source.with_suffix("")
    if args.out_dir:
        base = Path(args.out_dir) / base.name
    formats_requested = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    meta = formats.Meta(
        source=str(source),
        model=cfg.asr_kind,
        duration=result.audio_duration,
        words_per_sec=result.words_per_sec,
        audio_duration=result.audio_duration,
        extra={
            "segments_total": len(result.reports),
            "segments_kept": sum(1 for r in result.reports if r.kept),
            "processing_seconds": round(result.processing_seconds, 2),
        },
    )
    written: list[Path] = []
    for fmt in formats_requested:
        if fmt == "srt":
            written.append(formats.write_srt(result.cues, base.with_suffix(".srt")))
        elif fmt == "ass":
            written.append(formats.write_ass(result.cues, base.with_suffix(".ass")))
        elif fmt == "vtt":
            written.append(formats.write_vtt(result.cues, base.with_suffix(".vtt"), meta))
        elif fmt == "json":
            written.append(formats.write_json(result.cues, base.with_suffix(".json"), meta))
        elif fmt == "txt":
            written.append(formats.write_txt(result.cues, base.with_suffix(".txt")))
        else:
            raise SystemExit(f"Неизвестный формат: {fmt} (доступны: srt, ass, vtt, json, txt)")
    for path in written:
        log(f"Записан {path}")
    return written


def _print_report(result: Any) -> None:
    """Печатает разбор по сегментам: что попало в субтитры, а что было отброшено."""
    print("\nСегменты:")
    for r in result.reports:
        mark = "+" if r.kept else "-"
        print(
            f" {mark} [{r.start:7.2f}-{r.end:7.2f}] conf={r.confidence:7.2f} "
            f"{r.text[:70]}"
            + (f"   <- {r.reason}" if not r.kept else "")
        )


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Команда transcribe: полный цикл от файла до субтитров."""
    log = _make_logger(args.quiet)
    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"Файл не найден: {source}")
    cfg = _config_from_args(args)

    from subsong import pipeline
    from subsong.lyrics import load_lyrics

    lyrics = load_lyrics(args.lyrics) if args.lyrics else None
    if lyrics:
        log(f"Текст песни: {len(lyrics.lines)} строк" + (" (LRC с таймкодами)" if lyrics.timed else ""))

    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    result = pipeline.run(
        source,
        cfg,
        log=log,
        lyrics=lyrics,
        align_words=not args.no_align,
        start=start,
        end=end,
    )
    log(
        f"Готово: {len(result.cues)} реплик, {len(result.reports)} сегментов, "
        f"{result.processing_seconds:.1f} c "
        f"({result.words_per_sec} слов/с на реальном времени)"
    )
    _write_outputs(result, args, cfg, log)
    if args.report:
        _print_report(result)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """Команда models: список пресетов и их скачивание."""
    models_dir = Path(args.models_dir) if getattr(args, "models_dir", None) else _default_models_dir()
    if args.models_command == "list":
        for name in sorted(models_mod.PRESETS):
            print(models_mod.describe(name, models_dir))
        return 0
    names = args.names or [DEFAULT_PRESET, VAD_PRESET]
    log = _make_logger(False)
    for name in names:
        log(f"Скачиваю {name}...")
        path = models_mod.ensure_model(name, models_dir, log, force=args.force)
        log(f"Готово: {path}")
    return 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Проверяет окружение: ffmpeg, наличие моделей и согласие с sherpa-onnx."""
    models_dir = Path(args.models_dir) if getattr(args, "models_dir", None) else _default_models_dir()
    print(f"subsong {__version__}")
    try:
        print(f"ffmpeg: {audio_io.find_ffmpeg()}")
    except audio_io.AudioError as exc:
        print(f"ffmpeg: НЕ НАЙДЕН ({exc})")
        return 1
    try:
        import sherpa_onnx

        print(f"sherpa-onnx: {sherpa_onnx.__version__}")
    except ImportError as exc:
        print(f"sherpa-onnx: не установлен ({exc})")
        return 1

    preset = models_mod.PRESETS[DEFAULT_PRESET]
    asr_dir = models_mod.preset_dir(DEFAULT_PRESET, models_dir)
    vad_dir = models_mod.preset_dir(VAD_PRESET, models_dir)
    print(f"ASR {DEFAULT_PRESET}: {'ok' if models_mod.model_ready(preset, asr_dir) else 'нет'} ({asr_dir})")
    print(f"VAD {VAD_PRESET}: {'ok' if models_mod.model_ready(models_mod.PRESETS[VAD_PRESET], vad_dir) else 'нет'}")
    if not models_mod.model_ready(preset, asr_dir):
        print(f"Скачайте: subsong models download {DEFAULT_PRESET}")
        return 1

    import numpy as np

    from subsong.aligner import CtcAligner

    aligner = CtcAligner(
        asr_dir / "model.int8.onnx", asr_dir / "tokens.txt", num_threads=2
    )
    tone = np.sin(2 * np.pi * 220.0 * np.arange(16000 * 2) / 16000).astype(np.float32)
    res = aligner.transcribe(tone)
    print(f"Прогон ONNX: ок ({len(res.words)} слов на тоне, ожидаем 0-мусор)")
    print("Готово к работе.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    audio_io.enable_utf8_output()
    parser = build_parser()
    args_list = list(argv) if argv is not None else sys.argv[1:]
    known = {"transcribe", "models", "selfcheck"}
    # короткая форма: `subsong song.mp4` равносильно `subsong transcribe song.mp4`
    if args_list and not args_list[0].startswith("-") and args_list[0] not in known:
        args_list = ["transcribe", *args_list]
    args = parser.parse_args(args_list)

    try:
        if args.command == "transcribe":
            return cmd_transcribe(args)
        if args.command == "models":
            return cmd_models(args)
        if args.command == "selfcheck":
            return cmd_selfcheck(args)
    except KeyboardInterrupt:
        print("Прервано пользователем", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        # CLI показывает текст ошибки, а не traceback - так удобнее пользователю
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
