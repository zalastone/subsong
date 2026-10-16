"""Конфигурация пайплайна: пути к моделям, параметры VAD, стиль субтитров.

Конфиг собирается из трёх источников (по возрастанию приоритета):
  1) значения по умолчанию из этого модуля;
  2) YAML-файл (`--config`), где перечислены только те поля, которые нужно поменять;
  3) аргументы командной строки (`--max-chars`, `--num-threads` и т.п.).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# Частота дискретизации, на которой работают все модели sherpa-onnx
SR = 16000
# Базовое число мел-полос для CTC-моделей NeMo/GigaAM (n_fft=320, hop=160 -> 100 fps)
DEFAULT_FEATURE_DIM = 64


@dataclass
class VadSpec:
    """Параметры детектора речи (Silero VAD из sherpa-onnx).

    Порог по умолчанию ниже стандартного 0.5: на песне минусовка маскирует начало
    фразы, и при 0.5 часть строк теряется. Лишние срабатывания на инструментале
    отсекаются фильтром уверенности.
    """

    threshold: float = 0.35
    min_silence: float = 0.35
    min_speech: float = 0.2
    max_speech: float = 20.0
    buffer_sec: float = 60.0


@dataclass
class CueStyle:
    """Правила разбивки слов на реплики субтитров и таймингов."""

    max_chars: int = 42
    max_lines: int = 2
    min_duration: float = 1.0
    max_duration: float = 7.0
    max_cps: float = 20.0
    min_gap: float = 0.08
    lead_in: float = 0.0
    lead_out: float = 0.2
    uppercase: bool = False
    # Держать реплику целиком, если она целиком помещается в строку
    keep_short_lines: bool = True


@dataclass
class FilterSpec:
    """Фильтры мусорных реплик (инструментал, галлюцинации ASR на музыке)."""

    # Порог средней лог-вероятности пути выравнивания. Значение по умолчанию отсекает
    # галлюцинации ASR на инструментале (там уверенность около -4 и ниже),
    # не задевая настоящий вокал (обычно выше -2.5). Поставьте -99, чтобы отключить.
    min_confidence: float | None = -4.0
    # Сколько одинаковых слов подряд считать галлюцинацией (0 -> выключено)
    max_repeat_words: int = 0
    # Короткие сегменты короче N секунд отбрасываются (0 -> не отбрасывать)
    min_segment_sec: float = 0.0
    # Жёсткостьmusic-гейта: 0 - выключен, 1 - консервативный, 2 - агрессивный
    music_gate: int = 0
    # Уровень громкости сегмента ниже порога считается тишиной (dBFS)
    min_rms_db: float = -55.0


@dataclass
class SeparateSpec:
    """Опциональное выделение вокала из минусовки (sherpa-onnx source separation)."""

    enabled: bool = False
    model_dir: Path | None = None
    stem: str = "vocals"
    sample_rate: int = 44100


@dataclass
class DenoiseSpec:
    """Опциональное шумоподавление (GTCRN / DPDFNet из sherpa-onnx)."""

    enabled: bool = False
    model_path: Path | None = None


@dataclass
class RuntimeSpec:
    """Общие настройки вычислений и вывода."""

    num_threads: int = 2
    provider: str = "cpu"
    debug: bool = False
    quiet: bool = False
    keep_temp: bool = False
    temp_dir: Path | None = None
    normalize: str = "none"  # none | peak | rms


@dataclass
class Config:
    """Полная конфигурация одного запуска."""

    asr_dir: Path
    vad_model: Path
    asr_kind: str
    asr_files: dict[str, str] = field(default_factory=dict)
    feature_dim: int = DEFAULT_FEATURE_DIM
    alignable: bool = True
    language: str = ""
    punct_model: Path | None = None
    vad: VadSpec = field(default_factory=VadSpec)
    style: CueStyle = field(default_factory=CueStyle)
    filters: FilterSpec = field(default_factory=FilterSpec)
    separate: SeparateSpec = field(default_factory=SeparateSpec)
    denoise: DenoiseSpec = field(default_factory=DenoiseSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)

    def model_file(self, key: str) -> Path:
        """Возвращает абсолютный путь к файлу модели по семантическому имени."""
        try:
            name = self.asr_files[key]
        except KeyError as exc:
            raise KeyError(f"В модели ASR нет файла '{key}'. Ожидаются: {sorted(self.asr_files)}") from exc
        return self.asr_dir / name


def _apply(target: Any, values: dict[str, Any]) -> None:
    """Рекурсивно применяет словарь значений к dataclass-объекту (игнорируя лишние ключи)."""
    for f in fields(target):
        if f.name not in values:
            continue
        value = values[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        elif isinstance(current, Path) and isinstance(value, str):
            setattr(target, f.name, Path(value))
        else:
            setattr(target, f.name, value)


def load_config(path: Path | None) -> dict[str, Any]:
    """Читает YAML-конфиг и возвращает словарь переопределений (пустой, если файла нет)."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - опциональная зависимость
        raise RuntimeError(
            "Для чтения YAML-конфига установите pyyaml: pip install subsong[config]"
        ) from exc
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался словарь в начале файла {path}, получено: {type(data).__name__}")
    return data


def apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Применяет словарь переопределений к готовой конфигурации."""
    _apply(cfg, overrides)
    return cfg
