"""Реестр моделей и их загрузка с релизов sherpa-onnx (без внешних зависимостей)."""
from __future__ import annotations

import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RELEASE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

Logger = Callable[[str], None]


def _noop(msg: str) -> None:  # pylint: disable=unused-argument
    """Заглушка логгера по умолчанию."""


@dataclass(frozen=True)
class ModelSpec:
    """Описание одного загружаемого артефакта модели."""

    name: str
    url: str
    kind: str  # asr | vad | separation | denoise
    archive: bool = False  # true -> tar.bz2, который нужно распаковать
    files: dict[str, str] = field(default_factory=dict)
    feature_dim: int = 64
    alignable: bool = False
    language: str = ""
    note: str = ""


# Пресеты, которые знает CLI. Пути указаны относительно каталога модели.
PRESETS: dict[str, ModelSpec] = {
    "gigaam-v3-ru": ModelSpec(
        name="gigaam-v3-ru",
        url=f"{RELEASE_URL}/asr-models/sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16.tar.bz2",
        kind="asr",
        archive=True,
        files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
        feature_dim=64,
        alignable=True,
        language="ru",
        note="GigaAM v3 CTC + пунктуация, русский, ~215 МБ, эталон для песен",
    ),
    "gigaam-v3-ru-tensor": ModelSpec(
        name="gigaam-v3-ru-tensor",
        url=f"{RELEASE_URL}/asr-models/sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16.tar.bz2",
        kind="asr",
        archive=True,
        files={"encoder": "encoder.int8.onnx", "decoder": "decoder.onnx", "joiner": "joiner.onnx",
               "tokens": "tokens.txt"},
        alignable=False,
        language="ru",
        note="GigaAM v3 transducer, точнее на шуме, но без выравнивания по словам",
    ),
    "qwen3-0.6b": ModelSpec(
        name="qwen3-0.6b",
        url=f"{RELEASE_URL}/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        kind="asr",
        archive=True,
        files={"conv_frontend": "conv_frontend.onnx", "encoder": "encoder.int8.onnx",
               "decoder": "decoder.int8.onnx", "tokenizer_dir": "tokenizer"},
        feature_dim=128,
        alignable=False,
        language="multilingual",
        note="Qwen3-ASR 0.6B, многоязычный, медленнее",
    ),
    "silero-vad": ModelSpec(
        name="silero-vad",
        url=f"{RELEASE_URL}/asr-models/silero_vad.onnx",
        kind="vad",
        files={"model": "silero_vad.onnx"},
        note="Silero VAD, детектор речи",
    ),
    "spleeter-2stems": ModelSpec(
        name="spleeter-2stems",
        url=f"{RELEASE_URL}/source-separation-models/sherpa-onnx-spleeter-2stems-int8.tar.bz2",
        kind="separation",
        archive=True,
        files={"vocals": "vocals.int8.onnx", "accompaniment": "accompaniment.int8.onnx"},
        note="Выделение вокала из минусовки (2 stems), ~50 МБ",
    ),
    "uvr-main": ModelSpec(
        name="uvr-main",
        url=f"{RELEASE_URL}/source-separation-models/UVR_MDXNET_Main.onnx",
        kind="separation",
        files={"model": "UVR_MDXNET_Main.onnx"},
        note="UVR MDX-Net, выделение вокала",
    ),
    "gtcrn": ModelSpec(
        name="gtcrn",
        url=f"{RELEASE_URL}/speech-enhancement-models/gtcrn_simple.onnx",
        kind="denoise",
        files={"model": "gtcrn_simple.onnx"},
        note="GTCRN шумоподавление, 1 МБ",
    ),
}

# Куда внутри models_dir кладём каждый пресет
PRESET_DIRS = {
    "gigaam-v3-ru": "asr/gigaam-v3-ru",
    "gigaam-v3-ru-tensor": "asr/gigaam-v3-ru-transducer",
    "qwen3-0.6b": "asr/qwen3-0.6b",
    "silero-vad": "vad/silero-vad",
    "spleeter-2stems": "sep/spleeter-2stems",
    "uvr-main": "sep/uvr-main",
    "gtcrn": "sep/gtcrn",
}


def preset_dir(name: str, models_dir: Path) -> Path:
    """Возвращает каталог, в котором ожидается лежать модели пресета."""
    return models_dir / PRESET_DIRS.get(name, name)


def _download(url: str, dest: Path, log: Logger) -> None:
    """Скачивает файл с индикатором прогресса, без внешних HTTP-библиотек."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"Скачиваю {url}")
    try:
        with urllib.request.urlopen(url) as resp:  # noqa: S310 - фиксированный хост релизов
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last_pct = -5
            with open(tmp, "wb") as out:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total:
                        pct = int(done * 100 / total)
                        if pct >= last_pct + 5:
                            last_pct = pct
                            log(f"  {pct:3d}%  {done >> 20}/{total >> 20} МБ")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Не удалось скачать {url}: {exc}") from exc
    tmp.replace(dest)


def _extract(archive: Path, target_dir: Path, log: Logger) -> None:
    """Распаковывает tar.bz2: верхний каталог архива снимается, содержимое кладётся в target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        roots = {m.name.split("/")[0] for m in members if "/" in m.name}
        strip = len(roots) == 1 and all(m.name.startswith(next(iter(roots)) + "/") for m in members)
        for m in members:
            rel = m.name.split("/", 1)[1] if strip and "/" in m.name else m.name
            if not rel or rel.startswith("/") or ".." in rel.split("/"):
                continue
            out = target_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src is None:
                continue
            with open(out, "wb") as fh:
                shutil.copyfileobj(src, fh)
    log(f"Распаковано в {target_dir}")


def ensure_model(name: str, models_dir: Path, log: Logger = _noop, force: bool = False) -> Path:
    """Скачивает пресет при необходимости и возвращает каталог с моделями."""
    spec = PRESETS.get(name)
    if spec is None:
        raise KeyError(f"Неизвестный пресет '{name}'. Доступные: {', '.join(sorted(PRESETS))}")
    target = preset_dir(name, models_dir)
    if not force and model_ready(spec, target):
        log(f"Модель '{name}' уже на месте: {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    ext = ".tar.bz2" if spec.archive else Path(spec.url).suffix
    archive = target / f"{name}{ext}"
    if force or not archive.is_file():
        _download(spec.url, archive, log)
    if spec.archive:
        _extract(archive, target, log)
        archive.unlink(missing_ok=True)
    if not model_ready(spec, target):
        raise RuntimeError(
            f"После загрузки '{name}' не найдены файлы {list(spec.files.values())} в {target}"
        )
    return target


def model_ready(spec: ModelSpec, directory: Path) -> bool:
    """Проверяет, что все обязательные файлы пресета лежат в каталоге."""
    return all((directory / rel).exists() for rel in spec.files.values())


def describe(name: str, models_dir: Path) -> str:
    """Однострочная сводка о пресете и состоянии его файлов."""
    spec = PRESETS.get(name)
    if spec is None:
        return f"{name}: неизвестный пресет"
    where = preset_dir(name, models_dir)
    state = "установлена" if model_ready(spec, where) else "не установлена"
    files = ", ".join(f"{k}={v}" for k, v in spec.files.items())
    return f"{name:<18} [{spec.kind:>10}] {state:<14} {spec.note}\n    файлы: {files}\n    путь: {where}"


def detect_asr_kind(files: Iterable[Path]) -> str:
    """Определяет тип ASR по именам файлов в каталоге модели."""
    names = {p.name for p in files}

    def has(prefix: str) -> bool:
        return any(n.startswith(prefix) and n.endswith(".onnx") for n in names)

    if has("encoder") and has("decoder") and has("joiner"):
        return "transducer"
    if has("conv_frontend"):
        return "qwen3"
    if has("model") and "tokens.txt" in names:
        return "nemo_ctc"
    return "unknown"
