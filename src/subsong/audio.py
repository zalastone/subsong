"""Декодирование аудио через ffmpeg: видео/аудио любого формата -> mono float32 16 кГц."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from subsong.config import SR

# Прячем окно консоли, которое ffmpeg создаёт на Windows
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class AudioError(RuntimeError):
    """Ошибка чтения/кодирования аудио средствами ffmpeg."""


@dataclass(frozen=True)
class MediaInfo:
    """Сведения о медиафайле, полученные от ffprobe."""

    duration: float
    has_audio: bool
    has_video: bool


def find_ffmpeg(name: str = "ffmpeg") -> str:
    """Ищет исполняемый файл ffmpeg/ffprobe в PATH и в локальной папке bin."""
    local = Path(__file__).resolve().parent.parent.parent / "bin"
    if local.is_dir() and (local / f"{name}.exe").is_file():
        return str(local / f"{name}.exe")
    found = shutil.which(name)
    if not found:
        raise AudioError(
            f"Не найден {name}. Установите ffmpeg и добавьте его в PATH "
            f"(https://ffmpeg.org/download.html) либо положите {name}.exe в папку ./bin."
        )
    return found


def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess[bytes]:
    """Запускает процесс и превращает ненулевой код выхода в понятную ошибку."""
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, creationflags=_NO_WINDOW)
    except FileNotFoundError as exc:
        raise AudioError(f"{what}: исполняемый файл не найден ({cmd[0]})") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioError(f"{what} завершился с кодом {proc.returncode}:\n{err}")
    return proc


def probe(path: Path) -> MediaInfo:
    """Возвращает длительность и состав дорожек медиафайла."""
    cmd = [
        find_ffmpeg("ffprobe"), "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    data = json.loads(_run(cmd, "ffprobe").stdout.decode("utf-8", errors="replace") or "{}")
    duration = 0.0
    try:
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    streams = data.get("streams", []) or []
    kinds = {s.get("codec_type") for s in streams}
    if not duration:
        for s in streams:
            try:
                duration = max(duration, float(s.get("duration") or 0.0))
            except (TypeError, ValueError):
                continue
    return MediaInfo(duration=duration, has_audio="audio" in kinds, has_video="video" in kinds)


def _ffmpeg_cmd(
    path: Path,
    sr: int,
    start: float | None,
    end: float | None,
    extra_filters: list[str],
    channels: int = 1,
) -> list[str]:
    """Собирает команду ffmpeg для получения моно-PCM s16le в stdout."""
    cmd = [find_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None and end > 0:
        cmd += ["-t", f"{max(0.0, end - (start or 0.0)):.3f}"]
    cmd += ["-vn", "-sn", "-dn"]
    if extra_filters:
        cmd += ["-af", ",".join(extra_filters)]
    cmd += ["-ac", str(channels), "-ar", str(sr), "-acodec", "pcm_s16le", "-f", "s16le", "-"]
    return cmd


def decode(
    path: Path,
    sr: int = SR,
    start: float | None = None,
    end: float | None = None,
    filters: list[str] | None = None,
    channels: int = 1,
) -> np.ndarray:
    """Декодирует фрагмент файла: 1-D float32 для моно, 2-D [channels, N] для стерео."""
    cmd = _ffmpeg_cmd(path, sr, start, end, filters or [], channels)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
    assert proc.stdout is not None and proc.stderr is not None
    try:
        raw = proc.stdout.read()
        err = proc.stderr.read()
        code = proc.wait()
    finally:
        proc.stdout.close()
        proc.stderr.close()
    if code != 0:
        raise AudioError(
            f"ffmpeg не смог прочитать {path.name} (код {code}):\n"
            f"{err.decode('utf-8', errors='replace').strip()}"
        )
    pcm = np.frombuffer(raw, dtype="<i2")
    samples = pcm.astype(np.float32) / 32768.0
    if channels == 1:
        return cast("NDArray[np.float32]", samples)
    usable = (samples.size // channels) * channels
    reshaped = samples[:usable].reshape(channels, usable // channels)
    return cast("NDArray[np.float32]", reshaped)


def duration_of(samples: np.ndarray, sr: int = SR) -> float:
    """Длительность массива сэмплов в секундах."""
    return len(samples) / float(sr)


def normalize(samples: np.ndarray, mode: str = "none", target_db: float = -23.0) -> np.ndarray:
    """Нормализует громкость: none | peak | rms.

    Для тихих записей песен часто помогает rms до -23 dBFS, но на чистом инструментале
    это поднимает шум, поэтому по умолчанию ничего не меняется.
    """
    if mode == "none" or samples.size == 0:
        return samples
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1e-9:
        return samples
    if mode == "peak":
        return (samples / peak * 0.95).astype(np.float32)
    if mode == "rms":
        rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
        if rms <= 1e-9:
            return samples
        gain = (10.0 ** (target_db / 20.0)) / rms
        return cast("NDArray[np.float32]", np.clip(samples * gain, -1.0, 1.0).astype(np.float32))
    raise ValueError(f"Неизвестный режим нормализации: {mode}")


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    """Сохраняет массив сэмплов в 16-битный WAV (для отладочных артефактов)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    data = (clipped * 32767.0).astype("<i2")
    writer = wave.open(str(path), "wb")
    with writer as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
    return path


def enable_utf8_output() -> None:
    """Переключает stdout/stderr на UTF-8, иначе кириллица ломается в консоли Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - нестандартные потоки
            pass
    if os.name == "nt":  # приглушаем возможные сообщения ffmpeg про кодировку
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
