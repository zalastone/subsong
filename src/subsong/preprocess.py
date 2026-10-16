"""Предобработка аудио перед распознаванием: выделение вокала и шумоподавление.

Для песен это самый заметный ускоритель качества: на чистом вокале, а не на смеси
с минусовкой, ошибок распознавания заметно меньше. Обе операции выполняются
моделями sherpa-onnx (Spleeter / UVR и GTCRN) и потому не тянут за собой
torch и librosa.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from subsong.audio import decode
from subsong.config import SR

Logger = Callable[[str], None]


def _noop(msg: str) -> None:  # pylint: disable=unused-argument
    """Заглушка логгера."""


def separate_vocals_array(
    source: Path,
    model_dir: Path,
    stem: str = "vocals",
    work_dir: Path | None = None,
    log: Logger = _noop,
) -> tuple[np.ndarray, int]:
    """Выделяет вокал (или минусовку) из аудиофайла: mono-массив + частота дискретизации."""
    del work_dir  # разделение работает в памяти, временные файлы не нужны
    import sherpa_onnx

    spleeter_vocals = model_dir / "vocals.int8.onnx"
    accompaniment = model_dir / "accompaniment.int8.onnx"
    uvr = model_dir / "UVR_MDXNET_Main.onnx"
    sr = 44100
    if spleeter_vocals.is_file() and accompaniment.is_file():
        conf = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(
                    vocals=str(spleeter_vocals), accompaniment=str(accompaniment)
                )
            )
        )
        target = spleeter_vocals
    elif uvr.is_file():
        conf = sherpa_onnx.OfflineSourceSeparationConfig(
            model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(uvr))
            )
        )
        target = uvr
    else:
        raise FileNotFoundError(
            f"В каталоге {model_dir} нет ни Spleeter (vocals/accompaniment), ни UVR-модели"
        )
    if not conf.validate():
        raise RuntimeError(f"Некорректная конфигурация выделения вокала: {target}")
    log(f"Выделяю вокал моделью {target.name}")
    sep = sherpa_onnx.OfflineSourceSeparation(conf)

    pcm = decode(source, sr=sr, channels=2)
    if pcm.shape[0] != 2:  # модели разделения ждут ровно два канала
        pcm = np.repeat(pcm[:1], 2, axis=0)
    out = sep.process(sr, pcm.astype(np.float32))
    stems = [np.asarray(s.data, dtype=np.float32) for s in out.stems]
    if not stems:
        raise RuntimeError("Модель разделения не вернула ни одного потока")
    index = 1 if stem == "accompaniment" and len(stems) > 1 else 0
    chosen = stems[index]
    mono = chosen.mean(axis=0) if chosen.ndim == 2 else chosen
    return mono.astype(np.float32), int(out.sample_rate or sr)


def denoise_array(source: Path, model_path: Path, log: Logger = _noop) -> tuple[np.ndarray, int]:
    """Подавляет шум (GTCRN) и возвращает очищенные сэмплы 16 кГц."""
    import sherpa_onnx

    conf = sherpa_onnx.OfflineSpeechDenoiserConfig(
        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model_path))
        )
    )
    if not conf.validate():
        raise RuntimeError(f"Некорректная конфигурация шумоподавителя: {model_path}")
    log(f"Шумоподавление моделью {model_path.name}")
    denoiser = sherpa_onnx.OfflineSpeechDenoiser(conf)
    samples = decode(source, sr=SR)
    res = denoiser.run(samples.astype(np.float32), SR)
    clean = np.asarray(res.samples, dtype=np.float32)
    # модель может слегка менять длину - выравниваем по исходной длине
    if clean.size < samples.size:
        clean = np.pad(clean, (0, samples.size - clean.size))
    return clean[: samples.size], SR
