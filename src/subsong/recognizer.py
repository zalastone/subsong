"""Распознавание через sherpa-onnx для моделей без CTC-выравнивания.

Для CTC-моделей (GigaAM) используется `subsong.aligner`, который сам считает log_probs.
Здесь собраны остальные движки: transducer, Qwen3-ASR, Whisper, Paraformer, SenseVoice.
Если модель умеет отдавать тайминги слов (Whisper с enable_token_timestamps), берём их,
иначе раскладываем текст по времени сегмента пропорционально числу слогов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from subsong.config import SR, Config
from subsong.cues import TimedWord, distribute_words


@dataclass(frozen=True)
class Transcript:
    """Результат распознавания одного сегмента."""

    text: str
    words: list[TimedWord]


class SherpaRecognizer:
    """Обёртка над `sherpa_onnx.OfflineRecognizer` с несколькими бэкендами."""

    def __init__(self, cfg: Config) -> None:
        import sherpa_onnx

        self.kind = cfg.asr_kind
        self._rec = _build_recognizer(sherpa_onnx, cfg)
        self._punct = _build_punct(sherpa_onnx, cfg)
        self.language = cfg.language

    def transcribe(self, samples: np.ndarray, offset: float = 0.0) -> Transcript:
        """Распознаёт сэмплы; offset добавляется к абсолютным таймингам слов."""
        stream = self._rec.create_stream()
        stream.accept_waveform(SR, samples)
        self._rec.decode_stream(stream)
        result = stream.result
        text = str(getattr(result, "text", "")).strip()
        if self._punct is not None and text:
            text = str(self._punct.add_punctuation(text))
        words = self._extract_words(result, text, samples, offset)
        return Transcript(text=text, words=words)

    def _extract_words(self, result: object, text: str, samples: np.ndarray, offset: float) -> list[TimedWord]:
        """Достаёт тайминги слов: нативные, если модель их даёт, иначе пропорциональные."""
        native = _native_word_times(result)
        if native:
            return [TimedWord(w, offset + s, offset + e) for w, s, e in native]
        duration = len(samples) / SR
        return distribute_words(text, offset, offset + duration)


def _native_word_times(result: object) -> list[tuple[str, float, float]]:
    """Читает токены и их времена из результата sherpa-onnx и склеивает их в слова.

    sherpa-onnx отдаёт `tokens` и `timestamps` для CTC и Whisper-моделей, поэтому
    отдельный выравниватель для них не нужен. Пустой результат означает, что модель
    таймингов не даёт, и тогда слова распределяются равномерно.
    """
    tokens = list(getattr(result, "tokens", None) or [])
    stamps = list(getattr(result, "timestamps", None) or [])
    if not tokens or len(tokens) != len(stamps):
        return []
    words: list[tuple[str, float, float]] = []
    cur_parts: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    for tok, ts in zip(tokens, stamps):
        text = str(tok)
        if not text:
            continue
        starts_word = text[0].isspace()
        piece = text.strip()
        if not piece:
            # токен-разделитель: он закрывает текущее слово, но сам текста не несёт
            if starts_word and cur_parts:
                words.append(("".join(cur_parts), cur_start, cur_end))
                cur_parts = []
            continue
        if starts_word and cur_parts:
            words.append(("".join(cur_parts), cur_start, cur_end))
            cur_parts = []
        if not cur_parts:
            cur_start = float(ts)
        cur_parts.append(piece)
        cur_end = float(ts)
    if cur_parts:
        words.append(("".join(cur_parts), cur_start, cur_end))
    return [w for w in words if w[0] and w[2] >= w[1]]


def _build_recognizer(sherpa: Any, cfg: Config) -> Any:  # noqa: ANN401
    """Создаёт OfflineRecognizer под нужный тип модели."""
    threads = cfg.runtime.num_threads
    provider = cfg.runtime.provider
    if cfg.asr_kind == "transducer":
        return sherpa.OfflineRecognizer.from_transducer(
            encoder=str(cfg.model_file("encoder")),
            decoder=str(cfg.model_file("decoder")),
            joiner=str(cfg.model_file("joiner")),
            tokens=str(cfg.model_file("tokens")),
            num_threads=threads,
            sample_rate=SR,
            feature_dim=80,
            provider=provider,
        )
    if cfg.asr_kind == "qwen3":
        return sherpa.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(cfg.model_file("conv_frontend")),
            encoder=str(cfg.model_file("encoder")),
            decoder=str(cfg.model_file("decoder")),
            tokenizer=str(cfg.model_file("tokenizer_dir")),
            num_threads=threads,
            sample_rate=SR,
            feature_dim=128,
            provider=provider,
        )
    if cfg.asr_kind == "whisper":
        return sherpa.OfflineRecognizer.from_whisper(
            model=str(cfg.model_file("model")),
            tokens=str(cfg.model_file("tokens")),
            num_threads=threads,
            sample_rate=SR,
            enable_token_timestamps=True,
            provider=provider,
        )
    if cfg.asr_kind == "sense_voice":
        return sherpa.OfflineRecognizer.from_sense_voice(
            model=str(cfg.model_file("model")),
            tokens=str(cfg.model_file("tokens")),
            num_threads=threads,
            sample_rate=SR,
            use_itn=True,
            provider=provider,
        )
    if cfg.asr_kind == "paraformer":
        return sherpa.OfflineRecognizer.from_paraformer(
            model=str(cfg.model_file("model")),
            tokens=str(cfg.model_file("tokens")),
            num_threads=threads,
            sample_rate=SR,
            provider=provider,
        )
    if cfg.asr_kind == "nemo_ctc":
        # Режим --no-align: распознаёт sherpa-onnx, тайминги слов распределяются равномерно
        return sherpa.OfflineRecognizer.from_nemo_ctc(
            model=str(cfg.model_file("model")),
            tokens=str(cfg.model_file("tokens")),
            num_threads=threads,
            sample_rate=SR,
            feature_dim=cfg.feature_dim,
            provider=provider,
        )
    raise ValueError(
        f"Тип модели '{cfg.asr_kind}' не поддерживается для выравнивания слов. "
        "Для субтитров используйте CTC-модель (gigaam-v3-ru) или запустите с --no-align."
    )


def _build_punct(sherpa: Any, cfg: Config) -> Any:  # noqa: ANN401
    """Создаёт модель пунктуации, если она указана в конфиге."""
    if cfg.punct_model is None:
        return None
    model = sherpa.OfflinePunctuationModelConfig(ct_transformer=str(cfg.punct_model))
    conf = sherpa.OfflinePunctuationConfig(model=model)
    if not conf.validate():
        raise ValueError(f"Некорректная модель пунктуации: {cfg.punct_model}")
    return sherpa.OfflinePunctuation(conf)
