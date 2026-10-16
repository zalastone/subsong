"""CTC forced alignment: текст + тайминги слов + уверенность из одного прогона ONNX.

Для CTC-моделей (GigaAM v3 nemo_ctc) sherpa-onnx не отдаёт тайминги слов, поэтому мы
считаем log_probs сами через onnxruntime и делаем два прохода по ним:

  * жадное декодирование  -> текст реплики;
  * выравнивание Витерби  -> время каждого токена, а значит и каждого слова,
                            плюс средняя лог-вероятность для отсева мусора.

Один и тот же файл модели используется и для распознавания, и для выравнивания,
поэтому текст и тайминги гарантированно согласованы.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from subsong.config import SR
from subsong.features import log_mel

NEG_INF = -1e30
# Уверенность, которую отдаём, когда выравнивание невозможно (нет слов или слишком короткий сегмент)
POOR_LOGPROB = -8.0
# Служебные пики словаря SentencePiece, которые не являются текстом
SPECIAL_PIECES = frozenset({"<unk>", "<blk>", "<s>", "</s>", "<pad>", "▁"})
WORD_MARK = "▁"


@dataclass(frozen=True)
class TokenSpan:
    """Временной интервал одного токена (пика), в секундах от начала сегмента."""

    token_id: int
    piece: str
    start: float
    end: float
    logprob: float


@dataclass(frozen=True)
class WordSpan:
    """Слово с таймингом и уверенностью (средняя лог-вероятность его токенов)."""

    text: str
    start: float
    end: float
    logprob: float


@dataclass(frozen=True)
class AlignedResult:
    """Результат распознавания одного сегмента с выравниванием."""

    text: str
    words: list[WordSpan]
    mean_logprob: float
    blank_ratio: float  # доля кадров без токенов: косвенный признак шума/музыки


def load_tokens(tokens_path: Path) -> dict[int, str]:
    """Читает tokens.txt (формат `piece id`) в словарь id -> piece."""
    table: dict[int, str] = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            piece, idx = parts
            try:
                table[int(idx)] = piece
            except ValueError:
                continue
    return table


class CtcAligner:
    """Распознаёт сегменты CTC-моделью и выравнивает текст по времени."""

    def __init__(
        self,
        model_path: Path,
        tokens_path: Path,
        num_threads: int = 2,
        provider: str = "cpu",
        feature_dim: int = 64,
        n_fft: int = 320,
        hop: int = 160,
        win: int = 320,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, num_threads)
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"] if provider == "cpu" else [provider]
        self.session = ort.InferenceSession(str(model_path), opts, providers=providers)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.length_name = inputs[1].name if len(inputs) > 1 else None
        self.tokens = load_tokens(tokens_path)
        self.inverse = {p: i for i, p in self.tokens.items()}
        self.feature_dim = feature_dim
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.unk_id = self.inverse.get("<unk>", 0)
        self.blank_id = self.inverse.get("<blk>", max(self.tokens) if self.tokens else 0)
        self.max_piece_len = max((len(p) for p in self.tokens.values()), default=1)

    # ------------------------------------------------------------------ признаки
    def log_probs(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        """Прогоняет энкодер и возвращает (log_probs[T, V], длительность кадра в секундах)."""
        feats = log_mel(
            samples,
            n_mels=self.feature_dim,
            n_fft=self.n_fft,
            hop=self.hop,
            win=self.win,
            sr=SR,
        )
        if feats.shape[1] == 0:
            return np.zeros((0, max(1, len(self.tokens))), dtype=np.float32), 0.0
        batch = np.ascontiguousarray(feats[None, :, :].astype(np.float32))
        if self.length_name:
            lengths = np.array([feats.shape[1]], dtype=np.int64)
            out = self.session.run(None, {self.input_name: batch, self.length_name: lengths})[0]
        else:
            out = self.session.run(None, {self.input_name: batch})[0]
        lp = np.asarray(out[0], dtype=np.float32)  # (T, V)
        frame_sec = (len(samples) / SR) / max(1, lp.shape[0])
        return lp, frame_sec

    # ------------------------------------------------------------------ декодирование
    def greedy_ids(self, lp: np.ndarray) -> list[int]:
        """Жадное CTC-декодирование: argmax по кадрам со склейкой повторов и пропуском blank'а."""
        if lp.shape[0] == 0:
            return []
        ids = lp.argmax(axis=1).tolist()
        out: list[int] = []
        prev = self.blank_id
        for cur in ids:
            if cur != prev and cur != self.blank_id:
                out.append(cur)
            prev = cur
        return out

    def detokenize(self, ids: list[int]) -> str:
        """Склеивает токены в текст: `▁` даёт пробел (в том числе отдельный токен `▁`)."""
        parts: list[str] = []
        for tid in ids:
            piece = self.tokens.get(tid, "")
            if not piece or piece in {"<unk>", "<blk>", "<s>", "</s>", "<pad>"}:
                continue
            if piece == WORD_MARK:
                parts.append(" ")
            elif piece.startswith(WORD_MARK):
                parts.append(" " + piece[1:])
            else:
                parts.append(piece)
        return re.sub(r"\s+", " ", "".join(parts)).strip()

    def encode_text(self, text: str) -> list[int]:
        """Кодирует известный текст в токены жадным longest-match (нужно для режима --lyrics).

        В словаре GigaAM только 45 пиков с `▁`, поэтому остальные слова начинаются
        с отдельного токена `▁` - иначе границы слов теряются.
        """
        ids: list[int] = []
        for word in text.split():
            pos = 0
            first = WORD_MARK + word[0]
            if first in self.inverse:
                ids.append(self.inverse[first])
                pos = 1
            elif WORD_MARK in self.inverse:
                ids.append(self.inverse[WORD_MARK])
            while pos < len(word):
                best_end = -1
                for size in range(min(len(word), pos + self.max_piece_len), pos, -1):
                    cand = word[pos:size]
                    if cand in self.inverse and cand not in SPECIAL_PIECES:
                        best_end = size
                        break
                if best_end < 0:
                    ids.append(self.unk_id)
                    pos += 1
                else:
                    ids.append(self.inverse[word[pos:best_end]])
                    pos = best_end
        return ids

    def normalize_text(self, text: str) -> str:
        """Приводит внешний текст к тому виду, который понимает словарь модели."""
        return self.detokenize(self.encode_text(text))

    # ------------------------------------------------------------------ выравнивание
    def align_ids(
        self, lp: np.ndarray, target: list[int], frame_sec: float
    ) -> tuple[list[TokenSpan], float, float]:
        """Выравнивает последовательность токенов по кадрам (CTC forced alignment, Витерби).

        Возвращает интервалы токенов, долю blank-кадров и среднюю лог-вероятность пути.
        """
        n_frames = int(lp.shape[0])
        if not target or n_frames == 0 or len(target) > n_frames:
            return [], 1.0, POOR_LOGPROB
        u = len(target)
        neg = np.float32(NEG_INF)
        # Разворачиваем метки в путь [blank, y1, blank, y2, ..., yU, blank] и ищем
        # максимальный по сумме лог-вероятностей путь, который монотонно проходит по
        # этому состоянию от 0 до 2U. Так blank может стоять между любыми токенами,
        # а каждый токен обязан занять хотя бы один кадр.
        state_ids = np.empty(2 * u + 1, dtype=np.int64)
        state_ids[0::2] = self.blank_id
        state_ids[1::2] = target
        n_states = state_ids.size
        emit_all = lp[:, state_ids]  # (T, 2U+1)

        alpha = np.full(n_states, neg, dtype=np.float32)
        alpha[0] = 0.0
        prev_state = np.zeros((n_frames, n_states), dtype=np.int32)
        states = np.arange(n_states, dtype=np.int32)
        for t in range(n_frames):
            cand_prev = np.empty_like(alpha)
            cand_prev[0] = neg
            cand_prev[1:] = alpha[:-1]
            take_prev = cand_prev > alpha
            alpha = (np.maximum(alpha, cand_prev) + emit_all[t]).astype(np.float32)
            prev_state[t] = np.where(take_prev, np.maximum(states - 1, 0), states)

        # Обратный проход: из финального состояния (все токены излучены) восстанавливаем путь
        state = n_states - 1
        path = np.zeros(n_frames, dtype=np.int64)
        for t in range(n_frames - 1, -1, -1):
            path[t] = state
            if t > 0:
                state = int(prev_state[t, state])
        path = np.maximum.accumulate(path)  # путь монотонен по построению; чиним численные сбои
        blank_ratio = float(np.mean((path % 2) == 0))
        # Уверенность = средняя лог-вероятность всего пути (кадры blank тоже учитываются).
        # На настоящем вокале кадры объясняются почти идеально (значение около нуля),
        # а на инструментале выдуманный текст объясняет аудио плохо (значения сильно ниже).
        frames = np.arange(n_frames)
        path_logprob = float(np.mean(emit_all[frames, path]))

        spans: list[TokenSpan] = []
        for i in range(1, u + 1):
            mask = np.flatnonzero(path == 2 * i - 1)
            piece = self.tokens.get(target[i - 1], "")
            if mask.size == 0:
                # Токен не попал в путь (бывает при опечатках в тексте): ставим в середину
                mid = float(n_frames // 2) * frame_sec
                spans.append(TokenSpan(target[i - 1], piece, mid, mid, NEG_INF))
                continue
            start = float(mask[0]) * frame_sec
            end = float(mask[-1] + 1) * frame_sec
            lps = emit_all[mask, 2 * i - 1]
            spans.append(TokenSpan(target[i - 1], piece, start, end, float(lps.mean())))
        return make_contiguous(spans, frame_sec), blank_ratio, path_logprob

    # ------------------------------------------------------------------ верхний уровень
    def transcribe(self, samples: np.ndarray, text_hint: str | None = None) -> AlignedResult:
        """Распознаёт сегмент и возвращает текст, слова и уверенность.

        Если передан text_hint (режим --lyrics), текст не распознаётся, а выравнивается.
        """
        lp, frame_sec = self.log_probs(samples)
        if lp.shape[0] == 0:
            return AlignedResult("", [], POOR_LOGPROB, 1.0)

        if text_hint:
            ids = self.encode_text(text_hint)
            text = self.detokenize(ids)
        else:
            ids = self.greedy_ids(lp)
            text = self.detokenize(ids)
        if not ids or not text:
            return AlignedResult(text, [], POOR_LOGPROB, 1.0)

        spans, blank_ratio, path_logprob = self.align_ids(lp, ids, frame_sec)
        words = tokens_to_words(spans)
        if not spans:
            # текст есть, но выравнить его не удалось (слишком короткий отрезок)
            return AlignedResult(text, [], POOR_LOGPROB, blank_ratio)
        return AlignedResult(text=text, words=words, mean_logprob=path_logprob, blank_ratio=blank_ratio)


def make_contiguous(spans: list[TokenSpan], frame_sec: float) -> list[TokenSpan]:
    """Убирает перекрытия между токенами: интервалы идут строго по порядку."""
    if not spans:
        return []
    out: list[TokenSpan] = []
    prev_end = 0.0
    for span in spans:
        start = max(span.start, prev_end)
        end = max(span.end, start + frame_sec)
        out.append(TokenSpan(span.token_id, span.piece, start, end, span.logprob))
        prev_end = end
    return out


def tokens_to_words(spans: list[TokenSpan]) -> list[WordSpan]:
    """Группирует токены в слова: новое слово начинается с пика, помеченного `▁`."""
    words: list[WordSpan] = []
    cur_parts: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_lps: list[float] = []
    for span in spans:
        piece = span.piece
        if piece in {"<unk>", "<blk>", "<s>", "</s>", "<pad>"}:
            continue
        if piece == WORD_MARK:
            # одиночный `▁` не несёт текста, но означает границу слова
            starts_word = True
            clean = ""
        else:
            starts_word = piece.startswith(WORD_MARK)
            clean = piece[1:] if starts_word else piece
        if starts_word and cur_parts:
            words.append(make_word("".join(cur_parts), cur_start, cur_end, cur_lps))
            cur_parts = []
            cur_lps = []
        if not clean:
            continue
        if not cur_parts:
            cur_start = span.start
        cur_parts.append(clean)
        cur_end = span.end
        if span.logprob > NEG_INF / 2:
            cur_lps.append(span.logprob)
    if cur_parts:
        words.append(make_word("".join(cur_parts), cur_start, cur_end, cur_lps))
    return words


def make_word(text: str, start: float, end: float, lps: list[float]) -> WordSpan:
    """Собирает WordSpan из накопленных частей слова."""
    return WordSpan(
        text=text,
        start=start,
        end=max(end, start),
        logprob=float(np.mean(lps)) if lps else 0.0,
    )
