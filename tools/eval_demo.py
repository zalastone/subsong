"""Оценка качества subsong на демо-песне: WER + точность таймингов по словам.

Запуск:
    python tools/make_demo_song.py --out demo
    python tools/eval_demo.py
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, "src")


def norm(text: str) -> str:
    """Нормализует строку для подсчёта ошибок: только буквы и пробелы, нижний регистр."""
    return " ".join(re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", " ", text.lower()).split())


def levenshtein(ref: list[str], hyp: list[str]) -> int:
    """Расстояние Левенштейна между списками слов."""
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hw in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1]


def run(args: list[str], show_log: bool = False) -> pathlib.Path:
    """Запускает CLI и возвращает путь к JSON с результатом."""
    out = pathlib.Path("demo/eval") / "song.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-X", "utf8", "-m", "subsong", "transcribe", "demo/song.mp4",
        "--formats", "json", "-o", str(out.with_suffix("")),
    ] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if show_log:
        print(proc.stderr.strip())
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"subsong вернул код {proc.returncode}")
    return out


def evaluate(name: str, extra: list[str], ref_lines: list[dict], show_log: bool = False) -> None:
    """Прогоняет CLI с заданными флагами и печатает метрики + распознанный текст."""
    path = run(extra, show_log)
    data = json.loads(path.read_text(encoding="utf-8"))
    hyp = " ".join(c["text"].replace("\n", " ") for c in data["cues"])
    ref = " ".join(m["text"] for m in ref_lines)
    ref_words, hyp_words = norm(ref).split(), norm(hyp).split()
    wer = levenshtein(ref_words, hyp_words) / max(1, len(ref_words))

    # ошибка времени старта: сравниваем по позициям слов (грубая, но полезная метрика)
    hyp_starts = [w["start"] for c in data["cues"] for w in c["words"]]
    errs = []
    for i, rw in enumerate(ref_words):
        if i < len(hyp_starts):
            errs.append(abs(hyp_starts[i] - ref_line_starts(ref_lines)[i]))
    mean_err = sum(errs) / len(errs) if errs else float("nan")

    print(f"\n=== {name} ===")
    print(f"флаги: {' '.join(extra) or 'по умолчанию'}")
    print(f"реплик: {len(data['cues'])}, WER: {wer:.2f}, средняя ошибка времени слова: {mean_err:.2f} c")
    for c in data["cues"]:
        print(f"  [{c['start']:6.2f}-{c['end']:6.2f}] {c['text'].replace(chr(10), ' / ')}")


_STARTS: dict[str, list[float]] = {}


def ref_line_starts(lines: list[dict]) -> list[float]:
    """Возвращает время начала каждого слова эталона (для метрики по словам)."""
    key = id(lines)
    if key not in _STARTS:
        starts: list[float] = []
        for m in lines:
            words = m["text"].split()
            n = max(1, len(words))
            for k in range(n):
                starts.append(float(m["start"]) + (float(m["end"]) - float(m["start"])) * k / n)
        _STARTS[key] = starts
    return _STARTS[key]


if __name__ == "__main__":
    ref_lines = json.loads(pathlib.Path("demo/reference.json").read_text(encoding="utf-8"))
    print("Эталон:")
    for m in ref_lines:
        print(f"  {m['start']:6.2f}-{m['end']:6.2f} {m['text']}")

    for name, extra, verbose in [
        ("как есть", [], False),
        ("с выделением вокала", ["--separate"], False),
        ("с шумоподавлением", ["--denoise"], False),
        ("с фильтром инструментала", ["--music-gate", "1"], False),
        ("текст песни, LRC", ["--lyrics", "demo/lyrics.lrc"], True),
        ("текст песни, txt", ["--lyrics", "demo/lyrics.txt"], True),
        ("без выравнивания слов", ["--no-align"], False),
    ]:
        try:
            evaluate(name, extra, ref_lines, verbose)
        except SystemExit as exc:
            print(f"\n=== {name}: не удалось ({exc}) ===")