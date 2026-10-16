"""Собирает демо-песню для проверки subsong: синтез вокала + минусовка + видео.

Нужен для воспроизводимых тестов без чужих прав на музыку:
вокал синтезируется моделью Piper (ru_RU-irina), минусовка генерируется numpy,
а ffmpeg запекает всё в mp4 с картинкой.

Запуск:
    python tools/make_demo_song.py --out demo
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 44100
LYRICS = [
    "Мы идём по свету вместе",
    "И поём под дождём",
    "Эта ночь принадлежит нам",
    "И горит в окне огонь",
    "Ничего не будет завтра",
    "Только музыка и мы",
]


def tts_dir() -> Path:
    """Ищет каталог модели Piper среди типовых locations."""
    for candidate in (
        Path("models/tts/piper-irina/vits-piper-ru_RU-irina-medium-int8"),
        Path.home() / ".cache" / "subsong" / "piper-irina",
    ):
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "Модель Piper не найдена. Скачайте её:\n"
        "  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-piper-ru_RU-irina-medium-int8.tar.bz2\n"
        "и распакуйте в models/tts/piper-irina/"
    )


def synthesize_lyrics(out_dir: Path, noise_scale: float = 0.0, noise_w: float = 0.0) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Синтезирует вокал построчно и возвращает дорожку вместе с таймкодами строк.

    Шум VITS по умолчанию выключен (noise_scale=0): иначе Piper каждый раз даёт
    немного другой вокал и демо перестаёт быть воспроизводимым.
    """
    import sherpa_onnx

    model_dir = tts_dir()
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"В {model_dir} нет .onnx модели Piper")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(onnx_files[0]),
                lexicon=str(model_dir / "lexicon.txt") if (model_dir / "lexicon.txt").is_file() else "",
                tokens=str(model_dir / "tokens.txt"),
                data_dir=str(model_dir / "espeak-ng-data") if (model_dir / "espeak-ng-data").is_dir() else "",
                noise_scale=noise_scale,
                noise_scale_w=noise_w,
            )
        ),
        max_num_sentences=1,
    )
    if not cfg.validate():
        raise SystemExit(f"Некорректная конфигурация TTS для {onnx_files[0]}")
    tts = sherpa_onnx.OfflineTts(cfg)
    tts_sr = tts.sample_rate

    parts: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    cursor = 0.0
    for line in LYRICS:
        audio = tts.generate(line, sid=0, speed=1.0)
        audio = np.asarray(audio.samples, dtype=np.float32)
        start = cursor
        parts.append(audio)
        meta.append({"text": line, "start": round(start, 3), "end": round(start + len(audio) / tts_sr, 3)})
        gap = np.zeros(int(tts_sr * 0.55), dtype=np.float32)
        parts.append(gap)
        cursor += (len(audio) + gap.size) / tts_sr

    vocal = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    vocal = _resample(vocal, tts_sr, SR)
    return vocal, meta


def _resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Линейный ресемплинг - для синтеза тестовых данных этого достаточно."""
    if src_sr == dst_sr:
        return x
    n_out = int(round(len(x) * dst_sr / src_sr))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def make_instrumental(duration: float, seed: int = 7) -> np.ndarray:
    """Генерирует минусовку: аккорды, бас и ударные - намеренно 'музыкально' гармоничную."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SR

    # тональность ля минор: четыре аккорда по два такта
    chords = [[220.0, 261.6, 329.6], [196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [146.8, 220.0, 293.7]]
    bar = 4.0  # секунд на аккорд
    for bar_idx in range(int(duration // bar) + 1):
        start = int(bar_idx * bar * SR)
        stop = min(n, start + int(bar * SR))
        if stop <= start:
            break
        chord = chords[bar_idx % len(chords)]
        seg_t = t[start:stop] - t[start]
        env = np.minimum(1.0, seg_t / 0.4) * np.exp(-0.35 * seg_t / bar)
        for f in chord:
            out[start:stop] += (
                0.16 * env * np.sin(2 * np.pi * f * seg_t)
                + 0.07 * env * np.sin(2 * np.pi * 2 * f * seg_t + 0.4)
            )
        # бас
        bass = chord[0] / 2.0
        out[start:stop] += 0.22 * env * np.sin(2 * np.pi * bass * seg_t)
    # ударные: бочка на 1 и 3, малый барабан на 2 и 4
    beat = 0.5
    for i in range(int(duration / beat)):
        pos = int(i * beat * SR)
        if pos + 2200 >= n:
            break
        length = 2200
        tt = np.arange(length) / SR
        if i % 4 in (0, 2):
            kick = np.sin(2 * np.pi * (110 - 60 * tt) * tt) * np.exp(-28 * tt)
            out[pos : pos + length] += 0.35 * kick
        if i % 4 in (1, 3):
            snare = rng.standard_normal(length) * np.exp(-45 * tt)
            out[pos : pos + length] += 0.16 * snare
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.6).astype(np.float32)


def mix_song(vocal: np.ndarray, instrumental: np.ndarray, vocal_db: float = -6.0) -> np.ndarray:
    """Смешивает вокал и минусовку с заданным уровнем вокала."""
    n = max(len(vocal), len(instrumental))
    mix = np.zeros(n, dtype=np.float32)
    mix[: len(instrumental)] += instrumental
    gain = 10.0 ** (vocal_db / 20.0)
    vpeak = float(np.max(np.abs(vocal))) or 1.0
    inst_peak = float(np.max(np.abs(instrumental))) or 1.0
    vocal_norm = vocal / vpeak * inst_peak * gain
    mix[: len(vocal)] += vocal_norm
    return np.clip(mix, -0.98, 0.98).astype(np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Дублирует моно в стерео с небольшой расстройкой - так ближе к настоящей записи."""
    delay = int(SR * 0.006)
    left = x
    right = np.concatenate([np.zeros(delay, dtype=np.float32), x])[: len(x)]
    return np.stack([left, right])


def write_wav(path: Path, stereo: np.ndarray) -> Path:
    """Пишет 16-битный WAV."""
    import wave

    data = np.clip(stereo, -1.0, 1.0)
    pcm = (data.T * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(stereo.shape[0])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def mux_video(wav_path: Path, mp4_path: Path) -> None:
    """Запекает wav в mp4 с простым видеорядом через ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25",
        "-i", str(wav_path),
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k", str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    """Собирает демо-песню и файлы с текстом."""
    ap = argparse.ArgumentParser(description="Генератор демо-песни для тестов subsong")
    ap.add_argument("--out", default="demo", help="каталог для результатов")
    ap.add_argument("--vocal-db", type=float, default=-6.0, help="уровень вокала относительно минусовки")
    ap.add_argument("--intro", type=float, default=6.0, help="длительность инструментального вступления, с")
    ap.add_argument("--outro", type=float, default=4.0, help="длительность инструментального финала, с")
    ap.add_argument("--no-video", action="store_true", help="не создавать mp4")
    ap.add_argument("--tts-noise", type=float, default=0.0, help="шум VITS (0 = воспроизводимый вокал)")
    ap.add_argument("--tts-noise-w", type=float, default=0.0, help="шум длительности VITS")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Синтезирую вокал (Piper ru_RU-irina)...")
    vocal, meta = synthesize_lyrics(out_dir, args.tts_noise, args.tts_noise_w)
    print(f"  строк: {len(meta)}, длительность: {len(vocal) / SR:.1f} c")

    offset = args.intro
    padded_vocal = np.concatenate([np.zeros(int(offset * SR), dtype=np.float32), vocal])
    total = len(padded_vocal) / SR + args.outro
    print("Генерирую минусовку...")
    instrumental = make_instrumental(total)
    print(f"  длительность: {total:.1f} c")
    mix = mix_song(padded_vocal, instrumental, args.vocal_db)

    wav_path = write_wav(out_dir / "song.wav", to_stereo(mix))
    print(f"Записан {wav_path}")
    if not args.no_video:
        mux_video(wav_path, out_dir / "song.mp4")
        print(f"Записан {out_dir / 'song.mp4'}")

    # эталонные таймкоды считаем от начала итогового файла (вокал сдвинут на интро)
    for m in meta:
        m["start"] = round(float(m["start"]) + offset, 3)
        m["end"] = round(float(m["end"]) + offset, 3)
    plain = out_dir / "lyrics.txt"
    plain.write_text("\n".join(str(m["text"]) for m in meta) + "\n", encoding="utf-8")
    lrc_lines = ["[ti:Демо-песня]", "[ar:subsong]"]
    for m in meta:
        start = float(m["start"])
        lrc_lines.append(f"[{int(start // 60):02d}:{start % 60:05.2f}]{m['text']}")
    (out_dir / "lyrics.lrc").write_text("\n".join(lrc_lines) + "\n", encoding="utf-8")
    (out_dir / "reference.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Записаны {plain}, {out_dir / 'lyrics.lrc'}, {out_dir / 'reference.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())