# subsong — subtitles for song videos

A CLI that turns a song video (mp4/mkv/webm/mov + mp3/wav) into subtitles
**with word-level timing**: plain SRT, karaoke ASS, WebVTT and a detailed JSON.

The stack comes from [sherpa-onnx-russian-lab](https://github.com/akret00/sherpa-onnx-russian-lab)
(sherpa-onnx, Silero VAD, GigaAM v3), adapted to the harder problem of *singing over a backing track*:

* **CTC forced alignment** — the text and the word timings come from a single ONNX pass,
  so the karaoke highlight follows real sounds instead of a guess;
* **existing lyrics** (LRC or plain txt) are used as a hint — the most accurate mode:
  the words don't need to be guessed, they only need to be laid onto the audio;
* **confidence-based filtering** — on instrumentals the model invents plausible lyrics,
  but with a low log-probability; such cues are dropped;
* **vocal separation and denoising** — optional, via sherpa-onnx models (no torch).

CPU only, no torch, no librosa, no cloud. Python 3.10+.

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -e .
```

**ffmpeg** must be in PATH (or drop `ffmpeg.exe` into the project's `bin/` folder):
<https://ffmpeg.org/download.html>.

## Models

Downloaded automatically on first run (into `models/` or `$SUBSONG_MODELS`):

```bash
subsong models list                       # what's available and what's already installed
subsong models download gigaam-v3-ru      # GigaAM v3 CTC (~215 MB) - the default ASR model
subsong models download silero-vad        # speech detector (~1 MB)
subsong models download spleeter-2stems   # vocal separation (~50 MB), optional
subsong models download gtcrn              # denoising (~1 MB), optional
subsong selfcheck                          # verify ffmpeg, models and ONNX
```

## Quick start

```bash
subsong song.mp4                            # -> song.srt and song.ass
subsong song.mp4 --formats srt,ass,vtt,json # every format at once
subsong song.mp4 --lyrics words.lrc         # subtitles from existing lyrics
subsong song.mp4 -o subs/ru --uppercase     # custom names, karaoke-style caps
```

### Mode 1. Recognition only (no lyrics available)

```
video -> ffmpeg -> Silero VAD -> GigaAM CTC -> junk filter -> cue assembly -> SRT/ASS
```

On the demo track (vocal + backing track, 23 s, 6 chorus lines) this gives **WER 0.12**:
all six lines make it into the subtitles, while the instrumental intro and outro do not.

### Mode 2. Lyrics available (the accurate mode)

```bash
subsong song.mp4 --lyrics words.lrc   # LRC with timestamps: each line aligns to its own window
subsong song.mp4 --lyrics words.txt   # no timestamps: text is distributed over speech segments
```

On the same demo, LRC mode gives **WER 0.00** and a mean word timing error of **0.18 s** —
enough for karaoke.

When the model has nothing to guess but you do have the words, pass `--lyrics`: instrumentals
and noise stop interfering, and the model no longer invents words. For a `.txt` without
timestamps the text is spread across speech segments, so cue boundaries come out coarser —
accurate karaoke needs LRC with timestamps.

## How it works

```
ffmpeg ──► 16 kHz mono PCM
              │
              ├─ (optional) Spleeter/UVR: vocal separation
              ├─ (optional) GTCRN: denoising
              │
          Silero VAD ──► speech segments (merge, split, trim silence)
              │
          GigaAM v3 CTC (onnxruntime) ──► log_probs
              ├─► greedy decoding ──► text
              └─► forced alignment (Viterbi) ──► time of every word + confidence
              │
          filters: confidence, repetition, energy, spectral music gate
              │
          cue assembly (length, reading speed, line boundaries)
              │
          SRT / ASS (karaoke \k) / VTT / JSON
```

The key idea is **confidence from forced alignment**. We take the mean log-probability of
the entire alignment path (blank frames included). On live vocals it sits near zero and above;
on instrumentals the model invents words at around `-4` and below — those cues are dropped
automatically.

## Important options

### Recognition quality

| Option | What it does |
|---|---|
| `--lyrics FILE` | hint with the song text (LRC or txt) — the most accurate path |
| `--separate` | extract vocals from the backing track (Spleeter/UVR, downloaded automatically) |
| `--denoise` | GTCRN denoising (for noisy field recordings) |
| `--min-confidence -3` | drop doubtful cues more aggressively (default `-4.0`, `-99` disables) |
| `--max-repeat 3` | drop looping chunks of text |
| `--music-gate 1` | extra spectral filtering of instrumentals |
| `--vad-threshold 0.5` | raise the VAD threshold if cues merge together (default 0.35 works better for songs) |
| `--normalize rms` | level the loudness (useful for quiet recordings) |
| `--no-align` | fast mode without forced alignment (words are spread evenly) |

> `--separate` can fail on a synthetic backing track; subsong detects this and prints a
> warning, then continues with the original mix instead of returning an empty result.

### Subtitle style

| Option | Default | What it does |
|---|---|---|
| `--max-chars` | 42 | characters per line |
| `--max-lines` | 2 | lines per cue |
| `--max-duration` | 7.0 | seconds per cue |
| `--min-duration` | 1.0 | minimum on-screen time, so cues don't flash |
| `--max-cps` | 20.0 | characters per second |
| `--min-gap` | 0.08 | pause between cues, s |
| `--lead-in` / `--lead-out` | 0.0 / 0.2 | shift and "tail" of a cue |
| `--uppercase` | off | UPPERCASE |

Full list: `subsong transcribe --help`. Values can also live in YAML:
`cp config.yaml.sample subsong.yaml` and `subsong song.mp4 --config subsong.yaml`.

### Models

| Preset | Purpose |
|---|---|
| `gigaam-v3-ru` | default CTC model: Russian, with punctuation, supports word alignment |
| `gigaam-v3-ru-tensor` | transducer version: better on noise, no word alignment |
| `qwen3-0.6b` | multilingual ASR (recognises, but words are spread evenly) |
| `silero-vad` | speech detector |
| `spleeter-2stems`, `uvr-main` | vocal separation |
| `gtcrn` | denoising |

You can point at your own model directory:
`subsong song.mp4 --asr-model /path/to/model_dir`.

## Output formats

* **SRT** — plain subtitles;
* **ASS** — karaoke: `\k` tags highlight each word as it is sung
  (verified: the per-word duration in centiseconds comes from the alignment);
* **VTT** — for the web, with a `NOTE` header about the source;
* **JSON** — cues, words, times, per-word logprob and metadata;
* **TXT** — timecode + text, handy for proofreading.

## Tests and demo

```bash
python tools/make_demo_song.py --out demo   # synthesise the demo song (Piper + backing track)
python tools/eval_demo.py                    # WER and timing accuracy per run configuration
python -m pytest -q                          # 68 tests
python -m mypy                               # strict typing
python -m pylint src/subsong                 # code style
```

The demo song is built from synthesised vocals (Piper `ru_RU-irina`) and a numpy-generated
backing track, so it needs no third-party music rights.

## Limitations

* The default model is Russian GigaAM: for other languages use `qwen3-0.6b`, but then word
  timings are spread evenly (no alignment).
* On real singing recognition is worse than on speech: drawn-out vowels, melodic rises and
  a dense backing track. `--lyrics` is the cure.
* The spectral music gate is a heuristic; the confidence threshold does the heavy lifting
  when filtering out instrumentals.

## License

MIT, see [LICENSE](LICENSE). The models keep their own licenses.