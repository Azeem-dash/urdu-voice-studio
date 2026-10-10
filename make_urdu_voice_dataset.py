#!/usr/bin/env python3
"""
make_urdu_voice_dataset.py
Build an Urdu voice-note dataset in YOUR OWN voice, on your own computer.

    texts.csv (Urdu text + Roman Urdu)  +  a few short recordings of your voice
        -> OmniVoice (open-source zero-shot voice cloning; supports Urdu)
        -> audio/*.wav + metadata.csv + segments.csv + README.md

Once it's installed it runs offline, so you can generate as many notes as you like.

---------------------------------------------------------------- SETUP (once)
1. Python 3.10+. An NVIDIA GPU is best, Apple Silicon works, CPU works but is slow.
   No GPU? Run it in Google Colab or Kaggle with a GPU runtime (torch is preinstalled
   there, so you only need step 2b).
2a. NVIDIA GPU:
      pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    Mac or CPU only:
      pip install torch==2.8.0 torchaudio==2.8.0
2b. pip install omnivoice==0.2.1 scipy        (the version this script was written for)
3. Make a folder with 1-5 recordings of your voice, 3-10 seconds each, in a quiet room.
   Next to each one put a .txt file with EXACTLY what you said, in Urdu script:
       my_voice/calm.wav    my_voice/calm.txt
       my_voice/happy.wav   my_voice/happy.txt
   Different moods give more varied notes; each note uses one clip, taking turns.
   WAV is safest. Convert phone or WhatsApp audio with:
       ffmpeg -i note.ogg -ac 1 -ar 24000 calm.wav

---------------------------------------------------------------- RUN
    # 1) try 5 notes first and listen to them
    python make_urdu_voice_dataset.py --texts texts.csv --voice-dir my_voice --out urdu_dataset --limit 5
    # 2) then generate the rest (finished notes are skipped, so you can stop and resume)
    python make_urdu_voice_dataset.py --texts texts.csv --voice-dir my_voice --out urdu_dataset --check

texts.csv: one voice note per row, saved as UTF-8 (Excel: Save As > "CSV UTF-8").
    urdu_text   required, Urdu script. About 95-110 words gives a 40-45 second note.
    roman_urdu  optional, copied into metadata.csv as-is.
    id          optional, used as the file name (otherwise note_0001, note_0002, ...).

--check    transcribes every generated piece with Whisper (forced to Urdu), regenerates
           pieces that come out garbled (skipped or repeated words), and records a rough
           character error rate per clip. It needs about 2 GB more GPU memory.
--speed    speech speed: 0.9 is slower and longer, 1.1 is faster and shorter.
--dry-run  tests the whole pipeline without the model (it writes beeps, not speech).

Why this script splits the text itself: OmniVoice's built-in long-text splitter only
knows Latin and Chinese punctuation, not the Urdu full stop or question mark, so a
40-second Urdu paragraph would be generated in one go. Here every note is cut at Urdu
sentence boundaries into pieces of about 10-15 seconds, generated, and joined with
short pauses.

OmniVoice's weights are licensed CC-BY-NC 4.0: fine for coursework and research, not for
commercial use. The output is synthetic speech, and the README written into the output
folder says so.
"""

import argparse
import csv
import hashlib
import re
import sys
import time
import unicodedata
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}

# Arabic-keyboard look-alikes -> proper Urdu letters (ي and ى -> ی, ك -> ک)
URDU_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک"})
SENTENCE_END = set("۔؟!?")   # Urdu full stop ۔, Arabic question mark ؟, ! ?  ('.' handled separately)
TRAILING = set("۔؟!?.\"'”’)»")
SOFT_BREAK_RE = re.compile(r"(?<=[،,؛;:])\s+")   # after ، , ؛ ; :
MARKS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")   # Arabic-script diacritics

META_FIELDS = ["file_name", "id", "urdu_text", "roman_urdu", "speaker_id", "ref_clip",
               "duration_sec", "sample_rate", "asr_cer", "flags", "synthetic", "tts_model"]
SEG_FIELDS = ["file_name", "segment", "start_sec", "end_sec", "urdu_text", "asr_cer"]


# ------------------------------------------------------------------ text

def clean_urdu(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(URDU_FIX)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """Split at ۔ ؟ ! ? . and line breaks, keeping the punctuation (3.5 is not split)."""
    sentences, start, i, n = [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            if text[start:i].strip():
                sentences.append(text[start:i].strip())
            start = i = i + 1
            continue
        if ch in SENTENCE_END or (ch == "." and not (i + 1 < n and text[i + 1].isdigit())):
            j = i + 1
            while j < n and text[j] in TRAILING:
                j += 1
            if text[start:j].strip():
                sentences.append(text[start:j].strip())
            start = i = j
            continue
        i += 1
    if text[start:].strip():
        sentences.append(text[start:].strip())
    merged = []
    for s in sentences:   # glue punctuation-only fragments onto the previous sentence
        if merged and not any(c.isalnum() for c in s):
            merged[-1] += s
        else:
            merged.append(s)
    return merged


def _pack(pieces: list, max_chars: int) -> list:
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def make_chunks(text: str, max_chars: int = 160, min_chars: int = 25) -> list:
    """Cut a note into pieces of at most ~max_chars (about 10-15 s of speech):
    at sentence ends first, then after commas, then between words."""
    pieces = []
    for sentence in split_sentences(text):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for part in _pack(SOFT_BREAK_RE.split(sentence), max_chars):
            pieces.extend([part] if len(part) <= max_chars else _pack(part.split(), max_chars))
    chunks = _pack(pieces, max_chars)
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:   # no tiny last piece
        last = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {last}"
    return chunks


def _norm_for_cer(s: str) -> str:
    s = MARKS_RE.sub("", unicodedata.normalize("NFC", s).translate(URDU_FIX).lower())
    return "".join(c for c in s if c.isalnum())


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Edit distance / reference length, ignoring spaces, punctuation and diacritics."""
    ref, hyp = _norm_for_cer(reference), _norm_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def weighted_cer(chunks: list, cers: list):
    if not cers or any(c is None for c in cers):
        return None
    weights = [len(_norm_for_cer(c)) for c in chunks]
    total = sum(weights)
    return sum(w * c for w, c in zip(weights, cers)) / total if total else 0.0


# ------------------------------------------------------------------ audio

def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    from scipy.signal import resample_poly
    f = Fraction(sr_out, sr_in)
    return resample_poly(audio, f.numerator, f.denominator).astype(np.float32)


def join_pieces(pieces: list, sr: int, pause_sec: float):
    """Concatenate pieces with silence between them; return audio and (start, end) of each piece."""
    gap = np.zeros(int(round(pause_sec * sr)), dtype=np.float32)
    parts, spans, t = [], [], 0
    for k, a in enumerate(pieces):
        if k:
            parts.append(gap)
            t += len(gap)
        parts.append(a.astype(np.float32))
        spans.append((t / sr, (t + len(a)) / sr))
        t += len(a)
    return np.concatenate(parts), spans


# ------------------------------------------------------------------ engines

class OmniVoiceTTS:
    """Thin wrapper around OmniVoice (written against omnivoice 0.2.1)."""

    def __init__(self, model_id: str, device: str, asr_model=None):
        import torch
        from omnivoice import OmniVoice

        self.torch = torch
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        print(f"Loading {model_id} on {device} (the first run downloads a few GB) ...")
        self.model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
        self.sr = int(self.model.sampling_rate)
        self.asr = None
        if asr_model:
            from transformers import pipeline
            print(f"Loading {asr_model} for --check ...")
            self.asr = pipeline("automatic-speech-recognition", model=asr_model,
                                dtype=dtype, device=device)

    def voice(self, wav_path: Path, transcript: str):
        return self.model.create_voice_clone_prompt(ref_audio=str(wav_path), ref_text=transcript)

    def synth(self, texts: list, voice, seed: int, speed=None) -> list:
        self.torch.manual_seed(seed)
        audios = self.model.generate(text=list(texts), language="ur",
                                     voice_clone_prompt=voice, speed=speed)
        return [np.asarray(a, dtype=np.float32).reshape(-1) for a in audios]

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.asr({"array": audio, "sampling_rate": self.sr},
                          generate_kwargs={"language": "urdu", "task": "transcribe"})
        return result["text"].strip()

    def free_memory(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class FakeTTS:
    """--dry-run stand-in: beeps whose length follows the text, so the whole pipeline
    (files, CSVs, resume, --check retries) can be tested without the model."""

    sr = 24000

    def __init__(self):
        self._heard = {}

    @staticmethod
    def _key(audio: np.ndarray) -> bytes:
        return hashlib.blake2b(audio.tobytes(), digest_size=16).digest()

    def voice(self, wav_path, transcript):
        return str(wav_path)

    def synth(self, texts, voice, seed, speed=None):
        rng = np.random.default_rng(seed)
        out = []
        for t in texts:
            n = int(self.sr * 0.075 * len(t) / (speed or 1.0))
            x = (0.2 * np.sin(2 * np.pi * rng.uniform(150, 250) * np.arange(n) / self.sr)).astype(np.float32)
            # pretend about 1 piece in 4 comes out garbled, to exercise --check
            self._heard[self._key(x)] = t if rng.random() > 0.25 else t[: len(t) // 3]
            out.append(x)
        return out

    def transcribe(self, audio):
        return self._heard.get(self._key(audio), "")

    def free_memory(self):
        pass


def pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    print("No GPU found, so running on CPU. It works, but expect it to be slow.")
    return "cpu"


# ------------------------------------------------------------------ inputs and outputs

def read_texts(path: Path) -> list:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [(h or "").strip().lower() for h in (reader.fieldnames or [])]
            rows = list(reader)
    except FileNotFoundError:
        sys.exit(f"Text file not found: {path}")
    except UnicodeDecodeError:
        sys.exit(f"{path} is not UTF-8. In Excel use Save As > 'CSV UTF-8 (Comma delimited)'.")
    if "urdu_text" not in reader.fieldnames:
        sys.exit(f"{path} needs a 'urdu_text' column (found: {', '.join(reader.fieldnames)}).")
    notes, seen = [], set()
    for n, row in enumerate(rows, 1):
        text = clean_urdu(row.get("urdu_text") or "")
        if not text:
            continue
        note_id = re.sub(r"[^A-Za-z0-9_\-]", "_", (row.get("id") or "").strip()) or f"note_{n:04d}"
        if note_id in seen:
            sys.exit(f"Duplicate id '{note_id}' in {path} (row {n}). Ids must be unique.")
        seen.add(note_id)
        notes.append({"id": note_id, "urdu_text": text,
                      "roman_urdu": (row.get("roman_urdu") or "").strip()})
    if not notes:
        sys.exit(f"No rows with text in {path}.")
    return notes


def read_voices(folder: Path) -> list:
    if not folder.is_dir():
        sys.exit(f"Voice folder not found: {folder}")
    voices = []
    for clip in sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS):
        txt = clip.with_suffix(".txt")
        if not txt.exists():
            sys.exit(f"{clip.name} needs {txt.name} next to it, holding exactly what you said (Urdu script).")
        transcript = clean_urdu(txt.read_text(encoding="utf-8-sig"))
        if not transcript:
            sys.exit(f"{txt.name} is empty.")
        try:
            seconds = sf.info(str(clip)).duration
            if seconds < 3 or seconds > 12:
                print(f"Note: {clip.name} is {seconds:.1f}s long; 3-10 s clones best.")
        except Exception:
            pass   # format soundfile can't read (e.g. m4a); OmniVoice will still try
        voices.append((clip, transcript))
    if not voices:
        sys.exit(f"No recordings in {folder} (looked for {', '.join(sorted(AUDIO_EXTS))}).")
    return voices


def write_csv(path: Path, fields: list, rows) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tidy_previous_run(meta_path: Path, seg_path: Path, out_dir: Path) -> set:
    """Keep only rows whose WAV exists (newest wins), so re-runs never duplicate rows."""
    done = {}
    for r in read_csv(meta_path):
        if (out_dir / r["file_name"]).exists():
            done[r["id"]] = r
    keep = {r["file_name"] for r in done.values()}
    write_csv(meta_path, META_FIELDS, done.values())
    write_csv(seg_path, SEG_FIELDS, [r for r in read_csv(seg_path) if r["file_name"] in keep])
    return set(done)


def hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def write_readme(out_dir: Path, args, tts_model: str) -> None:
    rows = read_csv(out_dir / "metadata.csv")
    if not rows:
        return
    durations = [float(r["duration_sec"]) for r in rows]
    flagged = sum(1 for r in rows if r.get("flags"))
    speakers = sorted({r["speaker_id"] for r in rows})
    check_line = (
        f"Every piece was transcribed with Whisper ({args.asr_model}, forced to Urdu). "
        f"Pieces with a character error rate above {args.max_cer} were regenerated up to "
        f"{args.retries} times. `asr_cer` is that score (lower is better). It includes "
        "Whisper's own mistakes, so treat it as a rough guide and listen to flagged clips."
        if args.check else
        "No automatic check was run (`--check` was off), so listen to a sample of clips."
    )
    made_with = (
        "This was a dry run: the audio is test beeps, not speech." if args.dry_run else
        f"Speech was generated with {tts_model}, a zero-shot voice-cloning text-to-speech model, "
        "conditioned on short recordings of the speaker's own voice."
    )
    text = f"""# Urdu voice-note dataset (synthetic speech)

| | |
|---|---|
| Clips | {len(rows)} |
| Total audio | {hms(sum(durations))} |
| Clip length | {min(durations):.1f} to {max(durations):.1f} s (mean {sum(durations) / len(durations):.1f} s) |
| Speaker(s) | {", ".join(speakers)}: voice cloned from the speaker's own recordings |
| Audio | {args.sample_rate} Hz, mono, 16-bit WAV in `audio/` |
| Flagged for review | {flagged} (see the `flags` column) |

## Files
- `metadata.csv`: one row per clip with `file_name`, `id`, `urdu_text`, `roman_urdu`, `speaker_id`,
  `ref_clip` (which voice recording was cloned), `duration_sec`, `sample_rate`, `asr_cer`, `flags`,
  `synthetic`, `tts_model`.
- `segments.csv`: start and end time of every piece inside its clip, with its Urdu text. Each piece is
  well under 30 s, which matters if you train Whisper-style models, since they can't take 40-second clips.
- `audio/`: the WAV files.

## How it was made
Text came from `{Path(args.texts).name}` (Urdu script, plus Roman Urdu where provided). {made_with}
Each note was split at Urdu sentence boundaries, generated piece by piece, and the pieces were joined
with {args.pause} s pauses.

{check_line}

## Please note
- This is **synthetic speech**, not natural recordings. Say so wherever you use or share it.
- OmniVoice's model weights are licensed CC-BY-NC 4.0, so use this data for non-commercial purposes.

## Loading it
```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="{out_dir.name}")
```
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ generation

def synth_in_batches(tts, chunks, voice, seed, args) -> list:
    out, i = [], 0
    while i < len(chunks):
        batch = chunks[i:i + args.batch_size]
        try:
            out.extend(tts.synth(batch, voice, seed + i, args.speed))
            i += len(batch)
        except RuntimeError as e:   # torch's out-of-memory error is a RuntimeError
            if "out of memory" not in str(e).lower() or args.batch_size == 1:
                raise
            tts.free_memory()
            args.batch_size = max(1, args.batch_size // 2)
            print(f"  GPU memory full, continuing with --batch-size {args.batch_size}")
    return out


def make_note(tts, text: str, voice, seed: int, args):
    chunks = make_chunks(text, args.max_chars)
    pieces = synth_in_batches(tts, chunks, voice, seed, args)
    cers = [None] * len(chunks)
    if args.check:
        for i, (chunk, audio) in enumerate(zip(chunks, pieces)):
            best_audio, best_cer = audio, char_error_rate(chunk, tts.transcribe(audio))
            for attempt in range(1, args.retries + 1):
                if best_cer <= args.max_cer:
                    break
                retry = tts.synth([chunk], voice, seed + 7919 * attempt + i, args.speed)[0]
                retry_cer = char_error_rate(chunk, tts.transcribe(retry))
                if retry_cer < best_cer:
                    best_audio, best_cer = retry, retry_cer
            pieces[i], cers[i] = best_audio, best_cer
    audio, spans = join_pieces(pieces, tts.sr, args.pause)
    audio = np.clip(resample(audio, tts.sr, args.sample_rate), -1.0, 1.0)
    return chunks, audio, spans, cers


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an Urdu voice-note dataset in your own (cloned) voice with OmniVoice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--texts", required=True, help="CSV with a urdu_text column (optional: id, roman_urdu)")
    p.add_argument("--voice-dir", required=True, help="folder of 3-10 s recordings of your voice, each with a .txt transcript")
    p.add_argument("--out", default="urdu_voice_dataset", help="output folder")
    p.add_argument("--speaker-id", default="spk01")
    p.add_argument("--limit", type=int, default=0, help="only generate this many notes (0 = all)")
    p.add_argument("--check", action="store_true", help="verify each piece with Whisper and regenerate garbled ones")
    p.add_argument("--asr-model", default="openai/whisper-large-v3-turbo", help="Whisper model for --check")
    p.add_argument("--max-cer", type=float, default=0.35, help="--check: regenerate pieces above this character error rate")
    p.add_argument("--retries", type=int, default=2, help="--check: regeneration attempts per bad piece")
    p.add_argument("--speed", type=float, default=None, help="speaking speed factor (e.g. 0.9 slower, 1.1 faster)")
    p.add_argument("--pause", type=float, default=0.25, help="seconds of silence between pieces")
    p.add_argument("--max-chars", type=int, default=160, help="max characters per generated piece")
    p.add_argument("--batch-size", type=int, default=4, help="pieces generated at once (lowered automatically if GPU memory runs out)")
    p.add_argument("--sample-rate", type=int, default=16000, help="output sample rate (OmniVoice itself makes 24000)")
    p.add_argument("--min-sec", type=float, default=40.0, help="flag clips shorter than this")
    p.add_argument("--max-sec", type=float, default=45.0, help="flag clips longer than this")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto", help="auto, cuda:0, mps or cpu")
    p.add_argument("--model", default="k2-fsa/OmniVoice")
    p.add_argument("--dry-run", action="store_true", help="test everything with beeps instead of the model")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    meta_path, seg_path = out_dir / "metadata.csv", out_dir / "segments.csv"

    notes = read_texts(Path(args.texts))
    voices = read_voices(Path(args.voice_dir))
    done = tidy_previous_run(meta_path, seg_path, out_dir)
    todo = [(pos, note) for pos, note in enumerate(notes) if note["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(notes)} notes in {args.texts}, {len(done)} already done, generating {len(todo)} "
          f"with {len(voices)} voice clip(s).")
    tts_model = "a dry-run stand-in (beeps, not speech)" if args.dry_run else f"OmniVoice ({args.model})"
    if not todo:
        write_readme(out_dir, args, tts_model)
        return

    if args.dry_run:
        tts = FakeTTS()
    else:
        tts = OmniVoiceTTS(args.model, pick_device(args.device), args.asr_model if args.check else None)
    prompts = [(clip.name, tts.voice(clip, transcript)) for clip, transcript in voices]

    started = time.time()
    with open(meta_path, "a", encoding="utf-8", newline="") as meta_file, \
         open(seg_path, "a", encoding="utf-8", newline="") as seg_file:
        meta_writer = csv.DictWriter(meta_file, fieldnames=META_FIELDS)
        seg_writer = csv.DictWriter(seg_file, fieldnames=SEG_FIELDS)
        try:
            for k, (pos, note) in enumerate(todo, 1):
                ref_clip, voice = prompts[pos % len(prompts)]
                chunks, audio, spans, cers = make_note(tts, note["urdu_text"], voice, args.seed + 1000 * pos, args)
                file_name = f"audio/{note['id']}.wav"
                sf.write(str(out_dir / file_name), audio, args.sample_rate, subtype="PCM_16")
                duration = len(audio) / args.sample_rate
                cer = weighted_cer(chunks, cers) if args.check else None

                flags = []
                if args.check and any(c > args.max_cer for c in cers):
                    flags.append("check_audio")
                if duration < args.min_sec:
                    flags.append("too_short")
                elif duration > args.max_sec:
                    flags.append("too_long")

                for s, (chunk, (start, end), c) in enumerate(zip(chunks, spans, cers), 1):
                    seg_writer.writerow({"file_name": file_name, "segment": s,
                                         "start_sec": f"{start:.3f}", "end_sec": f"{end:.3f}",
                                         "urdu_text": chunk, "asr_cer": "" if c is None else f"{c:.3f}"})
                meta_writer.writerow({
                    "file_name": file_name, "id": note["id"], "urdu_text": note["urdu_text"],
                    "roman_urdu": note["roman_urdu"], "speaker_id": args.speaker_id, "ref_clip": ref_clip,
                    "duration_sec": f"{duration:.2f}", "sample_rate": args.sample_rate,
                    "asr_cer": "" if cer is None else f"{cer:.3f}", "flags": ";".join(flags),
                    "synthetic": "yes", "tts_model": tts_model,
                })
                seg_file.flush()
                meta_file.flush()

                eta = (time.time() - started) / k * (len(todo) - k)
                print(f"[{k}/{len(todo)}] {file_name}  {duration:5.1f}s  {len(chunks)} pieces"
                      + (f"  CER {cer:.2f}" if cer is not None else "")
                      + (f"  FLAG: {', '.join(flags)}" if flags else "")
                      + f"  (about {hms(eta)} left)")
        except KeyboardInterrupt:
            print("\nStopped. Everything finished so far is saved; run the same command again to continue.")

    rows = read_csv(meta_path)
    total = sum(float(r["duration_sec"]) for r in rows)
    flagged = [r["file_name"] for r in rows if r.get("flags")]
    print(f"\nDataset: {len(rows)} clips, {hms(total)} of audio in {out_dir}/")
    if flagged:
        print(f"{len(flagged)} clip(s) flagged (see the 'flags' column in metadata.csv). Listen to them. "
              "To redo a clip, delete its WAV and re-run with a different --seed. For too_short or "
              "too_long, edit that row's text or use --speed instead.")
    write_readme(out_dir, args, tts_model)


if __name__ == "__main__":
    main()
