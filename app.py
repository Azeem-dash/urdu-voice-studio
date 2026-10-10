#!/usr/bin/env python3
"""
Urdu Voice Studio: upload a clip of your voice, get 45-50 second Urdu voice notes
in that voice, one topic per note, with Urdu text and Roman Urdu.

    ./venv/bin/python app.py          then open http://127.0.0.1:7860

Texts are written by DeepSeek (key read from the .env file next to this script,
DEEPSEEK_API_KEY=...). Without a key the built-in stories are used instead.
Speech is made locally by OmniVoice.
"""

import csv
import json
import math
import tempfile
import os
import re
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf

from make_urdu_voice_dataset import (
    META_FIELDS, SEG_FIELDS, OmniVoiceTTS, char_error_rate, clean_urdu, join_pieces,
    make_chunks, pick_device, resample, weighted_cer, write_readme,
)
from stories import STORIES

ROOT = Path(__file__).resolve().parent
VOICES_DIR = ROOT / "voices"
DATASETS_DIR = ROOT / "datasets"
VOICES_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

TTS_MODEL = "k2-fsa/OmniVoice"
ASR_MODEL = "openai/whisper-large-v3-turbo"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
VOICE_NAME = "my_voice"
WORDS_PER_NOTE = 106          # about 47.5 s of natural Urdu speech
META_FIELDS_APP = META_FIELDS + ["topic"]
URDU_WORD_WEIGHT = 5.2       # average OmniVoice duration weight of one Urdu word (measured on 640 words)


def load_env():
    """Read KEY=value lines from .env next to this script into the environment."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
_tts = None


# ------------------------------------------------------------------ engine

def get_tts(need_asr: bool) -> OmniVoiceTTS:
    """Load OmniVoice once (and Whisper on first use)."""
    global _tts
    if _tts is None:
        _tts = OmniVoiceTTS(TTS_MODEL, pick_device("auto"), None)
    if need_asr and _tts.asr is None:
        import torch
        from transformers import pipeline
        device = next(_tts.model.parameters()).device
        _tts.asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, dtype=torch.float32, device=device)
    return _tts


# ------------------------------------------------------------------ voice

def voice_files():
    return VOICES_DIR / f"{VOICE_NAME}.wav", VOICES_DIR / f"{VOICE_NAME}.txt"


def to_wav_24k(src: str, dst: Path, max_sec: float = 20.0) -> float:
    """Convert any audio file (ogg, m4a, mp3, webm, wav) to 24 kHz mono WAV, at most max_sec long."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-t", str(max_sec),
                        "-ac", "1", "-ar", "24000", str(dst)], check=True)
    else:
        import librosa
        audio, _ = librosa.load(src, sr=24000, mono=True, duration=max_sec)
        sf.write(str(dst), audio, 24000)
    return sf.info(str(dst)).duration


def transcribe_reference(wav_path: Path) -> str:
    """Listen to the reference clip with Whisper, language detected automatically."""
    tts = get_tts(need_asr=True)
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    result = tts.asr({"array": audio, "sampling_rate": sr}, generate_kwargs={"task": "transcribe"})
    return clean_urdu(result["text"])


def save_voice(audio_paths) -> str:
    """Build one bounded reference using audio from every uploaded same-speaker clip."""
    paths = [audio_paths] if isinstance(audio_paths, str) else list(audio_paths)
    if not 1 <= len(paths) <= 5:
        raise gr.Error("Upload 1–5 recordings of the same person (including any microphone recording).")
    wav, txt = voice_files()
    gap = np.zeros(4800, dtype=np.float32)
    seconds_per_clip = (20.0 - 0.2 * (len(paths) - 1)) / len(paths)
    with tempfile.TemporaryDirectory(dir=VOICES_DIR) as temporary:
        pieces = []
        for index, path in enumerate(paths):
            converted = Path(temporary) / f"reference_{index}.wav"
            to_wav_24k(str(path), converted, max_sec=seconds_per_clip)
            audio, _ = sf.read(str(converted), dtype="float32")
            if len(audio) < 24000 or np.max(np.abs(audio)) < 0.001:
                raise gr.Error(f"Recording {index + 1} needs at least one second of audible speech.")
            if pieces:
                pieces.append(gap)
            pieces.append(audio)
        combined = Path(temporary) / "combined.wav"
        sf.write(str(combined), np.concatenate(pieces), 24000, subtype="PCM_16")
        transcript = transcribe_reference(combined)
        if not transcript.strip():
            raise gr.Error("Could not understand the recordings. Please upload clearer speech.")
        text_path = Path(temporary) / "combined.txt"
        text_path.write_text(transcript + "\n", encoding="utf-8")
        seconds = sf.info(str(combined)).duration
        combined.replace(wav)
        text_path.replace(txt)
    wav.with_suffix(".json").write_text(json.dumps({"recordings": len(paths)}))
    return f"Voice saved from {len(paths)} recording(s), {seconds:.1f} s combined reference. I heard: {transcript}"


def current_voice_status() -> str:
    wav, txt = voice_files()
    if not wav.exists():
        return "No voice yet. Upload or record a clip of yourself above."
    heard = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    info_path = wav.with_suffix(".json")
    recordings = json.loads(info_path.read_text()).get("recordings", 1) if info_path.exists() else 1
    return f"Current voice: {recordings} recording(s), {sf.info(str(wav)).duration:.1f} s reference" + (f', heard: "{heard}"' if heard else "") + "."


# ------------------------------------------------------------------ texts

def deepseek_client():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL, timeout=300, max_retries=2)


def deepseek_json(client, system: str, prompt: str) -> dict:
    import openai
    shape = json.dumps({"notes": [{"topic": "...", "urdu_text": "...", "roman_urdu": "..."}]}, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL, max_tokens=8192, temperature=1.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system + f"\n\nReply with JSON only, exactly this shape:\n{shape}"},
                      {"role": "user", "content": prompt}],
        )
    except openai.AuthenticationError:
        raise RuntimeError("DeepSeek rejected the API key in .env.")
    except openai.RateLimitError:
        raise RuntimeError("DeepSeek rate limit or insufficient balance.")
    except openai.APIConnectionError:
        raise RuntimeError("Could not reach api.deepseek.com.")
    except openai.APIStatusError as e:
        raise RuntimeError(f"DeepSeek error {e.status_code}: {e.message}")
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("DeepSeek's answer was cut off; ask for fewer notes at a time.")
    try:
        data = json.loads(choice.message.content or "")
    except json.JSONDecodeError:
        raise RuntimeError("DeepSeek did not return valid JSON.")
    if isinstance(data, list):
        data = {"notes": data}
    return data


def valid_count(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise gr.Error("Enter a whole number of notes between 1 and 100.")
    if not math.isfinite(number) or not number.is_integer() or not 1 <= number <= 100:
        raise gr.Error("Enter a whole number of notes between 1 and 100.")
    return int(number)


def topic_table(topics_text, count):
    count = valid_count(count)
    return [[topic.strip(" -•\t"), count] for topic in (topics_text or "").splitlines()
            if topic.strip(" -•\t")]


def topic_plan(table, default_count):
    plan = []
    for row in table or []:
        topic = str(row[0] or "").strip()
        if topic:
            plan.append((topic, valid_count(row[1])))
    if not plan:
        plan = [("", valid_count(default_count))]
    if sum(count for _, count in plan) > 100:
        raise gr.Error("Please request at most 100 notes in total per batch.")
    return plan


def write_topic_notes(plan, gender):
    rows, messages = [], []
    for topic, count in plan:
        # Keep requests small enough for the text model's output limit.
        for offset in range(0, count, 5):
            size = min(5, count - offset)
            batch, message = write_notes([topic] * size if topic else [], size, gender)
            rows.extend(batch)
            messages.append(message)
    return rows, " ".join(messages)


def write_notes(topics: list, count: int, gender: str, words: int = WORDS_PER_NOTE) -> tuple:
    """Return exactly one note per requested topic occurrence."""
    client = deepseek_client()
    if client is None:
        if topics:
            raise RuntimeError("Add a valid DEEPSEEK_API_KEY to .env to generate notes for your topics.")
        rows = [{"topic": s["title"], "urdu_text": s["urdu_text"], "roman_urdu": s["roman_urdu"]}
                for s in (STORIES * (count // len(STORIES) + 1))[:count]]
        return rows, "No DeepSeek key in .env, so built-in stories were used."
    forms = "male" if gender.startswith("Male") else "female"
    system = (
        "You write short Urdu voice notes for a text-to-speech dataset. Every note is natural, spoken, "
        "first-person Urdu, the way a person from Pakistan actually talks into a phone, not literary prose. "
        "Each note is about exactly ONE topic from start to finish; never mix topics inside a note. "
        "Rules: Urdu script with the Urdu full stop ۔ and question mark ؟ at every sentence end; "
        f"{forms} first-person verb forms; no English letters inside the Urdu text (write loanwords in Urdu "
        "script); spell numbers out in words; no headings, labels or quotes; sentences of 8-18 words. "
        "topic is the note's topic in English. roman_urdu is the same Urdu text in everyday Roman Urdu "
        "spelling, sentence by sentence."
    )
    given = [t for t in topics if t][:count]
    prompt = f"Write {count} voice notes, each {words} words of Urdu (between {int(words * 0.9)} and {int(words * 1.1)}).\n"
    if given:
        prompt += "Use these topics, one note per listed entry, in this order. Repeated entries need distinct stories about that same topic:\n" + "\n".join(f"- {t}" for t in given) + "\n"
    if len(given) < count:
        prompt += (f"Invent {count - len(given)} more everyday topics that a Pakistani person might send a voice "
                   "note about (family, work, food, travel, weather, friends, study, health, shopping, cricket, "
                   "festivals, small daily events), all different from each other and from the topics above.\n")
    prompt += "Vary the opening, the mood and the angle so the notes do not sound alike."
    data = deepseek_json(client, system, prompt)
    rows = []
    for n in data.get("notes", []):
        if isinstance(n, dict) and n.get("urdu_text"):
            rows.append({"topic": str(n.get("topic") or "").strip(), "urdu_text": clean_urdu(n["urdu_text"]),
                         "roman_urdu": str(n.get("roman_urdu") or "").strip()})
    if len(rows) != count:
        raise RuntimeError(f"DeepSeek returned {len(rows)} notes; expected {count}. Please retry.")
    for i, topic in enumerate(given):
        rows[i]["topic"] = topic
    return rows[:count], f"DeepSeek wrote {len(rows[:count])} note(s)."


# ------------------------------------------------------------------ speech

def synth(tts, chunks, voice, seed, durations=None):
    """Generate the pieces one by one (batching garbles pieces on this model)."""
    import torch
    out = []
    for i, chunk in enumerate(chunks):
        torch.manual_seed(seed + i)
        kw = {"duration": [durations[i]]} if durations else {}
        audio = tts.model.generate(text=[chunk], language="ur", voice_clone_prompt=voice, **kw)[0]
        out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    return out


def make_note(tts, text, voice, seed, target_sec, pause, check, max_cer, retries, sample_rate):
    chunks = make_chunks(text, 160)
    est = tts.model.duration_estimator
    w = np.array([est.calculate_total_weight(c) for c in chunks], dtype=float)
    speech = max(target_sec - pause * (len(chunks) - 1), 5.0)
    durations = list(speech * w / w.sum())
    pieces = synth(tts, chunks, voice, seed, durations)
    cers = [None] * len(chunks)
    if check:
        for i, (chunk, audio) in enumerate(zip(chunks, pieces)):
            best, best_cer = audio, char_error_rate(chunk, tts.transcribe(audio))
            for attempt in range(1, retries + 1):
                if best_cer <= max_cer:
                    break
                retry = synth(tts, [chunk], voice, seed + 7919 * attempt + i, [durations[i]])[0]
                cer = char_error_rate(chunk, tts.transcribe(retry))
                if cer < best_cer:
                    best, best_cer = retry, cer
            pieces[i], cers[i] = best, best_cer
    audio, spans = join_pieces(pieces, tts.sr, pause)
    audio = np.clip(resample(audio, tts.sr, sample_rate), -1.0, 1.0)
    return chunks, audio, spans, cers


def fit_duration(audio, spans, sample_rate, min_sec=45, max_sec=50, target_sec=None):
    """Keep speech intact; correct modest timing drift without changing pitch."""
    duration = len(audio) / sample_rate
    if min_sec < duration <= max_sec and (target_sec is None or abs(duration - target_sec) < 0.1):
        return audio, spans
    target = target_sec if target_sec is not None else (min_sec + max_sec) / 2
    rate = duration / target
    if not 0.8 <= rate <= 1.2:
        raise ValueError("Generated speech is too far from 45–50 seconds; regenerating it.")
    import librosa
    adjusted = librosa.effects.time_stretch(np.asarray(audio, dtype=np.float32), rate=rate)
    if not min_sec <= len(adjusted) / sample_rate <= max_sec:
        raise ValueError("Could not fit the generated speech into 45–50 seconds.")
    scale = len(adjusted) / len(audio)
    return adjusted, [(start * scale, end * scale) for start, end in spans]


def unique_id(base: str, taken: set) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", base.lower()).strip("_")[:40] or "note"
    cand, n = base, 2
    while cand in taken:
        cand, n = f"{base}_{n}", n + 1
    taken.add(cand)
    return cand


def run(audio_path, topics, count, gender, min_sec, max_sec, check, seed, batch_name, extra_audio=None, progress=gr.Progress()):
    """The whole flow: save voice -> write texts -> generate clips -> dataset files."""
    plan = topic_plan(topics, count)
    min_sec, max_sec = float(min_sec), float(max_sec)
    if not (45 <= min_sec < max_sec <= 50):
        raise gr.Error("Choose a duration range within 45–50 seconds.")
    pause, sample_rate = 0.25, 16000
    recordings = ([audio_path] if audio_path else []) + list(extra_audio or [])
    if len(recordings) > 5:
        raise gr.Error("Choose at most 5 recordings of the same person in total.")
    wav, txt = voice_files()
    if not recordings and not wav.exists():
        raise gr.Error("Upload or record a clip of your voice first.")
    log = []

    def show(extra=""):
        return "\n".join(log) + ("\n" + extra if extra else "")

    log.append("Loading the models (about a minute the first time) ...")
    yield show(), [], pd.DataFrame(), gr.update(value=None, interactive=False)
    progress(0, desc=log[-1])
    get_tts(need_asr=True)
    if recordings:
        log[-1] = save_voice(recordings)
    else:
        log[-1] = current_voice_status()
    yield show("Writing the texts ..."), [], pd.DataFrame(), gr.update(value=None, interactive=False)

    try:
        rows, msg = write_topic_notes(plan, gender)
    except RuntimeError as e:
        raise gr.Error(str(e)) from e
    log.append(msg)
    yield show("Generating speech ..."), [], pd.DataFrame(), gr.update(value=None, interactive=False)

    tts = get_tts(need_asr=check)
    voice = tts.voice(wav, txt.read_text(encoding="utf-8").strip() if txt.exists() else None)
    batch_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (batch_name or "").strip()).strip("_") or time.strftime("notes_%Y%m%d_%H%M")
    out_dir = Path(tempfile.mkdtemp(prefix=f"{batch_name}_{time.strftime('%Y%m%d_%H%M%S')}_", dir=DATASETS_DIR))
    batch_name = out_dir.name
    (out_dir / "audio").mkdir()
    (out_dir / "metadata").mkdir()
    meta_path, seg_path = out_dir / "metadata.csv", out_dir / "segments.csv"
    taken = set()
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as existing:
            taken = {row["id"] for row in csv.DictReader(existing)}
    args = SimpleNamespace(asr_model=ASR_MODEL, max_cer=0.35, retries=2, check=check, dry_run=False,
                           sample_rate=sample_rate, texts="DeepSeek-written topics", pause=pause)
    clips, started = [], time.time()
    duration_rng = np.random.default_rng(int(seed))
    # Stratify targets across the range so a batch has meaningfully varied lengths.
    targets = min_sec + (np.arange(len(rows)) + duration_rng.uniform(0.15, 0.85, len(rows))) / len(rows) * (max_sec - min_sec)
    duration_rng.shuffle(targets)
    with open(meta_path, "a", encoding="utf-8", newline="") as mf, open(seg_path, "a", encoding="utf-8", newline="") as sgf:
        mw, sw = csv.DictWriter(mf, fieldnames=META_FIELDS_APP), csv.DictWriter(sgf, fieldnames=SEG_FIELDS)
        if mf.tell() == 0:
            mw.writeheader()
            sw.writeheader()
        for k, note in enumerate(rows, 1):
            target = float(targets[k - 1])
            note_id = unique_id(note["topic"] or f"note_{k}", taken)
            progress((k - 1) / len(rows), desc=f"Note {k}/{len(rows)}: {note['topic']}")
            for attempt in range(3):
                chunks, audio, spans, cers = make_note(tts, note["urdu_text"], voice,
                    int(seed) + 1000 * k + 100003 * attempt, target, pause, check, 0.35, 2, sample_rate)
                try:
                    audio, spans = fit_duration(audio, spans, sample_rate, min_sec, max_sec, target_sec=target)
                    break
                except ValueError as error:
                    if attempt == 2:
                        raise gr.Error(f"Could not generate a 45–50 second note for {note['topic']}. Please retry.") from error
            file_name = f"audio/{note_id}.wav"
            sf.write(str(out_dir / file_name), audio, sample_rate, subtype="PCM_16")
            duration = len(audio) / sample_rate
            cer = weighted_cer(chunks, cers) if check else None
            flags = []
            if check and any(c > 0.35 for c in cers):
                flags.append("check_audio")
            if duration < min_sec:
                flags.append("too_short")
            elif duration > max_sec:
                flags.append("too_long")
            for s, (chunk, (st, en), c) in enumerate(zip(chunks, spans, cers), 1):
                sw.writerow({"file_name": file_name, "segment": s, "start_sec": f"{st:.3f}", "end_sec": f"{en:.3f}",
                             "urdu_text": chunk, "asr_cer": "" if c is None else f"{c:.3f}"})
            metadata_row = {"file_name": file_name, "id": note_id, "urdu_text": note["urdu_text"],
                         "roman_urdu": note["roman_urdu"], "speaker_id": VOICE_NAME, "ref_clip": wav.name,
                         "duration_sec": f"{duration:.2f}", "sample_rate": sample_rate,
                         "asr_cer": "" if cer is None else f"{cer:.3f}", "flags": ";".join(flags),
                         "synthetic": "yes", "tts_model": f"OmniVoice ({TTS_MODEL})", "topic": note["topic"]}
            mw.writerow(metadata_row)
            csv_path = out_dir / "metadata" / f"{note_id}.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as note_file:
                writer = csv.DictWriter(note_file, fieldnames=META_FIELDS_APP)
                writer.writeheader()
                writer.writerow(metadata_row)
            mf.flush(), sgf.flush()
            write_readme(out_dir, args, f"OmniVoice ({TTS_MODEL})")
            clips.append((note["topic"] or note_id, str(out_dir / file_name), duration, cer, flags,
                          note["urdu_text"], note["roman_urdu"], str(csv_path)))
            eta = (time.time() - started) / k * (len(rows) - k)
            log.append(f"{k}/{len(rows)} {note['topic']}: {duration:.1f} s" + (f", CER {cer:.2f}" if cer is not None else "")
                       + (f", FLAG {','.join(flags)}" if flags else ""))
            yield show(f"About {int(eta // 60)} min {int(eta % 60)} s left ..." if k < len(rows) else ""), clips, pd.read_csv(meta_path).fillna("")[["topic", "duration_sec", "urdu_text", "roman_urdu", "asr_cer", "flags"]], gr.update(value=archive_dataset(batch_name), interactive=True)

    zip_path = archive_dataset(batch_name)
    meta = pd.read_csv(meta_path).fillna("")
    log.append(f"Done. {len(meta)} clip(s) in datasets/{batch_name}/ ({meta['duration_sec'].sum() / 60:.1f} min of audio).")
    yield show(), clips, meta[["topic", "duration_sec", "urdu_text", "roman_urdu", "asr_cer", "flags"]], gr.update(value=str(zip_path), interactive=True)


def archive_dataset(name):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise gr.Error("Select a saved dataset first.")
    out_dir = DATASETS_DIR / name
    if not (out_dir / "metadata.csv").is_file():
        raise gr.Error("This dataset has no generated notes yet.")
    zip_path = DATASETS_DIR / f"{name}.zip"
    # Replace atomically so a download never receives a partly written archive.
    with tempfile.NamedTemporaryFile(dir=DATASETS_DIR, suffix=".zip", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out_dir))
        temp_path.replace(zip_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return str(zip_path)


def saved_datasets():
    return sorted((p.name for p in DATASETS_DIR.iterdir()
                   if p.is_dir() and (p / "metadata.csv").is_file()),
                  key=lambda name: (DATASETS_DIR / name / "metadata.csv").stat().st_mtime,
                  reverse=True)


def refresh_datasets():
    names = saved_datasets()
    selected = names[0] if names else None
    return gr.update(choices=names, value=selected), download_saved(selected)


def download_saved(name):
    if not name:
        return gr.update(value=None, interactive=False)
    return gr.update(value=archive_dataset(name), interactive=True)


# ------------------------------------------------------------------ UI

CSS = """
.urdu textarea, .urdu table { font-size: 1.1rem; }
#status { font-family: ui-monospace, monospace; white-space: pre-wrap; }
"""

with gr.Blocks(title="Urdu Voice Studio", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Urdu Voice Studio\nUpload a clip of your voice, press Generate, and get 45-50 second Urdu "
                "voice notes with varied lengths above 45 seconds, one topic per note, with Urdu text and Roman Urdu.")
    with gr.Row():
        with gr.Column(scale=1):
            rec = gr.Audio(sources=["upload", "microphone"], type="filepath",
                           label="1. Your voice: upload or record 5-15 seconds (any language; WhatsApp .ogg works)")
            uploads = gr.File(label="Additional recordings of the same person", file_count="multiple",
                              type="filepath", file_types=["audio"])
            gr.Markdown("Upload up to **5 recordings in total** of the **same speaker**. "
                        "You can use the microphone above, upload several files here, or both. "
                        "Use clear speech with different sentences and natural delivery. "
                        "We take a short portion from each recording to build a reference of up to 20 seconds.")
            voice_status = gr.Markdown(current_voice_status())
        with gr.Column(scale=1):
            topics_box = gr.Textbox(label="2. Paste multiple topics — one topic on each new line", lines=5,
                                    placeholder="Cricket\nTravel\nFood\nStudies\nWeather")
            gr.Markdown("Each pasted topic appears in the table below. Edit **Voice notes** beside each topic "
                        "to choose its count. For example: **Cricket → 3 notes, Travel → 2 notes = 5 notes**.")
            topics_grid = gr.Dataframe(headers=["Topic", "Voice notes"], datatype=["str", "number"],
                                       type="array", value=[], col_count=(2, "fixed"),
                                       label="Voice notes to generate for each topic", interactive=True, wrap=True)
            with gr.Row():
                count = gr.Number(label="Default count for every topic", value=3, precision=0, minimum=1, maximum=100)
                gender = gr.Radio(["Male voice (gaya tha)", "Female voice (gayi thi)"], value="Male voice (gaya tha)",
                                  label="Speaker")
            gr.Markdown("Changing the pasted topics or default count resets the table. "
                        "If you leave all topics empty, the default count generates random-topic notes.")
            topics_box.input(topic_table, [topics_box, count], topics_grid, queue=False)
            count.input(topic_table, [topics_box, count], topics_grid, queue=False)
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            min_sec = gr.Number(label="Min seconds", value=45, minimum=45, maximum=49)
            max_sec = gr.Number(label="Max seconds", value=50, minimum=46, maximum=50)
            seed = gr.Number(label="Seed", value=1234, precision=0)
            batch_name = gr.Textbox(label="Dataset name", value="urdu_notes")
        check = gr.Checkbox(label="Check every piece with Whisper and regenerate garbled ones (about 30% slower)", value=False)
    gen_btn = gr.Button("3. Generate voice notes", variant="primary", size="lg")
    status = gr.Markdown(elem_id="status")
    clips_state = gr.State([])

    @gr.render(inputs=clips_state)
    def show_clips(clips):
        for topic, path, dur, cer, flags, urdu, roman, csv_path in clips or []:
            label = f"{topic}  -  {dur:.1f} s" + (f"  -  CER {cer:.2f}" if cer is not None else "") + (f"  -  {', '.join(flags)}" if flags else "")
            with gr.Group():
                gr.Audio(path, label=label, interactive=False, show_download_button=True)
                gr.DownloadButton("Download this note’s metadata CSV", value=csv_path)
                gr.Markdown(f"<div dir='rtl' style='font-size:1.1rem'>{urdu}</div>\n\n{roman}")

    result_table = gr.Dataframe(label="Current generation only — metadata", wrap=True, interactive=False, elem_classes="urdu")
    zip_out = gr.DownloadButton("Download generated notes (ZIP)", interactive=False, variant="primary")
    generation = gen_btn.click(run, [rec, topics_grid, count, gender, min_sec, max_sec, check, seed, batch_name, uploads],
                              [status, clips_state, result_table, zip_out], concurrency_limit=1,
                              concurrency_id="generation")
    gr.Markdown("The ZIP becomes available after the first note and updates as more notes finish. "
                "It includes only this generation, with a separate metadata CSV for each note. You can also download each clip from its player.")
    with gr.Row():
        saved = gr.Dropdown(label="Previously saved datasets", choices=saved_datasets())
        refresh = gr.Button("Refresh saved datasets")
        saved_download = gr.DownloadButton("Download saved dataset (ZIP)", interactive=False)
    saved.input(download_saved, saved, saved_download, queue=False)
    refresh.click(refresh_datasets, outputs=[saved, saved_download], queue=False)
    generation.then(refresh_datasets, outputs=[saved, saved_download], queue=False)
    generation.then(current_voice_status, outputs=voice_status, queue=False)
    demo.load(refresh_datasets, outputs=[saved, saved_download], queue=False)
    gr.Markdown("Synthetic speech made locally with OmniVoice (weights CC-BY-NC 4.0, non-commercial use). "
                "Datasets are saved in the `datasets/` folder next to this app; each generation has its own folder, so previous notes stay separate.")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), inbrowser=False)
