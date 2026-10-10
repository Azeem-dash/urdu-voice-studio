#!/usr/bin/env python3
"""
Urdu Voice Studio: upload a clip of your voice, get 40-45 second Urdu voice notes
in that voice, one topic per note, with Urdu text and Roman Urdu.

    ./venv/bin/python app.py          then open http://127.0.0.1:7860

Texts are written by DeepSeek (key read from the .env file next to this script,
DEEPSEEK_API_KEY=...). Without a key the built-in stories are used instead.
Speech is made locally by OmniVoice.
"""

import csv
import json
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
WORDS_PER_NOTE = 95          # about 42 s of natural Urdu speech
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


def save_voice(audio_path: str) -> str:
    """Store the uploaded clip as the voice to clone; returns a one-line description."""
    wav, txt = voice_files()
    seconds = to_wav_24k(audio_path, wav)
    transcript = transcribe_reference(wav)
    txt.write_text(transcript + "\n", encoding="utf-8")
    heard = f' I heard: "{transcript}"' if transcript else " (no clear words, used as a voice sample only)"
    return f"Voice saved ({seconds:.1f} s).{heard}"


def current_voice_status() -> str:
    wav, txt = voice_files()
    if not wav.exists():
        return "No voice yet. Upload or record a clip of yourself above."
    heard = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    return f"Current voice: {sf.info(str(wav)).duration:.1f} s clip" + (f', heard: "{heard}"' if heard else "") + "."


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


def write_notes(topics: list, count: int, gender: str, words: int = WORDS_PER_NOTE) -> tuple:
    """Return (rows, message). One note per topic; topics are invented when fewer are given."""
    client = deepseek_client()
    if client is None:
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
        prompt += "Use these topics, one note per topic, in this order:\n" + "\n".join(f"- {t}" for t in given) + "\n"
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
    if not rows:
        raise RuntimeError("DeepSeek returned no notes.")
    for i, t in enumerate(given[:len(rows)]):
        if not rows[i]["topic"]:
            rows[i]["topic"] = t
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


def unique_id(base: str, taken: set) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "_", base.lower()).strip("_")[:40] or "note"
    cand, n = base, 2
    while cand in taken:
        cand, n = f"{base}_{n}", n + 1
    taken.add(cand)
    return cand


def run(audio_path, topics_text, count, gender, min_sec, max_sec, check, seed, batch_name, progress=gr.Progress()):
    """The whole flow: save voice -> write texts -> generate clips -> dataset files."""
    count = int(count or 1)
    min_sec, max_sec = float(min_sec), float(max_sec)
    if max_sec <= min_sec:
        raise gr.Error("Max seconds must be larger than min seconds.")
    target, pause, sample_rate = (min_sec + max_sec) / 2, 0.25, 16000
    wav, txt = voice_files()
    if not audio_path and not wav.exists():
        raise gr.Error("Upload or record a clip of your voice first.")
    log = []

    def show(extra=""):
        return "\n".join(log) + ("\n" + extra if extra else "")

    log.append("Loading the models (about a minute the first time) ...")
    yield show(), [], gr.update(), None
    progress(0, desc=log[-1])
    get_tts(need_asr=True)
    if audio_path:
        log[-1] = save_voice(audio_path)
    else:
        log[-1] = current_voice_status()
    yield show("Writing the texts ..."), [], gr.update(), None

    topics = [t.strip(" -•\t") for t in (topics_text or "").splitlines() if t.strip(" -•\t")]
    count = max(count, len(topics))
    try:
        rows, msg = write_notes(topics, count, gender)
    except RuntimeError as e:
        rows = [{"topic": s["title"], "urdu_text": s["urdu_text"], "roman_urdu": s["roman_urdu"]}
                for s in (STORIES * (count // len(STORIES) + 1))[:count]]
        msg = f"{e} Built-in stories were used instead."
    log.append(msg)
    yield show("Generating speech ..."), [], gr.update(), None

    tts = get_tts(need_asr=check)
    voice = tts.voice(wav, txt.read_text(encoding="utf-8").strip() if txt.exists() else None)
    batch_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", (batch_name or "").strip()).strip("_") or time.strftime("notes_%Y%m%d_%H%M")
    out_dir = DATASETS_DIR / batch_name
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    meta_path, seg_path = out_dir / "metadata.csv", out_dir / "segments.csv"
    taken = {r["id"] for r in csv.DictReader(open(meta_path, encoding="utf-8"))} if meta_path.exists() else set()
    clips, started = [], time.time()
    with open(meta_path, "a", encoding="utf-8", newline="") as mf, open(seg_path, "a", encoding="utf-8", newline="") as sgf:
        mw, sw = csv.DictWriter(mf, fieldnames=META_FIELDS_APP), csv.DictWriter(sgf, fieldnames=SEG_FIELDS)
        if mf.tell() == 0:
            mw.writeheader()
            sw.writeheader()
        for k, note in enumerate(rows, 1):
            note_id = unique_id(note["topic"] or f"note_{k}", taken)
            progress((k - 1) / len(rows), desc=f"Note {k}/{len(rows)}: {note['topic']}")
            chunks, audio, spans, cers = make_note(tts, note["urdu_text"], voice, int(seed) + 1000 * k, target,
                                                   pause, check, 0.35, 2, sample_rate)
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
            mw.writerow({"file_name": file_name, "id": note_id, "urdu_text": note["urdu_text"],
                         "roman_urdu": note["roman_urdu"], "speaker_id": VOICE_NAME, "ref_clip": wav.name,
                         "duration_sec": f"{duration:.2f}", "sample_rate": sample_rate,
                         "asr_cer": "" if cer is None else f"{cer:.3f}", "flags": ";".join(flags),
                         "synthetic": "yes", "tts_model": f"OmniVoice ({TTS_MODEL})", "topic": note["topic"]})
            mf.flush(), sgf.flush()
            clips.append((note["topic"] or note_id, str(out_dir / file_name), duration, cer, flags,
                          note["urdu_text"], note["roman_urdu"]))
            eta = (time.time() - started) / k * (len(rows) - k)
            log.append(f"{k}/{len(rows)} {note['topic']}: {duration:.1f} s" + (f", CER {cer:.2f}" if cer is not None else "")
                       + (f", FLAG {','.join(flags)}" if flags else ""))
            yield show(f"About {int(eta // 60)} min {int(eta % 60)} s left ..." if k < len(rows) else ""), clips, gr.update(), None

    args = SimpleNamespace(asr_model=ASR_MODEL, max_cer=0.35, retries=2, check=check, dry_run=False,
                           sample_rate=sample_rate, texts="DeepSeek-written topics", pause=pause)
    write_readme(out_dir, args, f"OmniVoice ({TTS_MODEL})")
    zip_path = DATASETS_DIR / f"{batch_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out_dir))
    meta = pd.read_csv(meta_path).fillna("")
    log.append(f"Done. {len(meta)} clip(s) in datasets/{batch_name}/ ({meta['duration_sec'].sum() / 60:.1f} min of audio).")
    yield show(), clips, meta[["topic", "duration_sec", "urdu_text", "roman_urdu", "asr_cer", "flags"]], str(zip_path)


# ------------------------------------------------------------------ UI

CSS = """
.urdu textarea, .urdu table { font-size: 1.1rem; }
#status { font-family: ui-monospace, monospace; white-space: pre-wrap; }
"""

with gr.Blocks(title="Urdu Voice Studio", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Urdu Voice Studio\nUpload a clip of your voice, press Generate, and get 40-45 second Urdu "
                "voice notes in your voice, one topic per note, with Urdu text and Roman Urdu.")
    with gr.Row():
        with gr.Column(scale=1):
            rec = gr.Audio(sources=["upload", "microphone"], type="filepath",
                           label="1. Your voice: upload or record 5-15 seconds (any language; WhatsApp .ogg works)")
            voice_status = gr.Markdown(current_voice_status())
        with gr.Column(scale=1):
            topics_box = gr.Textbox(label="2. Topics, one per line (optional: leave empty for random topics)", lines=5,
                                    placeholder="my first day at a new job\na rainy evening in Lahore\n...")
            with gr.Row():
                count = gr.Number(label="Number of voice notes", value=3, precision=0, minimum=1, maximum=100)
                gender = gr.Radio(["Male voice (gaya tha)", "Female voice (gayi thi)"], value="Male voice (gaya tha)",
                                  label="Speaker")
    with gr.Accordion("Advanced", open=False):
        with gr.Row():
            min_sec = gr.Number(label="Min seconds", value=40)
            max_sec = gr.Number(label="Max seconds", value=45)
            seed = gr.Number(label="Seed", value=1234, precision=0)
            batch_name = gr.Textbox(label="Dataset name", value="urdu_notes")
        check = gr.Checkbox(label="Check every piece with Whisper and regenerate garbled ones (about 30% slower)", value=False)
    gen_btn = gr.Button("3. Generate voice notes", variant="primary", size="lg")
    status = gr.Markdown(elem_id="status")
    clips_state = gr.State([])

    @gr.render(inputs=clips_state)
    def show_clips(clips):
        for topic, path, dur, cer, flags, urdu, roman in clips or []:
            label = f"{topic}  -  {dur:.1f} s" + (f"  -  CER {cer:.2f}" if cer is not None else "") + (f"  -  {', '.join(flags)}" if flags else "")
            with gr.Group():
                gr.Audio(path, label=label)
                gr.Markdown(f"<div dir='rtl' style='font-size:1.1rem'>{urdu}</div>\n\n{roman}")

    result_table = gr.Dataframe(label="metadata.csv", wrap=True, interactive=False, elem_classes="urdu")
    zip_out = gr.File(label="Download the dataset (zip with audio, metadata.csv, segments.csv, README.md)")
    gen_btn.click(run, [rec, topics_box, count, gender, min_sec, max_sec, check, seed, batch_name],
                  [status, clips_state, result_table, zip_out])
    gr.Markdown("Synthetic speech made locally with OmniVoice (weights CC-BY-NC 4.0, non-commercial use). "
                "Datasets are saved in the `datasets/` folder next to this app; generation resumes and adds to a dataset of the same name.")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), inbrowser=False)
