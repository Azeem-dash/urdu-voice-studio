# Urdu Voice Studio

Upload a clip of your voice and get 45-50 second Urdu voice notes in that voice, one topic per
note, each with Urdu text and Roman Urdu. Speech is made on this Mac by
[OmniVoice](https://github.com/k2-fsa/OmniVoice); the texts are written by DeepSeek.

## Start it

```bash
./venv/bin/python app.py
```

Then open http://127.0.0.1:7860.

## Use it

1. **Your voice.** Upload or record yourself talking. You can add up to five recordings of the
   same person using **Additional recordings of the same person**, with or without the microphone clip.
   A short portion of every clip contributes to a combined reference capped at 20 seconds. WhatsApp voice notes
   (.ogg), m4a, mp3 and wav all work. Nothing to type: the app listens to the clip itself.
   The last uploaded voice is remembered, so next time you can skip this step.
2. **Topics and counts.** Paste several topics, one per line. The table below lists each topic
   with its own editable voice-note count. For example, request 3 cricket notes and 2 travel notes
   to generate 5 notes in total. The default count applies to every newly entered topic; changing
   the pasted topics or default count resets the table. You can also edit topics directly in the table.
   Leave topics empty for random notes. Batches support up to 100 notes in total.
   Custom topics require a working DeepSeek key; errors are shown instead of substituting unrelated stories.
3. **Generate.** One button. Each note is written on its single topic (about 106 Urdu words),
   spoken in your voice, and fitted to 45-50 seconds. Expect roughly 1.5-2 minutes per note on an
   Apple Silicon Mac, plus a minute of model loading the first time.

Results appear on the page with a player, the Urdu text and the Roman Urdu for every clip,
and a **Download generated notes (ZIP)** button. The button becomes available after the first
completed clip and updates throughout generation. Each audio player also has its own download control.
Use **Previously saved datasets** to download an earlier dataset immediately after opening the app.
Each generation gets a separate folder and ZIP containing only its own notes. Each note has a
downloadable one-row CSV under `metadata/`; the batch `metadata.csv` summarizes only that run.

## Output

`datasets/<name>/`:

- `audio/*.wav` - 16 kHz mono clips, named after their topic
- `metadata.csv` - file name, id, topic, Urdu text, Roman Urdu, duration, flags
- `segments.csv` - start and end time of each sentence group inside every clip
- `README.md` - a description of the dataset
- `datasets/<name>.zip` - everything above

Running again creates a new timestamped folder, even with the same dataset name. Previous runs
remain available in the saved-dataset selector, but never appear in the current output.

Audio must be longer than 45 seconds and at most 50 seconds. Each note gets its own target
length, spread across that range for varied durations. Small timing deviations are corrected without changing pitch;
large deviations trigger up to two retries. A note that still cannot meet the range reports an error
instead of publishing an out-of-range clip.

## Settings

- The DeepSeek key lives in `.env` next to the app (`DEEPSEEK_API_KEY=...`). Without it, the
  six built-in stories in `stories.py` are used.
- **Advanced** on the page: length range, seed, dataset name, and an optional Whisper check that
  transcribes every piece and regenerates garbled ones (slower).
- `make_urdu_voice_dataset.py` is a command-line version of the same pipeline for a CSV of texts.

OmniVoice's weights are licensed CC-BY-NC 4.0, so the audio is for non-commercial use, and every
dataset README says the speech is synthetic.
