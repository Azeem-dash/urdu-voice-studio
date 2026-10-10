# Urdu Voice Studio

Upload a clip of your voice and get 40-45 second Urdu voice notes in that voice, one topic per
note, each with Urdu text and Roman Urdu. Speech is made on this Mac by
[OmniVoice](https://github.com/k2-fsa/OmniVoice); the texts are written by DeepSeek.

## Start it

```bash
./venv/bin/python app.py
```

Then open http://127.0.0.1:7860.

## Use it

1. **Your voice.** Upload (or record) 5-15 seconds of yourself talking. WhatsApp voice notes
   (.ogg), m4a, mp3 and wav all work. Nothing to type: the app listens to the clip itself.
   The last uploaded voice is remembered, so next time you can skip this step.
2. **Topics.** Optional. One topic per line; each topic becomes one note. Leave it empty and
   DeepSeek invents everyday topics. Set how many notes you want.
3. **Generate.** One button. Each note is written on its single topic (about 95 Urdu words),
   spoken in your voice, and fitted to 40-45 seconds. Expect roughly 1.5-2 minutes per note on an
   Apple Silicon Mac, plus a minute of model loading the first time.

Results appear on the page with a player, the Urdu text and the Roman Urdu for every clip,
and a zip of the whole dataset.

## Output

`datasets/<name>/`:

- `audio/*.wav` - 16 kHz mono clips, named after their topic
- `metadata.csv` - file name, id, topic, Urdu text, Roman Urdu, duration, flags
- `segments.csv` - start and end time of each sentence group inside every clip
- `README.md` - a description of the dataset
- `datasets/<name>.zip` - everything above

Running again with the same dataset name adds new notes to it.

## Settings

- The DeepSeek key lives in `.env` next to the app (`DEEPSEEK_API_KEY=...`). Without it, the
  six built-in stories in `stories.py` are used.
- **Advanced** on the page: length range, seed, dataset name, and an optional Whisper check that
  transcribes every piece and regenerates garbled ones (slower).
- `make_urdu_voice_dataset.py` is a command-line version of the same pipeline for a CSV of texts.

OmniVoice's weights are licensed CC-BY-NC 4.0, so the audio is for non-commercial use, and every
dataset README says the speech is synthetic.
