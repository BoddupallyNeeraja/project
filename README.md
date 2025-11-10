# Multilingual Voice Cloning Translator

End-to-end Python project that turns a single speech recording into translated, voice-cloned audio in multiple languages. The pipeline combines Whisper ASR, DeepL or Google Cloud Translation, and Coqui XTTS voice cloning to preserve the original speaker identity across languages.

## Features
- Transcribe any supported language with `faster-whisper`.
- Translate transcripts into any set of ISO language codes via DeepL or Google Translate APIs.
- Re-synthesize each translation with Coqui `xtts_v2`, cloning the original speaker voice.
- Persist translated audio tracks (`output_<lang>.wav`) and a `translations.json` manifest.

## Prerequisites
- Python 3.9 or later
- FFmpeg installed and on your `PATH` (required by Whisper and Coqui audio loaders)
- GPU is optional but recommended for faster inference

## Installation
```bash
git clone <this-repo>
cd <this-repo>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the Whisper and XTTS model weights automatically and stores them in the local cache (`~/.cache`). Expect a few gigabytes of model data.

## Configure Translation API Keys
Choose either DeepL or Google Cloud Translation and export the matching environment variables before running the pipeline. Replace `YOUR_API_KEY_HERE` with a real key.

### Option 1 — DeepL
```bash
export DEEPL_API_KEY=YOUR_API_KEY_HERE
# Optional: override the API endpoint (defaults to https://api-free.deepl.com/v2/translate)
# export DEEPL_API_URL=https://api.deepl.com/v2/translate
```

### Option 2 — Google Cloud Translation
```bash
export GOOGLE_TRANSLATE_API_KEY=YOUR_API_KEY_HERE
# Optional: override the API endpoint
# export GOOGLE_TRANSLATE_API_URL=https://translation.googleapis.com/language/translate/v2
```

You may alternatively pass the API key via the CLI arguments `--translator-provider` and `--translator-api-key` to avoid exporting environment variables.

## Usage
```bash
python main.py \
  --input input.wav \
  --languages en es fr hi \
  --translator-provider deepl \
  --translator-api-key YOUR_API_KEY_HERE
```

Key arguments:
- `--input`: path to the source audio file (WAV, MP3, FLAC, etc.).
- `--languages`: space-separated ISO 639-1 codes for the target languages.
- `--output-dir`: directory for generated outputs (defaults to `./outputs`).
- `--transcription-model`: Whisper checkpoint name (`tiny`, `base`, `small`, `medium`, `large-v2`, ...).
- `--transcription-device`: target device for faster-whisper (`cpu`, `cuda`, `auto`).
- `--transcription-compute-type`: precision mode (`int8`, `int8_float16`, `float16`, `float32`).
- `--tts-model`: Coqui XTTS model identifier (defaults to `tts_models/multilingual/multi-dataset/xtts_v2`).
- `--verbose`: enable progress prints.

## Outputs
- `outputs/output_<lang>.wav`: cloned speech for each requested language (e.g., `output_en.wav`, `output_es.wav`).
- `outputs/translations.json`: manifest containing the original transcript and translations. Example excerpt:

```json
{
  "source": {
    "language": "en",
    "transcript": "Hello world!"
  },
  "translations": {
    "es": "¡Hola mundo!",
    "fr": "Bonjour le monde!"
  }
}
```

## Notes and Troubleshooting
- Ensure FFmpeg is installed; Whisper and XTTS rely on it for audio decoding.
- When running on CPU-only systems, larger Whisper checkpoints may be slow. Start with `--transcription-model tiny` for quick experiments.
- XTTS expects the target language codes it supports (e.g., `en`, `es`, `fr`, `hi`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `zh`, `ja`). Check [Coqui documentation](https://tts.readthedocs.io/) for the full list.
- For batch processing you can wrap `main.py` in your own scripting logic—each invocation handles one input file.

## License
This project relies on third-party models/APIs (OpenAI Whisper, DeepL/Google Translate, Coqui XTTS). Consult their respective licenses and terms of service before commercial usage.
