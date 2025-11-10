"""
Command-line entry point for the multilingual audio-to-audio translation demo.

Pipeline overview:
1. Transcribe the input speech with faster-whisper.
2. Translate the transcript into the requested target languages via DeepL or Google.
3. Re-synthesize each translation using Coqui XTTS voice cloning so the speaker identity
   is preserved across languages.

Example
-------
python main.py --input input.wav --languages en es fr hi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from translation_pipeline import (
    TranslationClient,
    TranslationProviderNotConfigured,
    VoiceCloningSynthesizer,
    WhisperTranscriber,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Define and parse the command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Multilingual audio-to-audio translation with voice cloning.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source audio file containing speech.",
    )
    parser.add_argument(
        "--languages",
        required=True,
        nargs="+",
        help="Target language codes to synthesize (e.g., en es fr hi).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where generated audio and translation files will be saved.",
    )
    parser.add_argument(
        "--transcription-model",
        default="small",
        help="Whisper checkpoint identifier (tiny, base, small, medium, large-v2, ...).",
    )
    parser.add_argument(
        "--transcription-device",
        default=None,
        help="Device for faster-whisper (cpu, cuda, auto). Defaults to auto.",
    )
    parser.add_argument(
        "--transcription-compute-type",
        default="int8",
        help="Compute type for faster-whisper (int8, int8_float16, float16, float32).",
    )
    parser.add_argument(
        "--translator-provider",
        default="auto",
        choices=["auto", "deepl", "google"],
        help="Force a specific translation provider or auto-detect via environment.",
    )
    parser.add_argument(
        "--translator-api-key",
        default=None,
        help="Optional API key overriding environment variables.",
    )
    parser.add_argument(
        "--source-language",
        default=None,
        help="Optional ISO language code to hint the transcription language.",
    )
    parser.add_argument(
        "--tts-model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
        help="Coqui TTS model identifier for voice cloning.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(argv)


def configure_translation_env(provider: str, api_key: Optional[str]) -> None:
    """
    Inject API credentials into the process environment when supplied via CLI.
    """

    if not api_key or provider == "auto":
        return

    if provider == "deepl":
        os.environ.setdefault("DEEPL_API_KEY", api_key)
    elif provider == "google":
        os.environ.setdefault("GOOGLE_TRANSLATE_API_KEY", api_key)


def save_translations_json(
    output_dir: Path,
    transcript: str,
    source_language: str,
    translations: Dict[str, str],
) -> Path:
    """
    Persist the translation metadata to JSON for later inspection.
    """

    payload = {
        "source": {
            "language": source_language,
            "transcript": transcript,
        },
        "translations": translations,
    }

    json_path = output_dir / "translations.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return json_path


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entrypoint executed from the CLI.
    """

    args = parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input audio file was not found: {input_path}", file=sys.stderr)
        return 1

    configure_translation_env(args.translator_provider, args.translator_api_key)

    if args.verbose:
        print(f"[info] Loading Whisper model: {args.transcription_model}")

    transcriber = WhisperTranscriber(
        model_size=args.transcription_model,
        device=args.transcription_device,
        compute_type=args.transcription_compute_type,
    )

    transcript = transcriber.transcribe(
        audio_path=input_path,
        language=args.source_language,
    )

    if args.verbose:
        print(
            f"[info] Detected language={transcript.language}; "
            f"transcript length={len(transcript.text.split())} words",
        )

    try:
        translator = TranslationClient()
    except TranslationProviderNotConfigured as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    target_languages = [
        code.strip().lower() for code in args.languages if code.strip()
    ]
    if not target_languages:
        print("[error] No valid target language codes provided.", file=sys.stderr)
        return 1

    try:
        translations = translator.translate_batch(
            text=transcript.text,
            languages=target_languages,
            source_language=args.source_language or transcript.language,
        )
    except Exception as exc:  # noqa: BLE001 - surface the underlying translation error
        print(f"[error] Translation step failed: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"[info] Synthesizing speech with Coqui model: {args.tts_model}")

    synthesizer = VoiceCloningSynthesizer(
        model_name=args.tts_model,
        device=args.transcription_device,
    )

    generated_paths: Dict[str, str] = {}
    for lang_code, translated_text in translations.items():
        if args.verbose:
            print(f"[info] Synthesizing {lang_code} audio...")

        output_path = output_dir / f"output_{lang_code}.wav"
        try:
            synthesizer.synthesize(
                text=translated_text,
                language=lang_code,
                speaker_reference=input_path,
                output_path=output_path,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] Synthesis failed for {lang_code}: {exc}", file=sys.stderr)
            return 1
        generated_paths[lang_code] = str(output_path.resolve())

    translations_json = save_translations_json(
        output_dir=output_dir,
        transcript=transcript.text,
        source_language=transcript.language,
        translations=translations,
    )

    print("[done] Generated files:")
    for lang_code, path in generated_paths.items():
        print(f"  - {lang_code}: {path}")
    print(f"  - translations: {str(translations_json.resolve())}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
