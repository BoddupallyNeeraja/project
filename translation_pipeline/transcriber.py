"""
Audio transcription utilities powered by faster-whisper.

This module encapsulates the Whisper transcription model so the rest of the
pipeline can operate independently from the underlying ASR implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from faster_whisper import WhisperModel


@dataclass
class TranscriptSegment:
    """Lightweight representation of a single transcription segment."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    """
    Container bundling the global transcription text together with metadata.
    """

    text: str
    language: str
    segments: List[TranscriptSegment]


class WhisperTranscriber:
    """
    Convenience wrapper around the faster-whisper WhisperModel.

    Parameters
    ----------
    model_size:
        String identifier of the Whisper checkpoint to load (e.g. "base", "small").
    device:
        Target device passed to faster-whisper (defaults to "auto").
    compute_type:
        Mixed precision compute strategy. Use "int8_float16" or "float16" for GPU.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device or "auto"
        self.compute_type = compute_type

        # Lazily prepare the transcription model. Loading may take a few seconds
        # depending on the chosen checkpoint, so we surface configuration up-front.
        self._model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio_path: Path | str,
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> TranscriptResult:
        """
        Generate a text transcription for ``audio_path``.

        Parameters
        ----------
        audio_path:
            Path-like pointing at an audio file supported by Whisper.
        beam_size:
            Beam search width used during decoding.
        language:
            Optional language hint. When omitted, Whisper auto-detects.
        """

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=language,
            vad_filter=True,
        )

        segment_list = _collect_segments(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()

        return TranscriptResult(
            text=text,
            language=info.language,
            segments=segment_list,
        )


def _collect_segments(segments: Iterable) -> List[TranscriptSegment]:
    """
    Convert the generator returned by faster-whisper into dataclasses.
    """

    collected: List[TranscriptSegment] = []
    for segment in segments:
        collected.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
            )
        )
    return collected
