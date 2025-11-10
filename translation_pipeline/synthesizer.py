"""
Text-to-speech utilities based on the Coqui XTTS voice cloning model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:  # torch is optional but recommended
    torch = None

from TTS.api import TTS


class VoiceCloningSynthesizer:
    """
    Wraps the Coqui ``xtts_v2`` multilingual voice cloning model.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or "auto"
        gpu_enabled = self._should_use_gpu(self.device)
        self._tts = TTS(model_name=self.model_name, gpu=gpu_enabled)

    def synthesize(
        self,
        text: str,
        language: str,
        speaker_reference: Path | str,
        output_path: Path | str,
    ) -> Path:
        """
        Render ``text`` in ``language`` cloning the voice from ``speaker_reference``.
        """

        speaker_reference = Path(speaker_reference)
        output_path = Path(output_path)

        if not speaker_reference.exists():
            raise FileNotFoundError(
                f"Speaker reference audio not found: {speaker_reference}"
            )

        if not text.strip():
            raise ValueError("Cannot synthesize empty text payload.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._tts.tts_to_file(
            text=text,
            speaker_wav=str(speaker_reference),
            language=language,
            file_path=str(output_path),
        )
        return output_path

    @staticmethod
    def _should_use_gpu(device: str) -> bool:
        """
        Determine whether to enable GPU execution for the XTTS model.
        """

        if device == "cuda":
            return True

        if device in ("auto", None) and torch is not None:
            return torch.cuda.is_available()

        return False
