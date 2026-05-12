"""
models/audio_transcription.py
Audio transcription using OpenAI Whisper.
Uses scipy for audio processing — no ffmpeg required.
"""

import whisper
import numpy as np
import tempfile
import os
import scipy.io.wavfile as wavfile

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(audio_input) -> dict:
    if audio_input is None:
        return {
            "transcript": "",
            "language":   "unknown",
            "confidence": 0.0,
            "error":      "No audio provided",
        }

    try:
        model = _get_model()

        if isinstance(audio_input, tuple):
            sample_rate, audio_array = audio_input

            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)

            # Use a fixed path in current directory instead of temp
            tmp_path = os.path.join(os.getcwd(), "temp_audio.wav")
            wavfile.write(tmp_path, sample_rate, audio_array)

            # Confirm file exists before transcribing
            if not os.path.exists(tmp_path):
                return {
                    "transcript": "",
                    "language":   "unknown",
                    "confidence": 0.0,
                    "error":      f"WAV file was not created at {tmp_path}",
                }

        else:
            tmp_path = str(audio_input)

        result = model.transcribe(tmp_path, fp16=False)

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        transcript = result.get("text", "").strip()
        language   = result.get("language", "unknown")

        segments = result.get("segments", [])
        if segments:
            avg_logprob = np.mean([seg.get("avg_logprob", -1.0) for seg in segments])
            confidence  = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
        else:
            confidence = 0.5

        return {
            "transcript": transcript,
            "language":   language,
            "confidence": round(confidence, 4),
            "error":      None if transcript else "No speech detected",
        }

    except Exception as exc:
        import traceback
        return {
            "transcript": "",
            "language":   "unknown",
            "confidence": 0.0,
            "error":      f"{str(exc)} | {traceback.format_exc()}",
        }