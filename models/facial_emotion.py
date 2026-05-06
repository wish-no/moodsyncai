"""
models/facial_emotion.py
CNN-based facial emotion recognition using DeepFace (VGG-Face backbone).
"""

from deepface import DeepFace
import numpy as np
from PIL import Image

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def analyse_facial_emotion(image_input) -> dict:
    try:
        if isinstance(image_input, Image.Image):
            img_array = np.array(image_input.convert("RGB"))
        else:
            img_array = np.array(Image.open(image_input).convert("RGB"))

        result = DeepFace.analyze(
            img_path=img_array,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        face_data  = result[0] if isinstance(result, list) else result
        raw_scores = face_data["emotion"]
        total      = sum(raw_scores.values()) or 1.0
        normalised = {k: round(v / total, 4) for k, v in raw_scores.items()}
        dominant   = face_data["dominant_emotion"]
        confidence = normalised.get(dominant, 0.0)

        return {
            "dominant_emotion": dominant,
            "confidence":       confidence,
            "all_scores":       normalised,
            "error":            None,
        }

    except Exception as exc:
        fallback = {em: round(1.0 / 7, 4) for em in EMOTION_LABELS}
        return {
            "dominant_emotion": "neutral",
            "confidence":       round(1.0 / 7, 4),
            "all_scores":       fallback,
            "error":            str(exc),
        }