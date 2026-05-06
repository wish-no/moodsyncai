"""
models/generator.py
Generative summary using Flan-T5 (encoder-decoder transformer).
Lecture: DA-3-DeepLearning_TR.pdf — encoder-decoder transformers (slide 2)
"""

from transformers import pipeline

_gen_pipeline = None


def _get_pipeline():
    global _gen_pipeline
    if _gen_pipeline is None:
        _gen_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=120,
        )
    return _gen_pipeline


def _build_prompt(facial_result, text_result, fusion_result):
    face_emo  = facial_result.get("dominant_emotion", "neutral")
    face_pct  = int(facial_result.get("confidence", 0) * 100)
    text_sent = text_result.get("dominant_sentiment", "neutral")
    text_pct  = int(text_result.get("confidence", 0) * 100)
    mismatch  = fusion_result.get("mismatch", False)
    severity  = fusion_result.get("mismatch_severity", "none")

    if mismatch and severity == "strong":
        return (
            f"A person's face shows {face_emo} emotion at {face_pct}% confidence, "
            f"but their words express {text_sent} sentiment at {text_pct}% confidence. "
            f"There is a strong emotional mismatch between their face and their words. "
            f"Write a professional two-sentence psychological observation about this "
            f"incongruence and what it might indicate about the person's true emotional state."
        )
    elif mismatch:
        return (
            f"A person's face shows {face_emo} emotion at {face_pct}% confidence, "
            f"but their words express {text_sent} sentiment at {text_pct}% confidence. "
            f"There is a mild emotional mismatch between verbal and non-verbal signals. "
            f"Write a professional two-sentence observation about what this might suggest."
        )
    else:
        fused = fusion_result.get("fused_sentiment", text_sent)
        return (
            f"A person's face shows {face_emo} emotion at {face_pct}% confidence "
            f"and their words express {text_sent} sentiment at {text_pct}% confidence. "
            f"Both signals are aligned with a {fused} overall emotional state. "
            f"Write a professional two-sentence summary of this person's emotional state."
        )


def _template_fallback(facial_result, text_result, fusion_result):
    face_emo  = facial_result.get("dominant_emotion", "neutral").lower()
    text_sent = text_result.get("dominant_sentiment", "neutral").lower()
    severity  = fusion_result.get("mismatch_severity", "none")
    mismatch  = fusion_result.get("mismatch", False)

    if mismatch and severity == "strong":
        return (
            f"Despite expressing {text_sent} sentiment verbally, the speaker's facial "
            f"expression strongly indicates {face_emo}. This significant incongruence "
            f"may suggest emotional masking or suppressed distress."
        )
    elif mismatch:
        return (
            f"The speaker's words lean {text_sent} while their face shows {face_emo} — "
            f"a subtle divergence that may reflect underlying stress or social compliance."
        )
    else:
        return (
            f"Both facial expression and verbal tone consistently reflect a {text_sent} "
            f"emotional state, with facial cues confirming {face_emo}. "
            f"No significant incongruence detected."
        )


def generate_summary(facial_result, text_result, fusion_result):
    try:
        prompt = _build_prompt(facial_result, text_result, fusion_result)
        pipe   = _get_pipeline()
        output = pipe(prompt)[0]["generated_text"].strip()
        if len(output) < 30:
            return _template_fallback(facial_result, text_result, fusion_result)
        return output
    except Exception:
        return _template_fallback(facial_result, text_result, fusion_result)