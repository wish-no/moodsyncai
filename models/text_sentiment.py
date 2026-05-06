"""
models/text_sentiment.py
Transformer-based sentiment analysis using DistilBERT (BERT architecture).
Lecture: DA-3-DeepLearning_TR.pdf — BERT used for Sentiment Analysis (slide 5)
"""

from transformers import pipeline

_sentiment_pipeline = None
_emotion_pipeline   = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None,
        )
    return _sentiment_pipeline


def _get_emotion_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        _emotion_pipeline = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=None,
        )
    return _emotion_pipeline


def analyse_text_sentiment(text: str) -> dict:
    if not text or not text.strip():
        return {
            "dominant_sentiment": "neutral",
            "confidence":         1.0,
            "all_scores":         {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "dominant_emotion":   "neutral",
            "emotion_scores":     {},
            "error":              "Empty text provided",
        }

    try:
        raw         = _get_sentiment_pipeline()(text[:512])[0]
        sent_scores = {item["label"].lower(): round(item["score"], 4) for item in raw}

        pos = sent_scores.get("positive", 0.0)
        neg = sent_scores.get("negative", 0.0)
        sent_scores["neutral"] = round(max(0.0, 1.0 - pos - neg), 4)

        dominant_sent = max(sent_scores, key=sent_scores.get)
        sent_conf     = sent_scores[dominant_sent]

        try:
            raw_emo      = _get_emotion_pipeline()(text[:512])[0]
            emo_scores   = {item["label"].lower(): round(item["score"], 4) for item in raw_emo}
            dominant_emo = max(emo_scores, key=emo_scores.get)
        except Exception:
            emo_scores   = {}
            dominant_emo = "unknown"

        return {
            "dominant_sentiment": dominant_sent,
            "confidence":         sent_conf,
            "all_scores":         sent_scores,
            "dominant_emotion":   dominant_emo,
            "emotion_scores":     emo_scores,
            "error":              None,
        }

    except Exception as exc:
        return {
            "dominant_sentiment": "neutral",
            "confidence":         0.5,
            "all_scores":         {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
            "dominant_emotion":   "unknown",
            "emotion_scores":     {},
            "error":              str(exc),
        }