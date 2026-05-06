import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_sentiment import analyse_text_sentiment
from models.fusion import fuse
from models.generator import generate_summary


def test_positive_text():
    r = analyse_text_sentiment("This is amazing, I absolutely love it!")
    assert r["dominant_sentiment"] == "positive"
    print(f"[PASS] positive text  → {r['dominant_sentiment']} ({int(r['confidence']*100)}%)")


def test_negative_text():
    r = analyse_text_sentiment("I am so disappointed and upset about everything.")
    assert r["dominant_sentiment"] == "negative"
    print(f"[PASS] negative text  → {r['dominant_sentiment']} ({int(r['confidence']*100)}%)")


def test_empty_text():
    r = analyse_text_sentiment("")
    assert r["error"] is not None
    print(f"[PASS] empty text     → error handled gracefully")


def test_fusion_all_strategies():
    facial = {"dominant_emotion": "sad", "confidence": 0.7,
              "all_scores": {"sad": 0.6, "fear": 0.2, "neutral": 0.1,
                             "happy": 0.05, "angry": 0.02, "disgust": 0.02, "surprise": 0.01}}
    text   = {"dominant_sentiment": "positive", "confidence": 0.85,
              "all_scores": {"positive": 0.85, "negative": 0.05, "neutral": 0.10}}

    for strategy in ["late", "early", "attention"]:
        r = fuse(facial, text, strategy=strategy)
        print(f"[PASS] {strategy:9s} fusion → label={r['fused_sentiment']} "
              f"mismatch={r['mismatch']} score={r['mismatch_score']:.2f}")

    r = fuse(facial, text)
    assert r["mismatch"] is True
    print(f"[PASS] mismatch detect → severity={r['mismatch_severity']}")


def test_fusion_aligned():
    facial = {"dominant_emotion": "happy", "confidence": 0.8,
              "all_scores": {"happy": 0.8, "surprise": 0.1, "neutral": 0.05,
                             "sad": 0.02, "angry": 0.01, "disgust": 0.01, "fear": 0.01}}
    text   = {"dominant_sentiment": "positive", "confidence": 0.9,
              "all_scores": {"positive": 0.9, "negative": 0.03, "neutral": 0.07}}
    r = fuse(facial, text)
    assert r["mismatch"] is False
    print(f"[PASS] aligned detect  → score={r['mismatch_score']:.2f}")


def test_generator():
    facial = {"dominant_emotion": "sad",       "confidence": 0.7, "all_scores": {}}
    text   = {"dominant_sentiment": "positive", "confidence": 0.85, "all_scores": {}}
    fusion = {"mismatch": True, "mismatch_severity": "strong", "fused_sentiment": "negative"}
    s = generate_summary(facial, text, fusion)
    assert len(s) > 20
    print(f"[PASS] generator       → '{s[:60]}...'")


if __name__ == "__main__":
    print("\nRunning MoodSyncAI smoke tests...\n")
    test_positive_text()
    test_negative_text()
    test_empty_text()
    test_fusion_all_strategies()
    test_fusion_aligned()
    test_generator()
    print("\nAll tests passed.\n")