"""
models/webcam_timeline.py
=========================
Real-time webcam emotion timeline — Optional Extended Feature #1

Captures multiple frames from webcam feed, analyses emotion on each frame,
and displays an emotion timeline showing changes over time.

Lecture alignment:
- CNN lecture: applying CNN inference on video frames
- Multimodal lecture: temporal emotion analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from models.facial_emotion import analyse_facial_emotion

C_BG      = "#0D1117"
C_SURFACE = "#161B22"
C_BORDER  = "#30363D"
C_TEXT    = "#E6EDF3"
C_MUTED   = "#8B949E"

EMOTION_COLS = {
    "happy":    "#3FB950",
    "surprise": "#D29922",
    "neutral":  "#8B949E",
    "sad":      "#58A6FF",
    "fear":     "#BC8CFF",
    "angry":    "#F85149",
    "disgust":  "#56D364",
}


def analyse_frame(frame_array: np.ndarray) -> dict:
    """Analyse a single frame from webcam."""
    try:
        img = Image.fromarray(frame_array.astype(np.uint8))
        return analyse_facial_emotion(img)
    except Exception as e:
        return {
            "dominant_emotion": "neutral",
            "confidence": 0.0,
            "all_scores": {},
            "error": str(e)
        }


def build_timeline_chart(results: list) -> plt.Figure:
    """
    Build emotion timeline chart from a list of frame analysis results.
    Shows how emotion changes across captured frames.
    """
    if not results:
        return None

    emotions   = ["happy", "sad", "angry", "fear", "surprise", "neutral", "disgust"]
    frames     = list(range(1, len(results) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                    gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(C_BG)

    # ── Top: emotion scores over time ──────────────────────────
    ax1.set_facecolor(C_SURFACE)
    for em in emotions:
        scores = [r.get("all_scores", {}).get(em, 0.0) * 100 for r in results]
        if max(scores) > 1.0:   # only plot emotions with meaningful scores
            ax1.plot(frames, scores, marker="o", markersize=4,
                     color=EMOTION_COLS.get(em, "#888"),
                     label=em.capitalize(), linewidth=1.8)

    ax1.set_xlim(0.5, len(frames) + 0.5)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Confidence (%)", color=C_MUTED, fontsize=8)
    ax1.set_title("Emotion Timeline — CNN analysis across captured frames",
                  color=C_TEXT, fontsize=9.5, fontweight="bold", pad=7)
    ax1.tick_params(colors=C_MUTED, labelsize=8)
    ax1.legend(fontsize=7.5, labelcolor=C_MUTED,
               facecolor=C_SURFACE, edgecolor=C_BORDER,
               loc="upper right", ncol=4)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)
    ax1.spines["bottom"].set_color(C_BORDER)
    ax1.spines["left"].set_color(C_BORDER)

    # ── Bottom: dominant emotion per frame ─────────────────────
    ax2.set_facecolor(C_SURFACE)
    dominant_emotions = [r.get("dominant_emotion", "neutral") for r in results]
    bar_colors        = [EMOTION_COLS.get(em, "#888") for em in dominant_emotions]
    confidences       = [r.get("confidence", 0.0) * 100 for r in results]

    bars = ax2.bar(frames, confidences, color=bar_colors,
                   edgecolor="none", width=0.6)
    ax2.set_xlim(0.5, len(frames) + 0.5)
    ax2.set_ylim(0, 115)
    ax2.set_xlabel("Frame", color=C_MUTED, fontsize=8)
    ax2.set_ylabel("Confidence (%)", color=C_MUTED, fontsize=8)
    ax2.set_title("Dominant emotion per frame",
                  color=C_TEXT, fontsize=9, fontweight="bold", pad=5)
    ax2.tick_params(colors=C_MUTED, labelsize=8)

    for bar, em, conf in zip(bars, dominant_emotions, confidences):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 em[:3].upper(), ha="center",
                 color=C_TEXT, fontsize=7, fontweight="bold")

    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)
    ax2.spines["bottom"].set_color(C_BORDER)
    ax2.spines["left"].set_color(C_BORDER)

    plt.tight_layout()
    return fig


def analyse_webcam_frames(frames: list) -> tuple:
    """
    Analyse a list of frames (numpy arrays) from webcam.
    Returns (results list, timeline figure, summary string).
    """
    if not frames:
        return [], None, "No frames provided"

    results = []
    for i, frame in enumerate(frames):
        result = analyse_frame(frame)
        result["frame_number"] = i + 1
        results.append(result)

    timeline_fig = build_timeline_chart(results)

    # Summary
    dominant_counts = {}
    for r in results:
        em = r.get("dominant_emotion", "neutral")
        dominant_counts[em] = dominant_counts.get(em, 0) + 1

    most_common = max(dominant_counts, key=dominant_counts.get)
    summary = (
        f"Analysed {len(results)} frames.\n"
        f"Dominant emotion: {most_common.upper()} "
        f"({dominant_counts[most_common]}/{len(results)} frames)\n"
        f"Emotion distribution: " +
        ", ".join([f"{k}: {v}" for k, v in dominant_counts.items()])
    )

    return results, timeline_fig, summary