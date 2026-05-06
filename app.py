"""
app.py
MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser
Run:  python app.py
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.facial_emotion import analyse_facial_emotion
from models.text_sentiment import analyse_text_sentiment
from models.fusion import fuse
from models.generator import generate_summary

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG      = "#0D1117"
C_SURFACE = "#161B22"
C_BORDER  = "#30363D"
C_TEXT    = "#E6EDF3"
C_MUTED   = "#8B949E"

EMOTION_COLS = {
    "happy":    "#3FB950", "surprise": "#D29922", "neutral": "#8B949E",
    "sad":      "#58A6FF", "fear":     "#BC8CFF",
    "angry":    "#F85149", "disgust":  "#56D364",
}
SENT_COLS  = {"positive": "#3FB950", "neutral": "#8B949E", "negative": "#F85149"}
STRAT_COLS = {"late": "#58A6FF", "early": "#BC8CFF", "attention": "#F0883E"}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _setup_ax(ax):
    ax.set_facecolor(C_SURFACE)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.spines["left"].set_color(C_BORDER)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def bar_chart(scores: dict, title: str, col_map: dict):
    labels = list(scores.keys())
    values = [scores[l] * 100 for l in labels]
    colors = [col_map.get(l, "#8B949E") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(C_BG)
    _setup_ax(ax)

    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Confidence (%)", color=C_MUTED, fontsize=8)
    ax.set_title(title, color=C_TEXT, fontsize=9.5, fontweight="bold", pad=7)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color=C_TEXT, fontsize=7.5)
    plt.tight_layout()
    return fig


def fusion_comparison_chart(fusion_result: dict):
    strats = fusion_result.get("all_strategies", {})
    if not strats:
        return None

    categories     = ["Positive", "Neutral", "Negative"]
    x              = np.arange(len(categories))
    width          = 0.25
    strategy_names  = ["late", "early", "attention"]
    strategy_labels = ["Late Fusion", "Early Fusion", "Attention Fusion"]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor(C_BG)
    _setup_ax(ax)

    for i, (s_key, s_label) in enumerate(zip(strategy_names, strategy_labels)):
        vec = strats.get(s_key, {}).get("vector", [0, 0, 0])
        ax.bar(x + i * width, [v * 100 for v in vec],
               width, label=s_label, color=STRAT_COLS[s_key],
               alpha=0.85, edgecolor="none")

    mismatch   = fusion_result["mismatch"]
    badge_col  = "#F0883E" if mismatch else "#3FB950"
    badge_text = ("MISMATCH DETECTED" if mismatch else "ALIGNED") + \
                 f"  (divergence {fusion_result['mismatch_score']:.2f})"

    ax.set_title(badge_text, color=badge_col, fontsize=9, fontweight="bold", pad=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, color=C_MUTED, fontsize=8)
    ax.set_ylabel("Score (%)", color=C_MUTED, fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7.5, labelcolor=C_MUTED,
              facecolor=C_SURFACE, edgecolor=C_BORDER)

    plt.tight_layout()
    return fig


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_analysis(image, text, fusion_strategy):
    if image is None and (not text or not text.strip()):
        raise gr.Error("Please provide an image, text, or both.")

    facial_result = (
        analyse_facial_emotion(image) if image is not None
        else {"dominant_emotion": "neutral", "confidence": 0.5,
              "all_scores": {"neutral": 1.0}, "error": "No image"}
    )

    text_result = (
        analyse_text_sentiment(text) if text and text.strip()
        else {"dominant_sentiment": "neutral", "confidence": 0.5,
              "all_scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
              "dominant_emotion": "neutral", "emotion_scores": {}, "error": "No text"}
    )

    strategy_key = {
        "Late Fusion":               "late",
        "Early Fusion":              "early",
        "Attention Fusion (default)":"attention",
    }.get(fusion_strategy, "attention")

    fusion_result = fuse(facial_result, text_result, strategy=strategy_key)
    summary       = generate_summary(facial_result, text_result, fusion_result)

    face_chart = bar_chart(
        facial_result["all_scores"],
        f"Visual Emotion — {facial_result['dominant_emotion'].upper()} "
        f"({int(facial_result['confidence']*100)}%)",
        EMOTION_COLS,
    )
    text_chart = bar_chart(
        text_result["all_scores"],
        f"Text Sentiment — {text_result['dominant_sentiment'].upper()} "
        f"({int(text_result['confidence']*100)}%)",
        SENT_COLS,
    )
    fusion_chart = fusion_comparison_chart(fusion_result)

    status = fusion_result["alignment_label"]
    if fusion_result["mismatch"]:
        status += (f"\nVisual: {fusion_result['visual_polarity'].upper()}  |  "
                   f"Text: {fusion_result['text_polarity'].upper()}  |  "
                   f"Severity: {fusion_result['mismatch_severity'].upper()}")

    return face_chart, text_chart, fusion_chart, status, summary


# ── UI ────────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container        { background: #0D1117 !important; }
.gr-panel, .gr-box             { background: #161B22 !important;
                                  border: 1px solid #30363D !important;
                                  border-radius: 10px !important; }
.gr-button-primary             { background: linear-gradient(135deg,#238636,#2EA043) !important;
                                  border: none !important; font-weight: 600 !important;
                                  border-radius: 8px !important; }
textarea, input[type=text]     { background: #0D1117 !important;
                                  border: 1px solid #30363D !important;
                                  color: #E6EDF3 !important;
                                  border-radius: 8px !important; }
label, .label-wrap span        { color: #8B949E !important; font-size: 13px !important; }
"""

HEADER = """
<div style="text-align:center;padding:24px 0 8px">
  <h1 style="margin:0;font-size:1.9rem;font-weight:800;
      background:linear-gradient(135deg,#58A6FF,#BC8CFF,#3FB950);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    MoodSyncAI
  </h1>
  <p style="color:#8B949E;font-size:.9rem;margin-top:6px">
    Multi-Modal Sentiment &amp; Emotion Analyser &nbsp;·&nbsp;
    CNN + Transformer + Multimodal Fusion + Generative AI
  </p>
</div>
"""

FOOTER = """
<div style="text-align:center;padding:10px 0 4px;color:#30363D;font-size:11px">
  DA3 Deep Learning &amp; GenAI &nbsp;·&nbsp;
  DeepFace CNN &nbsp;·&nbsp; DistilBERT (BERT) &nbsp;·&nbsp;
  Late / Early / Attention Fusion &nbsp;·&nbsp; Flan-T5
</div>
"""


def build_app():
    with gr.Blocks(css=CSS, title="MoodSyncAI") as demo:
        gr.HTML(HEADER)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Input")
                img_in = gr.Image(type="pil", label="Face image (upload)", height=210)
                txt_in = gr.Textbox(
                    lines=3, label="What did they say?",
                    placeholder='"No, I think the project is going really well."',
                )
                fusion_selector = gr.Radio(
                    choices=["Attention Fusion (default)", "Late Fusion", "Early Fusion"],
                    value="Attention Fusion (default)",
                    label="Fusion strategy",
                )
                btn = gr.Button("Analyse", variant="primary", size="lg")

                gr.Markdown("#### Try these example texts")
                gr.Examples(
                    examples=[
                        [None, "I'm absolutely fine, don't worry about me at all."],
                        [None, "This is amazing, I love every minute of this project!"],
                        [None, "I guess it'll probably be okay, I don't know..."],
                    ],
                    inputs=[img_in, txt_in],
                    label="",
                )

            with gr.Column(scale=2):
                gr.Markdown("### Results")

                with gr.Row():
                    face_plot = gr.Plot(label="Visual emotion (CNN)")
                    text_plot = gr.Plot(label="Text sentiment (Transformer)")

                fusion_plot = gr.Plot(
                    label="Fusion comparison — Late vs Early vs Attention"
                )
                status_out = gr.Textbox(
                    label="Alignment status", interactive=False, lines=2
                )
                gr.Markdown("### Generative Summary (Flan-T5)")
                summary_out = gr.Textbox(
                    label="AI-generated emotional context",
                    lines=4, interactive=False,
                )

        btn.click(
            fn=run_analysis,
            inputs=[img_in, txt_in, fusion_selector],
            outputs=[face_plot, text_plot, fusion_plot, status_out, summary_out],
        )

        gr.HTML(FOOTER)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)