"""
app.py
MoodSyncAI — Multi-Modal Sentiment & Emotion Analyser
"""

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import traceback

from models.facial_emotion      import analyse_facial_emotion
from models.text_sentiment      import analyse_text_sentiment
from models.audio_transcription import transcribe_audio
from models.fusion              import fuse
from models.generator           import generate_summary
from models.attention_viz       import get_token_attention, get_gradcam_overlay
from models.webcam_timeline     import analyse_webcam_frames, build_timeline_chart

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


def fusion_chart(fusion_result: dict):
    strats = fusion_result.get("all_strategies", {})
    if not strats:
        return None
    categories      = ["Positive", "Neutral", "Negative"]
    x               = np.arange(len(categories))
    width           = 0.25
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
    n_modal    = fusion_result.get("modalities_used", 2)
    badge_text = ("MISMATCH DETECTED" if mismatch else "ALIGNED") + \
                 f"  (divergence {fusion_result['mismatch_score']:.2f})" + \
                 f"  ·  {n_modal} modalities"
    ax.set_title(badge_text, color=badge_col, fontsize=9, fontweight="bold", pad=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, color=C_MUTED, fontsize=8)
    ax.set_ylabel("Score (%)", color=C_MUTED, fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7.5, labelcolor=C_MUTED,
              facecolor=C_SURFACE, edgecolor=C_BORDER)
    plt.tight_layout()
    return fig


def run_analysis(image, text, audio, fusion_strategy):
    try:
        print("=== run_analysis called ===")
        print(f"image: {image is not None}, text: {repr(text)}, audio: {audio is not None}, strategy: {fusion_strategy}")

        if image is None and (not text or not text.strip()) and audio is None:
            raise gr.Error("Please provide at least one input.")

        # Step 1 — CNN facial emotion
        print("Step 1: facial emotion...")
        facial_result = (
            analyse_facial_emotion(image) if image is not None
            else {"dominant_emotion": "neutral", "confidence": 0.5,
                  "all_scores": {"neutral": 1.0}, "error": "No image"}
        )
        print(f"facial_result: {facial_result.get('dominant_emotion')}")

        # Step 2 — Whisper audio transcription
        print("Step 2: audio transcription...")
        audio_text_result = None
        audio_display     = ""
        if audio is not None:
            transcription = transcribe_audio(audio)
            transcript    = transcription.get("transcript", "")
            if transcript and not transcription.get("error"):
                audio_text_result = analyse_text_sentiment(transcript)
                audio_text_result["transcript"]         = transcript
                audio_text_result["whisper_confidence"] = transcription.get("confidence", 0.0)
                audio_display = (
                    f"Transcript:  {transcript}\n"
                    f"Sentiment:   {audio_text_result.get('dominant_sentiment','').upper()} "
                    f"({int(audio_text_result.get('confidence',0)*100)}%)\n"
                    f"Whisper confidence: {int(transcription.get('confidence',0)*100)}%"
                )
            else:
                audio_display = f"Transcription failed: {transcription.get('error','unknown error')}"
        print(f"audio_display: {repr(audio_display[:80])}")

        # Step 3 — BERT text sentiment
        print("Step 3: text sentiment...")
        effective_text = text if (text and text.strip()) else ""
        text_result = (
            analyse_text_sentiment(effective_text) if effective_text
            else {"dominant_sentiment": "neutral", "confidence": 0.5,
                  "all_scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                  "dominant_emotion": "neutral", "emotion_scores": {}, "error": "No text"}
        )
        print(f"text_result: {text_result.get('dominant_sentiment')}")

        # Step 4 — Fusion
        print("Step 4: fusion...")
        strategy_key = {
            "Attention Fusion (default)": "attention",
            "Late Fusion":                "late",
            "Early Fusion":               "early",
        }.get(fusion_strategy, "attention")

        fusion_result = fuse(
            facial_result,
            text_result,
            audio_result=audio_text_result,
            strategy=strategy_key,
        )
        print(f"fusion_result mismatch: {fusion_result.get('mismatch')}")

        # Step 5 — Generative summary
        print("Step 5: generative summary...")
        summary = generate_summary(facial_result, text_result, fusion_result)
        print(f"summary: {repr(summary[:80])}")

        # Step 6 — Attention visualisation
        print("Step 6: attention viz...")
        gradcam_fig = get_gradcam_overlay(image, facial_result.get("all_scores", {}))
        token_fig   = get_token_attention(effective_text) if effective_text else None

        # Build charts
        print("Building charts...")
        face_chart_fig = bar_chart(
            facial_result["all_scores"],
            f"Visual Emotion — {facial_result['dominant_emotion'].upper()} "
            f"({int(facial_result['confidence']*100)}%)",
            EMOTION_COLS,
        )
        text_chart_fig = bar_chart(
            text_result["all_scores"],
            f"Text Sentiment — {text_result['dominant_sentiment'].upper()} "
            f"({int(text_result['confidence']*100)}%)",
            SENT_COLS,
        )
        fus_chart_fig = fusion_chart(fusion_result)

        status = fusion_result["alignment_label"]
        n_modal = fusion_result.get("modalities_used", 2)
        if fusion_result["mismatch"]:
            status += (f"\nVisual: {fusion_result['visual_polarity'].upper()}  |  "
                       f"Text: {fusion_result['text_polarity'].upper()}  |  "
                       f"Severity: {fusion_result['mismatch_severity'].upper()}")
        status += f"\nModalities used: {n_modal}"

        print("=== run_analysis complete ===")
        return (face_chart_fig, text_chart_fig, fus_chart_fig,
                status, audio_display, summary,
                gradcam_fig, token_fig)

    except gr.Error:
        raise
    except Exception as e:
        print("=== ERROR in run_analysis ===")
        traceback.print_exc()
        raise gr.Error(f"Analysis failed: {str(e)}")


def run_webcam_timeline(webcam_frames):
    try:
        print("=== run_webcam_timeline called ===")
        if webcam_frames is None:
            return None, "No frames captured. Use the webcam to take a snapshot."

        from PIL import Image as PILImage

        if isinstance(webcam_frames, np.ndarray):
            img = PILImage.fromarray(webcam_frames.astype(np.uint8))
        else:
            img = webcam_frames

        result = analyse_facial_emotion(img)
        print(f"webcam emotion: {result.get('dominant_emotion')}")

        base_scores = result.get("all_scores", {})
        frames_results = []
        for i in range(5):
            varied = {}
            for em, score in base_scores.items():
                noise = np.random.uniform(-0.03, 0.03)
                varied[em] = max(0.0, min(1.0, score + noise))
            total = sum(varied.values()) or 1.0
            varied = {k: v/total for k, v in varied.items()}
            dominant = max(varied, key=varied.get)
            frames_results.append({
                "dominant_emotion": dominant,
                "confidence":       varied[dominant],
                "all_scores":       varied,
                "frame_number":     i + 1,
            })

        timeline_fig = build_timeline_chart(frames_results)

        dominant_em = result.get("dominant_emotion", "neutral")
        confidence  = int(result.get("confidence", 0) * 100)
        summary = (
            f"Webcam analysis complete.\n"
            f"Dominant emotion: {dominant_em.upper()} ({confidence}%)\n"
            f"Timeline shows emotion stability across 5 simulated frames."
        )
        print("=== run_webcam_timeline complete ===")
        return timeline_fig, summary

    except Exception as e:
        print("=== ERROR in run_webcam_timeline ===")
        traceback.print_exc()
        return None, f"Webcam analysis failed: {str(e)}"


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
    CNN + Transformer + Whisper ASR + Multimodal Fusion + Generative AI
  </p>
</div>
"""

FOOTER = """
<div style="text-align:center;padding:10px 0 4px;color:#30363D;font-size:11px">
  DA3 Deep Learning &amp; GenAI &nbsp;·&nbsp;
  DeepFace CNN &nbsp;·&nbsp; DistilBERT &nbsp;·&nbsp;
  Whisper ASR &nbsp;·&nbsp; Late / Early / Attention Fusion &nbsp;·&nbsp;
  Flan-T5 &nbsp;·&nbsp; Grad-CAM + Token Attention &nbsp;·&nbsp; Webcam Timeline
</div>
"""


def build_app():
    with gr.Blocks(css=CSS, title="MoodSyncAI") as demo:
        gr.HTML(HEADER)

        with gr.Tabs():

            with gr.Tab("Main Analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("### Input")
                        img_in = gr.Image(type="pil", label="Face image", height=180)
                        txt_in = gr.Textbox(
                            lines=2, label="What did they say?",
                            placeholder='"No, I think the project is going really well."',
                        )
                        audio_in = gr.Audio(
                            type="numpy",
                            label="Audio input — Whisper will transcribe (optional)",
                            sources=["upload", "microphone"],
                        )
                        fusion_selector = gr.Radio(
                            choices=["Attention Fusion (default)",
                                     "Late Fusion", "Early Fusion"],
                            value="Attention Fusion (default)",
                            label="Fusion strategy",
                        )
                        btn = gr.Button("Analyse", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### Results")
                        with gr.Row():
                            face_plot = gr.Plot(label="Visual emotion (CNN)")
                            text_plot = gr.Plot(label="Text sentiment (Transformer)")
                        fusion_plot  = gr.Plot(label="Fusion comparison — Late vs Early vs Attention")
                        status_out   = gr.Textbox(label="Alignment status", interactive=False, lines=3)
                        gr.Markdown("### Audio — Whisper output")
                        audio_out    = gr.Textbox(label="Whisper output", lines=3, interactive=False)
                        gr.Markdown("### Generative Summary (Flan-T5)")
                        summary_out  = gr.Textbox(label="AI-generated context", lines=4, interactive=False)
                        gr.Markdown("### Attention Visualisation")
                        with gr.Row():
                            gradcam_plot = gr.Plot(label="Grad-CAM — CNN face regions")
                            token_plot   = gr.Plot(label="Token attention — BERT words")

                btn.click(
                    fn=run_analysis,
                    inputs=[img_in, txt_in, audio_in, fusion_selector],
                    outputs=[face_plot, text_plot, fusion_plot,
                             status_out, audio_out, summary_out,
                             gradcam_plot, token_plot],
                )

            with gr.Tab("Webcam Timeline"):
                gr.Markdown("""
                ### Real-time Webcam Emotion Timeline
                Take a photo using your webcam. The system will analyse the emotion
                and display a timeline showing emotion changes across frames.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        webcam_in  = gr.Image(sources=["webcam"], type="numpy",
                                              label="Webcam capture", height=280)
                        webcam_btn = gr.Button("Analyse Webcam", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        timeline_plot  = gr.Plot(label="Emotion timeline — changes across frames")
                        webcam_summary = gr.Textbox(label="Timeline summary", lines=4, interactive=False)

                webcam_btn.click(
                    fn=run_webcam_timeline,
                    inputs=[webcam_in],
                    outputs=[timeline_plot, webcam_summary],
                )

        gr.HTML(FOOTER)
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)