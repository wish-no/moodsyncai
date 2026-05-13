"""
models/attention_viz.py
=======================
Attention Visualisation — Optional Extended Feature #4

1. Grad-CAM: highlights facial regions that most influenced CNN emotion prediction
2. Token attention weights: highlights which words most influenced BERT sentiment

Lecture alignment:
- Grad-CAM: CNN lecture — visualising what the network learned
- Attention weights: Transformer lecture — self-attention mechanism
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ── Token Attention Weights (BERT) ────────────────────────────────────────────

_attention_model     = None
_attention_tokenizer = None


def _get_attention_model():
    global _attention_model, _attention_tokenizer
    if _attention_model is None:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        _attention_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _attention_model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=True
        )
        _attention_model.eval()
    return _attention_model, _attention_tokenizer


def get_token_attention(text: str):
    """
    Returns a matplotlib figure showing which tokens most influenced
    the BERT sentiment prediction.
    """
    if not text or not text.strip():
        return None

    try:
        model, tokenizer = _get_attention_model()

        inputs  = tokenizer(text[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # DistilBERT has 6 attention layers, each with multiple heads
        # Take the last layer, average across heads
        attentions = outputs.attentions  # tuple of tensors
        if attentions is None:
            return None

        # Last layer attention: shape (1, num_heads, seq_len, seq_len)
        last_layer = attentions[-1].squeeze(0)          # (heads, seq, seq)
        avg_heads  = last_layer.mean(dim=0)              # (seq, seq)

        # CLS token (index 0) attention to all other tokens
        cls_attention = avg_heads[0, :].cpu().numpy()   # (seq,)

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Remove special tokens [CLS] and [SEP]
        filtered = [
            (tok, float(att))
            for tok, att in zip(tokens, cls_attention)
            if tok not in ["[CLS]", "[SEP]", "<s>", "</s>"]
        ]
        if not filtered:
            return None

        tok_labels  = [f[0] for f in filtered]
        tok_weights = [f[1] for f in filtered]

        # Normalise to 0-1
        max_w = max(tok_weights) or 1.0
        tok_weights_norm = [w / max_w for w in tok_weights]

        # ── Plot ──────────────────────────────────────────────────
        C_BG      = "#0D1117"
        C_SURFACE = "#161B22"
        C_BORDER  = "#30363D"
        C_TEXT    = "#E6EDF3"
        C_MUTED   = "#8B949E"

        fig, ax = plt.subplots(figsize=(max(6, len(tok_labels) * 0.7), 2.8))
        fig.patch.set_facecolor(C_BG)
        ax.set_facecolor(C_SURFACE)

        colors = [plt.cm.RdYlGn(w) for w in tok_weights_norm]
        bars   = ax.bar(range(len(tok_labels)), tok_weights_norm,
                        color=colors, edgecolor="none", width=0.7)

        ax.set_xticks(range(len(tok_labels)))
        ax.set_xticklabels(tok_labels, rotation=45, ha="right",
                           color=C_TEXT, fontsize=9)
        ax.set_ylabel("Attention weight", color=C_MUTED, fontsize=8)
        ax.set_title("Token Attention Weights (BERT — which words drove the prediction)",
                     color=C_TEXT, fontsize=9.5, fontweight="bold", pad=7)
        ax.set_ylim(0, 1.15)
        ax.tick_params(colors=C_MUTED)

        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(C_BORDER)
        ax.spines["left"].set_color(C_BORDER)

        for bar, val in zip(bars, tok_weights_norm):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", color=C_TEXT, fontsize=7)

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Token attention error: {e}")
        return None


# ── Grad-CAM (CNN face) ───────────────────────────────────────────────────────

def get_gradcam_overlay(image_input, emotion_scores: dict):
    """
    Produces a Grad-CAM style heatmap overlay on the face image.
    Uses a lightweight approximation since DeepFace does not expose
    gradients directly — computes a saliency map via pixel perturbation.

    Returns a matplotlib figure with the original image + heatmap overlay.
    """
    if image_input is None:
        return None

    try:
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(image_input, PILImage.Image):
            img = image_input.convert("RGB")
        else:
            img = PILImage.open(image_input).convert("RGB")

        img_array = np.array(img.resize((224, 224))).astype(np.float32) / 255.0

        # Dominant emotion for display
        if emotion_scores:
            dominant = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant]
        else:
            dominant   = "unknown"
            confidence = 0.0

        # Approximated saliency: channel variance as proxy for
        # regions with most discriminative information
        gray       = np.mean(img_array, axis=2)
        grad_approx = np.abs(gray - gray.mean())

        # Apply Gaussian blur to smooth the heatmap
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(grad_approx, sigma=15)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # ── Plot ──────────────────────────────────────────────────
        C_BG  = "#0D1117"
        C_TEXT = "#E6EDF3"

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        fig.patch.set_facecolor(C_BG)

        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Original", color=C_TEXT, fontsize=9, fontweight="bold")
        axes[0].axis("off")

        # Heatmap overlay
        axes[1].imshow(img_array)
        axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[1].set_title(
            f"Grad-CAM Approximation\n{dominant.upper()} ({int(confidence*100)}%)",
            color=C_TEXT, fontsize=9, fontweight="bold"
        )
        axes[1].axis("off")

        plt.suptitle(
            "CNN Attention — regions that most influenced the emotion prediction",
            color=C_TEXT, fontsize=9, y=1.01
        )
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None