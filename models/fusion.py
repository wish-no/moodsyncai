"""
models/fusion.py
Multimodal Fusion Layer — two or three modalities.

Supports:
  2-modal: visual + text         (base requirement)
  3-modal: visual + text + audio (extended feature)

Three strategies from lecture:
  Late      — weighted average
  Early     — concatenation + nn.Linear
  Attention — CrossAttentionFusion (nn.MultiheadAttention)

Lecture: DA-3-DeepLearning_MultiModal.pdf + advanced_multi-modal_model.ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

EMOTION_TO_POLARITY = {
    "happy":    "positive",
    "surprise": "positive",
    "neutral":  "neutral",
    "sad":      "negative",
    "fear":     "negative",
    "angry":    "negative",
    "disgust":  "negative",
}

MISMATCH_THRESHOLD = 0.38
DEVICE = torch.device("cpu")

VISUAL_W = 0.45
TEXT_W   = 0.35
AUDIO_W  = 0.20


# ── Strategy 1: Late Fusion ───────────────────────────────────────────────────

def late_fusion(vectors: list, weights: list) -> np.ndarray:
    total_w = sum(weights)
    fused   = sum(w * v for w, v in zip(weights, vectors)) / total_w
    return fused / (fused.sum() or 1.0)


# ── Strategy 2: Early Fusion ──────────────────────────────────────────────────

class EarlyFusionProjector(nn.Module):
    def __init__(self, input_dim: int = 6, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


class EarlyFusionProjector3(nn.Module):
    def __init__(self, input_dim: int = 9, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


# ── Strategy 3: Cross-Attention Fusion ───────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Exact pattern from professor's advanced_multi-modal_model.ipynb."""
    def __init__(self, embed_dim: int = 3, num_heads: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q  = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm(q + attn_out).squeeze(1)


_early2 = EarlyFusionProjector(input_dim=6,  output_dim=3).to(DEVICE).eval()
_early3 = EarlyFusionProjector3(input_dim=9, output_dim=3).to(DEVICE).eval()
_attn   = CrossAttentionFusion(embed_dim=3,  num_heads=1).to(DEVICE).eval()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _facial_to_polarity(emotion_scores: dict) -> np.ndarray:
    pos = neu = neg = 0.0
    for em, sc in emotion_scores.items():
        p = EMOTION_TO_POLARITY.get(em.lower(), "neutral")
        if   p == "positive": pos += sc
        elif p == "negative": neg += sc
        else:                 neu += sc
    total = pos + neu + neg or 1.0
    return np.array([pos/total, neu/total, neg/total], dtype=np.float32)


def _sentiment_to_polarity(all_scores: dict) -> np.ndarray:
    pos   = all_scores.get("positive", 0.0)
    neu   = all_scores.get("neutral",  0.0)
    neg   = all_scores.get("negative", 0.0)
    total = pos + neu + neg or 1.0
    return np.array([pos/total, neu/total, neg/total], dtype=np.float32)


def _cosine_dist(v1, v2) -> float:
    return float(1.0 - sk_cosine([v1], [v2])[0][0])


def _vec_label(vec: np.ndarray) -> str:
    return ["positive", "neutral", "negative"][int(np.argmax(vec))]


# ── Main entry point ──────────────────────────────────────────────────────────

def fuse(facial_result: dict,
         text_result:   dict,
         audio_result:  dict = None,
         strategy:      str  = "attention") -> dict:

    v_vec = _facial_to_polarity(facial_result.get("all_scores", {}))
    t_vec = _sentiment_to_polarity(text_result.get("all_scores", {}))

    use_audio = (
        audio_result is not None
        and audio_result.get("all_scores")
        and not audio_result.get("error")
    )

    if use_audio:
        a_vec   = _sentiment_to_polarity(audio_result.get("all_scores", {}))
        vectors = [v_vec, t_vec, a_vec]
        weights = [VISUAL_W, TEXT_W, AUDIO_W]
        n_modal = 3
    else:
        vectors = [v_vec, t_vec]
        weights = [0.55, 0.45]
        n_modal = 2

    # Late fusion
    late_vec = late_fusion(vectors, weights)

    # Early fusion
    concat   = np.concatenate(vectors)
    concat_t = torch.tensor(concat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        proj = _early3(concat_t) if n_modal == 3 else _early2(concat_t)
    early_vec = F.softmax(proj, dim=-1).cpu().numpy().squeeze()

    # Attention fusion
    kv_vec = (t_vec + a_vec) / 2.0 if use_audio else t_vec
    v_t    = torch.tensor(v_vec,  dtype=torch.float32).unsqueeze(0).to(DEVICE)
    kv_t   = torch.tensor(kv_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        attn_out = _attn(v_t, kv_t)
    attn_vec = F.softmax(attn_out, dim=-1).cpu().numpy().squeeze()

    strat_map   = {"late": late_vec, "early": early_vec, "attention": attn_vec}
    fused_vec   = strat_map.get(strategy, attn_vec)
    fused_label = _vec_label(fused_vec)
    fused_conf  = float(np.max(fused_vec))

    mismatch_score  = _cosine_dist(v_vec, t_vec)
    visual_polarity = EMOTION_TO_POLARITY.get(
        facial_result.get("dominant_emotion", "neutral").lower(), "neutral"
    )
    text_polarity = text_result.get("dominant_sentiment", "neutral")

    polarity_clash = (
        visual_polarity != text_polarity
        and visual_polarity != "neutral"
        and text_polarity   != "neutral"
    )
    mismatch = (mismatch_score > MISMATCH_THRESHOLD) or polarity_clash

    if   mismatch_score < 0.20: severity = "none"
    elif mismatch_score < MISMATCH_THRESHOLD: severity = "mild"
    else: severity = "strong"

    return {
        "fused_sentiment":   fused_label,
        "fused_confidence":  round(fused_conf, 4),
        "fused_vector":      [round(float(x), 4) for x in fused_vec],
        "visual_polarity":   visual_polarity,
        "text_polarity":     text_polarity,
        "mismatch":          mismatch,
        "mismatch_score":    round(mismatch_score, 4),
        "mismatch_severity": severity,
        "alignment_label":   "MISMATCH DETECTED" if mismatch else "ALIGNED",
        "strategy_used":     strategy,
        "modalities_used":   n_modal,
        "all_strategies": {
            "late":      {"vector": late_vec.tolist(),  "label": _vec_label(late_vec)},
            "early":     {"vector": early_vec.tolist(), "label": _vec_label(early_vec)},
            "attention": {"vector": attn_vec.tolist(),  "label": _vec_label(attn_vec)},
        },
    }