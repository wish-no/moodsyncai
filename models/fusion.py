"""
models/fusion.py
Multimodal Fusion Layer — three strategies as taught in lecture.

Lecture: DA-3-DeepLearning_MultiModal.pdf slide 6:
  Early  — mix inputs
  Middle — concentrate features
  Late   — combine final scores

Advanced notebook (professor's own code):
  EarlyFusionProjector   — concatenation + nn.Linear
  CrossAttentionFusion   — nn.MultiheadAttention (exact code from notebook)
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


# ── Strategy 1: Late Fusion ───────────────────────────────────────────────────

def late_fusion(vec_visual: np.ndarray,
                vec_text: np.ndarray,
                weight_visual: float = 0.55,
                weight_text: float = 0.45) -> np.ndarray:
    fused = weight_visual * vec_visual + weight_text * vec_text
    total = fused.sum()
    return fused / total if total > 0 else fused


# ── Strategy 2: Early Fusion Projector ───────────────────────────────────────

class EarlyFusionProjector(nn.Module):
    """Concatenate + linear projection. From professor's advanced notebook."""
    def __init__(self, input_dim: int = 6, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(x), dim=-1)


# ── Strategy 3: Cross-Attention Fusion ───────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Exact architecture from professor's advanced_multi-modal_model.ipynb:
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        attn_out, _ = self.attn(query, key_value, key_value)
        out = self.norm(query + attn_out)
    """
    def __init__(self, embed_dim: int = 3, num_heads: int = 1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        query     = query.unsqueeze(1)
        key_value = key_value.unsqueeze(1)
        attn_out, _ = self.attn(query, key_value, key_value)
        out = self.norm(query + attn_out)
        return out.squeeze(1)


_early_projector = EarlyFusionProjector(input_dim=6, output_dim=3).to(DEVICE)
_attention_fuser = CrossAttentionFusion(embed_dim=3, num_heads=1).to(DEVICE)
_early_projector.eval()
_attention_fuser.eval()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _facial_to_polarity_vec(emotion_scores: dict) -> np.ndarray:
    pos = neu = neg = 0.0
    for emotion, score in emotion_scores.items():
        p = EMOTION_TO_POLARITY.get(emotion.lower(), "neutral")
        if   p == "positive": pos += score
        elif p == "negative": neg += score
        else:                 neu += score
    total = pos + neu + neg or 1.0
    return np.array([pos / total, neu / total, neg / total], dtype=np.float32)


def _text_to_polarity_vec(all_scores: dict) -> np.ndarray:
    pos   = all_scores.get("positive", 0.0)
    neu   = all_scores.get("neutral",  0.0)
    neg   = all_scores.get("negative", 0.0)
    total = pos + neu + neg or 1.0
    return np.array([pos / total, neu / total, neg / total], dtype=np.float32)


def _cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(1.0 - sk_cosine([v1], [v2])[0][0])


def _vec_to_label(vec: np.ndarray) -> str:
    return ["positive", "neutral", "negative"][int(np.argmax(vec))]


# ── Main entry point ──────────────────────────────────────────────────────────

def fuse(facial_result: dict, text_result: dict, strategy: str = "attention") -> dict:
    v_vec = _facial_to_polarity_vec(facial_result.get("all_scores", {}))
    t_vec = _text_to_polarity_vec(text_result.get("all_scores", {}))

    # Late fusion
    late_vec = late_fusion(v_vec, t_vec, 0.55, 0.45)

    # Early fusion
    concat   = np.concatenate([v_vec, t_vec])
    concat_t = torch.tensor(concat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        early_t = _early_projector(concat_t)
    early_vec = F.softmax(early_t, dim=-1).cpu().numpy().squeeze()

    # Attention fusion
    v_t  = torch.tensor(v_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    tx_t = torch.tensor(t_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        attn_t = _attention_fuser(v_t, tx_t)
    attn_vec = F.softmax(attn_t, dim=-1).cpu().numpy().squeeze()

    strategy_map = {"late": late_vec, "early": early_vec, "attention": attn_vec}
    fused_vec    = strategy_map.get(strategy, attn_vec)
    fused_label  = _vec_to_label(fused_vec)
    fused_conf   = float(np.max(fused_vec))

    mismatch_score  = _cosine_distance(v_vec, t_vec)
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
        "all_strategies": {
            "late":      {"vector": late_vec.tolist(),  "label": _vec_to_label(late_vec)},
            "early":     {"vector": early_vec.tolist(), "label": _vec_to_label(early_vec)},
            "attention": {"vector": attn_vec.tolist(),  "label": _vec_to_label(attn_vec)},
        },
    }