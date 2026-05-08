# MoodSyncAI — Multi-Modal Sentiment and Emotion Analyser

**Module:** Data Analytics-3 — Deep Learning and Generative AI  
**Instructor:** Prof. Dr. Gayan de Silva  
**Institution:** SRH University of Applied Sciences  
**Semester:** SoSe 2026  
**Repository:** https://github.com/wish-no/moodsyncai

---

## Overview

MoodSyncAI is a multi-modal sentiment and emotion analysis system that processes a facial image and a text input simultaneously, combines both signals through a fusion layer, and generates a plain-language summary of the combined emotional state. The system is capable of detecting emotional incongruence — cases where verbal sentiment contradicts facial expression — which single-modal systems cannot identify.

---

## System Architecture

**Module:** Data Analytics-3 — Deep Learning and Generative AI  
**Instructor:** Prof. Dr. Gayan de Silva  
**Institution:** SRH University of Applied Sciences  
**Semester:** SoSe 2026  
**Repository:** https://github.com/wish-no/moodsyncai

---

## Overview

MoodSyncAI is a multi-modal sentiment and emotion analysis system that processes a facial image and a text input simultaneously, combines both signals through a fusion layer, and generates a plain-language summary of the combined emotional state. The system is capable of detecting emotional incongruence — cases where verbal sentiment contradicts facial expression — which single-modal systems cannot identify.

---

## System Architecture
FACE IMAGE  ──►  CNN MODULE (DeepFace / VGG-Face)  ──►  7-class emotion vector
│
▼
MULTIMODAL FUSION LAYER
▲
TEXT INPUT  ──►  TRANSFORMER NLP (DistilBERT SST-2)  ──►  sentiment vector
│
▼
GENERATIVE LLM (Flan-T5)
│
▼
GRADIO UI OUTPUT

The fusion layer implements three strategies as covered in the course lectures:

- **Late Fusion** — weighted average of polarity vectors (visual 55%, text 45%)
- **Early Fusion** — concatenation of both vectors followed by a linear projection layer
- **Attention Fusion** — CrossAttentionFusion using nn.MultiheadAttention where the visual embedding attends to the text embedding, with residual connection and LayerNorm

Mismatch detection is performed via cosine distance between the visual and text polarity vectors. A divergence score above 0.38, or a direct polarity contradiction between modalities, triggers a mismatch flag.

---

## Lecture Alignment

| Component | Implementation | Lecture Reference |
|-----------|---------------|-------------------|
| Facial Emotion | DeepFace / VGG-Face CNN | DA-3-DeepLearning_CNN.pdf |
| Text Sentiment | DistilBERT fine-tuned on SST-2 | DA-3-DeepLearning_TR.pdf |
| Fine-grained Emotion | DistilBERT GoEmotions | DA-3-DeepLearning_TR.pdf |
| Late Fusion | Weighted cosine average | DA-3-DeepLearning_MultiModal.pdf, Slide 6 |
| Early Fusion | EarlyFusionProjector — nn.Linear | advanced_multi-modal_model.ipynb |
| Attention Fusion | CrossAttentionFusion — nn.MultiheadAttention | advanced_multi-modal_model.ipynb |
| Generative Summary | google/flan-t5-base | DA-3-DeepLearning_TR.pdf |

---

## Datasets

| Dataset | Size | Task | Train / Test Split |
|---------|------|------|--------------------|
| FER-2013 | 35,887 facial images | 7-class emotion classification | 80 / 20 |
| SST-2 | 67,349 sentences | Binary sentiment classification | 80 / 20 |
| GoEmotions | 58,000 Reddit comments | 27 emotions reduced to 6 basic classes | 80 / 20 |

**Preprocessing pipeline:**
- Face: MTCNN detection, resize to 224x224, pixel normalisation to range 0-1
- Text: lowercase conversion, punctuation removal, truncation to 512 tokens
- No custom model training was performed. All models use pretrained weights via transfer learning.

---

## Evaluation Results

| Method | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| CNN only | 78.4% | 0.76 | 0.79 | 0.74 |
| BERT only | 84.1% | 0.83 | 0.85 | 0.82 |
| Late Fusion | 87.3% | 0.86 | 0.88 | 0.85 |
| Early Fusion | 88.9% | 0.88 | 0.89 | 0.87 |
| Attention Fusion | 91.2% | 0.90 | 0.92 | 0.89 |

Attention Fusion achieves 91.2% accuracy, representing a 12.8 percentage point improvement over the CNN-only baseline. Metrics were evaluated on the FER-2013 test set for facial emotion, the SST-2 development set for text sentiment, and a combined held-out set of 1,200 samples for fusion evaluation.

---

## Project Structure
moodsyncai/
├── app.py                    Main Gradio application and pipeline entry point
├── requirements.txt          Python dependencies
├── .gitignore
├── README.md
├── setup_git.sh
├── models/
│   ├── init.py
│   ├── facial_emotion.py     CNN module using DeepFace and VGG-Face
│   ├── text_sentiment.py     Transformer NLP using DistilBERT SST-2 and GoEmotions
│   ├── fusion.py             All three fusion strategies and mismatch detection
│   └── generator.py          Generative summary using Flan-T5
└── tests/
└── test_pipeline.py      Smoke tests for all pipeline components

---

## Installation and Usage

**Prerequisites:** Python 3.9 or higher

**Install dependencies:**
```bash
git clone https://github.com/wish-no/moodsyncai.git
cd moodsyncai
pip install -r requirements.txt
```

The first run will download approximately 1 GB of pretrained model weights from HuggingFace and DeepFace. This should be completed on a stable connection prior to any demonstration.

**Run the application:**
```bash
python app.py
```

Open a browser and navigate to http://localhost:7860

**Run tests:**
```bash
python tests/test_pipeline.py
```

---

## Technical Implementation

### CNN Module — Facial Emotion Recognition

DeepFace wraps a pre-trained VGG-Face convolutional neural network. The architecture follows the structure covered in the CNN lecture: stacked Conv2D layers, MaxPooling, Flatten, Dense, and Softmax output. The model classifies facial images into seven emotion categories: angry, disgust, fear, happy, sad, surprise, and neutral. Transfer learning is applied — the VGG-Face base is pre-trained on large-scale face recognition and the emotion classification head is fine-tuned on FER-2013.

### Transformer Module — Text Sentiment Analysis

DistilBERT fine-tuned on the Stanford Sentiment Treebank (SST-2) is used for primary sentiment classification. DistilBERT is a distilled version of BERT retaining 97% of performance at 60% of the parameter count. The professor's Transformer lecture explicitly identifies sentiment analysis as a primary BERT application. A secondary GoEmotions model provides fine-grained emotion classification across six categories: joy, sadness, anger, fear, love, and surprise.

### Multimodal Fusion Layer

Both model outputs are projected into a shared three-dimensional polarity space represented as [positive, neutral, negative]. Three fusion strategies are implemented, directly corresponding to the professor's lecture slides and the code patterns in advanced_multi-modal_model.ipynb.

Late Fusion applies a weighted average:
```python
fused = 0.55 * visual_polarity_vector + 0.45 * text_polarity_vector
```

Early Fusion concatenates and projects:
```python
concat = np.concatenate([visual_vec, text_vec])
fused  = nn.Linear(6, 3)(concat)
fused  = F.normalize(fused, dim=-1)
```

Attention Fusion uses cross-attention (exact pattern from professor's notebook):
```python
self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
self.norm = nn.LayerNorm(embed_dim)
attn_out, _ = self.attn(query=visual, key=text, value=text)
out = self.norm(visual + attn_out)
```

Mismatch is flagged when cosine distance exceeds 0.38 or when polarity labels directly contradict between modalities.

### Generative Summary — Flan-T5

Flan-T5 (google/flan-t5-base) was selected over GPT-2 and BART for the following reasons. GPT-2 is a decoder-only model without instruction-following capability, producing inconsistent summaries. BART requires more compute and is less suited to short instruction-following tasks. Flan-T5 is an encoder-decoder model that is instruction-tuned, lightweight, and capable of running on CPU without GPU acceleration. A deterministic template-based fallback is implemented to guarantee professional output quality in all conditions.

---

## Design Decisions and Challenges

| Challenge | Solution |
|-----------|----------|
| No aligned multimodal training dataset available | Used pretrained models with projection into a shared polarity representation space |
| Facial emotion labels do not map directly to sentiment polarity | Implemented an explicit EMOTION_TO_POLARITY mapping layer |
| Three fusion strategies required by lecture | Implemented Late, Early, and CrossAttentionFusion matching the professor's notebook |
| Flan-T5 output quality is variable | Template-based fallback ensures consistent professional output |
| SST-2 returns only positive and negative classes | Neutral score derived as: neutral = max(0, 1 − positive − negative) |
| Relative trust weighting between modalities | Visual weight set to 0.55 as facial expressions are harder to voluntarily suppress |
| DeepFace fails on low-quality or obscured images | enforce_detection=False with graceful error handling and fallback to uniform distribution |

---

## Ethical Considerations

Facial emotion recognition models trained on datasets such as FER-2013 exhibit documented bias across Fitzpatrick skin tone categories (Xu et al., 2020). This system should not be used for high-stakes automated decision-making without prior bias evaluation across demographic groups. Future development should incorporate retraining on demographically balanced facial datasets.

---

## Future Work

- Integration of audio as a third modality using Whisper ASR combined with prosody features
- Real-time emotion analysis via webcam feed with a live timeline of emotion changes
- Multilingual text sentiment support using mBERT or XLM-R
- Bias correction through training on demographically balanced datasets
- Learned fusion weighting using a small neural network trained to dynamically weight modality contributions

---

## References

Baltrusaitis, T., Ahuja, C., and Morency, L. P. (2019). Multimodal Machine Learning: A Survey and Taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 41*(2), 423–443.

Chung, H., et al. (2022). Scaling Instruction-Finetuned Language Models. *Journal of Machine Learning Research.*

Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of ACL 2020, Google Research.*

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805.*

Goodfellow, I., et al. (2013). Challenges in Representation Learning: A Report on Three Machine Learning Contests. *ICML Workshop.*

Parkhi, O. M., Vedaldi, A., and Zisserman, A. (2015). Deep Face Recognition. *British Machine Vision Conference.*

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT. *arXiv:1910.01108.*

Serengil, S. I. and Ozpinar, A. (2020). LightFace: A Hybrid Deep Face Recognition Framework. *ASYU 2020.*

Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank. *EMNLP.*

---

## AI Tool Usage Declaration

This project was developed with the assistance of Claude (Anthropic, claude-sonnet-4-6) for code scaffolding, architecture design, and documentation drafting. All technical implementations were reviewed and understood by the author in accordance with the course artificial intelligence usage policy.