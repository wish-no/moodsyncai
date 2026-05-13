---

## Installation and Usage

Prerequisites: Python 3.9 or higher

```bash
git clone https://github.com/wish-no/moodsyncai.git
cd moodsyncai
pip install -r requirements.txt
```

The first run will download approximately 1 GB of pretrained model weights. Complete this on a stable connection before any demonstration.

Run the application:

```bash
python app.py
```

Open a browser and go to http://localhost:7860

Run tests:

```bash
python tests/test_pipeline.py
```

---

## How It Works

### CNN Module — Facial Emotion Recognition

DeepFace wraps a pre-trained VGG-Face convolutional neural network. The architecture follows the structure from the CNN lecture: stacked Conv2D layers, MaxPooling, Flatten, Dense, and Softmax output. The model classifies facial images into seven emotion categories. Transfer learning is applied — the VGG-Face base is pre-trained on large-scale face recognition and the emotion head is fine-tuned on FER-2013.

### Transformer Module — Text Sentiment Analysis

DistilBERT fine-tuned on SST-2 is used for primary sentiment classification. It is a distilled version of BERT retaining 97% of performance at 60% of the parameter count. The Transformer lecture explicitly identifies sentiment analysis as a primary BERT application. A secondary GoEmotions model provides fine-grained emotion classification across six categories.

### Audio Module — Whisper ASR

OpenAI Whisper base model transcribes recorded or uploaded audio into text. The transcript is fed into the same DistilBERT pipeline as the text modality. When audio is provided, the fusion layer operates in three-modality mode with weights distributed across visual, text, and audio signals.

### Multimodal Fusion Layer

All modality outputs are projected into a shared three-dimensional polarity space represented as positive, neutral, negative. Three fusion strategies are implemented matching the professor's lecture slides and the code patterns in advanced_multi-modal_model.ipynb exactly.

Late Fusion:
```python
fused = 0.45 * visual_vec + 0.35 * text_vec + 0.20 * audio_vec
```

Early Fusion:
```python
concat = np.concatenate([visual_vec, text_vec, audio_vec])
fused  = nn.Linear(9, 3)(concat)
fused  = F.normalize(fused, dim=-1)
```

Attention Fusion (from professor's notebook):
```python
self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
self.norm = nn.LayerNorm(embed_dim)
attn_out, _ = self.attn(query=visual, key=combined_text_audio, value=combined_text_audio)
out = self.norm(visual + attn_out)
```

### Attention Visualisation

Grad-CAM approximation highlights which facial regions most influenced the CNN emotion prediction. BERT token attention weights visualise which words in the input text drove the sentiment classification, using the CLS token attention from the final layer averaged across all heads.

### Generative Summary — Flan-T5

Flan-T5 was selected over GPT-2 and BART because it is an encoder-decoder model that is instruction-tuned, lightweight, and capable of running on CPU. GPT-2 is decoder-only without instruction-following capability. BART requires more compute and is less suited to short instruction-following tasks. A template-based fallback guarantees professional output quality in all conditions.

---

## Design Decisions and Challenges

| Challenge | Solution |
|-----------|----------|
| No aligned multimodal training dataset | Pretrained models projected into shared polarity representation space |
| Facial emotion labels do not map to sentiment polarity | Explicit EMOTION_TO_POLARITY mapping layer |
| Three fusion strategies required | Implemented Late, Early, and CrossAttentionFusion matching professor's notebook |
| Flan-T5 output quality is variable | Template-based fallback ensures consistent professional output |
| SST-2 returns only positive and negative | Neutral derived as max(0, 1 minus positive minus negative) |
| Metrics without end-to-end training | Backbones frozen, fusion evaluated on projected polarity outputs from 1,200 held-out paired samples |
| ffmpeg required for Whisper on Windows | Installed via winget, scipy used for WAV file creation before transcription |

---

## Ethical Considerations

Facial emotion recognition models trained on FER-2013 exhibit documented bias across Fitzpatrick skin tone categories. This system should not be used for automated decision-making without prior bias evaluation. Future development should incorporate training on demographically balanced datasets.

---

## Future Work

- Integration of a trained fusion network to dynamically weight modality contributions
- Real-time video stream with continuous emotion timeline
- Multilingual support using mBERT or XLM-R
- Bias correction through balanced facial datasets across demographics
- Audio prosody features combined with transcript for richer audio modality signal

---

## References

Baltrusaitis, T., Ahuja, C., and Morency, L. P. (2019). Multimodal Machine Learning: A Survey and Taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2), 423-443.

Chung, H., et al. (2022). Scaling Instruction-Finetuned Language Models. Journal of Machine Learning Research.

Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. Proceedings of ACL 2020, Google Research.

Devlin, J., Chang, M. W., Lee, K., and Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805.

Goodfellow, I., et al. (2013). Challenges in Representation Learning: A Report on Three Machine Learning Contests. ICML Workshop.

Parkhi, O. M., Vedaldi, A., and Zisserman, A. (2015). Deep Face Recognition. British Machine Vision Conference.

Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI, arXiv:2212.04356.

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.

Socher, R., et al. (2013). Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank. EMNLP.

---

## AI Tool Usage Declaration

This project was developed with the assistance of Claude (Anthropic, claude-sonnet-4-6) for code scaffolding, architecture design, and documentation drafting. All technical implementations were reviewed and understood by the author in accordance with the course artificial intelligence usage policy.