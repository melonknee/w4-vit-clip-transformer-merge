<h1 align="center">🧠 Multimodal Transfer Learning for Image Captioning</h1>

<p align="center">
  <em>A modular image captioning system using CLIP and Qwen for domain-specialized caption generation.</em>
</p>

---

## 🔍 Project Overview

This project explores **multimodal transfer learning** by bridging a pre-trained <strong>CLIP-based ViT encoder</strong> with a <strong>Qwen language decoder</strong> (Qwen or Qwen1.5-0.5B). The goal is to generate meaningful image captions by transferring general-purpose vision and language knowledge into a modular encoder–decoder setup.

---

## ⚙️ Architecture & Approach

- **Encoder**: CLIP ViT (frozen at first)
- **Decoder**: Qwen1.5-0.5B
- **Bridge**: Trainable projection layer between vision and language spaces

I initially froze both encoder and decoder, only training the projector. As performance was limited, I later **partially unfroze Qwen decoder layers**, which significantly improved output quality.

---

## 🧪 Domain-Specific Fine-Tuning

To improve performance and training efficiency, I focused on a **narrow domain**:
- Dataset: [`alexg99/captioned_flickr_faces`](https://huggingface.co/datasets/alexg99/captioned_flickr_faces) and ['alexg99/captioned_flickr_faces_2'](https://huggingface.co/datasets/alexg99/captioned_flickr_faces_2) (~60k face images with captions)
- Benefits:
  - Smaller vocabulary (not asking them to comment on every aspect of the image, only the face)
  - Consistent visual structure of patterns with less surprises (each face has a pair of eyes, a nose, a mouth, glasses/no glasses)
  - Faster convergence

### 🧭 Prompt Engineering

To steer the model toward relevant outputs, I used **grounded prefixes** such as:
> "Describe the facial expression of the person in the image"

instead of generic instructions like:
> "Describe the image"

---

## ⚡ Training Optimisations

The model trained with a limited compute budget, hence why training optimisation was important in this project.
- **Batch processing** for efficiency
- **Top-K sampling** for better diversity and fluency
- **Beam search** for generating more fluent and accurate captions (better diversity, but takes longer to train


---

## 📂 Datasets Used

| Dataset | Description |
|--------|-------------|
| [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k) | For initial prototyping and model baseline |
| [`alexg99/captioned_flickr_faces`](https://huggingface.co/datasets/alexg99/captioned_flickr_faces) | Main training dataset |
| [`alexg99/captioned_flickr_faces_2`](https://huggingface.co/datasets/alexg99/captioned_flickr_faces_2) | Supplementary testing dataset |

---

## ✅ Results & Reflections

- Final outputs are **mostly relevant**, especially when using grounded prompts
- Captions can still occasionally be repetitive or generic
- Project demonstrates how **transfer learning + domain narrowing + smart prompting** can produce solid results even with constrained resources

---

## 📌 Summary

> A multimodal transformer architecture connecting CLIP and Qwen, trained to generate facial image captions via domain-specialised fine-tuning and prompt engineering.

---

