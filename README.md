# 🌾 Agri-Conversation — Plant Disease Diagnosis in Bengali

Fine-tuning small LLMs to diagnose crop diseases through conversation — in Bengali, for farmers.

---

## What It Does

An AI agricultural assistant that:
- Takes a **plant/leaf image** and farmer's symptom description
- Asks follow-up questions to narrow down the diagnosis
- Identifies the disease and gives **practical treatment advice**
- Responds entirely in **Bengali** (Bangla), in simple farmer-friendly language

---

## Models Fine-Tuned

| Notebook | Model | Type |
|---|---|---|
| `Gemma3-1B.ipynb` | Gemma 3 1B | Text only |
| `Gemma3-4B.ipynb` | Gemma 3 4B | Text only |
| `Ministral-3-3B.ipynb` | Ministral 3B | Text only |
| `Qwen3_1B.ipynb` | Qwen3 1.7B | Text only |
| `VLM_Gemma3_VL-4B.ipynb` | Gemma 3 VL 4B | Vision + Text |
| `VLM_-_Qwen3-VL-4B.ipynb` | Qwen3 VL 4B | Vision + Text |
| `VLM__-_Ministral-3-3B-Instruct.ipynb` | Ministral 3B Instruct | Vision + Text |

---

## Dataset

A custom JSONL dataset (`finetune_sharegpt.jsonl`) with multi-turn medical-style conversations:
- `symptom_text` — farmer's initial complaint
- `qa_flow` — follow-up Q&A turns
- `advice` + `notes` — final diagnosis and treatment
- `disease_label` — disease class (e.g., *Rice Tungro*, *Leaf Blight*)

A separate image dataset (`dataset.zip`) pairs disease labels with real leaf photos for the VLM notebooks.

**Split:** 80% train / 20% test

---

## Training Setup

All notebooks use **[Unsloth](https://github.com/unslothai/unsloth)** for fast, memory-efficient fine-tuning via **LoRA / QLoRA** (4-bit quantization) — runnable on Google Colab free tier.

```python
model = FastModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 8,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    ...
)
```

---

## Quick Start

1. Open any notebook in **Google Colab**
2. Run the dataset download cells (`gdown ...`)
3. Run training — checkpoints save automatically
4. Evaluate on the test split

---

## Stack

`Python` · `Unsloth` · `HuggingFace Transformers` · `TRL` · `LoRA/QLoRA` · `Google Colab`
