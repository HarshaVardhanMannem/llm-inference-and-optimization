# LLM Inference and Optimization

A collection of experiments exploring techniques to make large language models smaller, faster, and more efficient — without significant accuracy loss. This repository will continue to grow with new experiments and optimization strategies over time.

---

## Contents

| Folder | Topic | Notebook |
|--------|-------|----------|
| [`distillation/`](distillation/) | Knowledge Distillation (BERT → DistilBERT) | [`bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb) |
| [`distillation/`](distillation/) | Code Distillation (Qwen2.5-Coder 14B → 7B, GKD + QLoRA) | [`Code_Distillation.ipynb`](distillation/Code_Distillation.ipynb) |
| [`purning/`](purning/) | LLM Pruning from First Principles | [`llm_pruning_from_first_principles.ipynb`](purning/llm_pruning_from_first_principles.ipynb) |

---

## Experiments

### 1. Knowledge Distillation — BERT → DistilBERT

**Notebook:** [`distillation/bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb)

Demonstrates how to compress a large pre-trained transformer (teacher) into a smaller, faster model (student) using **knowledge distillation**, while preserving most of the teacher's task performance.

#### Task
Binary sentiment classification on the **SST-2** dataset (Stanford Sentiment Treebank, ~67K training samples).

#### Models
| Role | Model | Parameters |
|------|-------|------------|
| Teacher | `bert-base-uncased` fine-tuned on SST-2 | ~110M |
| Student | `distilbert-base-uncased` | ~67M (39% fewer) |

#### Distillation Setup
- **Temperature scaling** (T = 4.0) to soften the teacher's output distribution and expose "dark knowledge"
- **Combined loss:** `α × CrossEntropyLoss + (1 − α) × KLDivergenceLoss` with α = 0.5
- **Custom `DistillationTrainer`** built on top of the Hugging Face `Trainer` API
- Training: 3 epochs, batch size 32, learning rate 2e-5, max sequence length 128

#### Results

| Model | Accuracy | Size (MB) | GPU Latency (ms) |
|-------|----------|-----------|-----------------|
| Teacher — BERT (FP32) | 92.43% | 417.7 | 11.86 |
| Student — DistilBERT (FP32) | **90.37%** | **255.4** | **4.36** |
| Student — DistilBERT (INT8) | 90.94% | 132.3 | 22.11 (CPU) |
| Raw DistilBERT (no distillation) | 49.08% | 255.4 | 4.29 |

**Key takeaways:**
- The distilled student retains **97.77% of the teacher's accuracy** (90.37% vs 92.43%)
- **2.7× faster** inference on GPU (4.36 ms vs 11.86 ms)
- **39% smaller** model footprint (255.4 MB vs 417.7 MB)
- The raw DistilBERT baseline (49%) confirms that the distillation process is responsible for the knowledge transfer — not just the smaller architecture

---

### 2. Code Distillation — Qwen2.5-Coder 14B → 7B (GKD + QLoRA)

**Notebook:** [`distillation/Code_Distillation.ipynb`](distillation/Code_Distillation.ipynb)
**Adapter on the Hub:** [`Harsha901/qwen2.5-coder-7b-distilled-from-14b`](https://huggingface.co/Harsha901/qwen2.5-coder-7b-distilled-from-14b)

Scales the same idea up to a billion-parameter code model: distill `Qwen/Qwen2.5-Coder-14B-Instruct` into `Qwen/Qwen2.5-Coder-7B-Instruct` using **TRL's `DistillationTrainer`** (Generalized Knowledge Distillation, GKD) on top of **QLoRA**, so both models fit on a single A100 80 GB.

#### Task
Python instruction-following / code generation on the **`iamtarun/python_code_instructions_18k_alpaca`** dataset (~18 K Alpaca-formatted Python coding tasks, 95/5 train/eval split).

#### Models

| Role | Model | Total params | Trainable params |
|------|-------|-------------:|-----------------:|
| Teacher (frozen, 4-bit NF4) | `Qwen/Qwen2.5-Coder-14B-Instruct` | 14.77 B | 0 |
| Student (4-bit + LoRA) | `Qwen/Qwen2.5-Coder-7B-Instruct` | 7.66 B | **40.4 M (0.53 %)** |

#### Distillation Setup
- **GKD loss** (TRL `DistillationTrainer`) with `lmbda = 1.0` (pure on-policy student sampling) and `beta = 0.5` (symmetric Jensen–Shannon divergence)
- **QLoRA**: NF4 4-bit base + double-quant + bf16 compute, LoRA `r=16`, `α=32` on all attention and MLP projections
- Optimiser: `paged_adamw_8bit`, lr `1e-4`, cosine schedule, 20-step warmup
- Effective batch size 32 (`per_device=16 × grad_accum=2`), `max_steps=50` for the published run

#### Architecture comparison

| Property | Teacher (14B) | Student (7B) | Ratio |
|---|---:|---:|---:|
| Total parameters | 14.77 B | 7.66 B | **0.52×** |
| Hidden size | 5 120 | 3 584 | 0.70× |
| Transformer layers | 48 | 28 | 0.58× |
| Attention heads | 40 | 28 | 0.70× |
| KV heads (GQA) | 8 | 4 | 0.50× |
| Max context | 32 768 | 32 768 | 1.00× |
| 4-bit storage footprint | ~7.4 GB | ~3.8 GB | **0.51×** |

**Key takeaways:**
- The whole pipeline (4-bit teacher + 4-bit student + LoRA) trains end-to-end on a single A100 80 GB at ~36 GB allocated VRAM
- Only **0.53 %** of the student updates — the published artifact is a ~160 MB LoRA adapter, not a full 15 GB checkpoint
- The published adapter is a **first-pass run (50 optimiser steps)** that demonstrates the pipeline; longer training (3+ epochs ≈ 1 656 steps) is the natural next step

---

## Techniques Covered (so far)

- [x] Knowledge Distillation (classic Hinton-style: BERT → DistilBERT)
- [x] Knowledge Distillation (GKD via TRL: Qwen2.5-Coder 14B → 7B)
- [x] Post-training Quantization (INT8, 4-bit via BitsAndBytes / NF4)
- [x] LoRA / QLoRA fine-tuning
- [x] Pruning (from first principles)

## Planned / Coming Soon

- [ ] Speculative Decoding
- [ ] KV-Cache optimizations
- [ ] Benchmarking across hardware targets (CPU, GPU, edge devices)

---

## Getting Started

### Prerequisites

For the BERT → DistilBERT notebook:
```bash
pip install transformers datasets accelerate evaluate torch bitsandbytes scipy
```

For the Code Distillation notebook (Qwen 14B → 7B, requires an A100 80 GB):
```bash
pip install -q trl transformers datasets accelerate bitsandbytes peft torch
pip install -q -U huggingface_hub
```

### Run a Notebook

Open any notebook in Jupyter or directly in Google Colab:

```bash
jupyter notebook distillation/bert_distillbert_knowledge_distillation.ipynb
jupyter notebook distillation/Code_Distillation.ipynb
jupyter notebook purning/llm_pruning_from_first_principles.ipynb
```

---

## Repository Structure

```
llm-inference-and-optimization/
├── distillation/
│   ├── bert_distillbert_knowledge_distillation.ipynb   # BERT → DistilBERT (classic KD)
│   ├── Code_Distillation.ipynb                          # Qwen2.5-Coder 14B → 7B (GKD + QLoRA)
│   └── KNOWLEDGE_DISTILLATION_GUIDE.md                  # Background reading
├── purning/
│   └── llm_pruning_from_first_principles.ipynb
└── README.md
```

---

## References

- [DistilBERT paper](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015
- [GLUE Benchmark / SST-2](https://gluebenchmark.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
