# LLM Inference and Optimization

A collection of experiments exploring techniques to make large language models smaller, faster, and more efficient — without significant accuracy loss. This repository will continue to grow with new experiments and optimization strategies over time.

---

## Contents

| Folder | Topic | Notebook |
|--------|-------|----------|
| [`distillation/`](distillation/) | Knowledge Distillation (BERT → DistilBERT) | [`bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb) |

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

## Techniques Covered (so far)

- [x] Knowledge Distillation
- [x] Post-training Quantization (INT8, 4-bit via BitsAndBytes)

## Planned / Coming Soon

- [ ] Pruning
- [ ] Speculative Decoding
- [ ] KV-Cache optimizations
- [ ] LoRA / QLoRA fine-tuning
- [ ] Benchmarking across hardware targets (CPU, GPU, edge devices)

---

## Getting Started

### Prerequisites

```bash
pip install transformers datasets accelerate evaluate torch bitsandbytes scipy
```

### Run a Notebook

Open any notebook in Jupyter or directly in Google Colab:

```bash
jupyter notebook distillation/bert_distillbert_knowledge_distillation.ipynb
```

---

## Repository Structure

```
llm-inference-and-optimization/
├── distillation/
│   └── bert_distillbert_knowledge_distillation.ipynb
└── README.md
```

---

## References

- [DistilBERT paper](https://arxiv.org/abs/1910.01108) — Sanh et al., 2019
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton et al., 2015
- [GLUE Benchmark / SST-2](https://gluebenchmark.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
