# LLM Inference and Optimization

A collection of experiments exploring techniques to make large language models smaller, faster, and more efficient — without significant accuracy loss. This repository will continue to grow with new experiments and optimization strategies over time.

---

## Contents

| Folder | Topic | Notebook |
|--------|-------|----------|
| [`distillation/`](distillation/) | Knowledge Distillation (BERT → DistilBERT) | [`bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb) |
| [`purning/`](purning/) | LLM Pruning From First Principles | [`llm_pruning_from_first_principles.ipynb`](purning/llm_pruning_from_first_principles.ipynb) |

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

### 2. LLM Pruning — From First Principles

**Notebook:** [`purning/llm_pruning_from_first_principles.ipynb`](purning/llm_pruning_from_first_principles.ipynb)

A complete pruning pipeline for transformer-based language models built from scratch, covering theory, implementation, benchmarking, and industry context.

#### Model
`distilgpt2` — small enough to run on CPU; architecturally identical to GPT-2 and representative of modern transformer decoders.

#### Techniques Implemented
| Technique | Description |
|-----------|-------------|
| **Unstructured magnitude pruning** | Global weight zeroing by magnitude threshold (no `torch.nn.utils.prune`) |
| **Structured pruning** | Remove entire attention heads (by output-projection L1 norm) and MLP neurons (by activation magnitude) |
| **Wanda-style activation-weighted pruning** | Score weights by `\|w\| × \|activation\|` for calibration-guided sparsity |
| **Iterative pruning** | Incremental sparsity steps to minimise accuracy degradation |

#### Key Highlights
- **Why pruning works:** over-parameterisation, the Lottery Ticket Hypothesis, and weight redundancy explained
- **Benchmarks:** latency, model size, and throughput measured at every sparsity level (0 % → 90 %)
- **Industry survey:** SparseGPT, Wanda, NVIDIA 2:4 structured sparsity, LLM Pruner, ShortGPT/Layer Dropping

#### Industry Techniques Covered

| Technique | Sparsity Type | Needs Fine-tune? | Best For |
|-----------|--------------|-----------------|----------|
| SparseGPT | Unstructured | No | Large models, single-shot |
| Wanda | Unstructured | No | Fast application, near-SparseGPT quality |
| NVIDIA 2:4 | Semi-structured | Yes (brief) | Ampere+ GPUs, production |
| LLM Pruner | Structured | Yes | Edge deployment |
| ShortGPT | Layer-level | No | Latency-critical inference |

---

## Techniques Covered (so far)

- [x] Knowledge Distillation
- [x] Post-training Quantization (INT8, 4-bit via BitsAndBytes)
- [x] Pruning (unstructured, structured, activation-guided, iterative)

## Planned / Coming Soon

- [ ] Pruning (advanced — SparseGPT, 2:4 structured)
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
