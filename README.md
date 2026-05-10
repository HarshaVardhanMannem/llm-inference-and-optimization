<div align="center">

# LLM Inference & Optimization

**A reproducible playbook of techniques for making large language models smaller, faster, and cheaper to serve — without losing the qualities that make them useful.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.x-yellow.svg)](https://huggingface.co/docs/transformers)
[![TRL](https://img.shields.io/badge/TRL-experimental-orange.svg)](https://huggingface.co/docs/trl)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Harsha901%2Fqwen2.5--coder--7b--distilled--from--14b-yellow.svg)](https://huggingface.co/Harsha901/qwen2.5-coder-7b-distilled-from-14b)
[![License](https://img.shields.io/badge/license-MIT%20(proposed)-lightgrey.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)

</div>

---

## Table of Contents

- [Why this repo](#why-this-repo)
- [Experiments at a glance](#experiments-at-a-glance)
- [Quickstart](#quickstart)
- [Experiments in detail](#experiments-in-detail)
  - [1. Knowledge Distillation — BERT → DistilBERT (SST-2)](#1-knowledge-distillation--bert--distilbert-sst-2)
  - [2. Code Distillation — Qwen2.5-Coder 14B → 7B (GKD + QLoRA)](#2-code-distillation--qwen25-coder-14b--7b-gkd--qlora)
  - [3. LLM Pruning from First Principles](#3-llm-pruning-from-first-principles)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [References](#references)

---

## Why this repo

Production LLM systems live or die by three numbers: **latency, memory, and cost per token**. This repo is a curated set of *runnable* experiments that target those numbers using the standard compression toolkit — distillation, quantization, LoRA / QLoRA, and pruning — applied to real models on real hardware.

Each experiment is a single self-contained Jupyter notebook with:

- The full training / evaluation pipeline (no hidden helpers)
- Saved outputs so the results can be inspected without re-running
- Honest reporting of compute used, parameters touched, and trade-offs made
- A short, scannable summary section in this README

The goal is for each technique to be reproducible end-to-end, and for the README alone to be enough to answer *"what was tried, on what model, and what changed."*

---

## Experiments at a glance

| # | Technique | Model pair | Trainable params | Result | Notebook |
|---|---|---|---:|---|---|
| 1 | Classic Knowledge Distillation | BERT-base → DistilBERT | ~67 M (full) | 90.4 % SST-2 acc, **2.7× faster** GPU inference, 39 % smaller | [`distillation/bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb) |
| 2 | GKD + QLoRA Distillation | Qwen2.5-Coder 14B → 7B | 40.4 M LoRA (**0.53 %**) | LoRA adapter [published on the Hub](https://huggingface.co/Harsha901/qwen2.5-coder-7b-distilled-from-14b); fits on a single A100 80 GB | [`distillation/Code_Distillation.ipynb`](distillation/Code_Distillation.ipynb) |
| 3 | Pruning (from first principles) | — | — | Walk-through of magnitude / structured pruning intuition | [`purning/llm_pruning_from_first_principles.ipynb`](purning/llm_pruning_from_first_principles.ipynb) |

---

## Quickstart

Clone the repo and pick the experiment you want to run.

```bash
git clone https://github.com/HarshaVardhanMannem/llm-inference-and-optimization.git
cd llm-inference-and-optimization
```

### Environments

Two install profiles, depending on which notebook you open:

<details>
<summary><strong>BERT → DistilBERT (works on a single GPU or even CPU)</strong></summary>

```bash
pip install transformers datasets accelerate evaluate torch bitsandbytes scipy
jupyter notebook distillation/bert_distillbert_knowledge_distillation.ipynb
```
</details>

<details>
<summary><strong>Qwen2.5-Coder 14B → 7B distillation (requires an A100 80 GB)</strong></summary>

```bash
pip install -q trl transformers datasets accelerate bitsandbytes peft torch
pip install -q -U huggingface_hub
jupyter notebook distillation/Code_Distillation.ipynb
```

> Set your Hugging Face token via the `HF_TOKEN` env var, the Colab Secrets sidebar (`Tools → Secrets → HF_TOKEN`), or the interactive `getpass()` prompt the notebook will fall back to. **Never paste the token into the notebook directly.**
</details>

<details>
<summary><strong>Pruning notebook</strong></summary>

```bash
pip install transformers torch numpy matplotlib
jupyter notebook purning/llm_pruning_from_first_principles.ipynb
```
</details>

### Open in Colab

Each notebook can be opened directly on Colab; the Code Distillation notebook needs an A100 80 GB runtime (Colab Pro+).

---

## Experiments in detail

### 1. Knowledge Distillation — BERT → DistilBERT (SST-2)

**Notebook:** [`distillation/bert_distillbert_knowledge_distillation.ipynb`](distillation/bert_distillbert_knowledge_distillation.ipynb)

Compresses a fine-tuned BERT-base teacher into a DistilBERT student on SST-2 sentiment classification using classic Hinton-style logit distillation.

| Component | Choice |
|---|---|
| Task | Binary sentiment classification (SST-2, ~67 K train) |
| Teacher | `bert-base-uncased` fine-tuned on SST-2 (~110 M params) |
| Student | `distilbert-base-uncased` (~67 M params, 39 % fewer) |
| Loss | `α · CE + (1 − α) · KL(student ‖ teacher / T)` with `α = 0.5`, `T = 4.0` |
| Trainer | Custom `DistillationTrainer` on top of HF `Trainer` |
| Schedule | 3 epochs, batch 32, lr 2e-5, max-len 128 |

#### Results

| Model | Accuracy | Size (MB) | GPU latency (ms) |
|---|---:|---:|---:|
| Teacher — BERT (FP32) | 92.43 % | 417.7 | 11.86 |
| **Student — DistilBERT (FP32)** | **90.37 %** | **255.4** | **4.36** |
| Student — DistilBERT (INT8) | 90.94 % | 132.3 | 22.11 (CPU) |
| Raw DistilBERT (no distillation) | 49.08 % | 255.4 | 4.29 |

**Takeaways:** retains **97.8 %** of the teacher's accuracy at **2.7× faster** GPU inference and **39 % smaller** footprint. The raw-DistilBERT baseline (49 %) confirms the lift is from distillation, not just architecture.

---

### 2. Code Distillation — Qwen2.5-Coder 14B → 7B (GKD + QLoRA)

**Notebook:** [`distillation/Code_Distillation.ipynb`](distillation/Code_Distillation.ipynb)
**Adapter on the Hub:** [`Harsha901/qwen2.5-coder-7b-distilled-from-14b`](https://huggingface.co/Harsha901/qwen2.5-coder-7b-distilled-from-14b)

Scales the same idea to a billion-parameter code model: distill `Qwen/Qwen2.5-Coder-14B-Instruct` into `Qwen/Qwen2.5-Coder-7B-Instruct` with **TRL's `DistillationTrainer` (GKD)** on top of **QLoRA**, so both models fit on a single A100 80 GB.

| Component | Choice | Why |
|---|---|---|
| Teacher | `Qwen2.5-Coder-14B-Instruct`, frozen, 4-bit NF4 | strong code reasoner, fits in <30 GB quantised |
| Student | `Qwen2.5-Coder-7B-Instruct` + LoRA (r=16, α=32) | trainable head, only adapters update |
| Loss | GKD with `lmbda=1.0`, `beta=0.5` | on-policy student sampling + symmetric JSD |
| Optimiser | `paged_adamw_8bit`, lr 1e-4, cosine schedule, 20-step warmup | memory-light optimiser state |
| Precision | bf16 compute, NF4 + double-quant storage | A100-native, big VRAM savings |
| Dataset | `iamtarun/python_code_instructions_18k_alpaca` | ~18 K Python instruction → code pairs (95/5 split) |
| Run length | `max_steps=50` (effective batch 32, ~1.6 K examples seen) | first usable run; raise for longer training |

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

#### Trainable parameter budget

| | Params | Share |
|---|---:|---:|
| Frozen 4-bit base | 7 615 616 512 | 99.47 % |
| **LoRA adapters (trainable)** | **40 370 176** | **0.53 %** |
| Total | 7 655 986 688 | 100.00 % |

**Takeaways:**
- Whole pipeline (4-bit teacher + 4-bit student + LoRA) trains end-to-end on a single A100 80 GB at ~36 GB allocated VRAM
- Published artifact is a **~160 MB LoRA adapter**, not a 15 GB checkpoint — cheap to store, version, and ship
- This is a **first-pass run (50 optimiser steps)** that demonstrates the pipeline; longer training (3+ epochs ≈ 1 656 steps) is the natural next step

#### Loading the published adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base, "Harsha901/qwen2.5-coder-7b-distilled-from-14b"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
```

---

### 3. LLM Pruning from First Principles

**Notebook:** [`purning/llm_pruning_from_first_principles.ipynb`](purning/llm_pruning_from_first_principles.ipynb)

A from-first-principles walk-through of pruning for transformers — magnitude pruning, structured vs unstructured sparsity, and how each interacts with the underlying linear-algebra kernels at inference time. Useful as a primer before reaching for SparseGPT / Wanda-style methods.

---

## Repository layout

```
llm-inference-and-optimization/
├── distillation/
│   ├── bert_distillbert_knowledge_distillation.ipynb   # Experiment 1 — classic KD on SST-2
│   ├── Code_Distillation.ipynb                          # Experiment 2 — Qwen 14B → 7B (GKD + QLoRA)
│   └── KNOWLEDGE_DISTILLATION_GUIDE.md                  # Background reading on KD techniques
├── purning/
│   └── llm_pruning_from_first_principles.ipynb         # Experiment 3 — pruning primer
└── README.md
```

---

## Reproducibility

| | Detail |
|---|---|
| **Hardware (Experiment 1)** | Single GPU (any modern NVIDIA) or CPU |
| **Hardware (Experiment 2)** | NVIDIA A100 80 GB (Colab Pro+ A100 runtime works) |
| **Python** | 3.10+ |
| **Core libraries** | `torch`, `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `bitsandbytes` |
| **Determinism** | Seeds set where it matters (e.g. `train_test_split(seed=42)`); full bit-exact reproducibility across hardware is not guaranteed for bf16/4-bit kernels |
| **Saved outputs** | Notebook outputs are committed so you can review results without re-running |

If a notebook fails to reproduce on your setup, please [open an issue](https://github.com/HarshaVardhanMannem/llm-inference-and-optimization/issues) with your hardware, library versions, and the cell that failed.

---

## Roadmap

| Status | Technique |
|---|---|
| ✅ Done | Classic Knowledge Distillation (BERT → DistilBERT) |
| ✅ Done | GKD + QLoRA Distillation (Qwen2.5-Coder 14B → 7B) |
| ✅ Done | Post-training Quantization (INT8, 4-bit NF4 via BitsAndBytes) |
| ✅ Done | LoRA / QLoRA fine-tuning |
| ✅ Done | Pruning — first-principles walk-through |
| 🚧 Planned | Speculative Decoding |
| 🚧 Planned | KV-Cache optimizations (paged attention, prefix caching) |
| 🚧 Planned | Cross-hardware benchmarking (CPU, GPU, edge) |
| 🚧 Planned | SparseGPT / Wanda-style structured pruning at scale |

---

## Contributing

Contributions are welcome — especially:

- New compression / inference-optimization experiments (one notebook = one technique)
- Reproductions on different hardware with the timing & memory numbers attached
- Bug fixes or clearer explanations in existing notebooks

**Workflow:**

1. Fork and create a feature branch (`git checkout -b experiment/<short-name>`).
2. Add your notebook under the appropriate folder (or create a new one).
3. **Strip secrets and large outputs** before committing — never hard-code Hugging Face / API tokens; use `os.environ`, `google.colab.userdata`, or `getpass()`.
4. Add a short summary section to this README following the format used by Experiments 1 and 2.
5. Open a pull request describing the technique, the model pair, and the headline result.

---

## Citation

If you use this repository in academic work, please cite:

```bibtex
@misc{mannem_llm_inference_optimization,
  author       = {Mannem, Harsha Vardhan},
  title        = {LLM Inference and Optimization: a reproducible playbook
                  of distillation, quantization, LoRA, and pruning experiments},
  year         = {2026},
  howpublished = {\url{https://github.com/HarshaVardhanMannem/llm-inference-and-optimization}}
}
```

---

## License

No `LICENSE` file is currently committed to this repository. **MIT** is recommended (and assumed by the badge above) — to make it official, add a `LICENSE` file at the repo root. Until a license is added, default copyright applies and the code cannot be redistributed under another license.

---

## References

### Distillation
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton, Vinyals, Dean (2015)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) — Sanh et al. (2019)
- [On-Policy Distillation of Language Models (GKD)](https://arxiv.org/abs/2306.13649) — Agarwal et al. (2023)
- [TRL — `DistillationTrainer` docs](https://huggingface.co/docs/trl/main/en/distillation_trainer)

### Quantization & PEFT
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al. (2023)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al. (2021)
- [BitsAndBytes — 8-bit / 4-bit quantization](https://github.com/bitsandbytes-foundation/bitsandbytes)

### Datasets & benchmarks
- [GLUE / SST-2](https://gluebenchmark.com/)
- [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

### Tooling
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [Hugging Face TRL](https://huggingface.co/docs/trl)

---

<div align="center">

⭐ If this repo helped you, a star is the easiest way to say thanks.

</div>
