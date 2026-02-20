"""Simplify markdown cells - remove styling, emojis, HTML"""
import json
import re

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Simplified markdown content
simplified = {
    'title': '''# BERT to DistilBERT Knowledge Distillation Tutorial

## Overview
This notebook demonstrates Knowledge Distillation - a technique to compress a large, accurate model (teacher) into a smaller, faster model (student) while preserving performance.

### What is Knowledge Distillation?
- Teacher Model: Large, pre-trained BERT model fine-tuned on SST-2 (sentiment analysis)
- Student Model: Smaller DistilBERT model that learns from the teacher
- Goal: Transfer the teacher's knowledge to the student, achieving similar accuracy with ~60% fewer parameters

### Key Concepts
1. Temperature Scaling: Softens probability distributions to reveal "dark knowledge"
2. KL Divergence Loss: Measures how well student matches teacher's predictions
3. Combined Loss: Balances hard labels (ground truth) and soft labels (teacher predictions)

### Notebook Structure
- Setup: Install dependencies and load models
- Distillation: Custom trainer implementing knowledge distillation loss
- Training: Train student model using teacher's soft predictions
- Evaluation: Benchmark teacher vs student performance
- Deployment: Upload distilled model to Hugging Face Hub

Run cells sequentially to complete the distillation pipeline.
''',
    'step1': '''## Step 1: Load Teacher and Student Models

### Teacher Model: BERT-base-uncased (Fine-tuned on SST-2)
- Model: textattack/bert-base-uncased-SST-2
- Parameters: ~110M parameters
- Purpose: Pre-trained and fine-tuned on Stanford Sentiment Treebank (SST-2)
- Role: Provides "soft labels" (probability distributions) instead of just hard labels

### Student Model: DistilBERT-base-uncased
- Model: distilbert-base-uncased
- Parameters: ~67M parameters (~60% smaller)
- Purpose: Smaller, faster model that will learn from teacher
- Initialization: Starts with generic pre-trained weights, not fine-tuned

### Why This Pair?
- DistilBERT is architecturally similar to BERT but with fewer layers
- Both use the same tokenizer, making knowledge transfer easier
- Size reduction enables faster inference and lower memory usage
''',
    'step2': '''## Step 2: Custom Distillation Trainer

### DistillationTrainer Class
This custom trainer extends Hugging Face's Trainer to implement knowledge distillation.

### Key Components:

#### 1. Temperature Scaling (temperature=2.0)
- Divides logits by temperature before softmax
- Higher temperature = softer probability distributions
- Reveals relationships between classes that hard labels miss
- Example: Teacher might assign [0.7, 0.3] instead of [1.0, 0.0]

#### 2. KL Divergence Loss (loss_distill)
- Measures how well student's probability distribution matches teacher's
- Formula: KL(student_softmax || teacher_softmax)
- Multiplied by temperature² to scale back to original magnitude

#### 3. Cross-Entropy Loss (loss_ce)
- Standard classification loss using ground truth labels
- Ensures student still learns from actual labels

#### 4. Combined Loss (alpha=0.5)
- loss = α × loss_ce + (1-α) × loss_distill
- alpha controls balance: higher = more weight on hard labels
- Typical range: 0.3-0.7 (here: 0.5 = equal weight)

### Why Freeze Teacher?
- Teacher weights are frozen (requires_grad_(False))
- Only student learns; teacher provides guidance
- Prevents teacher from being modified during training
''',
    'step3': '''## Step 3: Training with Knowledge Distillation

### Dataset: SST-2 (Stanford Sentiment Treebank)
- Task: Binary sentiment classification (positive/negative)
- Samples: ~67K training, 872 validation, 1.8K test
- Format: Movie review sentences with sentiment labels

### Training Process:
1. Tokenization: Convert sentences to token IDs (max_length=128)
2. Forward Pass: 
   - Student processes inputs → student logits
   - Teacher processes inputs (no gradients) → teacher logits
3. Loss Calculation:
   - Apply temperature scaling to both logits
   - Compute KL divergence between student and teacher distributions
   - Compute cross-entropy with ground truth
   - Combine losses with alpha weighting
4. Backward Pass: Update only student model weights

### Training Configuration:
- Epochs: 3
- Batch Size: 32
- Learning Rate: 2e-5
- Temperature: 2.0
- Alpha: 0.5

### Expected Results:
- Student should achieve ~90-92% accuracy (close to teacher's ~93%)
- Model size: ~67M params vs teacher's ~110M params
- Inference speed: ~2x faster than teacher
''',
    'step4': '''## Step 4: Evaluation and Benchmarking

### Benchmarking Function
Compares multiple model variants across key metrics:

#### Metrics Measured:
1. Accuracy: Classification accuracy on validation set
2. Model Size: Disk size in MB
3. Latency: Average inference time per sample (ms)

#### Models Evaluated:
1. Teacher (BERT): Baseline - original fine-tuned model
2. Student (FP32): Distilled model in full precision
3. Student (INT8): Quantized to 8-bit integers for CPU inference
4. Student (4-bit): Quantized using BitsAndBytes for GPU memory efficiency
5. DistilBERT (Raw): Untrained baseline (should be ~50% accuracy)

### Key Fix: DistilBERT Token Type IDs
- DistilBERT doesn't use token_type_ids (unlike BERT)
- Must remove this input to avoid errors
- Check: if "DistilBert" in model class name → del inputs["token_type_ids"]

### Expected Performance:
- Teacher: ~93% accuracy, ~440MB, baseline latency
- Student (FP32): ~90-92% accuracy, ~268MB, ~2x faster
- Student (INT8): Similar accuracy, ~67MB, fastest CPU inference
- Student (4-bit): Similar accuracy, ~34MB, lowest VRAM usage
''',
    'step5': '''## Step 5: Upload to Hugging Face Hub

### Model Deployment
After successful distillation, upload the student model to Hugging Face Hub for:
- Sharing: Make model publicly available
- Versioning: Track model iterations
- Integration: Easy loading with from_pretrained()
- Documentation: Add model card with performance metrics

### Upload Process:
1. Login: Authenticate with Hugging Face token
2. Push: Upload model files (config.json, model weights, tokenizer)
3. Verify: Check model appears on your Hugging Face profile

### Model Card Best Practices:
- Document distillation parameters (temperature, alpha)
- Report accuracy metrics (teacher vs student)
- Include inference benchmarks
- Note model size and speed improvements
- Specify use cases and limitations
'''
}

# Map markdown cells to simplified content
markdown_map = {
    0: 'title',
    3: 'step1',
    5: 'step2',
    7: 'step3',
    11: 'step4',
    13: 'step5'
}

# Update markdown cells
updated = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and i in markdown_map:
        key = markdown_map[i]
        cell['source'] = simplified[key].strip().split('\n')
        updated += 1
        print(f'Updated cell {i}: {key}')

# Save
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nSimplified {updated} markdown cells')
print('Removed: HTML styling, emojis, colors, gradients')
print('Kept: Plain markdown headers, lists, and text')
