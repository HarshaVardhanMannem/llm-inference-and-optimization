# Knowledge Distillation for Code Models: Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Setup for A100 80GB](#setup)
3. [Distillation Techniques](#techniques)
4. [Best Practices](#best-practices)
5. [Advanced Methods](#advanced)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

## Overview

Knowledge distillation transfers capabilities from a large "teacher" model to a smaller "student" model. For coding tasks, this enables deployment of efficient models that retain most of the teacher's programming ability.

### Key Benefits
- **5-10x faster inference** with 70-95% performance retention
- **Lower deployment costs** (fewer GPUs needed)
- **On-device deployment** possible with smaller models
- **Maintained code quality** through teacher supervision

### Supported Model Pairs

| Teacher | Student | Memory (4-bit) | Speedup |
|---------|---------|---------------|---------|
| CodeLlama-70B | CodeLlama-13B | ~40GB | 5x |
| CodeLlama-34B | CodeLlama-7B | ~20GB | 5x |
| DeepSeek-Coder-33B | DeepSeek-Coder-6.7B | ~18GB | 5x |
| StarCoder2-15B | StarCoder2-7B | ~9GB | 2x |

## Setup for A100 80GB

### Installation

```bash
# Install dependencies
pip install torch==2.3.0 \
    transformers==4.41.0 \
    datasets==2.19.0 \
    accelerate==0.30.0 \
    bitsandbytes==0.43.1 \
    peft==0.11.0 \
    sentencepiece==0.2.0
```

### Memory Optimization

```python
# 4-bit quantization for teacher (fits 70B models)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-70b-hf",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

## Distillation Techniques

### 1. Logit-Based Distillation

**Standard KD (Forward KL):**
```python
# Student learns from teacher's probability distribution
kl_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T ** 2)
```

**Pros:**
- Simple to implement
- Works well for classification tasks
- Minimal computational overhead

**Cons:**
- Can suffer from exposure bias in generation
- May overfit to teacher's mistakes

**When to use:** Code classification, bug detection, code search

---

**Reverse KL (MiniLLM approach):**
```python
# Better for generation tasks
reverse_kl = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1).detach(),
    reduction='batchmean',
    log_target=False
) * (T ** 2)
```

**Pros:**
- Reduces exposure bias
- Better for long-form code generation
- More stable training

**Cons:**
- Slightly more complex
- May require more training steps

**When to use:** Code generation, completion, translation

### 2. Feature-Based Distillation

Match intermediate representations between teacher and student:

```python
def feature_distillation_loss(student_features, teacher_features):
    """
    Align hidden states from multiple layers
    """
    loss = 0.0
    for s_feat, t_feat in zip(student_features, teacher_features):
        # MSE on hidden states
        loss += F.mse_loss(s_feat, t_feat.detach())
    return loss / len(student_features)
```

**Layer Selection Strategies:**

1. **Uniform spacing:** Every Nth layer
   ```python
   layers = list(range(0, 32, 4))  # Layers 0, 4, 8, 12, ...
   ```

2. **Late layers:** Focus on high-level representations
   ```python
   layers = list(range(24, 32))  # Last 8 layers
   ```

3. **Manual selection:** Based on analysis
   ```python
   layers = [0, 8, 16, 24, 31]  # Input, middle, output
   ```

**When to use:** When student and teacher have similar architectures

### 3. Attention Distillation

Transfer attention patterns:

```python
def attention_distillation_loss(student_attn, teacher_attn):
    """
    Match attention distributions
    """
    # Normalize attention weights
    student_attn = F.softmax(student_attn, dim=-1)
    teacher_attn = F.softmax(teacher_attn, dim=-1)
    
    # MSE on attention
    loss = F.mse_loss(student_attn, teacher_attn.detach())
    return loss
```

**Pros:**
- Captures structural understanding of code
- Helps with syntax and dependencies

**Cons:**
- Very memory intensive
- Requires architectural compatibility

**When to use:** When memory allows and architectures are similar

### 4. Data-Augmented Distillation

Generate synthetic training data from teacher:

```python
def generate_teacher_data(teacher_model, prompts, temperature=0.7):
    """
    Generate high-quality code examples from teacher
    """
    outputs = teacher_model.generate(
        prompts,
        max_new_tokens=512,
        temperature=temperature,
        do_sample=True,
        top_p=0.95
    )
    return outputs
```

**Strategy:**
1. Start with seed instructions/problems
2. Generate multiple solutions from teacher
3. Filter for quality (syntax check, test cases)
4. Use for distillation

**When to use:** Limited training data, domain-specific tasks

## Best Practices

### 1. Loss Weighting

**Recommended starting points:**

```python
# For code generation
alpha = 0.7  # Distillation loss (soft labels)
beta = 0.3   # Hard label loss (ground truth)
gamma = 0.5  # Feature distillation

# For code understanding
alpha = 0.5
beta = 0.5
gamma = 0.3
```

**Tuning strategy:**
- Start with equal weights (0.5/0.5)
- Increase alpha if student underfits
- Increase beta if student diverges from truth
- Adjust gamma based on validation loss

### 2. Temperature Selection

Temperature controls "softness" of probability distributions:

```python
# Low temperature (1.0-1.5): Sharp distributions
# - Use for: Classification, precise matching
# - Risk: Less information transfer

# Medium temperature (2.0-3.0): Balanced
# - Use for: Most tasks (recommended)
# - Sweet spot for code generation

# High temperature (4.0-6.0): Smooth distributions
# - Use for: When teacher is very confident
# - Risk: May lose precision
```

**Finding optimal temperature:**
```python
temperatures = [1.0, 2.0, 3.0, 4.0]
for T in temperatures:
    kl_loss = distillation_loss(logits, T)
    print(f"T={T}: KL={kl_loss:.4f}")
# Choose T with best validation performance
```

### 3. Training Hyperparameters

**For A100 80GB:**

```python
config = {
    # Batch sizing
    "batch_size": 4,  # Per device
    "gradient_accumulation_steps": 4,  # Effective = 16
    
    # Learning rate
    "learning_rate": 2e-5,  # Lower than pre-training
    "warmup_ratio": 0.1,    # 10% warmup
    "lr_scheduler": "cosine",
    
    # Training duration
    "num_epochs": 3,  # Usually sufficient
    "max_steps": None,  # Or set explicit step count
    
    # Optimization
    "fp16": False,  # A100 prefers bf16
    "bf16": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    
    # Saving
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,
}
```

### 4. Data Quality

**Dataset recommendations:**

1. **Code generation:**
   - CodeAlpaca-20K: Instruction following
   - MBPP: Python programming
   - HumanEval: Coding problems
   - Mix for best results

2. **Code understanding:**
   - CodeSearchNet: Multi-language
   - CodeXGLUE: Various tasks
   - Stack Overflow: Real-world code

**Data filtering:**
```python
def filter_high_quality(examples):
    """Keep only high-quality examples"""
    filtered = []
    for ex in examples:
        code = ex['output']
        
        # Check syntax
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError:
            continue
        
        # Check length
        if len(code) < 10 or len(code) > 2000:
            continue
        
        # Check for comments/docstrings
        if '"""' in code or "'''" in code or '#' in code:
            filtered.append(ex)
    
    return filtered
```

### 5. Monitoring Training

**Key metrics to track:**

```python
metrics = {
    'total_loss': total_loss,
    'kl_loss': kl_divergence,
    'ce_loss': cross_entropy,
    'feature_loss': feature_distillation,
    'learning_rate': current_lr,
    'perplexity': torch.exp(ce_loss),
}
```

**Warning signs:**
- KL loss not decreasing: Temperature too low/high
- CE loss diverging: Alpha too high
- Feature loss stuck: Projection issue
- Perplexity increasing: Overfitting to teacher

## Advanced Methods

### 1. Multi-Teacher Distillation

Combine knowledge from multiple teachers:

```python
def multi_teacher_distillation(
    student_logits,
    teacher_logits_list,  # List of teacher outputs
    weights=None  # Optional teacher weights
):
    if weights is None:
        weights = [1.0 / len(teacher_logits_list)] * len(teacher_logits_list)
    
    # Ensemble teacher distribution
    teacher_ensemble = sum(
        w * F.softmax(t_logits / T, dim=-1)
        for w, t_logits in zip(weights, teacher_logits_list)
    )
    
    # KL divergence
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        teacher_ensemble,
        reduction='batchmean'
    ) * (T ** 2)
    
    return kl_loss
```

**When to use:**
- Have multiple strong teachers (e.g., CodeLlama + DeepSeek)
- Want robustness across different coding styles
- Teachers excel at different subtasks

### 2. On-Policy Distillation

Generate training data from student's own distribution:

```python
def on_policy_distillation_step(student, teacher, prompts):
    """
    1. Student generates completions
    2. Teacher provides feedback (logits)
    3. Student learns from teacher's correction
    """
    # Generate from student
    student_outputs = student.generate(prompts)
    
    # Get teacher's logits for student's generations
    with torch.no_grad():
        teacher_logits = teacher(student_outputs).logits
    
    # Train student to match teacher on its own outputs
    student_logits = student(student_outputs).logits
    loss = kl_div(student_logits, teacher_logits)
    
    return loss
```

**Benefits:**
- Reduces distribution shift
- Student learns from its own mistakes
- Better long-term performance

**Drawbacks:**
- More complex implementation
- Requires more compute
- Slower training

### 3. Task-Specific Distillation

Optimize for specific coding tasks:

**Code completion:**
```python
# Focus on next-token prediction
# Use low temperature (sharper distributions)
config = {
    'temperature': 1.5,
    'alpha': 0.8,  # High distillation weight
    'beta': 0.2,
    'use_feature_distillation': False  # Speed over accuracy
}
```

**Bug fixing:**
```python
# Focus on understanding code structure
# Use feature distillation heavily
config = {
    'temperature': 2.0,
    'alpha': 0.5,
    'beta': 0.3,
    'gamma': 0.7,  # High feature weight
    'use_attention_distillation': True
}
```

**Code translation:**
```python
# Balance between languages
# Use multi-teacher if available
config = {
    'temperature': 2.5,
    'alpha': 0.7,
    'beta': 0.3,
    'use_multi_teacher': True,
    'teachers': ['python_expert', 'java_expert']
}
```

### 4. Progressive Distillation

Train student in stages:

```python
def progressive_distillation(student, teachers):
    """
    Stage 1: Distill from smallest teacher
    Stage 2: Distill from medium teacher  
    Stage 3: Distill from largest teacher
    """
    for teacher in teachers:
        print(f"Training with teacher: {teacher.name}")
        train_distillation(student, teacher, epochs=1)
        
        # Evaluate
        performance = evaluate(student)
        print(f"Performance: {performance}")
```

**Benefits:**
- Easier optimization (smaller gap per stage)
- Better final performance
- More stable training

## Evaluation

### 1. Code Generation Quality

```python
from datasets import load_dataset

# HumanEval benchmark
def evaluate_humaneval(model, tokenizer):
    """Test on HumanEval coding problems"""
    dataset = load_dataset("openai_humaneval")
    
    correct = 0
    total = 0
    
    for example in dataset['test']:
        prompt = example['prompt']
        test_cases = example['test']
        
        # Generate solution
        generated = generate_code(model, tokenizer, prompt)
        
        # Run test cases
        try:
            exec(generated + '\n' + test_cases)
            correct += 1
        except:
            pass
        
        total += 1
    
    pass_at_1 = correct / total
    return pass_at_1

# Run evaluation
score = evaluate_humaneval(student_model, tokenizer)
print(f"HumanEval Pass@1: {score:.2%}")
```

### 2. Code Understanding Tasks

```python
# CodeXGLUE benchmarks
tasks = [
    'code_search',      # Find code from description
    'code_clone',       # Detect duplicate code
    'defect_detection', # Find bugs
    'code_to_code',     # Translation
]

for task in tasks:
    score = evaluate_codexglue(student_model, task)
    print(f"{task}: {score:.2%}")
```

### 3. Perplexity on Code Corpus

```python
def calculate_perplexity(model, dataloader):
    """Lower is better"""
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            
            # Count non-padding tokens
            mask = batch['attention_mask']
            num_tokens = mask.sum()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()
```

### 4. Performance Comparison

```python
import time

def benchmark_inference(model, prompts, num_runs=100):
    """Measure inference speed"""
    start = time.time()
    
    for _ in range(num_runs):
        model.generate(prompts, max_new_tokens=50)
    
    end = time.time()
    avg_time = (end - start) / num_runs
    return avg_time

teacher_time = benchmark_inference(teacher_model, test_prompts)
student_time = benchmark_inference(student_model, test_prompts)

print(f"Teacher: {teacher_time:.3f}s per generation")
print(f"Student: {student_time:.3f}s per generation")
print(f"Speedup: {teacher_time/student_time:.1f}x")
```

## Troubleshooting

### Problem: Student not learning

**Symptoms:**
- KL loss not decreasing
- Student outputs random/nonsensical code

**Solutions:**
1. Lower temperature (try 1.5-2.0)
2. Increase alpha (distillation weight)
3. Check data quality
4. Reduce learning rate
5. Add warmup steps

### Problem: Student overfits to teacher

**Symptoms:**
- Train loss low, eval loss high
- Student mimics teacher's mistakes
- Poor generalization

**Solutions:**
1. Increase beta (ground truth weight)
2. Add regularization (dropout, weight decay)
3. Use more diverse training data
4. Reduce training epochs
5. Use validation-based early stopping

### Problem: OOM (Out of Memory)

**Solutions:**
1. **Reduce batch size:**
   ```python
   batch_size = 2
   gradient_accumulation_steps = 8
   ```

2. **Enable gradient checkpointing:**
   ```python
   student_model.gradient_checkpointing_enable()
   ```

3. **Quantize teacher further:**
   ```python
   # 8-bit instead of 4-bit
   load_in_8bit=True
   ```

4. **Disable feature distillation:**
   ```python
   use_feature_distillation = False
   ```

5. **Use CPU offloading:**
   ```python
   device_map = "balanced"  # Spread across GPU/CPU
   ```

### Problem: Training too slow

**Solutions:**
1. **Use compiled model (PyTorch 2.0+):**
   ```python
   student_model = torch.compile(student_model)
   ```

2. **Optimize data loading:**
   ```python
   num_workers = 4
   pin_memory = True
   persistent_workers = True
   ```

3. **Reduce feature matching:**
   ```python
   feature_matching_layers = [0, 16, 31]  # Just 3 layers
   ```

4. **Use larger batch size:**
   ```python
   batch_size = 8
   gradient_accumulation_steps = 2
   ```

### Problem: Poor code quality

**Symptoms:**
- Syntax errors
- Incomplete functions
- Logic errors

**Solutions:**
1. **Increase max_length:**
   ```python
   max_length = 1024  # Allow longer completions
   ```

2. **Filter training data:**
   - Remove examples with syntax errors
   - Keep only test-passing examples

3. **Use task-specific fine-tuning after distillation:**
   ```python
   # First distill, then fine-tune on high-quality data
   ```

4. **Adjust generation parameters:**
   ```python
   temperature = 0.7  # Lower for more conservative output
   top_p = 0.95       # Nucleus sampling
   repetition_penalty = 1.1
   ```

## Resources

### Datasets
- **CodeAlpaca-20K**: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k
- **MBPP**: https://huggingface.co/datasets/mbpp
- **HumanEval**: https://huggingface.co/datasets/openai_humaneval
- **CodeSearchNet**: https://huggingface.co/datasets/code_search_net
- **APPS**: https://huggingface.co/datasets/codeparrot/apps

### Pre-trained Models
- **CodeLlama**: https://huggingface.co/codellama
- **DeepSeek-Coder**: https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct
- **StarCoder2**: https://huggingface.co/bigcode/starcoder2-15b
- **CodeGen**: https://huggingface.co/Salesforce/codegen-16B-mono

### Papers
- **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
- **MiniLLM** (Gu et al., 2023)
- **Self-Paced Knowledge Distillation** (Chen et al., 2023)
- **An Empirical Study of KD for Code** (Wang et al., 2025)

### Tools
- **Transformers**: https://github.com/huggingface/transformers
- **Accelerate**: https://github.com/huggingface/accelerate
- **BitsAndBytes**: https://github.com/TimDettmers/bitsandbytes
- **PEFT**: https://github.com/huggingface/peft

---

## Quick Start Checklist

- [ ] Install dependencies
- [ ] Check GPU memory (need 40-80GB for large teachers)
- [ ] Load and prepare dataset
- [ ] Configure quantization for teacher
- [ ] Set distillation hyperparameters
- [ ] Start training with small subset (validate setup)
- [ ] Monitor losses (KL, CE, feature)
- [ ] Run evaluation on validation set
- [ ] Full training run
- [ ] Benchmark final model (speed & quality)
- [ ] Save and deploy

**Estimated time:** 2-4 hours for 10K examples with CodeLlama models on A100 80GB
