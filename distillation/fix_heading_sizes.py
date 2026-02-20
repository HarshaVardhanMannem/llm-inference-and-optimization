"""Fix heading sizes for consistency and readability"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fixed styling with proper heading sizes
fixed_styling = {
    'title': '''# 🎓 BERT to DistilBERT Knowledge Distillation Tutorial

<div style="text-align: center; padding: 20px 0; margin-bottom: 30px; border-bottom: 3px solid #667eea;">
<h1 style="color: #667eea; font-size: 2em; margin: 0; font-weight: bold; line-height: 1.2;">🎓 BERT to DistilBERT</h1>
<h2 style="color: #764ba2; font-size: 1.5em; margin: 10px 0 0 0; font-weight: 600; line-height: 1.3;">Knowledge Distillation Tutorial</h2>
</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h2 style="color: white; margin-top: 0; font-size: 1.3em; font-weight: 600;">📚 Overview</h2>
<p style="font-size: 1em; line-height: 1.6; color: white; margin-bottom: 0;">
This notebook demonstrates <strong>Knowledge Distillation</strong> - a powerful technique to compress a large, accurate model (teacher) into a smaller, faster model (student) while preserving performance.
</p>
</div>

## 🎯 What is Knowledge Distillation?

<div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">👨‍🏫 Teacher Model</strong>: Large, pre-trained BERT model fine-tuned on SST-2 (sentiment analysis)</li>
<li style="color: #212529;"><strong style="color: #000;">👨‍🎓 Student Model</strong>: Smaller DistilBERT model that learns from the teacher</li>
<li style="color: #212529;"><strong style="color: #000;">🎯 Goal</strong>: Transfer the teacher's knowledge to the student, achieving similar accuracy with ~60% fewer parameters</li>
</ul>
</div>

## 🔑 Key Concepts

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-top: 3px solid #2196f3;">
<h3 style="margin-top: 0; color: #1976d2; font-size: 1.1em; font-weight: 600;">🌡️ Temperature Scaling</h3>
<p style="margin-bottom: 0; color: #212529; font-size: 0.95em;">Softens probability distributions to reveal "dark knowledge"</p>
</div>
<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-top: 3px solid #9c27b0;">
<h3 style="margin-top: 0; color: #7b1fa2; font-size: 1.1em; font-weight: 600;">📊 KL Divergence Loss</h3>
<p style="margin-bottom: 0; color: #212529; font-size: 0.95em;">Measures how well student matches teacher's predictions</p>
</div>
<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-top: 3px solid #ff9800;">
<h3 style="margin-top: 0; color: #e65100; font-size: 1.1em; font-weight: 600;">⚖️ Combined Loss</h3>
<p style="margin-bottom: 0; color: #212529; font-size: 0.95em;">Balances hard labels (ground truth) and soft labels (teacher predictions)</p>
</div>
</div>

## 📋 Notebook Structure

<div style="background-color: #fff; border: 2px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">🔧 Setup</strong>: Install dependencies and load models</li>
<li style="color: #212529;"><strong style="color: #000;">🧠 Distillation</strong>: Custom trainer implementing knowledge distillation loss</li>
<li style="color: #212529;"><strong style="color: #000;">🚀 Training</strong>: Train student model using teacher's soft predictions</li>
<li style="color: #212529;"><strong style="color: #000;">📈 Evaluation</strong>: Benchmark teacher vs student performance</li>
<li style="color: #212529;"><strong style="color: #000;">☁️ Deployment</strong>: Upload distilled model to Hugging Face Hub</li>
</ol>
</div>

<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 20px 0;">
<strong style="color: #155724;">💡 Tip:</strong> <span style="color: #155724;">Run cells sequentially to complete the distillation pipeline.</span>
</div>
''',
    
    'step1': '''## 📥 Step 1: Load Teacher and Student Models

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; font-size: 1.2em; font-weight: 600;">👨‍🏫 Teacher Model: BERT-base-uncased (Fine-tuned on SST-2)</h3>
</div>

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">Model</strong>: <code style="background: #fff; padding: 2px 6px; border-radius: 3px; color: #d32f2f;">textattack/bert-base-uncased-SST-2</code></li>
<li style="color: #212529;"><strong style="color: #000;">Parameters</strong>: <span style="color: #d32f2f; font-weight: bold;">~110M parameters</span></li>
<li style="color: #212529;"><strong style="color: #000;">Purpose</strong>: Pre-trained and fine-tuned on Stanford Sentiment Treebank (SST-2)</li>
<li style="color: #212529;"><strong style="color: #000;">Role</strong>: Provides "soft labels" (probability distributions) instead of just hard labels</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; font-size: 1.2em; font-weight: 600;">👨‍🎓 Student Model: DistilBERT-base-uncased</h3>
</div>

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">Model</strong>: <code style="background: #fff; padding: 2px 6px; border-radius: 3px; color: #388e3c;">distilbert-base-uncased</code></li>
<li style="color: #212529;"><strong style="color: #000;">Parameters</strong>: <span style="color: #388e3c; font-weight: bold;">~67M parameters</span> (<span style="color: #d32f2f;">~60% smaller</span>)</li>
<li style="color: #212529;"><strong style="color: #000;">Purpose</strong>: Smaller, faster model that will learn from teacher</li>
<li style="color: #212529;"><strong style="color: #000;">Initialization</strong>: Starts with generic pre-trained weights, not fine-tuned</li>
</ul>
</div>

### 🤔 Why This Pair?

<div style="background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px; color: #212529;">
<li>DistilBERT is architecturally similar to BERT but with fewer layers</li>
<li>Both use the same tokenizer, making knowledge transfer easier</li>
<li>Size reduction enables <strong style="color: #000;">faster inference</strong> and <strong style="color: #000;">lower memory usage</strong></li>
</ul>
</div>
''',
    
    'step2': '''## 🛠️ Step 2: Custom Distillation Trainer

<div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border: 2px solid #ff9800; margin: 20px 0;">
<h3 style="color: #e65100; margin-top: 0; font-size: 1.3em; font-weight: 600;">DistillationTrainer Class</h3>
<p style="color: #212529; margin-bottom: 0;">This custom trainer extends Hugging Face's <code style="background: #f5f5f5; padding: 2px 6px; border-radius: 3px; color: #000;">Trainer</code> to implement knowledge distillation.</p>
</div>

### 🔧 Key Components:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0; font-size: 1.1em; font-weight: 600;">🌡️ Temperature Scaling</h4>
<p style="margin-bottom: 5px; color: white; font-size: 0.9em;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px; color: white;">temperature=2.0</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em; color: white;">
<li>Divides logits by temperature before softmax</li>
<li>Higher temperature = softer probability distributions</li>
<li>Reveals relationships between classes</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0; font-size: 1.1em; font-weight: 600;">📊 KL Divergence Loss</h4>
<p style="margin-bottom: 5px; color: white; font-size: 0.9em;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px; color: white;">loss_distill</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em; color: white;">
<li>Measures student-teacher distribution match</li>
<li>Formula: <code style="background: rgba(255,255,255,0.2); color: white;">KL(student_softmax || teacher_softmax)</code></li>
<li>Multiplied by <code style="background: rgba(255,255,255,0.2); color: white;">temperature²</code> to scale back</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0; font-size: 1.1em; font-weight: 600;">⚖️ Combined Loss</h4>
<p style="margin-bottom: 5px; color: white; font-size: 0.9em;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px; color: white;">alpha=0.5</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em; color: white;">
<li><code style="background: rgba(255,255,255,0.2); color: white;">loss = α × loss_ce + (1-α) × loss_distill</code></li>
<li>Balances hard labels vs teacher predictions</li>
<li>Typical range: 0.3-0.7</li>
</ul>
</div>
</div>

<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 20px 0;">
<strong style="color: #155724;">🔒 Why Freeze Teacher?</strong> <span style="color: #155724;">Teacher weights are frozen - only student learns. Teacher provides guidance without being modified.</span>
</div>
''',
    
    'step3': '''## 🚀 Step 3: Training with Knowledge Distillation

<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
<h3 style="margin-top: 0; color: #212529; font-size: 1.3em; font-weight: 600;">📊 Dataset: SST-2 (Stanford Sentiment Treebank)</h3>
<ul style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">Task</strong>: Binary sentiment classification (positive/negative)</li>
<li style="color: #212529;"><strong style="color: #000;">Samples</strong>: ~67K training, 872 validation, 1.8K test</li>
<li style="color: #212529;"><strong style="color: #000;">Format</strong>: Movie review sentences with sentiment labels</li>
</ul>
</div>

### 🔄 Training Process:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">Tokenization</strong>: Convert sentences to token IDs (max_length=128)</li>
<li style="color: #212529;"><strong style="color: #000;">Forward Pass</strong>: 
   <ul style="color: #212529;">
   <li>Student processes inputs → student logits</li>
   <li>Teacher processes inputs (no gradients) → teacher logits</li>
   </ul>
</li>
<li style="color: #212529;"><strong style="color: #000;">Loss Calculation</strong>:
   <ul style="color: #212529;">
   <li>Apply temperature scaling to both logits</li>
   <li>Compute KL divergence between distributions</li>
   <li>Compute cross-entropy with ground truth</li>
   <li>Combine losses with alpha weighting</li>
   </ul>
</li>
<li style="color: #212529;"><strong style="color: #000;">Backward Pass</strong>: Update only student model weights</li>
</ol>
</div>

### ⚙️ Training Configuration:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #1976d2;">3</div>
<div style="font-size: 0.9em; color: #212529;">Epochs</div>
</div>
<div style="background: #f3e5f5; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #7b1fa2;">32</div>
<div style="font-size: 0.9em; color: #212529;">Batch Size</div>
</div>
<div style="background: #fff3e0; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #e65100;">2e-5</div>
<div style="font-size: 0.9em; color: #212529;">Learning Rate</div>
</div>
<div style="background: #e8f5e9; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #388e3c;">2.0</div>
<div style="font-size: 0.9em; color: #212529;">Temperature</div>
</div>
</div>

### 📈 Expected Results:

<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;">
<ul style="margin: 0; padding-left: 20px; color: #155724;">
<li>Student should achieve <strong style="color: #155724;">~90-92% accuracy</strong> (close to teacher's ~93%)</li>
<li>Model size: <strong style="color: #155724;">~67M params</strong> vs teacher's ~110M params</li>
<li>Inference speed: <strong style="color: #155724;">~2x faster</strong> than teacher</li>
</ul>
</div>
''',
    
    'step4': '''## 📊 Step 4: Evaluation and Benchmarking

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; font-size: 1.2em; font-weight: 600;">Benchmarking Function</h3>
<p style="margin-bottom: 0; color: white; font-size: 0.95em;">Compares multiple model variants across key metrics</p>
</div>

### 📏 Metrics Measured:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">📈</div>
<div style="color: #212529; font-weight: 600;">Accuracy</div>
<div style="font-size: 0.85em; color: #666;">Classification accuracy</div>
</div>
<div style="background: #f3e5f5; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">💾</div>
<div style="color: #212529; font-weight: 600;">Model Size</div>
<div style="font-size: 0.85em; color: #666;">Disk size in MB</div>
</div>
<div style="background: #fff3e0; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">⚡</div>
<div style="color: #212529; font-weight: 600;">Latency</div>
<div style="font-size: 0.85em; color: #666;">Inference time (ms)</div>
</div>
</div>

### 🔬 Models Evaluated:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">👨‍🏫 Teacher (BERT)</strong>: Baseline - original fine-tuned model</li>
<li style="color: #212529;"><strong style="color: #000;">👨‍🎓 Student (FP32)</strong>: Distilled model in full precision</li>
<li style="color: #212529;"><strong style="color: #000;">⚡ Student (INT8)</strong>: Quantized to 8-bit integers for CPU inference</li>
<li style="color: #212529;"><strong style="color: #000;">💾 Student (4-bit)</strong>: Quantized using BitsAndBytes for GPU memory efficiency</li>
<li style="color: #212529;"><strong style="color: #000;">🔬 DistilBERT (Raw)</strong>: Untrained baseline (should be ~50% accuracy)</li>
</ol>
</div>

<div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 20px 0;">
<strong style="color: #856404;">🔧 Key Fix:</strong> <span style="color: #856404;">DistilBERT doesn't use <code style="background: #fff; padding: 2px 6px; border-radius: 3px; color: #000;">token_type_ids</code> (unlike BERT). Must remove this input to avoid errors.</span>
</div>

### 📊 Expected Performance:

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
<tr style="background-color: #667eea; color: white;">
<th style="padding: 10px; text-align: left; color: white; font-weight: 600;">Model</th>
<th style="padding: 10px; text-align: center; color: white; font-weight: 600;">Accuracy</th>
<th style="padding: 10px; text-align: center; color: white; font-weight: 600;">Size</th>
<th style="padding: 10px; text-align: center; color: white; font-weight: 600;">Speed</th>
</tr>
<tr style="background-color: #f5f5f5;">
<td style="padding: 10px; color: #212529;"><strong>Teacher</strong></td>
<td style="padding: 10px; text-align: center; color: #212529;">~93%</td>
<td style="padding: 10px; text-align: center; color: #212529;">~440MB</td>
<td style="padding: 10px; text-align: center; color: #212529;">Baseline</td>
</tr>
<tr>
<td style="padding: 10px; color: #212529;"><strong>Student (FP32)</strong></td>
<td style="padding: 10px; text-align: center; color: #212529;">~90-92%</td>
<td style="padding: 10px; text-align: center; color: #212529;">~268MB</td>
<td style="padding: 10px; text-align: center; color: #212529;">~2x faster</td>
</tr>
<tr style="background-color: #f5f5f5;">
<td style="padding: 10px; color: #212529;"><strong>Student (INT8)</strong></td>
<td style="padding: 10px; text-align: center; color: #212529;">Similar</td>
<td style="padding: 10px; text-align: center; color: #212529;">~67MB</td>
<td style="padding: 10px; text-align: center; color: #212529;">Fastest CPU</td>
</tr>
<tr>
<td style="padding: 10px; color: #212529;"><strong>Student (4-bit)</strong></td>
<td style="padding: 10px; text-align: center; color: #212529;">Similar</td>
<td style="padding: 10px; text-align: center; color: #212529;">~34MB</td>
<td style="padding: 10px; text-align: center; color: #212529;">Lowest VRAM</td>
</tr>
</table>
''',
    
    'step5': '''## ☁️ Step 5: Upload to Hugging Face Hub

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h3 style="color: white; margin-top: 0; font-size: 1.3em; font-weight: 600;">Model Deployment</h3>
<p style="color: white; margin-bottom: 0;">After successful distillation, upload the student model to Hugging Face Hub for:</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-top: 3px solid #2196f3;">
<h4 style="margin-top: 0; color: #1976d2; font-size: 1.1em; font-weight: 600;">🔗 Sharing</h4>
<p style="margin-bottom: 0; color: #212529;">Make model publicly available</p>
</div>
<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-top: 3px solid #9c27b0;">
<h4 style="margin-top: 0; color: #7b1fa2; font-size: 1.1em; font-weight: 600;">📝 Versioning</h4>
<p style="margin-bottom: 0; color: #212529;">Track model iterations</p>
</div>
<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-top: 3px solid #ff9800;">
<h4 style="margin-top: 0; color: #e65100; font-size: 1.1em; font-weight: 600;">🔌 Integration</h4>
<p style="margin-bottom: 0; color: #212529;">Easy loading with <code style="background: #f5f5f5; padding: 2px 6px; border-radius: 3px; color: #000;">from_pretrained()</code></p>
</div>
<div style="background: #e8f5e9; padding: 15px; border-radius: 8px; border-top: 3px solid #4caf50;">
<h4 style="margin-top: 0; color: #388e3c; font-size: 1.1em; font-weight: 600;">📚 Documentation</h4>
<p style="margin-bottom: 0; color: #212529;">Add model card with performance metrics</p>
</div>
</div>

### 📋 Upload Process:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px; color: #212529;">
<li style="color: #212529;"><strong style="color: #000;">🔐 Login</strong>: Authenticate with Hugging Face token</li>
<li style="color: #212529;"><strong style="color: #000;">⬆️ Push</strong>: Upload model files (config.json, model weights, tokenizer)</li>
<li style="color: #212529;"><strong style="color: #000;">✅ Verify</strong>: Check model appears on your Hugging Face profile</li>
</ol>
</div>

### 📝 Model Card Best Practices:

<div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0;">
<ul style="margin: 0; padding-left: 20px; color: #155724;">
<li>Document distillation parameters (temperature, alpha)</li>
<li>Report accuracy metrics (teacher vs student)</li>
<li>Include inference benchmarks</li>
<li>Note model size and speed improvements</li>
<li>Specify use cases and limitations</li>
</ul>
</div>
'''
}

# Update markdown cells with fixed heading sizes
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        
        # Identify which explanation this is
        if 'BERT to DistilBERT Knowledge Distillation Tutorial' in src and 'Overview' in src:
            cell['source'] = fixed_styling['title'].split('\n')
        elif 'Step 1: Load Teacher and Student Models' in src:
            cell['source'] = fixed_styling['step1'].split('\n')
        elif 'Step 2: Custom Distillation Trainer' in src:
            cell['source'] = fixed_styling['step2'].split('\n')
        elif 'Step 3: Training with Knowledge Distillation' in src:
            cell['source'] = fixed_styling['step3'].split('\n')
        elif 'Step 4: Evaluation and Benchmarking' in src:
            cell['source'] = fixed_styling['step4'].split('\n')
        elif 'Step 5: Upload to Hugging Face Hub' in src:
            cell['source'] = fixed_styling['step5'].split('\n')

# Save
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Fixed heading sizes!')
print('  - H1: 2em (main title)')
print('  - H2: 1.5em (subtitle), 1.3em (section headers)')
print('  - H3: 1.2-1.3em (subsection headers)')
print('  - H4: 1.1em (card headers)')
print('  - Consistent font-weight: 600 for headers')
print('  - Proper line-height for readability')
