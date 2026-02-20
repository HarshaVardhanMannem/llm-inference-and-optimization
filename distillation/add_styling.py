"""Add styling and visual improvements to notebook markdown cells"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Enhanced markdown content with styling
styled_content = {
    'title': '''# 🎓 BERT to DistilBERT Knowledge Distillation Tutorial

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h2 style="color: white; margin-top: 0;">📚 Overview</h2>
<p style="font-size: 1.1em; line-height: 1.6;">
This notebook demonstrates <strong>Knowledge Distillation</strong> - a powerful technique to compress a large, accurate model (teacher) into a smaller, faster model (student) while preserving performance.
</p>
</div>

## 🎯 What is Knowledge Distillation?

<div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li><strong>👨‍🏫 Teacher Model</strong>: Large, pre-trained BERT model fine-tuned on SST-2 (sentiment analysis)</li>
<li><strong>👨‍🎓 Student Model</strong>: Smaller DistilBERT model that learns from the teacher</li>
<li><strong>🎯 Goal</strong>: Transfer the teacher's knowledge to the student, achieving similar accuracy with ~60% fewer parameters</li>
</ul>
</div>

## 🔑 Key Concepts

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-top: 3px solid #2196f3;">
<h3 style="margin-top: 0; color: #1976d2;">🌡️ Temperature Scaling</h3>
<p style="margin-bottom: 0;">Softens probability distributions to reveal "dark knowledge"</p>
</div>
<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-top: 3px solid #9c27b0;">
<h3 style="margin-top: 0; color: #7b1fa2;">📊 KL Divergence Loss</h3>
<p style="margin-bottom: 0;">Measures how well student matches teacher's predictions</p>
</div>
<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-top: 3px solid #ff9800;">
<h3 style="margin-top: 0; color: #e65100;">⚖️ Combined Loss</h3>
<p style="margin-bottom: 0;">Balances hard labels (ground truth) and soft labels (teacher predictions)</p>
</div>
</div>

## 📋 Notebook Structure

<div style="background-color: #fff; border: 2px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px;">
<li><strong>🔧 Setup</strong>: Install dependencies and load models</li>
<li><strong>🧠 Distillation</strong>: Custom trainer implementing knowledge distillation loss</li>
<li><strong>🚀 Training</strong>: Train student model using teacher's soft predictions</li>
<li><strong>📈 Evaluation</strong>: Benchmark teacher vs student performance</li>
<li><strong>☁️ Deployment</strong>: Upload distilled model to Hugging Face Hub</li>
</ol>
</div>

<div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 12px; margin: 20px 0;">
<strong>💡 Tip:</strong> Run cells sequentially to complete the distillation pipeline.
</div>
''',
    
    'step1': '''## 📥 Step 1: Load Teacher and Student Models

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0;">👨‍🏫 Teacher Model: BERT-base-uncased (Fine-tuned on SST-2)</h3>
</div>

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li><strong>Model</strong>: <code style="background: #fff; padding: 2px 6px; border-radius: 3px;">textattack/bert-base-uncased-SST-2</code></li>
<li><strong>Parameters</strong>: <span style="color: #d32f2f; font-weight: bold;">~110M parameters</span></li>
<li><strong>Purpose</strong>: Pre-trained and fine-tuned on Stanford Sentiment Treebank (SST-2)</li>
<li><strong>Role</strong>: Provides "soft labels" (probability distributions) instead of just hard labels</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0;">👨‍🎓 Student Model: DistilBERT-base-uncased</h3>
</div>

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li><strong>Model</strong>: <code style="background: #fff; padding: 2px 6px; border-radius: 3px;">distilbert-base-uncased</code></li>
<li><strong>Parameters</strong>: <span style="color: #388e3c; font-weight: bold;">~67M parameters</span> (<span style="color: #d32f2f;">~60% smaller</span>)</li>
<li><strong>Purpose</strong>: Smaller, faster model that will learn from teacher</li>
<li><strong>Initialization</strong>: Starts with generic pre-trained weights, not fine-tuned</li>
</ul>
</div>

### 🤔 Why This Pair?

<div style="background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 15px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li>DistilBERT is architecturally similar to BERT but with fewer layers</li>
<li>Both use the same tokenizer, making knowledge transfer easier</li>
<li>Size reduction enables <strong>faster inference</strong> and <strong>lower memory usage</strong></li>
</ul>
</div>
''',
    
    'step2': '''## 🛠️ Step 2: Custom Distillation Trainer

<div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border: 2px solid #ff9800; margin: 20px 0;">
<h3 style="color: #e65100; margin-top: 0;">DistillationTrainer Class</h3>
<p>This custom trainer extends Hugging Face's <code>Trainer</code> to implement knowledge distillation.</p>
</div>

### 🔧 Key Components:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0;">🌡️ Temperature Scaling</h4>
<p style="margin-bottom: 5px;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">temperature=2.0</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
<li>Divides logits by temperature before softmax</li>
<li>Higher temperature = softer probability distributions</li>
<li>Reveals relationships between classes</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0;">📊 KL Divergence Loss</h4>
<p style="margin-bottom: 5px;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">loss_distill</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
<li>Measures student-teacher distribution match</li>
<li>Formula: <code>KL(student_softmax || teacher_softmax)</code></li>
<li>Multiplied by <code>temperature²</code> to scale back</li>
</ul>
</div>

<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 15px; border-radius: 8px; color: white;">
<h4 style="color: white; margin-top: 0;">⚖️ Combined Loss</h4>
<p style="margin-bottom: 5px;"><code style="background: rgba(255,255,255,0.2); padding: 2px 6px; border-radius: 3px;">alpha=0.5</code></p>
<ul style="margin: 0; padding-left: 20px; font-size: 0.9em;">
<li><code>loss = α × loss_ce + (1-α) × loss_distill</code></li>
<li>Balances hard labels vs teacher predictions</li>
<li>Typical range: 0.3-0.7</li>
</ul>
</div>
</div>

<div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 12px; margin: 20px 0;">
<strong>🔒 Why Freeze Teacher?</strong> Teacher weights are frozen - only student learns. Teacher provides guidance without being modified.
</div>
''',
    
    'step3': '''## 🚀 Step 3: Training with Knowledge Distillation

<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
<h3 style="margin-top: 0;">📊 Dataset: SST-2 (Stanford Sentiment Treebank)</h3>
<ul style="margin: 0; padding-left: 20px;">
<li><strong>Task</strong>: Binary sentiment classification (positive/negative)</li>
<li><strong>Samples</strong>: ~67K training, 872 validation, 1.8K test</li>
<li><strong>Format</strong>: Movie review sentences with sentiment labels</li>
</ul>
</div>

### 🔄 Training Process:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px;">
<li><strong>Tokenization</strong>: Convert sentences to token IDs (max_length=128)</li>
<li><strong>Forward Pass</strong>: 
   <ul>
   <li>Student processes inputs → student logits</li>
   <li>Teacher processes inputs (no gradients) → teacher logits</li>
   </ul>
</li>
<li><strong>Loss Calculation</strong>:
   <ul>
   <li>Apply temperature scaling to both logits</li>
   <li>Compute KL divergence between distributions</li>
   <li>Compute cross-entropy with ground truth</li>
   <li>Combine losses with alpha weighting</li>
   </ul>
</li>
<li><strong>Backward Pass</strong>: Update only student model weights</li>
</ol>
</div>

### ⚙️ Training Configuration:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #1976d2;">3</div>
<div style="font-size: 0.9em;">Epochs</div>
</div>
<div style="background: #f3e5f5; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #7b1fa2;">32</div>
<div style="font-size: 0.9em;">Batch Size</div>
</div>
<div style="background: #fff3e0; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #e65100;">2e-5</div>
<div style="font-size: 0.9em;">Learning Rate</div>
</div>
<div style="background: #e8f5e9; padding: 10px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.5em; font-weight: bold; color: #388e3c;">2.0</div>
<div style="font-size: 0.9em;">Temperature</div>
</div>
</div>

### 📈 Expected Results:

<div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 20px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li>Student should achieve <strong>~90-92% accuracy</strong> (close to teacher's ~93%)</li>
<li>Model size: <strong>~67M params</strong> vs teacher's ~110M params</li>
<li>Inference speed: <strong>~2x faster</strong> than teacher</li>
</ul>
</div>
''',
    
    'step4': '''## 📊 Step 4: Evaluation and Benchmarking

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0;">Benchmarking Function</h3>
<p style="margin-bottom: 0;">Compares multiple model variants across key metrics</p>
</div>

### 📏 Metrics Measured:

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">📈</div>
<div><strong>Accuracy</strong></div>
<div style="font-size: 0.85em; color: #666;">Classification accuracy</div>
</div>
<div style="background: #f3e5f5; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">💾</div>
<div><strong>Model Size</strong></div>
<div style="font-size: 0.85em; color: #666;">Disk size in MB</div>
</div>
<div style="background: #fff3e0; padding: 12px; border-radius: 5px; text-align: center;">
<div style="font-size: 1.2em; font-weight: bold;">⚡</div>
<div><strong>Latency</strong></div>
<div style="font-size: 0.85em; color: #666;">Inference time (ms)</div>
</div>
</div>

### 🔬 Models Evaluated:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px;">
<li><strong>👨‍🏫 Teacher (BERT)</strong>: Baseline - original fine-tuned model</li>
<li><strong>👨‍🎓 Student (FP32)</strong>: Distilled model in full precision</li>
<li><strong>⚡ Student (INT8)</strong>: Quantized to 8-bit integers for CPU inference</li>
<li><strong>💾 Student (4-bit)</strong>: Quantized using BitsAndBytes for GPU memory efficiency</li>
<li><strong>🔬 DistilBERT (Raw)</strong>: Untrained baseline (should be ~50% accuracy)</li>
</ol>
</div>

<div style="background-color: #fff3e0; border-left: 4px solid #ff9800; padding: 12px; margin: 20px 0;">
<strong>🔧 Key Fix:</strong> DistilBERT doesn't use <code>token_type_ids</code> (unlike BERT). Must remove this input to avoid errors.
</div>

### 📊 Expected Performance:

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
<tr style="background-color: #667eea; color: white;">
<th style="padding: 10px; text-align: left;">Model</th>
<th style="padding: 10px; text-align: center;">Accuracy</th>
<th style="padding: 10px; text-align: center;">Size</th>
<th style="padding: 10px; text-align: center;">Speed</th>
</tr>
<tr style="background-color: #f5f5f5;">
<td><strong>Teacher</strong></td>
<td style="text-align: center;">~93%</td>
<td style="text-align: center;">~440MB</td>
<td style="text-align: center;">Baseline</td>
</tr>
<tr>
<td><strong>Student (FP32)</strong></td>
<td style="text-align: center;">~90-92%</td>
<td style="text-align: center;">~268MB</td>
<td style="text-align: center;">~2x faster</td>
</tr>
<tr style="background-color: #f5f5f5;">
<td><strong>Student (INT8)</strong></td>
<td style="text-align: center;">Similar</td>
<td style="text-align: center;">~67MB</td>
<td style="text-align: center;">Fastest CPU</td>
</tr>
<tr>
<td><strong>Student (4-bit)</strong></td>
<td style="text-align: center;">Similar</td>
<td style="text-align: center;">~34MB</td>
<td style="text-align: center;">Lowest VRAM</td>
</tr>
</table>
''',
    
    'step5': '''## ☁️ Step 5: Upload to Hugging Face Hub

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h3 style="color: white; margin-top: 0;">Model Deployment</h3>
<p>After successful distillation, upload the student model to Hugging Face Hub for:</p>
</div>

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
<div style="background: #e3f2fd; padding: 15px; border-radius: 8px; border-top: 3px solid #2196f3;">
<h4 style="margin-top: 0; color: #1976d2;">🔗 Sharing</h4>
<p style="margin-bottom: 0;">Make model publicly available</p>
</div>
<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-top: 3px solid #9c27b0;">
<h4 style="margin-top: 0; color: #7b1fa2;">📝 Versioning</h4>
<p style="margin-bottom: 0;">Track model iterations</p>
</div>
<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-top: 3px solid #ff9800;">
<h4 style="margin-top: 0; color: #e65100;">🔌 Integration</h4>
<p style="margin-bottom: 0;">Easy loading with <code>from_pretrained()</code></p>
</div>
<div style="background: #e8f5e9; padding: 15px; border-radius: 8px; border-top: 3px solid #4caf50;">
<h4 style="margin-top: 0; color: #388e3c;">📚 Documentation</h4>
<p style="margin-bottom: 0;">Add model card with performance metrics</p>
</div>
</div>

### 📋 Upload Process:

<div style="background-color: #f5f5f5; padding: 15px; border-radius: 8px; margin: 15px 0;">
<ol style="margin: 0; padding-left: 20px;">
<li><strong>🔐 Login</strong>: Authenticate with Hugging Face token</li>
<li><strong>⬆️ Push</strong>: Upload model files (config.json, model weights, tokenizer)</li>
<li><strong>✅ Verify</strong>: Check model appears on your Hugging Face profile</li>
</ol>
</div>

### 📝 Model Card Best Practices:

<div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 15px; margin: 20px 0;">
<ul style="margin: 0; padding-left: 20px;">
<li>Document distillation parameters (temperature, alpha)</li>
<li>Report accuracy metrics (teacher vs student)</li>
<li>Include inference benchmarks</li>
<li>Note model size and speed improvements</li>
<li>Specify use cases and limitations</li>
</ul>
</div>
'''
}

# Update markdown cells with styled content
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        
        # Identify which explanation this is
        if 'BERT to DistilBERT Knowledge Distillation Tutorial' in src and 'Overview' in src:
            cell['source'] = styled_content['title'].split('\n')
        elif 'Step 1: Load Teacher and Student Models' in src:
            cell['source'] = styled_content['step1'].split('\n')
        elif 'Step 2: Custom Distillation Trainer' in src:
            cell['source'] = styled_content['step2'].split('\n')
        elif 'Step 3: Training with Knowledge Distillation' in src:
            cell['source'] = styled_content['step3'].split('\n')
        elif 'Step 4: Evaluation and Benchmarking' in src:
            cell['source'] = styled_content['step4'].split('\n')
        elif 'Step 5: Upload to Hugging Face Hub' in src:
            cell['source'] = styled_content['step5'].split('\n')

# Save
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Added styling to notebook markdown cells!')
print('  - Gradient backgrounds')
print('  - Color-coded sections')
print('  - Icons and emojis')
print('  - Grid layouts')
print('  - Tables and cards')
print('  - Visual hierarchy')
