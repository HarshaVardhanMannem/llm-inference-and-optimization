"""Fix top section visibility in first cell"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved title cell with better visibility
improved_title = '''# 🎓 BERT to DistilBERT Knowledge Distillation Tutorial

<div style="text-align: center; padding: 20px 0; margin-bottom: 30px;">
<h1 style="color: #667eea; font-size: 2.5em; margin: 0; font-weight: bold;">🎓 BERT to DistilBERT</h1>
<h2 style="color: #764ba2; font-size: 1.8em; margin: 10px 0; font-weight: 600;">Knowledge Distillation Tutorial</h2>
</div>

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 20px 0;">
<h2 style="color: white; margin-top: 0;">📚 Overview</h2>
<p style="font-size: 1.1em; line-height: 1.6; color: white;">
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
<h3 style="margin-top: 0; color: #1976d2;">🌡️ Temperature Scaling</h3>
<p style="margin-bottom: 0; color: #212529;">Softens probability distributions to reveal "dark knowledge"</p>
</div>
<div style="background: #f3e5f5; padding: 15px; border-radius: 8px; border-top: 3px solid #9c27b0;">
<h3 style="margin-top: 0; color: #7b1fa2;">📊 KL Divergence Loss</h3>
<p style="margin-bottom: 0; color: #212529;">Measures how well student matches teacher's predictions</p>
</div>
<div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-top: 3px solid #ff9800;">
<h3 style="margin-top: 0; color: #e65100;">⚖️ Combined Loss</h3>
<p style="margin-bottom: 0; color: #212529;">Balances hard labels (ground truth) and soft labels (teacher predictions)</p>
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
'''

# Update first markdown cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell.get('source', []))
        if 'BERT to DistilBERT Knowledge Distillation Tutorial' in src and i == 0:
            cell['source'] = improved_title.split('\n')
            print(f'Updated cell {i} with improved title visibility')
            break

# Save
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Fixed top section visibility!')
print('  - Added prominent centered title')
print('  - Larger font sizes for title')
print('  - Better spacing and visibility')
