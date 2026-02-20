"""Check notebook for GitHub rendering issues"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

issues = []

# Check nbformat
if nb.get('nbformat') != 4:
    issues.append(f'nbformat should be 4, got {nb.get("nbformat")}')
if nb.get('nbformat_minor') != 4:
    issues.append(f'nbformat_minor should be 4, got {nb.get("nbformat_minor")}')

# Check metadata.widgets
if 'metadata' in nb and 'widgets' in nb['metadata']:
    issues.append('metadata.widgets still exists (should be removed)')

# Check cell structure
for i, cell in enumerate(nb['cells']):
    # Check markdown cells
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        
        # Source should be a list
        if not isinstance(source, list):
            issues.append(f'Cell {i}: markdown source is not a list')
        
        # Check for HTML that might cause issues
        full_text = ''.join(source)
        if '<div' in full_text or '<style' in full_text:
            issues.append(f'Cell {i}: Contains HTML div/style tags')
        if 'style=' in full_text:
            issues.append(f'Cell {i}: Contains inline styles')
    
    # Check code cells
    elif cell['cell_type'] == 'code':
        # Check for required fields
        if 'source' not in cell:
            issues.append(f'Cell {i}: Missing source')
        if 'metadata' not in cell:
            issues.append(f'Cell {i}: Missing metadata')
        if 'outputs' not in cell:
            issues.append(f'Cell {i}: Missing outputs')

# Check for very large outputs
large_outputs = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for output in cell['outputs']:
            if 'data' in output:
                for key, value in output['data'].items():
                    if isinstance(value, list):
                        size = len(str(value))
                        if size > 1000000:  # >1MB
                            large_outputs += 1

if large_outputs > 0:
    issues.append(f'Found {large_outputs} very large outputs (>1MB) that may slow rendering')

print('GitHub Rendering Check:')
print('=' * 50)
if issues:
    print('Issues found:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('No issues found!')
    print('\nNotebook should render on GitHub.')
    print('\nIf it still doesn\'t render, try:')
    print('1. Ensure file is committed and pushed')
    print('2. Wait a few minutes for GitHub to process')
    print('3. Check GitHub\'s raw file view to verify JSON is valid')
    print('4. Try viewing in a different browser')
