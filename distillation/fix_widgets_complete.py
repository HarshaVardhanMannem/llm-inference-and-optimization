"""Completely fix widget metadata issues in notebook"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Checking notebook for widget issues...')

# 1. Remove metadata.widgets if it exists
if 'metadata' in nb:
    if 'widgets' in nb['metadata']:
        print(f'Removing metadata.widgets ({len(nb["metadata"]["widgets"])} entries)')
        del nb['metadata']['widgets']
    
    # Also ensure metadata structure is clean
    if 'widgets' in nb['metadata']:
        print('Warning: widgets still in metadata after deletion')

# 2. Check and clean cell-level widget metadata
widget_refs_removed = 0
for i, cell in enumerate(nb['cells']):
    if 'metadata' in cell:
        # Remove widgets from cell metadata
        if 'widgets' in cell['metadata']:
            del cell['metadata']['widgets']
            widget_refs_removed += 1
        
        # Check for widget-related keys
        widget_keys = [k for k in cell['metadata'].keys() if 'widget' in k.lower()]
        for key in widget_keys:
            del cell['metadata'][key]
            widget_refs_removed += 1
    
    # Check outputs for widget references
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'data' in output and 'application/vnd.jupyter.widget-view+json' in output['data']:
                # Keep the widget output but ensure it's properly formatted
                widget_data = output['data']['application/vnd.jupyter.widget-view+json']
                if isinstance(widget_data, dict):
                    # Ensure it has required fields for static rendering
                    if 'model_id' not in widget_data:
                        # This is fine - GitHub will render as static
                        pass

if widget_refs_removed > 0:
    print(f'Removed widget metadata from {widget_refs_removed} cells')

# 3. Ensure metadata structure is minimal and clean
if 'metadata' not in nb:
    nb['metadata'] = {}

# Keep only essential metadata
essential_metadata = {
    'kernelspec': nb['metadata'].get('kernelspec', {
        'display_name': 'Python 3',
        'name': 'python3'
    }),
    'language_info': nb['metadata'].get('language_info', {
        'name': 'python'
    })
}

# Preserve colab metadata if it exists (harmless)
if 'colab' in nb['metadata']:
    essential_metadata['colab'] = nb['metadata']['colab']

nb['metadata'] = essential_metadata

# 4. Verify no widgets remain
if 'widgets' in nb.get('metadata', {}):
    print('ERROR: widgets still found in metadata!')
else:
    print('Confirmed: No widgets in metadata')

# Save fixed notebook
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\nFixed! Notebook cleaned of all widget metadata.')
print('Notebook should now render on GitHub.')
