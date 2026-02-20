"""Fix markdown cell formatting for GitHub"""
import json

with open('bert_distillbert_knowledge_distillation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixed = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        
        # Ensure source is a list
        if isinstance(source, str):
            # Convert string to list of lines
            lines = source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]]
            if lines[-1]:
                cell['source'].append(lines[-1] + '\n')
            fixed += 1
            print(f'Fixed cell {i}: converted string to list')
        
        # Ensure each line ends with \n (except possibly last)
        elif isinstance(source, list):
            new_source = []
            for j, line in enumerate(source):
                if isinstance(line, str):
                    # Ensure line ends with \n (except last line if empty)
                    if j < len(source) - 1 or line:
                        if not line.endswith('\n'):
                            new_source.append(line + '\n')
                        else:
                            new_source.append(line)
                    else:
                        new_source.append(line)
                else:
                    new_source.append(str(line) + '\n' if j < len(source) - 1 else str(line))
            
            if new_source != source:
                cell['source'] = new_source
                fixed += 1
                print(f'Fixed cell {i}: normalized line endings')

# Ensure metadata structure is clean
if 'metadata' not in nb:
    nb['metadata'] = {}

# Remove any problematic metadata
if 'widgets' in nb['metadata']:
    del nb['metadata']['widgets']
    print('Removed metadata.widgets')

# Ensure kernelspec and language_info exist
if 'kernelspec' not in nb['metadata']:
    nb['metadata']['kernelspec'] = {
        'display_name': 'Python 3',
        'name': 'python3'
    }

if 'language_info' not in nb['metadata']:
    nb['metadata']['language_info'] = {
        'name': 'python'
    }

# Save
with open('bert_distillbert_knowledge_distillation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nFixed {fixed} markdown cells')
print('Notebook structure cleaned for GitHub rendering')
