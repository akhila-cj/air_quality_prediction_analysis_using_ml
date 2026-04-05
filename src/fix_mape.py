import json

with open('AQI_Prediction.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'nonzero_mask = y_test.values != 0' in line:
                continue
            if '/ y_test.values[nonzero_mask])) * 100' in line:
                continue
            new_source.append(line)
        cell['source'] = new_source

with open('AQI_Prediction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed!')
