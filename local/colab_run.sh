#!/bin/bash
set -e

echo "=== Step 1: Install dependencies ==="
pip install tiktoken wandb datasets numpy

echo "=== Step 2: Download dataset ==="
cd /content
python -c "
from datasets import load_dataset
from pathlib import Path

data_dir = Path('data/wikitext-103')
data_dir.mkdir(parents=True, exist_ok=True)

if (data_dir / 'train.txt').exists():
    print('Dataset already downloaded, skipping...')
else:
    print('Downloading wikitext-103...')
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    for split in ['train', 'validation', 'test']:
        print(f'Saving {split} split...')
        with open(data_dir / f'{split}.txt', 'w') as f:
            for example in dataset[split]:
                f.write(example['text'])
                f.write('\n')
    print('Done downloading.')
"

echo "=== Step 3: Tokenize ==="
python prepare.py

echo "=== Step 4: Login to wandb ==="
wandb login

echo "=== Step 5: Train ==="
python train.py
