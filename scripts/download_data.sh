#!/bin/bash
set -e

# Download CoNLL-2003 via Hugging Face
python -c "
from datasets import load_dataset
dataset = load_dataset('conll2003', trust_remote_code=True)
dataset.save_to_disk('data/raw/conll2003')
print('CoNLL-2003 downloaded to data/raw/conll2003')
"

# Download CrossNER Politics
echo "Downloading CrossNER..."
cd data/raw
git clone --depth 1 https://github.com/zhouxiangru/CrossNER.git
cp -r CrossNER/data/politics crossner/politics
rm -rf CrossNER
echo "CrossNER Politics extracted to data/raw/crossner/politics"
