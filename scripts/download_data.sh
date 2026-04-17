#!/bin/bash
set -e

# Ensure we run from the repository root, regardless of where the script is invoked.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Download CoNLL-2003 via Hugging Face
python -c "
from datasets import load_dataset
dataset = load_dataset('conll2003', trust_remote_code=True)
dataset.save_to_disk('data/raw/conll2003')
print('CoNLL-2003 downloaded to data/raw/conll2003')
"

# Download CrossNER Politics
echo "Downloading CrossNER..."
mkdir -p data/raw
cd data/raw

rm -rf CrossNER
git clone --depth 1 https://github.com/zliucr/CrossNER.git

mkdir -p crossner
rm -rf crossner/politics
cp -r CrossNER/ner_data/politics crossner/

rm -rf CrossNER
echo "CrossNER Politics extracted to data/raw/crossner/politics"
