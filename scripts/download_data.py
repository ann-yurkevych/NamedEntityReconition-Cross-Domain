"""One-time script to download CoNLL-2003, CrossNER Politics, and DAPT unlabeled data."""
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONLL_DIR    = ROOT / 'data' / 'raw' / 'conll2003'
POLITICS_DIR = ROOT / 'data' / 'raw' / 'crossner' / 'politics'
UNLABELED_DIR = ROOT / 'data' / 'raw' / 'unlabeled'

# Google Drive folder ID from shared link
GDRIVE_FOLDER_ID = '1xDAaTwruESNmleuIsln7IaNleNsYlHGn'

# CoNLL-2003
print('Downloading CoNLL-2003 via Hugging Face...')
from datasets import load_dataset
dataset = load_dataset('conll2003', trust_remote_code=True)
CONLL_DIR.mkdir(parents=True, exist_ok=True)
tag_names = dataset['train'].features['ner_tags'].feature.names # get tag names from the dataset features
for split_name, split in dataset.items(): # loops over each split (train, validation, test)
    with open(CONLL_DIR / f'{split_name}.txt', 'w', encoding='utf-8') as f:
        for example in split:
            for token, tag_id in zip(example['tokens'], example['ner_tags']):
                f.write(f'{token} {tag_names[tag_id]}\n')
            f.write('\n')
print(f'  CoNLL-2003 saved to {CONLL_DIR}')

# CrossNER Politics
print('Downloading CrossNER Politics from GitHub...')
POLITICS_DIR.mkdir(parents=True, exist_ok=True)
zip_url = 'https://github.com/zliucr/CrossNER/archive/refs/heads/main.zip'
with urllib.request.urlopen(zip_url) as resp:
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        for member in zf.namelist():
            if member.startswith('CrossNER-main/ner_data/politics/') and not member.endswith('/'):
                with zf.open(member) as src, open(POLITICS_DIR / Path(member).name, 'wb') as dst:
                    dst.write(src.read())
print(f'CrossNER Politics saved to {POLITICS_DIR}')

# DAPT unlabeled data
print('Downloading DAPT unlabeled data from Google Drive...')
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'gdown', '-q'], check=True)
    import gdown

UNLABELED_DIR.mkdir(parents=True, exist_ok=True)
gdown.download_folder(
    id=GDRIVE_FOLDER_ID,
    output=str(UNLABELED_DIR),
    quiet=False,
    use_cookies=False,
)
print(f'  Unlabeled data saved to {UNLABELED_DIR}')

# fix folder names to be lowercase for consistency
for p in UNLABELED_DIR.iterdir():
    if p.is_dir() and p.name != p.name.lower():
        p.rename(p.parent / p.name.lower())

print('\nDone. All data is in data/raw/.')
