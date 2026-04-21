"""One-time script to download CoNLL-2003, CrossNER Politics, and DAPT unlabeled data."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONLL_DIR    = ROOT / 'data' / 'raw' / 'conll2003'
POLITICS_DIR = ROOT / 'data' / 'raw' / 'crossner' / 'politics'
CLONE_DIR    = ROOT / 'data' / 'raw' / '_crossner_repo'
UNLABELED_DIR = ROOT / 'data' / 'raw' / 'unlabeled'

# Google Drive folder ID from shared link
GDRIVE_FOLDER_ID = '1xDAaTwruESNmleuIsln7IaNleNsYlHGn'

# CoNLL-2003
print('Downloading CoNLL-2003 via Hugging Face...')
from datasets import load_dataset
dataset = load_dataset('conll2003', trust_remote_code=True)
CONLL_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(CONLL_DIR))
print(f'  CoNLL-2003 saved to {CONLL_DIR}')

# CrossNER Politics
print('Downloading CrossNER Politics from GitHub...')
POLITICS_DIR.mkdir(parents=True, exist_ok=True)
if not CLONE_DIR.exists():
    subprocess.run(
        ['git', 'clone', '--depth', '1',
         'https://github.com/zliucr/CrossNER.git', str(CLONE_DIR)],
        check=True
    )
src = CLONE_DIR / 'ner_data' / 'politics'
for f in src.iterdir():
    shutil.copy(f, POLITICS_DIR / f.name)
def _remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

shutil.rmtree(CLONE_DIR, onerror=_remove_readonly)
print(f'  CrossNER Politics saved to {POLITICS_DIR}')

# DAPT unlabeled data
print('Downloading DAPT unlabeled data from Google Drive...')
try:
    import gdown
except ImportError:
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

print('\nDone. All data is in data/raw/.')
