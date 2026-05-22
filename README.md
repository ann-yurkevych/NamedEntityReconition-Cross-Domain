# CrossNER: Domain-Adaptive NER for Politics

## Description

This project explores cross-domain Named Entity Recognition (NER), training a model on the general-domain CoNLL-2003 dataset and evaluating its ability to generalize to the Politics domain from the [CrossNER benchmark](https://github.com/zliucr/CrossNER). The goal is to understand domain shift in NER and experiment with adaptation strategies.

## Setup

### Step 1 - Clone the repo

```bash
git clone https://github.com/ann-yurkevych/NamedEntityReconition-Cross-Domain.git
cd NamedEntityReconition-Cross-Domain
```

> **Important:** always `cd` into the repo folder first. All commands below assume you are inside `NamedEntityReconition-Cross-Domain/`.

---

### Step 2 - Install dependencies

**Option A: conda (recommended)**

Open a terminal where `conda` is available (on Windows: open a new PowerShell after running `conda init powershell`, or use Anaconda Prompt):

```bash
conda create -n crossner311 python=3.11 && \
conda activate crossner311 && \
pip install -r requirements.txt
```

To verify it worked:

```bash
python --version   # should say Python 3.11
```

**Option B: pip**

```bash
pip install -r requirements.txt
```

---

### Step 3 - Download datasets

```bash
python scripts/download_data.py
```

This script fetches CoNLL-2003 via Hugging Face, downloads CrossNER Politics directly from GitHub, and downloads the DAPT unlabeled corpus from Google Drive via `gdown`. Everything is placed in `data/raw/`. It only needs to be run once. Data is gitignored and never committed to the repo.

---

### Step 4 - Run experiments

All five modes can be run independently. The DAPT model (`bert-dapt-politics`) is hosted on [HuggingFace](https://huggingface.co/daradage/bert-dapt-politics) and downloaded automatically on first use — no pretraining step needed.

```bash
python -m scripts.main_run --mode crossner       # train + test on CrossNER politics
python -m scripts.main_run --mode zero_shot      # train on CoNLL, test on CrossNER (no finetune)
python -m scripts.main_run --mode transfer       # CoNLL pretraining → CrossNER finetune
python -m scripts.main_run --mode jointly_train  # joint CoNLL + CrossNER training
python -m scripts.main_run --mode dapt           # loads DAPT model from HuggingFace automatically
```

> **Note:** if you want to reproduce the DAPT pretraining from scratch instead of using the shared weights, run `python -m scripts.run_dapt` first (≈25 000 training steps, requires a GPU).

The script uses CUDA automatically if a GPU is available. If you have a GPU, reinstall torch with the CUDA build after activating the environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade
```

Results are saved to `results/metrics/` with timestamps.

---

### Step 5 - Register the kernel and open notebooks

Register the conda environment as a Jupyter kernel (once):

```bash
python -m ipykernel install --user --name crossner311 --display-name "crossner311"
```

Then open any notebook in VS Code, click the kernel selector in the top right, and choose **crossner311** from the list. If it doesn't appear, reload VS Code.

---

## Step 6 - Running Utility Scripts

### Domain-Adaptive Pretraining (DAPT)

Run this script in order to reproduce the DAPT step from the paper by further pretraining BERT on the unlabeled politics corpus with span-style masking:

```bash
python scripts/run_dapt.py
```
**What it does:**
- Loads the unlabeled politics corpus from `politics_integrated.txt`
- Fine-tunes a masked language model with span-style masking
- Saves the resulting checkpoint to `bert-dapt-politics`

**Notes:**
- This is a long-running, GPU-heavy job


### Emerging Entity Analysis

Use this script to compare performance on seen versus unseen entities:

```bash
python scripts/emerging_entity_analysis.py
```

**What it does:**
- Loads the CoNLL-2003 training data and CrossNER Politics test data
- Reads a saved predictions JSON file from metrics
- Splits entities into seen and unseen groups
- Prints classification reports and F1 scores for both groups

**Notes:**
- Before running it, in the line 105 in the "emerging_entity_analysis.py" file change the PRED_PATH for the one which you received from the MAIN run with the DAPT mode, you can find it in the results/metrics folder and it should be named like preds_dapt_20260511_224820.json (the timestamp will differ):

```bash
PRED_PATH = "results/metrics/preds_dapt_20260511_224820.json" # change the PRED_PATH
```



## Folder Structure

```
.
├── data/
│   ├── raw/
│   │   ├── conll2003/          # CoNLL-2003 source-domain data
│   │   ├── crossner/
│   │   │   └── politics/       # CrossNER Politics target-domain data
│   │   └── unlabeled/          # DAPT unlabeled corpus (politics)
│   └── processed/              # Tokenized / preprocessed splits
├── notebooks/                  # Exploratory and evaluation notebooks
├── scripts/
│   ├── download_data.py        # One-time data download script
│   ├── run_experiment.py       # Main training + evaluation entry point
│   └── run_dapt.py             # Domain-adaptive pretraining script
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # BERT tagger model
│   ├── training/               # Trainer, evaluator, losses
│   └── utils/                  # Label mapping, metrics
├── results/
│   ├── logs/                   # Training logs
│   ├── models/                 # Saved model checkpoints (gitignored)
│   └── metrics/                # Evaluation metric outputs
├── requirements.txt
├── environment.yml
└── README.md
```

## Team

| Member                | ITU email   |
| --------------------- | ----------- |
| Dara Georgieva        | dage@itu.dk |
| Nina Osifova          | nios@itu.dk |
| Hanna Yurkevych       | hayu@itu.dk |
| Anna Weronika Lekston | awle@itu.dk |

## Label Mapping

CoNLL-2003 trains on four coarse types (`PER`, `ORG`, `LOC`, `MISC`).
CrossNER Politics evaluates on nine fine-grained types. The mapping is:

| CoNLL-2003 | CrossNER Politics                      |
| :--------: | :------------------------------------- |
|    PER     | person, **politician**                  |
|    ORG     | organisation, **politicalparty**        |
|    LOC     | location, **country**                   |
|    MISC    | misc, **event**, **election**           |

**Bold** types are fine-grained specialisations absent from CoNLL-2003 training data.

[src/utils/label_mapping.py](src/utils/label_mapping.py) provides the utilities for the mapping.
