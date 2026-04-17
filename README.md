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

**Option A: conda**

Open a terminal where `conda` is available (on Windows: open a new PowerShell after running `conda init powershell`, or use Anaconda Prompt):

```bash
conda env create -f environment.yml   # create the environment (once)
conda activate crossner               # activate it (every session)
```

To verify it worked:

```bash
python --version   # should say Python 3.10.x
```

**Option B: pip**

```bash
pip install -r requirements.txt
```

---

### Step 3 - Download datasets

```powershell
python scripts/download_data.py
```

This script fetches CoNLL-2003 via Hugging Face and clones CrossNER Politics from GitHub, placing everything in the correct `data/raw/` subdirectories. It only needs to be run once. Data is gitignored and never committed to the repo.

---

### Step 4 - Register the kernel and open notebooks

Register the conda environment as a Jupyter kernel (once):

```powershell
python -m ipykernel install --user --name crossner --display-name "Python (crossner)"
```

Then open any notebook in VS Code, click the kernel selector in the top right, and choose **Python (crossner)** from the list. If it doesn't appear, reload VS Code.

---

## Folder Structure

```
.
├── data/
│   ├── raw/
│   │   ├── conll2003/          # CoNLL-2003 source-domain data
│   │   └── crossner/
│   │       └── politics/       # CrossNER Politics target-domain data
│   └── processed/              # Tokenized / preprocessed splits
├── notebooks/                  # Exploratory and evaluation notebooks
├── scripts/
│   └── download_data.sh        # One-time data download script
├── src/
│   └── label_mapping.py        # CoNLL-2003 ↔ CrossNER Politics label mapping
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
|    PER     | person, **politician**                 |
|    ORG     | organization, **political-party**      |
|    LOC     | location, **country**                  |
|    MISC    | miscellaneous, **event**, **election** |

**Bold** types are fine-grained specialisations absent from CoNLL-2003 training data.

`src/label_mapping.py` provides the utilities for the mapping.
