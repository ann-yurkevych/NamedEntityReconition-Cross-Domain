# CrossNER: Domain-Adaptive NER for Politics

## Description

This project explores cross-domain Named Entity Recognition (NER), training a model on the general-domain CoNLL-2003 dataset and evaluating its ability to generalize to the Politics domain from the [CrossNER benchmark](https://github.com/zliucr/CrossNER). The goal is to understand domain shift in NER and experiment with adaptation strategies.

## Setup

### Clone the repository

```bash
git clone https://github.com/ann-yurkevych/NamedEntityReconition-Cross-Domain.git
cd NamedEntityReconition-Cross-Domain
```

### Install dependencies

**Using pip:**

```bash
pip install -r requirements.txt
```

**Using conda:**

```bash
conda env create -f environment.yml
conda activate crossner
```

### Download datasets

Place the raw dataset files as follows:

- **CoNLL-2003**: Download from the official source and place files in `data/raw/conll2003/`
  - Expected files: `train.txt`, `valid.txt`, `test.txt`
- **CrossNER Politics**: Download from the [CrossNER repository](https://github.com/zliucr/CrossNER) and place files in `data/raw/crossner/politics/`
  - Expected files: `train.txt`, `dev.txt`, `test.txt`

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
├── src/                        # Source code (data loading, model, training, eval)
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
