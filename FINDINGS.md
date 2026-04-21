# Findings

## Experiment: BERT CrossNER (Politics, in-domain)

**Model:** `bert-base-cased`  
**Mode:** `crossner` (train and test on CrossNER Politics)  
**Epochs:** 5  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.57      | 0.42   | 0.48 | 418     |
| election       | 0.75      | 0.91   | 0.82 | 434     |
| event          | 0.31      | 0.24   | 0.27 | 195     |
| location       | 0.72      | 0.85   | 0.78 | 599     |
| misc           | 0.32      | 0.47   | 0.38 | 258     |
| organisation   | 0.50      | 0.69   | 0.58 | 513     |
| **person**     | 0.00      | 0.00   | 0.00 | 354     |
| politicalparty | 0.71      | 0.88   | 0.79 | 953     |
| politician     | 0.55      | 0.96   | 0.70 | 485     |
| **micro avg**  | 0.60      | 0.69   | **0.644** | 4209 |
| macro avg      | 0.49      | 0.60   | 0.53 | 4209    |
| weighted avg   | 0.56      | 0.69   | 0.61 | 4209    |

---

### Finding: `person` collapses to zero

The model predicts `person` with 0.00 precision and recall despite 354 test instances.

**Root cause:** severe class imbalance in the training data.

| Label      | B- count | I- count |
|------------|----------|----------|
| person     | 14       | 5        |
| politician | 359      | 341      |

`person` appears only 14 times in training vs 359 for `politician`. Since both share the same coarse parent type (`PER`), the model learns to always predict `politician` for person-like entities and never `person`.

**Possible fixes:**
- Upsample `person` training sentences
- Increase the loss weight for the `person` label
- Accept it as a domain property — in politics text, most people mentioned are politicians
