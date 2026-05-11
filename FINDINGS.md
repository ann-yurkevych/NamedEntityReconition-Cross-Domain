# Findings

## Experiment: BERT CrossNER (Politics, in-domain)

**Model:** `bert-base-cased`  
**Mode:** `crossner` (train and test on CrossNER Politics)  
**Epochs:** 15  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.64      | 0.52   | 0.57 | 418     |
| election       | 0.86      | 0.93   | 0.89 | 434     |
| event          | 0.35      | 0.26   | 0.30 | 195     |
| location       | 0.72      | 0.88   | 0.79 | 599     |
| misc           | 0.27      | 0.38   | 0.31 | 258     |
| organisation   | 0.55      | 0.70   | 0.61 | 513     |
| **person**     | 0.25      | 0.02   | 0.04 | 354     |
| politicalparty | 0.78      | 0.81   | 0.80 | 953     |
| politician     | 0.55      | 0.94   | 0.69 | 485     |
| **micro avg**  | 0.63      | 0.69   | **0.659** | 4209 |
| macro avg      | 0.55      | 0.60   | 0.56 | 4209    |
| weighted avg   | 0.61      | 0.69   | 0.63 | 4209    |

---

### Finding: `person` near-zero despite 354 test instances

The model predicts `person` with F1 0.04 (precision 0.25, recall 0.02) — nearly useless despite 354 test instances.

**Root cause:** severe class imbalance in the training data.

| Label      | B- count | I- count |
|------------|----------|----------|
| person     | 14       | 5        |
| politician | 359      | 341      |

`person` appears only 14 times in training vs 359 for `politician`. Since both share the same coarse parent type (`PER`), the model strongly favours `politician` for person-like entities, almost never predicting `person`.

**This is a data issue, not a code issue.** The model training is correct. With only 14 `person` examples and 359 `politician` examples sharing the same coarse parent (`PER`), the model never learns a reliable decision boundary. More training epochs help marginally (F1 improved from 0.00 at 5 epochs to 0.04 at 15 epochs) but 14 instances is simply too few. This is a property of the CrossNER Politics dataset itself.

**Possible fixes:**
- Upsample `person` training sentences
- Increase the loss weight for the `person` label
- Accept it as a domain property — in politics text, most people mentioned are politicians

---

## Experiment: BERT Zero-Shot (CoNLL → CrossNER Politics)

**Model:** `bert-base-cased`  
**Mode:** `zero_shot` (train on CoNLL-2003, evaluate on CrossNER Politics with no fine-tuning)  
**Epochs:** 5 (CoNLL only)  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.00      | 0.00   | 0.00 | 418     |
| election       | 0.00      | 0.00   | 0.00 | 434     |
| event          | 0.00      | 0.00   | 0.00 | 195     |
| location       | 0.40      | 0.85   | 0.54 | 599     |
| misc           | 0.22      | 0.58   | 0.31 | 258     |
| organisation   | 0.25      | 0.85   | 0.38 | 513     |
| person         | 0.38      | 0.97   | 0.55 | 354     |
| politicalparty | 0.00      | 0.00   | 0.00 | 953     |
| politician     | 0.00      | 0.00   | 0.00 | 485     |
| **micro avg**  | 0.31      | 0.34   | **0.325** | 4209 |
| macro avg      | 0.14      | 0.36   | 0.20 | 4209    |
| weighted avg   | 0.08      | 0.25   | 0.12 | 4209    |

### Finding: fine-grained types score zero by construction

`politician`, `politicalparty`, `country`, `election`, `event` all score 0.00. These types do not exist in CoNLL-2003 — the model has never seen them during training and never predicts them. Every entity in these categories gets predicted as its coarse generic equivalent (`politician` → `person`, `politicalparty` → `organisation`, `country` → `location`).

The four types that do score (`person`, `location`, `organisation`, `misc`) show high recall but low precision — the model casts a wide net, predicting the generic type for everything in that coarse category.

**The F1 gap between zero_shot (0.325) and crossner (0.659) measures the value of the 200 CrossNER labelled training sentences.** `transfer` and `dapt` aim to close this gap while using the same 200 sentences more effectively.

---

## Experiment: BERT Transfer Learning (CoNLL → CrossNER Politics)

**Model:** `bert-base-cased`  
**Mode:** `transfer` (CoNLL-2003 pre-training → CrossNER Politics fine-tuning)  
**Epochs:** 3 (CoNLL) + 15 (CrossNER)  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.73      | 0.57   | 0.64 | 418     |
| election       | 0.86      | 0.92   | 0.89 | 434     |
| event          | 0.48      | 0.43   | 0.45 | 195     |
| location       | 0.73      | 0.88   | 0.79 | 599     |
| misc           | 0.41      | 0.58   | 0.48 | 258     |
| organisation   | 0.60      | 0.74   | 0.66 | 513     |
| **person**     | 0.23      | 0.04   | 0.06 | 354     |
| politicalparty | 0.79      | 0.85   | 0.82 | 953     |
| politician     | 0.54      | 0.91   | 0.68 | 485     |
| **micro avg**  | 0.66      | 0.72   | **0.691** | 4209 |
| macro avg      | 0.60      | 0.66   | 0.61 | 4209    |
| weighted avg   | 0.64      | 0.72   | 0.67 | 4209    |

### Finding: transfer is the best-performing mode

Test F1 0.691 — the highest across all five modes. CoNLL pre-training gives the encoder 3 epochs of NER-structured supervision on 14k sentences before CrossNER fine-tuning. By the time the model sees CrossNER, both the encoder and the classification head already have strong NER representations, making the 200 CrossNER sentences more effective.

**Notable improvements over crossner:**

| Label | crossner F1 | transfer F1 | Δ |
|---|---|---|---|
| event | 0.30 | 0.45 | +0.15 |
| misc | 0.31 | 0.48 | +0.17 |
| country | 0.57 | 0.64 | +0.07 |

`event` and `misc` — the rarest fine-grained types — benefit most. CoNLL `MISC` pre-training provides a stronger prior for these low-frequency types.

**`person` remains near-zero (F1 0.06)** — CoNLL pre-training does not fix the 14-instance class imbalance problem.

---

## Experiment: BERT Jointly Train (CoNLL + CrossNER Politics)

**Model:** `bert-base-cased`  
**Mode:** `jointly_train` (CoNLL-2003 + CrossNER Politics combined, CrossNER upsampled to match CoNLL size)  
**Epochs:** 15  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.72      | 0.21   | 0.32 | 418     |
| election       | 0.78      | 0.86   | 0.82 | 434     |
| event          | 0.27      | 0.14   | 0.19 | 195     |
| location       | 0.51      | 0.84   | 0.64 | 599     |
| misc           | 0.32      | 0.55   | 0.41 | 258     |
| organisation   | 0.37      | 0.68   | 0.48 | 513     |
| person         | 0.43      | 0.38   | 0.41 | 354     |
| politicalparty | 0.68      | 0.62   | 0.65 | 953     |
| politician     | 0.52      | 0.75   | 0.62 | 485     |
| **micro avg**  | 0.52      | 0.61   | **0.562** | 4209 |
| macro avg      | 0.51      | 0.56   | 0.50 | 4209    |
| weighted avg   | 0.55      | 0.61   | 0.55 | 4209    |

### Finding: jointly_train underperforms crossner and transfer

Test F1 0.562 is below both crossner (0.659) and transfer (0.691), despite using more data.

**The code is correct** — `load_conll2003` returns train split only, labels are mapped before combining, and CrossNER is upsampled to match CoNLL size (~50/50 split). The underperformance is algorithmic.

**Root cause: label dilution from 70× upsampling.** Each of the 200 CrossNER sentences is repeated 70× per epoch while each CoNLL sentence appears once. The CoNLL signal (mapping everything to generic types — `person`, `organisation`, `location`, `misc`) constantly competes with the fine-grained CrossNER signal. The result is a tug-of-war: the model is pulled toward generic predictions and becomes conservative on fine-grained types. This is visible in the pattern of high precision but very low recall on fine-grained types (`country`: precision 0.72, recall 0.21; `event`: precision 0.27, recall 0.14).

**Notable exception: `person` improves** to F1 0.41 (vs 0.04 in crossner). The large volume of CoNLL `PER` entities mapped to `person` gives the model enough signal to distinguish `person` from `politician` — the one case where CoNLL data helps rather than hurts.

This result is consistent with the CrossNER paper, which reports jointly_train underperforming transfer and DAPT on small target domains.

---

## Experiment: BERT + DAPT (Domain-Adaptive Pre-Training → CrossNER Politics)

**Model:** `bert-base-cased` MLM-adapted on politics corpus → fine-tuned on CrossNER Politics  
**Mode:** `dapt`  
**DAPT:** 10,000 steps, span-level masking (p=0.15), integrated corpus, block_size=256  
**Fine-tuning epochs:** 15  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.70      | 0.50   | 0.59 | 418     |
| election       | 0.85      | 0.94   | 0.89 | 434     |
| event          | 0.49      | 0.35   | 0.41 | 195     |
| location       | 0.68      | 0.86   | 0.76 | 599     |
| misc           | 0.30      | 0.48   | 0.37 | 258     |
| organisation   | 0.56      | 0.62   | 0.59 | 513     |
| **person**     | 0.37      | 0.03   | 0.05 | 354     |
| politicalparty | 0.79      | 0.89   | 0.83 | 953     |
| politician     | 0.54      | 0.91   | 0.68 | 485     |
| **micro avg**  | 0.64      | 0.70   | **0.670** | 4209 |
| macro avg      | 0.59      | 0.62   | 0.58 | 4209    |
| weighted avg   | 0.64      | 0.70   | 0.64 | 4209    |

### Finding: DAPT beats crossner but not transfer

DAPT (0.670) outperforms the in-domain baseline crossner (0.659) — the MLM pre-training on political text gives the encoder better domain representations. However it does not beat transfer (0.691).

**Final scoreboard:**

| Mode | Test F1 |
|---|---|
| transfer | **0.691** |
| dapt | 0.670 |
| crossner | 0.659 |
| jointly_train | 0.562 |
| zero_shot | 0.325 |

**Why transfer beats dapt:** Transfer gets 3 epochs of CoNLL NER supervision (14k sentences) before CrossNER fine-tuning, giving both the encoder and classification head a strong NER-structured starting point. DAPT starts with a domain-adapted encoder but a randomly initialised classification head — the 200 CrossNER sentences must simultaneously train the head and specialise the encoder for NER, which is a harder learning problem.

**Why dapt could be improved:**
- 10,000 MLM steps covers ~41% of the integrated corpus — more steps would strengthen domain adaptation
- No warmup in the fine-tuning phase; jumping to lr=5e-5 with a random head can destabilise early training
- Lowering the fine-tuning LR to 2e-5 was tried but hurt performance (0.663), suggesting the original 5e-5 is already well-calibrated for this setting

**Fast convergence observed:** Training loss dropped from 1.62 (epoch 1) to 0.085 (epoch 6), much faster than crossner — consistent with the DAPT encoder having already learned strong political text representations.
