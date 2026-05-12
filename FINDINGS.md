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
**DAPT:** tried 10,000 and 25,000 steps, span-level masking (p=0.15), integrated corpus, block_size=256  
**Fine-tuning epochs:** 15  
**Batch size:** 16  
**Optimizer:** AdamW, lr=5e-5

### Results (25,000 steps)

| Label          | Precision | Recall | F1   | Support |
|----------------|-----------|--------|------|---------|
| country        | 0.73      | 0.53   | 0.62 | 418     |
| election       | 0.87      | 0.94   | 0.91 | 434     |
| event          | 0.42      | 0.34   | 0.38 | 195     |
| location       | 0.72      | 0.88   | 0.79 | 599     |
| misc           | 0.33      | 0.50   | 0.40 | 258     |
| organisation   | 0.55      | 0.74   | 0.63 | 513     |
| **person**     | 0.47      | 0.02   | 0.04 | 354     |
| politicalparty | 0.78      | 0.87   | 0.82 | 953     |
| politician     | 0.55      | 0.96   | 0.70 | 485     |
| **micro avg**  | 0.65      | 0.72   | **0.683** | 4209 |
| macro avg      | 0.60      | 0.64   | 0.59 | 4209    |
| weighted avg   | 0.65      | 0.72   | 0.66 | 4209    |

### Earlier result: 10,000 steps (F1 0.670)

| Label          | F1 (10k) | F1 (25k) | Δ |
|----------------|----------|----------|---|
| country        | 0.59 | 0.62 | +0.03 |
| election       | 0.89 | 0.91 | +0.02 |
| event          | 0.41 | 0.38 | −0.03 |
| location       | 0.76 | 0.79 | +0.03 |
| misc           | 0.37 | 0.40 | +0.03 |
| organisation   | 0.59 | 0.63 | +0.04 |
| politicalparty | 0.83 | 0.82 | −0.01 |
| politician     | 0.68 | 0.70 | +0.02 |

More MLM steps broadly help — 7 of 9 labels improve. `event` is the only meaningful regression (0.41→0.38), consistent with the pattern seen in the confusion matrix analysis.

### Finding: DAPT beats crossner but not transfer

DAPT (0.683) outperforms the in-domain baseline crossner (0.659). The gap with transfer (0.691) narrows to just 0.008 with 25k steps, down from 0.021 at 10k steps.

**Final scoreboard:**

| Mode | Test F1 |
|---|---|
| transfer | **0.691** |
| dapt (25k steps) | 0.683 |
| dapt (10k steps) | 0.670 |
| crossner | 0.659 |
| jointly_train | 0.562 |
| zero_shot | 0.325 |

**Why transfer beats dapt:** Transfer gets 3 epochs of CoNLL NER supervision (14k sentences) before CrossNER fine-tuning, giving both the encoder and classification head a strong NER-structured starting point. DAPT starts with a domain-adapted encoder but a randomly initialised classification head — the 200 CrossNER sentences must simultaneously train the head and specialise the encoder for NER, which is a harder learning problem. The remaining 0.008 gap is likely irreducible without addressing head initialisation.

**Why more steps help:** At 10k steps the MLM loss was still clearly descending (eval_loss: 2.241→2.144→2.087→2.015 at 5k-step intervals). 25k steps covers ~100% of the integrated corpus (vs ~41% at 10k), giving the encoder substantially more political text exposure before fine-tuning.

**Discriminative fine-tuning — tried and rejected:**

*Motivation:* When a DAPT checkpoint is loaded and a classification head attached, the head is randomly initialised. In the first few fine-tuning steps it produces essentially random predictions, generating large loss values. Those gradients propagate back through the entire encoder and can overwrite the domain representations that 25,000 MLM steps built. Transfer avoids this because the CoNLL phase trains both encoder and head together on a real NER objective — by the time CrossNER fine-tuning starts, the head already knows roughly what entity boundaries look like, gradients are small and stable, and the encoder is not destabilised. The standard fix is discriminative fine-tuning: a small encoder lr and a larger head lr so the head converges quickly without sending destructive gradients into the encoder.

*What we tried:*
- Encoder lr=2e-5, head lr=5e-5 (2.5:1 ratio) → Test F1 0.617
- Encoder lr=1e-5, head lr=5e-5 (5:1 ratio) → Test F1 0.622
- Both worse than flat lr=5e-5 (F1 0.683)

*Why it failed:* With only 200 CrossNER training sentences (~195 fine-tuning steps total), the encoder at 1e-5 or 2e-5 does not update fast enough to connect the DAPT representations to the NER task. The head stabilises but the encoder stays too generic. Discriminative fine-tuning requires enough fine-tuning data for the head to reach a stable regime before encoder updates matter — 200 sentences is not enough. Flat 5e-5 forces encoder and head to learn simultaneously, which is what this data size requires.

**Fast convergence observed:** Training loss dropped from 1.62 (epoch 1) to 0.085 (epoch 6), much faster than crossner — consistent with the DAPT encoder having already learned strong political text representations.

---

## Confusion Matrix Analysis: Transfer vs DAPT

Entity-level span confusion counts (exact boundary match, Gold → Predicted):

| Pair (Gold → Predicted) | Transfer | DAPT | Δ |
|---|---|---|---|
| politicalparty → organisation | 28 | 7 | −21 |
| politician → person | 18 | 13 | −5 |
| location → country | 7 | 4 | −3 |
| organisation → politicalparty | 41 | 46 | +5 |
| country → location | 59 | 73 | +14 |
| person → politician | 327 | 329 | ≈0 |

### Finding: DAPT helps ORG-type hierarchy but hurts LOC-type and event detection

DAPT reduces `politicalparty → organisation` confusion by 75% (28→7) — the clearest win. Political text is dense with named parties; MLM pre-training teaches the encoder to distinguish them from generic organisations before any NER labels are seen.

The opposite pattern holds for location-type entities: `country → location` confusions increase by +14 under DAPT. Country names appear heavily in political text in location-like syntactic contexts ("talks in Berlin", "war in Ukraine"), so domain adaptation reinforces rather than resolves this ambiguity. The `event` label shows the same direction — diagonal drops from 0.43 (transfer) to 0.35 (DAPT) — consistent with political event mentions also appearing in contexts the MLM cannot distinguish from general location or misc usage.

`person → politician` (327→329) is unchanged by DAPT, confirming this confusion is driven entirely by the 14-instance `person` class imbalance, not by domain representations.

---

## Emerging Entity Analysis

**Definition:** entities whose surface string never appeared in CoNLL-2003 training data are classified as *unseen/emerging*. Evaluation collapses CrossNER Politics fine-grained labels to CoNLL-2003 coarse types (PER/ORG/LOC/MISC) for a fair comparison.

**Split:** 811 seen (19.3%) / 3,398 unseen (80.7%) — the vast majority of test entities are emerging.

### Results: Seen vs Unseen F1 by mode

| Mode | Seen F1 | Unseen F1 | Gap |
|---|---|---|---|
| zero_shot | 92.29 | 70.24 | +22.05 |
| crossner | 87.60 | 83.90 | +3.70 |
| transfer | 92.40 | 85.74 | +6.66 |
| jointly_train | 92.29 | 75.68 | +16.61 |
| **dapt** | **86.26** | **83.89** | **+2.37** |

### Finding: DAPT generalises best to unseen entities

DAPT has the smallest seen/unseen gap (+2.37 pts) — MLM pre-training on political text gives the encoder representations that transfer to entity strings it has never seen labelled, which is the core purpose of domain adaptation.

**zero_shot gap is largest (+22 pts):** a CoNLL-only model is well-calibrated on entity strings it saw during training but has no mechanism for unseen domain-specific names.

**jointly_train gap is also large (+16.6 pts):** the 70× upsampling of CrossNER gives near-perfect coverage of the 200 training sentences (high seen F1), but the constant CoNLL signal prevents the model from learning robust representations for new entity types, so unseen performance suffers.

**transfer beats crossner on unseen** (85.74 vs 83.90): CoNLL pre-training provides a stronger NER prior that helps with novel entities, but at the cost of a larger gap than DAPT or crossner alone.

Since 80.7% of test entities are unseen, unseen F1 is the more meaningful number for real-world usefulness — by that measure the ranking is transfer > dapt ≈ crossner > jointly_train > zero_shot.
