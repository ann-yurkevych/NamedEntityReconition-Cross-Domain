#!/usr/bin/env python
# coding: utf-8

# ### libraries

# In[3]:


import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)


# ### load the data

# In[3]:


def load_iob2(filepath):
    sentences = []
    current = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()  
                word  = parts[0]          
                label = parts[-1]          
                current.append((word, label))
    if current:                            
        sentences.append(current)
    return sentences


# In[4]:


train_set = load_iob2("./project/en_ewt-ud-train.iob2")
dev_set = load_iob2("./project/en_ewt-ud-dev.iob2")


# ### label mapping

# In[5]:


def build_label_map(sentences):
    """
    Collects every unique label in the dataset and assigns an integer ID.
    Returns:  label2id  {"B-PER": 0, "I-PER": 1, "O": 2, ...}
              id2label  {0: "B-PER", 1: "I-PER", 2: "O", ...}
    """
    labels = sorted(set(label for sent in sentences for _, label in sent))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label  = {i: l for l, i in label2id.items()}
    return label2id, id2label

label2id, id2label = build_label_map(train_set)


# ### BERT

# In[6]:


model_name = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)


# ### splitting into labels and tokens

# In[7]:


def split_tokens_labels(dataset):
    tokens = []
    labels = []

    for sentence in dataset:
        tok = []
        lab = []
        for word, tag in sentence:
            tok.append(word)
            lab.append(tag)
        tokens.append(tok)
        labels.append(lab)

    return tokens, labels

train_tokens, train_labels = split_tokens_labels(train_set)
dev_tokens, dev_labels = split_tokens_labels(dev_set)


# ### convert labels to ids

# In[8]:


def encode_labels(labels, label2id):
    return [[label2id[l] for l in sent] for sent in labels]

train_labels_enc = encode_labels(train_labels, label2id)
dev_labels_enc = encode_labels(dev_labels, label2id)


# ### tokenization + aligning labels

# In[9]:


def tokenize_and_align_labels(tokens, labels_enc):
    tokenized_inputs = tokenizer(
        tokens,
        truncation=True,
        padding=True,
        is_split_into_words=True
    )
    aligned_labels = []
    for i in range(len(tokens)):
        word_ids  = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)              # [CLS], [SEP] → ignore
            elif word_idx != prev_word:
                label_ids.append(labels_enc[i][word_idx])  # first subword → real label
            else:
                label_ids.append(-100)              # continuation subword → ignore
            prev_word = word_idx
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

train_tokenized = tokenize_and_align_labels(train_tokens, train_labels_enc)
dev_tokenized   = tokenize_and_align_labels(dev_tokens,   dev_labels_enc)


# ### building the dataset

# In[10]:


train_dataset = Dataset.from_dict(train_tokenized)
dev_dataset   = Dataset.from_dict(dev_tokenized)


# ### computing metrics

# In[11]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    true_labels = []
    true_preds  = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:                        # skip padding/special tokens
                true_labels.append(id2label[l])
                true_preds.append(id2label[p])

    correct = sum(p == l for p, l in zip(true_preds, true_labels))
    total   = len(true_labels)
    return {"token_accuracy": correct / total if total > 0 else 0.0}


# ### training

# In[14]:


training_args = TrainingArguments(
    output_dir="./ner_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()


# ### predicting on test dataset

# In[ ]:


test_set    = load_iob2("./project/en_ewt-ud-test-masked.iob2")
test_tokens, test_labels = split_tokens_labels(test_set)
test_labels_enc          = encode_labels(test_labels, label2id)
test_tokenized           = tokenize_and_align_labels(test_tokens, test_labels_enc)
test_dataset             = Dataset.from_dict(test_tokenized)

predictions, _, _ = trainer.predict(test_dataset)
predictions       = np.argmax(predictions, axis=-1)

with open("./project/predictions.iob2", "w", encoding="utf-8") as f:
    for sent, pred_seq, label_seq in zip(test_set, predictions, test_tokenized["labels"]):
        word_idx = 0
        for pred, label in zip(pred_seq, label_seq):
            if label != -100 and word_idx < len(sent):
                word  = sent[word_idx][0]
                f.write(f"{word} {id2label[pred]}\n")
                word_idx += 1
        f.write("\n")

