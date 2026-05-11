'''
Main supervised experiment entrypoint.
Builds tokenizer, label mapping, model, optimizer, trainer.

Supports three modes:
zero_shot: train on CoNLL, evaluate on CrossNER politics (no CrossNER finetune).
crossner: train and test on CrossNER politics (in-domain).
transfer: train on CoNLL, then finetune on CrossNER politics, then test.

Uses a class-weight trick: label O gets low weight (0.1), so missing entities is penalized relatively more.

Saves final F1 and classification report as JSON.
'''


import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim


from src.data.loader import load_conll2003, load_crossner
from src.data.preprocess import tokenize_and_align_labels
from src.data.dataset import NERDataset
from src.models.bert_tagger import BertForNER
from src.training.trainer import Trainer
from src.utils.metrics import compute_metrics
from src.utils.label_mapping import map_conll_to_politics, get_crossner_labels
from src.training.evaluator import Evaluator


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[Using device: {DEVICE}]")

def build_dataloader(
    texts,
    labels,
    tokenizer,
    label2id,
    batch_size=16,
    label_mapper=None,
    shuffle=True,
):

    if label_mapper is not None:
        labels = [label_mapper(seq) for seq in labels]

    encodings, aligned_labels = tokenize_and_align_labels(
            texts, labels, tokenizer, label2id
        )
    dataset = NERDataset(encodings, aligned_labels, label_mapper=None)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def save_metrics(f1, report, config, preds=None, refs=None, texts=None, labels=None,
                 dev_f1=None, dev_report=None):
    os.makedirs("results/metrics", exist_ok=True)

    filename = f"results/metrics/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "f1": float(f1),
        "config": config,
        "report": report,
    }
    if dev_f1 is not None:
        output["dev_f1"] = float(dev_f1)
        output["dev_report"] = dev_report

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Saved metrics to {filename}]")

    # added saving raw predictions for confusion matrix analysis
    if preds is not None and refs is not None:
        preds_filename = f"results/metrics/preds_{config['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(preds_filename, "w") as f:
            preds_list = [[int(p) for p, l in zip(pred, ref) if l != -100] for pred, ref in zip(preds, refs)]
            refs_list  = [[int(r) for r in ref if r != -100] for ref in refs]
            payload = {"preds": preds_list, "refs": refs_list, "mode": config["mode"]}
            if texts is not None:
                payload["texts"] = [" ".join(sentence) for sentence in texts]
            if labels is not None:
                payload["gold_labels"] = labels
            json.dump(payload, f)
        print(f"[Saved raw predictions to {preds_filename}]")

def run(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"]) # use HuggingFace tokenizer BERT

    # Example label space (should be unified beforehand)
    label_list = config["labels"]
    label2id = {l: i for i, l in enumerate(label_list)} # defines a global label space - important for CrossNER and CoNLL alignment 
    id2label = {i: l for l, i in label2id.items()}

    model = BertForNER(config["model_name"], len(label_list)).to(DEVICE) #BertForNER → encoder + token classification head
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    trainer = Trainer(model, optimizer, DEVICE, label_weights=None)

    if config["mode"] == "zero_shot":
        '''
        1. BERT (CoNLL --> CrossNER zero-shot)

        Pipeline:
        Train: CoNLL2003
        Test: CrossNER
        No finetuning
        '''
        texts, labels = load_conll2003(config["data_dir"])
        train_loader = build_dataloader(texts, labels, tokenizer, label2id, label_mapper=map_conll_to_politics, shuffle=True)

        trainer.train(train_loader, epochs=5)

        _, (dev_texts, dev_labels), (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )
        dev_loader  = build_dataloader(dev_texts,  dev_labels,  tokenizer, label2id, label_mapper=None, shuffle=False)
        test_loader = build_dataloader(test_texts, test_labels, tokenizer, label2id, label_mapper=None, shuffle=False)

        dev_preds,  dev_refs  = trainer.evaluate(dev_loader)
        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "crossner":
        '''
        2. BERT (CrossNER only)

        Pipeline:
        Train: CrossNER
        Test: CrossNER
        '''
        (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id, shuffle=True)
        dev_loader   = build_dataloader(dev_texts,   dev_labels,   tokenizer, label2id, shuffle=False)
        test_loader  = build_dataloader(test_texts,  test_labels,  tokenizer, label2id, shuffle=False)

        trainer.train(train_loader, epochs=15)
        dev_preds,  dev_refs  = trainer.evaluate(dev_loader)
        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "transfer":
    #  train on CoNLL
    # 3. BERT + transfer learning (CoNLL --> CrossNER)
        texts, labels = load_conll2003(config["data_dir"])
        train_loader = build_dataloader(
            texts, labels, tokenizer, label2id, label_mapper=map_conll_to_politics, shuffle=True
        )
        trainer.train(train_loader, epochs=3)

        # Save CoNLL-finetuned encoder
        conll_save_path = "results/models/bert-conll-politics"
        os.makedirs(conll_save_path, exist_ok=True)
        model.bert.save_pretrained(conll_save_path)
        tokenizer.save_pretrained(conll_save_path)
        print(f"[CoNLL-finetuned encoder saved to {conll_save_path}]")

        # finetune on CrossNER politics
        (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )
        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id, shuffle=True)
        dev_loader   = build_dataloader(dev_texts,   dev_labels,   tokenizer, label2id, shuffle=False)
        test_loader  = build_dataloader(test_texts,  test_labels,  tokenizer, label2id, shuffle=False)
        trainer.train(train_loader, epochs=15)
        dev_preds,  dev_refs  = trainer.evaluate(dev_loader)
        preds, refs = trainer.evaluate(test_loader)
        
    elif config["mode"] == "jointly_train":
        '''
        4. BERT (Jointly Train on Both Source and Target Domains)

        Pipeline:
        Train: CoNLL2003 + CrossNER politics simultaneously
        Test: CrossNER politics
        
        Following the paper: upsample CrossNER target domain data 
        to balance source and target domain data samples.
        '''
        # Load both datasets
        conll_texts, conll_labels = load_conll2003(config["data_dir"])
        (train_texts, train_labels), _, (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        # Map CoNLL labels to CrossNER politics label space
        conll_labels_mapped = [map_conll_to_politics(seq) for seq in conll_labels]

        # Upsample CrossNER to match CoNLL size (as described in the paper)
        crossner_size = len(train_texts)
        conll_size = len(conll_texts)

        upsample_factor = conll_size // crossner_size
        remainder = conll_size % crossner_size

        upsampled_texts = train_texts * upsample_factor + train_texts[:remainder]
        upsampled_labels = train_labels * upsample_factor + train_labels[:remainder]

        print(f"[CoNLL size: {conll_size}, CrossNER size: {crossner_size}, "
            f"Upsampled CrossNER size: {len(upsampled_texts)}]")

        # Combine CoNLL and upsampled CrossNER
        combined_texts = conll_texts + upsampled_texts
        combined_labels = conll_labels_mapped + upsampled_labels

        # Shuffle combined dataset by creating paired list
        combined = list(zip(combined_texts, combined_labels))
        import random
        random.seed(42)
        random.shuffle(combined)
        combined_texts, combined_labels = zip(*combined)
        combined_texts = list(combined_texts)
        combined_labels = list(combined_labels)

        train_loader = build_dataloader(
            combined_texts, combined_labels, tokenizer, label2id,
            label_mapper=None, shuffle=True
        )
        test_loader = build_dataloader(
            test_texts, test_labels, tokenizer, label2id,
            label_mapper=None, shuffle=False
        )

        trainer.train(train_loader, epochs=15)
        dev_preds,  dev_refs  = trainer.evaluate(dev_loader)
        preds, refs = trainer.evaluate(test_loader)
        
    elif config["mode"] == "jointly_train":
        '''
        4. BERT (Jointly Train on Both Source and Target Domains)

        Pipeline:
        Train: CoNLL2003 + CrossNER politics simultaneously
        Test: CrossNER politics
        
        Following the paper: upsample CrossNER target domain data 
        to balance source and target domain data samples.
        '''
        # Load both datasets
        conll_texts, conll_labels = load_conll2003(config["data_dir"])
        (train_texts, train_labels), _, (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        # Map CoNLL labels to CrossNER politics label space
        conll_labels_mapped = [map_conll_to_politics(seq) for seq in conll_labels]

        # Upsample CrossNER to match CoNLL size (as described in the paper)
        crossner_size = len(train_texts)
        conll_size = len(conll_texts)

        upsample_factor = conll_size // crossner_size
        remainder = conll_size % crossner_size

        upsampled_texts = train_texts * upsample_factor + train_texts[:remainder]
        upsampled_labels = train_labels * upsample_factor + train_labels[:remainder]

        print(f"[CoNLL size: {conll_size}, CrossNER size: {crossner_size}, "
            f"Upsampled CrossNER size: {len(upsampled_texts)}]")

        # Combine CoNLL and upsampled CrossNER
        combined_texts = conll_texts + upsampled_texts
        combined_labels = conll_labels_mapped + upsampled_labels

        # Shuffle combined dataset by creating paired list
        combined = list(zip(combined_texts, combined_labels))
        import random
        random.seed(42)
        random.shuffle(combined)
        combined_texts, combined_labels = zip(*combined)
        combined_texts = list(combined_texts)
        combined_labels = list(combined_labels)

        train_loader = build_dataloader(
            combined_texts, combined_labels, tokenizer, label2id,
            label_mapper=None, shuffle=True
        )
        test_loader = build_dataloader(
            test_texts, test_labels, tokenizer, label2id,
            label_mapper=None, shuffle=False
        )

        trainer.train(train_loader, epochs=15)
        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "dapt":
        '''
        5. BERT + DAPT (domain-adapted BERT --> CrossNER)

        Pipeline:
        Step 1: MLM pretraining on domain corpus (run separately via run_dapt.py)
        Step 2: Load DAPT model weights, tokenizer from CoNLL fine-tuned BERT
        Step 3: Finetune on CrossNER
        '''

        tokenizer = AutoTokenizer.from_pretrained(config["conll_model_path"])
        model = BertForNER(config["dapt_model_path"], len(label_list)).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5) 
        trainer = Trainer(model, optimizer, DEVICE, label_weights=None)

        (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id, shuffle=True)
        dev_loader   = build_dataloader(dev_texts,   dev_labels,   tokenizer, label2id, shuffle=False)
        test_loader  = build_dataloader(test_texts,  test_labels,  tokenizer, label2id, shuffle=False)

        trainer.train(train_loader, epochs=15)
        dev_preds,  dev_refs  = trainer.evaluate(dev_loader)
        preds, refs = trainer.evaluate(test_loader)

    evaluator = Evaluator(id2label)

    # Dev evaluation is observational only — consistent with the CrossNER paper's
    # fixed-step protocol (no checkpoint selection on dev). Logged for monitoring.
    dev_f1, dev_report = evaluator.evaluate(dev_preds, dev_refs)
    print("Dev F1:", dev_f1)
    print(dev_report)

    f1, report = evaluator.evaluate(preds, refs)
    print("Test F1:", f1)
    print(report)

    save_metrics(f1, report, config, preds=preds, refs=refs, texts=test_texts, labels=test_labels,
                 dev_f1=dev_f1, dev_report=dev_report)


if __name__ == "__main__":
    config = {
        "model_name": "bert-base-cased",
        "data_dir": "data/raw",
        "domain": "politics",
        "mode": "transfer", # change to: crossner, transfer, jointly_train, dapt
        "labels": get_crossner_labels(),
        "dapt_model_path": "results/models/bert-dapt-politics",
        "conll_model_path": "results/models/bert-conll-politics",
    }

    run(config)

