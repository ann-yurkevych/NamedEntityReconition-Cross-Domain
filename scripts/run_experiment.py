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

def build_dataloader(texts, labels, tokenizer, label2id, batch_size=16, label_mapper=None):
    encodings, aligned_labels = tokenize_and_align_labels(
        texts, labels, tokenizer, label2id
    )
    dataset = NERDataset(encodings, aligned_labels, label_mapper=label_mapper)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def save_metrics(f1, report, config, preds=None, refs=None):
    os.makedirs("results/metrics", exist_ok=True)

    filename = f"results/metrics/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output = {
        "f1": float(f1),
        "config": config,
        "report": report
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[Saved metrics to {filename}]")

    # added saving raw predictions for confusion matrix analysis
    if preds is not None and refs is not None:
        preds_filename = f"results/metrics/preds_{config['mode']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(preds_filename, "w") as f:
            preds_list = [list(map(int, p)) for p in preds]
            refs_list = [list(map(int, r)) for r in refs]
            json.dump(
                {"preds": preds_list, "refs": refs_list, "mode": config["mode"]},
                f,
            )
        print(f"[Saved raw predictions to {preds_filename}]")

def run(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Example label space (should be unified beforehand)
    label_list = config["labels"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    # Give O a lower weight so the model is penalized more for missing entities
    label_weights = torch.ones(len(label_list))
    label_weights[label2id["O"]] = 0.1
    label_weights = label_weights.to(DEVICE)

    model = BertForNER(config["model_name"], len(label_list)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    trainer = Trainer(model, optimizer, DEVICE, label_weights=label_weights)

    if config["mode"] == "zero_shot":
        '''
        1. BERT (CoNLL --> CrossNER zero-shot)

        Pipeline:
        Train: CoNLL2003
        Test: CrossNER
        No finetuning
        '''
        texts, labels = load_conll2003(config["data_dir"])
        train_loader = build_dataloader(texts, labels, tokenizer, label2id, label_mapper=map_conll_to_politics)

        trainer.train(train_loader, epochs=5)

        (test_texts, test_labels), _, _ = load_crossner(
            config["data_dir"], config["domain"]
        )
        test_loader = build_dataloader(test_texts, test_labels, tokenizer, label2id, label_mapper=None)

        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "crossner":
        '''
        2. BERT (CrossNER only)

        Pipeline:
        Train: CrossNER
        Test: CrossNER
        '''
        (train_texts, train_labels), _, (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id)
        test_loader = build_dataloader(test_texts, test_labels, tokenizer, label2id)

        trainer.train(train_loader)
        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "transfer":
        '''
        3. Transfer (CoNLL --> CrossNER)
        '''
        # Step 1: train on CoNLL
        texts, labels = load_conll2003(config["data_dir"])
        train_loader = build_dataloader(
            texts, labels, tokenizer, label2id, label_mapper=map_conll_to_politics
        )
        trainer.train(train_loader)

        # Save CoNLL-finetuned model so DAPT mode can reuse its tokenizer
        conll_save_path = "results/models/bert-conll-politics"
        os.makedirs(conll_save_path, exist_ok=True)
        model.bert.save_pretrained(conll_save_path)
        tokenizer.save_pretrained(conll_save_path)
        print(f"[CoNLL-finetuned encoder saved to {conll_save_path}]")

        # Step 2: finetune on CrossNER politics
        (train_texts, train_labels), _, (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )
        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id)
        test_loader = build_dataloader(test_texts, test_labels, tokenizer, label2id)
        trainer.train(train_loader)
        preds, refs = trainer.evaluate(test_loader)

    elif config["mode"] == "dapt":
        '''
        4. BERT + DAPT (domain-adapted BERT --> CrossNER)

        Pipeline:
        Step 1: MLM pretraining on domain corpus (run separately via run_dapt.py)
        Step 2: Load DAPT model weights, tokenizer from CoNLL fine-tuned BERT
        Step 3: Finetune on CrossNER
        '''

        tokenizer = AutoTokenizer.from_pretrained(config["conll_model_path"])
        model = BertForNER(config["dapt_model_path"], len(label_list)).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5) 
        trainer = Trainer(model, optimizer, DEVICE, label_weights=label_weights)

        (train_texts, train_labels), _, (test_texts, test_labels) = load_crossner(
            config["data_dir"], config["domain"]
        )

        train_loader = build_dataloader(train_texts, train_labels, tokenizer, label2id)
        test_loader = build_dataloader(test_texts, test_labels, tokenizer, label2id)

        trainer.train(train_loader)
        preds, refs = trainer.evaluate(test_loader)

    evaluator = Evaluator(id2label)
    f1, report = evaluator.evaluate(preds, refs)
    print("F1:", f1)
    print(report)
    save_metrics(f1, report, config, preds=preds, refs=refs)


if __name__ == "__main__":
    config = {
        "model_name": "bert-base-cased",
        "data_dir": "data/raw",
        "domain": "politics",
        "mode": "crossner", # change to: crossner, zero_shot, transfer, dapt
        "labels": get_crossner_labels(),
        "dapt_model_path": "results/models/bert-dapt-politics",
        "conll_model_path": "results/models/bert-conll-politics",
    }

    run(config)

    
    

'''
4. BERT + DAPT

Pipeline:
Step 1: LM pretraining on domain corpus
Step 2: Load weights
Step 3: Train on CrossNER
'''

