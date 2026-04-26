'''
Trainer.train:
    forward pass, cross-entropy over flattened token logits.
    backward, optimizer step.
    prints average epoch loss.
Trainer.evaluate:
    argmax predictions.
    returns raw predicted label ids and reference label ids, sequence by sequence.
'''


import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, label_weights=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = CrossEntropyLoss(ignore_index=-100, weight=label_weights)

    def train(self, dataloader, epochs=5):
        self.model.train()
        total_loss = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask) # forward pass
                loss = self.loss_fn(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(dataloader) # prints epoch-average loss and accumulates it.
            print(f"Epoch {epoch+1} loss: {avg:.4f}")
            total_loss += avg

        return total_loss / epochs # returns average loss across epochs.

    def evaluate(self, dataloader):
        self.model.eval() # dropout off, no backprop, etc.
        preds, refs = [], []

        with torch.no_grad(): # no grad for efficent inference
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask) # forward pass then argmax over label dimension.
                predictions = torch.argmax(logits, dim=-1)

                # Extend as lists of lists, not flat arrays
                preds.extend(predictions.cpu().numpy().tolist())
                refs.extend(labels.cpu().numpy().tolist())

        return preds, refs # returns raw integer predictions and references.