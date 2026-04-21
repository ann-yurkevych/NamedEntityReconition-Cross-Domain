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

                logits = self.model(input_ids, attention_mask)
                loss = self.loss_fn(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1)
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} loss: {avg:.4f}")
            total_loss += avg

        return total_loss / epochs

    def evaluate(self, dataloader):
        self.model.eval()
        preds, refs = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)

                # Extend as lists of lists, not flat arrays
                preds.extend(predictions.cpu().numpy().tolist())
                refs.extend(labels.cpu().numpy().tolist())

        return preds, refs