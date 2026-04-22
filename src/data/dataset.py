'''
NERDataset wraps tokenized encodings and aligned labels into a PyTorch Dataset.
Optional label_mapper lets you remap tag schema before training (important for CoNLL → CrossNER compatibility)
'''


import torch
from src.utils.label_mapping import (
    conll_to_crossner,
    collapse_to_coarse,
    map_conll_to_politics
)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, label_mapper=None):
        """
        encodings: tokenizer outputs
        labels: list of label sequences
        label_mapper: optional function to normalize labels
        """
        self.encodings = encodings

        if label_mapper is not None:
            self.labels = [label_mapper(seq) for seq in labels]
        else:
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item