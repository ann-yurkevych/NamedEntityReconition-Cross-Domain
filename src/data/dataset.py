'''
NERDataset wraps tokenized encodings and aligned labels into a PyTorch Dataset, accessible sample-by-sample for the training loop.
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
                encodings = dictionary-like structure after running preprocess.py
                    encodings = {
                        "input_ids": [...],
                        "attention_mask": [...],
                        # sometimes also:
                        "token_type_ids": [...]
                        }
        labels: list of label sequences
        label_mapper: optional function to normalize labels
        """
        self.encodings = encodings

        if label_mapper is not None:
            self.labels = [label_mapper(seq) for seq in labels] # Used when training on CoNLL but targeting CrossNER --> example: "B-LOC → B-LOCATION" or "collapsing hierarchy" 
        else:
            self.labels = labels

    def __len__(self): # Number of samples = number of sentences
        return len(self.labels)

    def __getitem__(self, idx): # This returns training examples        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item