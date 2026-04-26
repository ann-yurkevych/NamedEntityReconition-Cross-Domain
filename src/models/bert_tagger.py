'''
BertForNER = a standard token-classification architecture for NER::
    backbone: AutoModel (BERT encoder).
    dropout.
    linear classifier from hidden_size → num_labels.
Outputs token-level logits for each sequence position
'''


import torch.nn as nn
from transformers import AutoModel

class BertForNER(nn.Module): # BertForNER inherits from nn.Module, so it behaves like any PyTorch model: train(), eval(), parameters(), forward(), etc.
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name) # Load pre-trained encoder- BERT model --> AutoModel gives hidden states, not task head. --> So this file defines its own NER head manually.
        hidden_size = self.bert.config.hidden_size # Read backbone embedding size
        self.dropout = nn.Dropout(0.1) # add dropout for regularization - During training: randomly zeroes 10% of features. druing eval: does nothing. helps prevent overfitting.
        self.classifier = nn.Linear(hidden_size, num_labels) # Add token classifier head - linear layer that maps from hidden_size → num_labels (number of entity tags)
        '''
        Intuition (shape):
            Input token representation at each position: [hidden_size]
            Output per token: [num_labels]
        '''

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) # run encoder
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output) # classify each token
        return logits