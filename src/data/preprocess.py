'''
tokenize_and_align_labels:
tokenizes pre-split word lists with is_split_into_words.
assigns true label only to first subword.
uses -100 for special tokens and non-first subwords so loss ignores them.
'''


def tokenize_and_align_labels(texts, labels, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        texts, # after running the loader.py
        truncation=True,
        padding=True,
        is_split_into_words=True
    )
    '''
    Tokenized_inputs - This is a dictionary-like object (specifically a HuggingFace BatchEncoding) that contains multiple aligned arrays.
    The most important fields are:
    
    - input_ids: integers representing tokens in BERT’s vocabulary.
    input_ids = [
    [101, 4715, 118, 3943, 1234, ..., 102, 0, 0],
    [101, 12345, 6789, 102, 0, 0, 0]
    ]
    
    -attention_mask: binary mask indicating which tokens are real (1) vs padding (0).
    attention_mask = [
    [1, 1, 1, 1, 1, ..., 1, 0, 0],
    [1, 1, 1, 1, 0, 0, 0]
    ]
    
    -word_ids: for each token, indicates which original word it came from (None for special tokens).
    word_ids = [None, 0, 1, 2, 3, 3, 4, ..., None, None]
    where: 
    "None" : special tokens ([CLS], [SEP], padding)
    "0" : belongs to word 0
    "3, 3" : two subwords from the same original word
    
    WE NEED THIS BECAUSE:
    -->  BERT does subword tokenization.
    Example: "unbelievable" → ["un", "##bel", "##ievable"]
    So: 
        texts = [["unbelievable"]]
        labels = [["O"]]
    Becomes: 
        tokens = ["[CLS]", "un", "##bel", "##ievable", "[SEP]"]
        word_ids = [None, 0, 0, 0, None]
    '''
    

    aligned_labels = []

    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], label2id["O"])) # fallback to 0
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    return tokenized_inputs, aligned_labels

'''
what does our alignment look like after tokenization?

"None" --> ignore (-100)
first subword of a word --> assign label
subsequent subwords --> ignore (-100)

EXAMPLE:

input:
    tokens = ["unbelievable"]
    labels = ["O"]
    
after tokenization:
    word_ids = [None, 0, 0, 0, None]

aligned labels:
    [-100, label2id["O"], -100, -100, -100]
    
    
    
ANOTHER EXAMPLE:

input:
    texts = [["Nadim", "Ladki"]]
    labels = [["B-PER", "I-PER"]]
    
tokenized (simplified):
    tokens = ["[CLS]", "Na", "##dim", "Lad", "##ki", "[SEP]"]
    word_ids = [None, 0, 0, 1, 1, None]

output tensors:
    input_ids = [101, 1234, 5678, 9101, 1121, 102]
    attention_mask = [1, 1, 1, 1, 1, 1]
    label_ids = [-100, label2id["B-PER"], -100, label2id["I-PER"], -100, -100]
    
    

WHY "-100" SPECIFICALLY?
--> Because PyTorch's CrossEntropyLoss ignores targets with the value -100 by default.
'''
