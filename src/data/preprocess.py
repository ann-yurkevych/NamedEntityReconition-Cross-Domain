'''
tokenize_and_align_labels:
tokenizes pre-split word lists with is_split_into_words.
assigns true label only to first subword.
uses -100 for special tokens and non-first subwords so loss ignores them.
'''


def tokenize_and_align_labels(texts, labels, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        is_split_into_words=True
    )

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