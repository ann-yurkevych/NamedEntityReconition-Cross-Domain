import os

def read_conll_format(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    texts.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                tokens.append(splits[0])
                tags.append(splits[-1])

    return texts, labels


def load_conll2003(data_dir):
    return read_conll_format(os.path.join(data_dir, "train.txt"))


def load_crossner(data_dir, domain):
    base = os.path.join(data_dir, "crossner", domain)
    train = read_conll_format(os.path.join(base, "train.txt"))
    dev = read_conll_format(os.path.join(base, "dev.txt"))
    test = read_conll_format(os.path.join(base, "test.txt"))
    return train, dev, test