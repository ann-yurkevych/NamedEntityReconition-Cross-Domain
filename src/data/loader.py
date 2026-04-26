'''
read_conll_format parses sentence-separated token/tag files.
load_conll2003 reads only CoNLL train split from data/raw/conll2003/train.txt.
load_crossner reads train/dev/test for one domain.
load_unlabeled reads unlabeled domain text variants for DAPT.
'''


import os
from pathlib import Path


def load_unlabeled(data_dir: str, domain: str, variants: list[str] | None = None) -> list[str]:
    if variants is None:
        variants = ['domainlevel', 'entitylevel', 'integrated', 'tasklevel']

    domain_dir = Path(data_dir) / 'unlabeled' / domain
    texts: list[str] = []

    for variant in variants:
        '''
        Removes leading/trailing whitespace
        Skips empty lines
        Appends each remaining line as a full string
        
        Example Output: [
        "The International became a significant political force , and Karl Marx a leading figure and a member of its General Council .",
        "In the Spanish Civil War , anarchists and syndicalists ( CNT and Federación Anarquista Ibérica ) once again allied themselves with various currents of leftists .",
        "In response to the army rebellion , an anarchist-inspired movement of peasants and workers , supported by armed militias , took control of Barcelona and of large areas of rural Spain where they collectivised the land ."
        ]
        --> This is suitable for domain-adaptive pretraining inputs!!!!!!!!!
        '''
        fname = domain_dir / f"{domain}_{variant}.txt"
        if not fname.exists():
            continue
        with open(fname, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    return texts


def read_conll_format(path):
    '''
    Example output:
    texts = [
    ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",",
    "CHINA", "IN", "SURPRISE", "DEFEAT", "."],

    ["Nadim", "Ladki"],

    ["AL-AIN", ",", "United", "Arab", "Emirates", "1996-12-06"]
    ]
    
    labels = [
    ["O", "O", "B-LOC", "O", "O", "O", "O",
    "B-PER", "O", "O", "O", "O"],

    ["B-PER", "I-PER"],

    ["B-LOC", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
    ]
    '''
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
    return read_conll_format(os.path.join(data_dir, "conll2003", "train.txt"))


def load_crossner(data_dir, domain):
    base = os.path.join(data_dir, "crossner", domain)
    train = read_conll_format(os.path.join(base, "train.txt"))
    dev = read_conll_format(os.path.join(base, "dev.txt"))
    test = read_conll_format(os.path.join(base, "test.txt"))
    return train, dev, test