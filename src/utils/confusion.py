from typing import List, Tuple, Dict
import numpy as np


def extract_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """Convert BIO tag sequence -> list of (start, end, type) spans."""
    spans = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            ent_type = lab[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{ent_type}":
                j += 1
            spans.append((i, j, ent_type))
            i = j
        else:
            i += 1
    return spans


def build_entity_confusion_matrix(
    preds: List[List[str]],
    refs: List[List[str]],
    entity_types: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build entity-level confusion matrix.
    Rows = gold types, cols = predicted types. Last row/col = 'O' (missed/spurious).
    """
    labels = entity_types + ["O"]
    label2idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for pred_sent, ref_sent in zip(preds, refs):
        gold_spans = {(s, e): t for s, e, t in extract_spans(ref_sent)}
        pred_spans = {(s, e): t for s, e, t in extract_spans(pred_sent)}

        all_positions = set(gold_spans) | set(pred_spans)
        for pos in all_positions:
            g_type = gold_spans.get(pos, "O")
            p_type = pred_spans.get(pos, "O")
            if g_type in label2idx and p_type in label2idx:
                matrix[label2idx[g_type], label2idx[p_type]] += 1

    return matrix, labels


def ids_to_bio(pred_id_seqs, ref_id_seqs, id2label):
    """
    Convert integer ID sequences to BIO strings.
    Drops positions where ref == -100 (subword continuations / special tokens)
    from BOTH preds and refs to maintain alignment.
    Returns (pred_bio_seqs, ref_bio_seqs).
    """
    pred_out, ref_out = [], []
    for p_seq, r_seq in zip(pred_id_seqs, ref_id_seqs):
        p_bio, r_bio = [], []
        for p, r in zip(p_seq, r_seq):
            if r == -100:
                continue
            p_bio.append(id2label.get(p, "O"))
            r_bio.append(id2label.get(r, "O"))
        pred_out.append(p_bio)
        ref_out.append(r_bio)
    return pred_out, ref_out


def highlight_hierarchy_confusions(
    matrix: np.ndarray,
    labels: List[str],
    hierarchy_pairs: List[Tuple[str, str]],
) -> Dict[str, int]:
    """Count confusions for specific (fine, coarse) label pairs."""
    label2idx = {lab: i for i, lab in enumerate(labels)}
    result = {}
    for fine, coarse in hierarchy_pairs:
        if fine in label2idx and coarse in label2idx:
            count = int(matrix[label2idx[fine], label2idx[coarse]])
            result[f"{fine} -> {coarse}"] = count
    return result