"""
Label mapping between CoNLL-2003 (coarse) and CrossNER Politics (fine-grained).

CoNLL-2003 trains on 4 types: PER, ORG, LOC, MISC.
CrossNER Politics evaluates on 9 fine-grained types.
"""

# Hierarchy mapping: CoNLL-2003 → CrossNER Politics
HIERARCHY: dict[str, list[str]] = {
    "PER":  ["person", "politician"],
    "ORG":  ["organisation", "politicalparty"],
    "LOC":  ["location", "country"],
    "MISC": ["misc", "event", "election"],
}

# Reverse mapping: CrossNER → parent CoNLL type
CROSSNER_TO_PARENT: dict[str, str] = {
    "person":        "PER",
    "politician":    "PER",
    "organisation":  "ORG",
    "politicalparty":"ORG",
    "location":      "LOC",
    "country":       "LOC",
    "misc":          "MISC",
    "event":         "MISC",
    "election":      "MISC",
}

# Fine-grained types absent from CoNLL-2003
FINEGRAINED: set[str] = {"politician", "politicalparty", "country", "event", "election"}


def are_hierarchically_related(pred_label: str, gold_label: str) -> bool:
    """Check if pred and gold labels share a parent category."""
    parent_pred = CROSSNER_TO_PARENT.get(pred_label, pred_label)
    parent_gold = CROSSNER_TO_PARENT.get(gold_label, gold_label)
    return parent_pred == parent_gold and pred_label != gold_label


def is_finegrained(label: str) -> bool:
    return label in FINEGRAINED


# ---------------------------------------------------------------------------
# IOB2 sequence helpers
# ---------------------------------------------------------------------------

def _map_tag(tag: str, mapping: dict[str, str]) -> str:
    if tag == "O":
        return "O"
    prefix, etype = tag.split("-", 1)
    return f"{prefix}-{mapping.get(etype.lower(), etype)}"


def collapse_to_coarse(tags: list[str]) -> list[str]:
    """
    Map CrossNER Politics IOB2 tags → CoNLL-2003 coarse types.

    e.g. ["B-politician", "I-politician", "O"] → ["B-PER", "I-PER", "O"]

    Use this when evaluating a CoNLL-trained model on the Politics test set
    at the coarse level.
    """
    return [_map_tag(t, CROSSNER_TO_PARENT) for t in tags]


def map_conll_to_politics(tags: list[str]) -> list[str]:
    """
    Map CoNLL-2003 IOB2 predictions → CrossNER Politics generic types.

    Each coarse type maps to the generic (non-specialised) fine-grained type:
        PER  → person
        ORG  → organisation
        LOC  → location
        MISC → misc
    """
    default_map = {coarse: fine_list[0] for coarse, fine_list in HIERARCHY.items()}
    return [_map_tag(t, default_map) for t in tags]


def coarse_eval_pairs(
    gold_tags: list[list[str]], pred_tags: list[list[str]]
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Collapse both gold (Politics fine-grained) and pred (CoNLL coarse) tag
    sequences to CoNLL-2003 coarse types, ready for seqeval.

    Returns (gold_coarse, pred_coarse).
    """
    return (
        [collapse_to_coarse(seq) for seq in gold_tags],
        [collapse_to_coarse(seq) for seq in pred_tags],
    )
