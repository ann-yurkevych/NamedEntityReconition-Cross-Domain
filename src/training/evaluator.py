'''
Decodes integer ids back to tag strings while filtering out -100.
Computes entity-level F1 and detailed seqeval report.
'''


from seqeval.metrics import classification_report, f1_score

class Evaluator:
    '''
    Evaluator expects input in the form:
        preds: List[List[int]]   # predicted label IDs
        labels: List[List[int]]  # true label IDs (with -100)
    Example:
        preds = [
        [1, 0, 0, 2, 0],
        [0, 1, 2, 0]
        ]

        labels = [
        [-100, 1, -100, 2, -100],
        [-100, 1, 2, -100]
        ]
    And mapping:
        id2label = {
        0: "O",
        1: "B-PER",
        2: "I-PER"
        }
        
    The decode method will filter out -100 and convert IDs to tags:
        Example output of decode(from above) - after filtering:
            filtered_preds  = ["O", "I-PER"]
            filtered_labels = ["B-PER", "I-PER"]
    
    Then evaluate 
    '''
    
    def __init__(self, id2label):
        self.id2label = id2label # stores id2label dictionary for integer-to-tag conversion.

    def decode(self, preds, labels):
        true_preds = []
        true_labels = []

        for pred, label in zip(preds, labels): # Iterates sentence-by-sentence over preds and labels.
            filtered_preds = []
            filtered_labels = []
            for p, l in zip(pred, label): #Iterates token-by-token with zip(pred, label).
                if l != -100: #Keeps only tokens where ground-truth label is not -100:
                    filtered_preds.append(self.id2label[p])
                    filtered_labels.append(self.id2label[l])
            true_preds.append(filtered_preds)
            true_labels.append(filtered_labels)

        return true_labels, true_preds # returns true labels(gold) and true predictions(pred) as lists of lists of tag strings, aligned and filtered

    def evaluate(self, preds, labels):
        gold, pred = self.decode(preds, labels) #Calls decode to get filtered string tags

        f1 = f1_score(gold, pred) #Computes entity-level F1 with seqeval.
        report = classification_report(gold, pred) #Computes full classification report with seqeval.
        # seqeval is a Python library specifically for sequence labeling evaluation, like: NER or POS tagging. It provides functions to compute metrics like F1, precision, recall, and detailed classification reports that include per-entity-type performance.
        # Example:
        #    B-PER I-PER   → correct entity
        #    B-PER O       → incorrect (broken span)
        
        return f1, report