'''
Decodes integer ids back to tag strings while filtering out -100.
Computes entity-level F1 and detailed seqeval report.
'''


from seqeval.metrics import classification_report, f1_score

class Evaluator:
    def __init__(self, id2label):
        self.id2label = id2label

    def decode(self, preds, labels):
        true_preds = []
        true_labels = []

        for pred, label in zip(preds, labels):
            filtered_preds = []
            filtered_labels = []
            for p, l in zip(pred, label):
                if l != -100:
                    filtered_preds.append(self.id2label[p])
                    filtered_labels.append(self.id2label[l])
            true_preds.append(filtered_preds)
            true_labels.append(filtered_labels)

        return true_labels, true_preds

    def evaluate(self, preds, labels):
        gold, pred = self.decode(preds, labels)

        f1 = f1_score(gold, pred)
        report = classification_report(gold, pred)

        return f1, report