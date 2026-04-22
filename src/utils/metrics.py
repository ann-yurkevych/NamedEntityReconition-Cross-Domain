'''
Has a compute_metrics helper similar to evaluator.
Currently has issues (uses save_path/json without defining/importing them), and it is not the active evaluation path in run_experiment.
'''

from seqeval.metrics import classification_report, f1_score

def compute_metrics(preds, labels, id2label):
    true_preds = []
    true_labels = []

    for pred, label in zip(preds, labels):
        curr_preds = []
        curr_labels = []

        for p, l in zip(pred, label):
            if l != -100:
                curr_preds.append(id2label[p])
                curr_labels.append(id2label[l])

        true_preds.append(curr_preds)
        true_labels.append(curr_labels)

    f1 = f1_score(true_labels, true_preds)
    report = classification_report(true_labels, true_preds)
    report_dict = classification_report(true_labels, true_preds, output_dict=True)
    report_str = classification_report(true_labels, true_preds)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(report_dict, f, indent=4)
            
    return f1, report, report_dict