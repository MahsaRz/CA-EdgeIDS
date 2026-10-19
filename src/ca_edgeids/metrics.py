"""Metrics always use an explicit fixed class list and probability scores."""
import numpy as np
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score


def evaluate(y,probabilities,names):
    p = np.asarray(probabilities)
    if p.shape != (len(y),len(names)) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError("Invalid probability array")
    pred = p.argmax(1)
    ids = np.arange(len(names))
    report = classification_report(y,pred,labels=ids,target_names=names,output_dict=True,zero_division=0)
    ap = {}
    for i,name in enumerate(names):
        binary = (y == i).astype(int)
        ap[name] = float(average_precision_score(binary,p[:,i])) if len(np.unique(binary))==2 else None
    cm = confusion_matrix(y,pred,labels=ids)
    fpr = {}
    for i,name in enumerate(names):
        fp = cm[:,i].sum()-cm[i,i]
        tn = cm.sum()-cm[i,:].sum()-fp
        fpr[name] = float(fp/(fp+tn)) if fp+tn else None
    benign = names.index("Benign") if "Benign" in names else None
    benign_n = int(np.sum(y==benign)) if benign is not None else 0
    return {
        "test_n":len(y), "macro_f1":float(f1_score(y,pred,labels=ids,average="macro",zero_division=0)),
        "weighted_f1":float(f1_score(y,pred,labels=ids,average="weighted",zero_division=0)),
        "macro_precision":report["macro avg"]["precision"],"macro_recall":report["macro avg"]["recall"],
        "weighted_precision":report["weighted avg"]["precision"],"weighted_recall":report["weighted avg"]["recall"],
        "per_class":report,"confusion_matrix":cm.tolist(),"average_precision_ovr":ap,
        "fpr_ovr":fpr,"attack_false_positive_rate":float(np.sum((y==benign)&(pred!=benign))/benign_n) if benign_n else None,
        "missing_test_classes":[names[i] for i in ids if not np.any(y==i)],
        "pr_metric_definition":"non-interpolated average precision, not trapezoidal PR-AUC",
    }
