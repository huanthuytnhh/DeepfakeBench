"""FAS metrics — APCER / BPCER / ACER / EER / AUC (ISO/IEC 30107-3 style).

Convention: spoof/attack = POSITIVE class (label 1), live/bona-fide = NEGATIVE (label 0)
— consistent with the deepfake detector (fake=1). A sample is predicted "attack" iff prob >= thr.
"""
import numpy as np
from sklearn import metrics as skm


def auc_score(labels, probs):
    return float(skm.roc_auc_score(np.asarray(labels), np.asarray(probs)))


def eer_threshold(labels, probs):
    """Return (EER, threshold) at the operating point where FAR == FRR."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    fpr, tpr, thr = skm.roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def apcer_bpcer(labels, probs, thr):
    """APCER = attacks(1) wrongly accepted as bona-fide; BPCER = bona-fide(0) wrongly rejected."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    pred_attack = probs >= thr
    attack, bona = labels == 1, labels == 0
    apcer = float(np.mean(~pred_attack[attack])) if attack.any() else float("nan")
    bpcer = float(np.mean(pred_attack[bona])) if bona.any() else float("nan")
    return apcer, bpcer


def evaluate(dev_labels, dev_probs, test_labels, test_probs):
    """Fix threshold @ EER on DEV, then report TEST APCER/BPCER/ACER + threshold-free EER/AUC.
    (At this threshold ACER == HTER since spoof is the positive class.)"""
    _, thr = eer_threshold(dev_labels, dev_probs)
    apcer, bpcer = apcer_bpcer(test_labels, test_probs, thr)
    eer, _ = eer_threshold(test_labels, test_probs)
    return {
        "threshold@dev_eer": thr,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "eer": eer,
        "auc": auc_score(test_labels, test_probs),
    }
