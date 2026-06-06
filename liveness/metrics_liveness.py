"""metrics_liveness.py — chỉ số FAS chuẩn ISO/IEC 30107-3 (độc lập, đuôi _liveness).

Quy ước: spoof/attack = lớp DƯƠNG (label 1), live/bona-fide = âm (0). Đoán "attack" khi prob >= thr.
"""
import numpy as np
from sklearn import metrics as skm


def auc_liveness(labels, probs):
    return float(skm.roc_auc_score(np.asarray(labels), np.asarray(probs)))


def eer_threshold_liveness(labels, probs):
    """(EER, threshold) tại điểm FAR == FRR."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    fpr, tpr, thr = skm.roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def apcer_bpcer_liveness(labels, probs, thr):
    """APCER = attack(1) bị nhận nhầm là bona-fide; BPCER = bona-fide(0) bị từ chối."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    pred_attack = probs >= thr
    attack, bona = labels == 1, labels == 0
    apcer = float(np.mean(~pred_attack[attack])) if attack.any() else float("nan")
    bpcer = float(np.mean(pred_attack[bona])) if bona.any() else float("nan")
    return apcer, bpcer


def evaluate_liveness(dev_labels, dev_probs, test_labels, test_probs):
    """Cố định ngưỡng @EER trên DEV, báo cáo TEST APCER/BPCER/ACER + EER/AUC (độc lập ngưỡng)."""
    _, thr = eer_threshold_liveness(dev_labels, dev_probs)
    apcer, bpcer = apcer_bpcer_liveness(test_labels, test_probs, thr)
    eer, _ = eer_threshold_liveness(test_labels, test_probs)
    return {
        "threshold@dev_eer": thr,
        "apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2,
        "eer": eer, "auc": auc_liveness(test_labels, test_probs),
    }
