"""
eval_and_viz.py — run a TRAINED checkpoint on the test sets, DUMP raw scores, and SAVE result figures.
Mirrors test.py's loading/inference (run from the REPO ROOT, like test.py):

  python training/eval_and_viz.py \
      --detector_path ./training/config/detector/efficientnetb4_sfdct.yaml \
      --weights_path  <log_dir>/.../ckpt_best.pth \
      --test_dataset  FaceForensics++ Celeb-DF-v2 \
      --out           ./viz_out/sfdct

Produces in --out:
  scores_<dataset>.npz   (prob, label, feat — for any later plot)
  roc_ekyc.png           (C2: ROC + FPR≤5% eKYC line, TPR@FPR table)
  auc_bar.png            (C1: per-dataset AUC, within vs cross)
  tsne.png               (C3: fused-feature separation, optional)
  gate_alpha.png         (B1: the zero-init gate after training)
  results.json           (auc/eer/ap + video-auc + TPR@FPR=5%/1% per dataset)
"""
import os, json, argparse
import numpy as np
import yaml
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, average_precision_score

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR
from metrics.utils import get_test_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--detector_path", required=True)
ap.add_argument("--weights_path", required=True)
ap.add_argument("--test_dataset", nargs="+", default=None)
ap.add_argument("--out", default="./viz_out/eval")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def load_config():
    with open(args.detector_path) as f:
        config = yaml.safe_load(f)
    with open("./training/config/test_config.yaml") as f:
        config.update(yaml.safe_load(f))
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    return config


def make_loaders(config):
    loaders = {}
    for name in config["test_dataset"]:
        c = config.copy(); c["test_dataset"] = name
        ds = DeepfakeAbstractBaseDataset(config=c, mode="test")
        loaders[name] = torch.utils.data.DataLoader(
            ds, batch_size=config["test_batchSize"], shuffle=False,
            num_workers=int(config["workers"]), collate_fn=ds.collate_fn, drop_last=False)
    return loaders


@torch.no_grad()
def infer(model, loader):
    probs, labels, feats = [], [], []
    for data_dict in tqdm(loader, total=len(loader)):
        label = torch.where(data_dict["label"] != 0, 1, 0)
        data_dict["image"], data_dict["label"] = data_dict["image"].to(device), label.to(device)
        for k in ("mask", "landmark"):
            if data_dict.get(k) is not None: data_dict[k] = data_dict[k].to(device)
        pred = model(data_dict, inference=True)
        probs += list(pred["prob"].cpu().numpy())
        labels += list(data_dict["label"].cpu().numpy())
        feats += list(pred["feat"].cpu().numpy())
    return np.array(probs), np.array(labels), np.array(feats)


def tpr_at_fpr(fpr, tpr, target):
    return float(np.interp(target, fpr, tpr))


def main():
    config = load_config()
    if config["cudnn"]: cudnn.benchmark = True
    model = DETECTOR[config["model_name"]](config).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt: ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True); model.eval()
    print("===> checkpoint loaded:", args.weights_path)

    loaders = make_loaders(config)
    results, roc_data, auc_bar = {}, {}, {}
    for name, loader in loaders.items():
        img_paths = loader.dataset.data_dict["image"]
        prob, label, feat = infer(model, loader)
        np.savez(os.path.join(args.out, f"scores_{name}.npz"), prob=prob, label=label, feat=feat)
        m = get_test_metrics(y_pred=prob, y_true=label, img_names=img_paths)
        fpr, tpr, _ = roc_curve(label, prob)
        results[name] = {
            "frame_auc": float(sk_auc(fpr, tpr)),
            "ap": float(average_precision_score(label, prob)),
            "eer": float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]),
            "tpr@fpr=5%": tpr_at_fpr(fpr, tpr, 0.05),
            "tpr@fpr=1%": tpr_at_fpr(fpr, tpr, 0.01),
            "video_auc": float(m.get("video_auc", float("nan"))),
            "n": int(len(label)),
        }
        roc_data[name] = (fpr, tpr); auc_bar[name] = results[name]["frame_auc"]
        print(f"[{name}] {json.dumps(results[name])}")

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    # C2 — ROC + eKYC operating point
    fig, axx = plt.subplots(figsize=(6, 5.2))
    for name, (fpr, tpr) in roc_data.items():
        axx.plot(fpr, tpr, lw=2, label=f"{name} (AUC={results[name]['frame_auc']:.3f})")
    axx.axvline(0.05, color="red", ls="--", lw=1.2, label="eKYC FPR≤5% (TT17/2024)")
    axx.plot([0, 1], [0, 1], "k:", lw=0.8)
    axx.set_xlabel("FPR"); axx.set_ylabel("TPR"); axx.set_title("ROC + eKYC operating point")
    axx.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "roc_ekyc.png"), dpi=130); plt.close(fig)

    # C1 — AUC bar
    fig, axx = plt.subplots(figsize=(5.5, 4))
    ks = list(auc_bar.keys())
    axx.bar(ks, [auc_bar[k] for k in ks], color=["tab:blue", "tab:orange", "tab:green"][:len(ks)])
    for i, k in enumerate(ks): axx.text(i, auc_bar[k] + 0.005, f"{auc_bar[k]:.3f}", ha="center")
    axx.set_ylim(0.5, 1.0); axx.set_ylabel("frame AUC"); axx.set_title("AUC: within vs cross-dataset")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "auc_bar.png"), dpi=130); plt.close(fig)

    # C3 — t-SNE of fused features (optional, first dataset, subsampled)
    try:
        from sklearn.manifold import TSNE
        name0 = list(loaders.keys())[0]
        d = np.load(os.path.join(args.out, f"scores_{name0}.npz"))
        feat, label = d["feat"], d["label"]
        idx = np.random.RandomState(0).permutation(len(feat))[:2000]
        emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(feat[idx])
        fig, axx = plt.subplots(figsize=(5.5, 5))
        for lab, col, nm in [(0, "tab:green", "real"), (1, "tab:red", "fake")]:
            sel = label[idx] == lab
            axx.scatter(emb[sel, 0], emb[sel, 1], s=6, c=col, alpha=0.5, label=nm)
        axx.set_title(f"t-SNE fused features — {name0}"); axx.legend(); axx.set_xticks([]); axx.set_yticks([])
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "tsne.png"), dpi=130); plt.close(fig)
    except Exception as e:
        print("t-SNE skipped:", e)

    # B1 — gate alpha (if present)
    alphas = {k: v for k, v in model.state_dict().items() if k.endswith("alpha") or ".alpha" in k}
    for k, v in alphas.items():
        v = v.detach().cpu().float().flatten()
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(v.numpy(), bins=60, color="tab:blue", alpha=0.8); ax.axvline(0, color="k", ls="--", label="init=0")
        ax.set_title(f"Zero-init gate after training — mean|α|={v.abs().mean():.3f}, "
                     f"{(v.abs()>1e-3).float().mean()*100:.0f}% engaged"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "gate_alpha.png"), dpi=130); plt.close(fig)

    print("\n==> all figures + scores written to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
