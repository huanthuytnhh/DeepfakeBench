"""
eval_and_viz.py — run a TRAINED checkpoint on the test sets, DUMP raw scores, and SAVE the full
Evaluation & Analysis figure set (one per run). Mirrors test.py's loading (run from the REPO ROOT):

  python training/eval_and_viz.py \
      --detector_path ./training/config/detector/efficientnetb4_sfdct.yaml \
      --weights_path  <log_dir>/.../ckpt_best.pth \
      --test_dataset  FaceForensics++ Celeb-DF-v2 \
      --out           ./viz_out/sfdct

Figures in --out (each wrapped so one failure can't kill the rest):
  roc_auc.png        ROC + eKYC FPR<=5% line + TPR@FPR table
  pr_curve.png       Precision-Recall per dataset
  radar.png          multi-metric radar (AUC / AP / 1-EER / TPR@5% / TPR@1%) per dataset
  ap_bar.png         AUC & AP bars, within vs cross
  heatmap.png        metric x dataset heat map
  tsne.png           t-SNE of fused features (real vs fake)
  frequency.png      mean log|DCT| spectrum real vs fake (the block-DCT signal)
  gradcam.png        Grad-CAM on one fake sample (spatial evidence)
  gate_alpha.png     the zero-init fusion gate after training
  results.json       all metrics ;  scores_<dataset>.npz  raw prob/label/feat
"""
import os, json, math, argparse
import numpy as np
import yaml
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, average_precision_score, precision_recall_curve

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
def save(fig, name): fig.tight_layout(); fig.savefig(os.path.join(args.out, name), dpi=130); plt.close(fig)
def guard(fn, *a):
    try: fn(*a)
    except Exception as e: print(f"[skip] {fn.__name__}: {type(e).__name__}: {e}")


def load_config():
    with open(args.detector_path) as f: config = yaml.safe_load(f)
    with open("./training/config/test_config.yaml") as f: config.update(yaml.safe_load(f))
    if args.test_dataset: config["test_dataset"] = args.test_dataset
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
    for dd in tqdm(loader, total=len(loader)):
        dd["label"] = torch.where(dd["label"] != 0, 1, 0)
        dd["image"], dd["label"] = dd["image"].to(device), dd["label"].to(device)
        for k in ("mask", "landmark"):
            if dd.get(k) is not None: dd[k] = dd[k].to(device)
        p = model(dd, inference=True)
        probs += list(p["prob"].cpu().numpy()); labels += list(dd["label"].cpu().numpy())
        f = p["feat"]; f = f.mean((2, 3)) if f.dim() == 4 else f   # GAP -> [B,C]; raw 4D feat would be ~10GB in npz
        feats += list(f.cpu().numpy())
    return np.array(probs), np.array(labels), np.array(feats)

def tpr_at(fpr, tpr, t): return float(np.interp(t, fpr, tpr))

# ---------------- figures ----------------
def fig_roc(roc, res):
    fig, a = plt.subplots(figsize=(6, 5.2))
    for n, (fpr, tpr) in roc.items(): a.plot(fpr, tpr, lw=2, label=f"{n} (AUC={res[n]['frame_auc']:.3f})")
    a.axvline(0.05, color="red", ls="--", lw=1.2, label="eKYC FPR<=5% (TT17/2024)")
    a.plot([0, 1], [0, 1], "k:", lw=.8); a.set_xlabel("FPR"); a.set_ylabel("TPR")
    a.set_title("ROC-AUC + eKYC operating point"); a.legend(fontsize=8); save(fig, "roc_auc.png")

def fig_pr(pr, res):
    fig, a = plt.subplots(figsize=(6, 5))
    for n, (rc, pc) in pr.items(): a.plot(rc, pc, lw=2, label=f"{n} (AP={res[n]['ap']:.3f})")
    a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.set_title("Precision-Recall"); a.legend(fontsize=8)
    save(fig, "pr_curve.png")

def fig_radar(res):
    axes = ["AUC", "AP", "1-EER", "TPR@5%", "TPR@1%"]
    ang = np.linspace(0, 2*math.pi, len(axes), endpoint=False).tolist(); ang += ang[:1]
    fig, a = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for n, r in res.items():
        v = [r["frame_auc"], r["ap"], 1-r["eer"], r["tpr@fpr=5%"], r["tpr@fpr=1%"]]; v += v[:1]
        a.plot(ang, v, lw=2, label=n); a.fill(ang, v, alpha=0.1)
    a.set_xticks(ang[:-1]); a.set_xticklabels(axes); a.set_ylim(0, 1)
    a.set_title("Multi-metric radar"); a.legend(loc="lower right", fontsize=8); save(fig, "radar.png")

def fig_ap_bar(res):
    ks = list(res.keys()); x = np.arange(len(ks)); w = 0.38
    fig, a = plt.subplots(figsize=(5.8, 4))
    a.bar(x-w/2, [res[k]["frame_auc"] for k in ks], w, label="AUC", color="tab:blue")
    a.bar(x+w/2, [res[k]["ap"] for k in ks], w, label="AP", color="tab:orange")
    a.set_xticks(x); a.set_xticklabels(ks); a.set_ylim(0.5, 1.0); a.set_title("AUC / AP bar"); a.legend()
    save(fig, "ap_bar.png")

def fig_heatmap(res):
    metrics = ["frame_auc", "ap", "eer", "tpr@fpr=5%", "tpr@fpr=1%"]
    ks = list(res.keys()); M = np.array([[res[k][m] for k in ks] for m in metrics])
    fig, a = plt.subplots(figsize=(1.6*len(ks)+2.5, 4))
    im = a.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(ks))); a.set_xticklabels(ks); a.set_yticks(range(len(metrics))); a.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(ks)): a.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    a.set_title("Metric x dataset heat map"); fig.colorbar(im, ax=a, fraction=0.046); save(fig, "heatmap.png")

def fig_tsne(name, npz):
    from sklearn.manifold import TSNE
    d = np.load(npz); feat, label = d["feat"], d["label"]
    if feat.ndim > 2: feat = feat.reshape(feat.shape[0], -1)       # safety: flatten if not already [N,C]
    idx = np.random.RandomState(0).permutation(len(feat))[:2000]
    emb = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(feat[idx])
    fig, a = plt.subplots(figsize=(5.5, 5))
    for lab, col, nm in [(0, "tab:green", "real"), (1, "tab:red", "fake")]:
        s = label[idx] == lab; a.scatter(emb[s, 0], emb[s, 1], s=6, c=col, alpha=.5, label=nm)
    a.set_title(f"t-SNE fused features — {name}"); a.legend(); a.set_xticks([]); a.set_yticks([]); save(fig, "tsne.png")

def fig_frequency(loader):
    """mean log|DCT| 8x8 spectrum, real vs fake, from one batch of crops."""
    import sys; sys.path.insert(0, "training/detectors")
    from sfdct_core import dct_matrix
    dd = next(iter(loader)); x = dd["image"]; y = torch.where(dd["label"] != 0, 1, 0)
    M = dct_matrix(8)
    def spec(imgs):
        g = imgs.mean(1, keepdim=True)                       # to gray
        blk = g.unfold(2, 8, 8).unfold(3, 8, 8)
        c = torch.einsum("pa,ncijab,qb->ncijpq", M, blk, M)
        return torch.log1p(c.abs()).mean((0,1,2,3)).numpy()
    sr, sf = spec(x[y == 0]), spec(x[y == 1]); diff = np.abs(sf - sr)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (s, t) in zip(ax, [(sr, "REAL log|DCT|"), (sf, "FAKE log|DCT|"), (diff, "|fake-real|")]):
        im = a.imshow(s, cmap="viridis"); a.set_title(t); fig.colorbar(im, ax=a, fraction=.046)
    fig.suptitle("Frequency: fake energy leaks into high DCT band"); save(fig, "frequency.png")

def fig_gradcam(model, loader):
    dd = next(iter(loader)); y = torch.where(dd["label"] != 0, 1, 0)
    fake = (y == 1).nonzero(as_tuple=True)[0]
    if len(fake) == 0: return
    i = int(fake[0]); img = dd["image"][i:i+1].to(device).requires_grad_(True)
    feat = model.features({"image": img})                    # [1,C,h,w]
    feat.retain_grad()
    logit = model.classifier(feat); score = logit[0, 1]
    model.zero_grad(); score.backward()
    w = feat.grad.mean((2, 3), keepdim=True)                 # channel weights
    cam = torch.relu((w * feat).sum(1, keepdim=True))        # [1,1,h,w]
    im = img.detach().cpu()[0].permute(1, 2, 0).numpy(); im = (im - im.min()) / (np.ptp(im) + 1e-8)
    H, W = im.shape[:2]
    cam = torch.nn.functional.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy(); cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    fig, a = plt.subplots(1, 2, figsize=(7, 3.6))
    a[0].imshow(im); a[0].set_title("input (fake)"); a[0].axis("off")
    a[1].imshow(im); a[1].imshow(cam, cmap="jet", alpha=0.45); a[1].set_title("Grad-CAM"); a[1].axis("off")
    save(fig, "gradcam.png")

def fig_gate(model):
    al = {k: v for k, v in model.state_dict().items() if k.endswith("alpha") or ".alpha" in k}
    for k, v in al.items():
        v = v.detach().cpu().float().flatten()
        fig, a = plt.subplots(figsize=(6, 3.8))
        a.hist(v.numpy(), bins=60, color="tab:blue", alpha=.8); a.axvline(0, color="k", ls="--", label="init=0")
        a.set_title(f"Zero-init gate after training — mean|a|={v.abs().mean():.3f}, "
                    f"{(v.abs()>1e-3).float().mean()*100:.0f}% engaged"); a.legend(); save(fig, "gate_alpha.png")


def main():
    config = load_config()
    if config["cudnn"]: cudnn.benchmark = True
    model = DETECTOR[config["model_name"]](config).to(device)
    ck = torch.load(args.weights_path, map_location=device)
    if isinstance(ck, dict) and "state_dict" in ck: ck = ck["state_dict"]
    model.load_state_dict(ck, strict=True); model.eval()
    print("===> checkpoint:", args.weights_path)

    loaders = make_loaders(config)
    res, roc, pr = {}, {}, {}
    first = None
    for name, loader in loaders.items():
        if first is None: first = (name, loader)
        prob, label, feat = infer(model, loader)
        # lưu thêm 'img' (đường dẫn frame, order khớp do shuffle=False) → suy video-id cho bootstrap CẤP-VIDEO
        img_names = loader.dataset.data_dict["image"]
        np.savez(os.path.join(args.out, f"scores_{name}.npz"),
                 prob=prob, label=label, feat=feat,
                 img=np.array(img_names[:len(prob)], dtype=object))
        m = get_test_metrics(y_pred=prob, y_true=label, img_names=img_names)
        fpr, tpr, _ = roc_curve(label, prob); prec, rec, _ = precision_recall_curve(label, prob)
        res[name] = {"frame_auc": float(sk_auc(fpr, tpr)), "ap": float(average_precision_score(label, prob)),
                     "eer": float(fpr[np.nanargmin(np.abs((1-tpr)-fpr))]),
                     "tpr@fpr=5%": tpr_at(fpr, tpr, .05), "tpr@fpr=1%": tpr_at(fpr, tpr, .01),
                     "video_auc": float(m.get("video_auc", float("nan"))), "n": int(len(label))}
        roc[name] = (fpr, tpr); pr[name] = (rec, prec)
        print(f"[{name}] {json.dumps(res[name])}")
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=2)

    guard(fig_roc, roc, res); guard(fig_pr, pr, res); guard(fig_radar, res)
    guard(fig_ap_bar, res); guard(fig_heatmap, res); guard(fig_gate, model)
    guard(fig_tsne, first[0], os.path.join(args.out, f"scores_{first[0]}.npz"))
    guard(fig_frequency, first[1]); guard(fig_gradcam, model, first[1])
    print("\n==> figures + scores in", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
