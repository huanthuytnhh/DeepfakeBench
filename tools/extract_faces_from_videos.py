"""
Extract face crops 256x256 from raw deepfake videos (FF++ / Celeb-DF) into
the folder layout that DeepfakeBench's rearrange.py expects.

Faster than the official preprocess.py (uses MTCNN on GPU instead of dlib CPU).
Use on Kaggle T4 to convert raw video Kaggle Datasets into DeepfakeBench format.

Output layout for FF++ (compression=c23):
  <out>/FaceForensics++/original_sequences/youtube/c23/frames/<vid>/<idx>.png
  <out>/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/<vid>/<idx>.png
  <out>/FaceForensics++/manipulated_sequences/Face2Face/c23/frames/<vid>/<idx>.png
  <out>/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames/<vid>/<idx>.png
  <out>/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames/<vid>/<idx>.png
  <out>/FaceForensics++/train.json, val.json, test.json   (copy from official splits)

Output layout for Celeb-DF-v2:
  <out>/Celeb-DF-v2/Celeb-real/frames/<vid>/<idx>.png
  <out>/Celeb-DF-v2/Celeb-synthesis/frames/<vid>/<idx>.png
  <out>/Celeb-DF-v2/YouTube-real/frames/<vid>/<idx>.png
  <out>/Celeb-DF-v2/List_of_testing_videos.txt   (copy from official)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
except ImportError:
    sys.stderr.write("pip install facenet-pytorch\n")
    raise


FF_SUBSETS = {
    "youtube": ("original_sequences/youtube", "real"),
    "Deepfakes": ("manipulated_sequences/Deepfakes", "fake"),
    "Face2Face": ("manipulated_sequences/Face2Face", "fake"),
    "FaceSwap": ("manipulated_sequences/FaceSwap", "fake"),
    "NeuralTextures": ("manipulated_sequences/NeuralTextures", "fake"),
}
CDF_SUBSETS = ("Celeb-real", "Celeb-synthesis", "YouTube-real")


def evenly_sample_indices(total, n):
    if total <= n:
        return list(range(total))
    step = total / n
    return [int(i * step) for i in range(n)]


def crop_with_margin(frame_bgr, box, margin=0.3, out_size=256):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + margin)
    nx1 = max(0, int(cx - side / 2))
    ny1 = max(0, int(cy - side / 2))
    nx2 = min(w, int(cx + side / 2))
    ny2 = min(h, int(cy + side / 2))
    crop = frame_bgr[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def process_video(video_path, out_dir, mtcnn, num_frames=32, out_size=256):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0
    sample_idx = set(evenly_sample_indices(total, num_frames))

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    out_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in sample_idx:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(rgb)
            if boxes is not None and len(boxes) > 0 and probs[0] is not None and probs[0] > 0.9:
                # pick largest face
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                best = boxes[int(np.argmax(areas))]
                crop = crop_with_margin(frame, best, out_size=out_size)
                if crop is not None:
                    cv2.imwrite(str(out_dir / f"{out_idx:03d}.png"), crop)
                    out_idx += 1
                    saved += 1
        frame_idx += 1
    cap.release()
    return saved


def walk_videos(root):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            yield p


def process_ffpp(video_root, out_root, mtcnn, num_frames, compression):
    """video_root has subfolders: youtube/, Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/"""
    for sub_name, (rel_path, _label) in FF_SUBSETS.items():
        src = Path(video_root) / sub_name
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        out_dir = Path(out_root) / "FaceForensics++" / rel_path / compression / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        vids = list(walk_videos(src))
        print(f"[FF++/{sub_name}] {len(vids)} videos -> {out_dir}")
        for vp in tqdm(vids, desc=sub_name):
            vid_name = vp.stem.split(".")[0]
            target = out_dir / vid_name
            if target.exists() and len(list(target.iterdir())) >= num_frames - 2:
                continue
            try:
                process_video(vp, target, mtcnn, num_frames=num_frames)
            except Exception as e:
                print(f"  fail {vp}: {e}")


def process_celebdfv2(video_root, out_root, mtcnn, num_frames):
    for sub in CDF_SUBSETS:
        src = Path(video_root) / sub
        if not src.exists():
            print(f"[skip] {src} not found")
            continue
        out_dir = Path(out_root) / "Celeb-DF-v2" / sub / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        vids = list(walk_videos(src))
        print(f"[Celeb-DF-v2/{sub}] {len(vids)} videos -> {out_dir}")
        for vp in tqdm(vids, desc=sub):
            vid_name = vp.stem
            target = out_dir / vid_name
            if target.exists() and len(list(target.iterdir())) >= num_frames - 2:
                continue
            try:
                process_video(vp, target, mtcnn, num_frames=num_frames)
            except Exception as e:
                print(f"  fail {vp}: {e}")
    # Copy List_of_testing_videos.txt if found
    list_file = Path(video_root) / "List_of_testing_videos.txt"
    if list_file.exists():
        import shutil
        dest = Path(out_root) / "Celeb-DF-v2" / "List_of_testing_videos.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(list_file, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["FaceForensics++", "Celeb-DF-v2"], required=True)
    ap.add_argument("--video_root", required=True, help="Folder containing the dataset's video subfolders")
    ap.add_argument("--out_root", required=True, help="Output root where DeepfakeBench-format face crops are written")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    mtcnn = MTCNN(image_size=256, margin=0, device=args.device, post_process=False, select_largest=True)

    if args.dataset == "FaceForensics++":
        process_ffpp(args.video_root, args.out_root, mtcnn, args.num_frames, args.compression)
    else:
        process_celebdfv2(args.video_root, args.out_root, mtcnn, args.num_frames)

    print("Done.")


if __name__ == "__main__":
    main()
