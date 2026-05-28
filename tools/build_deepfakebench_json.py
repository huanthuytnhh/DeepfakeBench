"""
Build DeepfakeBench JSON manifests from face-crop folders.

Reads the layout produced by tools/extract_faces_from_videos.py and writes:
  preprocessing/dataset_json_v3/FaceForensics++.json
  preprocessing/dataset_json_v3/FF-DF.json
  preprocessing/dataset_json_v3/FF-F2F.json
  preprocessing/dataset_json_v3/FF-FS.json
  preprocessing/dataset_json_v3/FF-NT.json
  preprocessing/dataset_json_v3/Celeb-DF-v2.json

For FF++ uses the official train/val/test splits (must be downloaded separately:
  https://github.com/ondyari/FaceForensics/tree/master/dataset/splits  )

For Celeb-DF-v2 uses the official List_of_testing_videos.txt (rest -> train).
"""
import argparse
import json
import os
import urllib.request
from pathlib import Path


FFPP_SPLIT_URLS = {
    "train.json": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json",
    "val.json":   "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json",
    "test.json":  "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json",
}

FF_FAKE_LABELS = {
    "Deepfakes":      "FF-DF",
    "Face2Face":      "FF-F2F",
    "FaceSwap":       "FF-FS",
    "NeuralTextures": "FF-NT",
}


def download_ffpp_splits(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FFPP_SPLIT_URLS.items():
        target = out_dir / name
        if target.exists():
            continue
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)


def load_video_to_mode(splits_dir):
    """Build a video_id -> {train,val,test} mapping from FF++ official splits."""
    video_to_mode = {}
    for mode_file, mode in [("train.json", "train"), ("val.json", "val"), ("test.json", "test")]:
        with open(Path(splits_dir) / mode_file) as f:
            data = json.load(f)
        for a, b in data:
            video_to_mode[a] = mode
            video_to_mode[b] = mode
            video_to_mode[f"{a}_{b}"] = mode
            video_to_mode[f"{b}_{a}"] = mode
    return video_to_mode


def list_videos(frames_dir):
    """Return dict: video_id -> sorted list of frame paths."""
    out = {}
    p = Path(frames_dir)
    if not p.exists():
        return out
    for vid_dir in sorted(p.iterdir()):
        if not vid_dir.is_dir():
            continue
        frames = sorted(str(x) for x in vid_dir.iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if frames:
            out[vid_dir.name] = frames
    return out


def build_ffpp(rgb_root, splits_dir, compression, output_dir):
    """
    rgb_root: e.g. /kaggle/working/processed
              (must contain FaceForensics++/{original_sequences,manipulated_sequences}/...)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_to_mode = load_video_to_mode(splits_dir)

    ff_root = Path(rgb_root) / "FaceForensics++"
    if not ff_root.exists():
        raise FileNotFoundError(f"{ff_root} not found")

    # Collect real videos
    real_frames_dir = ff_root / "original_sequences" / "youtube" / compression / "frames"
    real_videos = list_videos(real_frames_dir)
    print(f"Real videos: {len(real_videos)}")

    real_block = {"train": {compression: {}}, "val": {compression: {}}, "test": {compression: {}}}
    for vid, frames in real_videos.items():
        mode = video_to_mode.get(vid, "train")
        real_block[mode][compression][vid] = {"label": "FF-real", "frames": frames}

    # Build combined FaceForensics++.json (all 4 fakes + real, c23)
    ffpp_dict = {"FaceForensics++": {"FF-real": real_block}}

    for fake_dir_name, label_key in FF_FAKE_LABELS.items():
        frames_dir = ff_root / "manipulated_sequences" / fake_dir_name / compression / "frames"
        fake_videos = list_videos(frames_dir)
        print(f"{fake_dir_name} videos: {len(fake_videos)}")
        block = {"train": {compression: {}}, "val": {compression: {}}, "test": {compression: {}}}
        for vid, frames in fake_videos.items():
            mode = video_to_mode.get(vid, "train")
            block[mode][compression][vid] = {"label": label_key, "frames": frames, "masks": []}
        ffpp_dict["FaceForensics++"][label_key] = block

        # Also write per-fake JSON: <label>.json with FF-real + that fake only
        per_fake = {label_key: {"FF-real": real_block, label_key: block}}
        with open(output_dir / f"{label_key}.json", "w") as f:
            json.dump(per_fake, f)
        print(f"Wrote {label_key}.json")

    with open(output_dir / "FaceForensics++.json", "w") as f:
        json.dump(ffpp_dict, f)
    print("Wrote FaceForensics++.json")


def build_celebdfv2(rgb_root, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cdf_root = Path(rgb_root) / "Celeb-DF-v2"
    if not cdf_root.exists():
        raise FileNotFoundError(f"{cdf_root} not found")

    real_subs = ["Celeb-real", "YouTube-real"]
    fake_subs = ["Celeb-synthesis"]

    all_real, all_fake = {}, {}
    for sub in real_subs:
        videos = list_videos(cdf_root / sub / "frames")
        for vid, frames in videos.items():
            all_real[vid] = (sub, frames)
    for sub in fake_subs:
        videos = list_videos(cdf_root / sub / "frames")
        for vid, frames in videos.items():
            all_fake[vid] = (sub, frames)
    print(f"Celeb-DF-v2 real: {len(all_real)}, fake: {len(all_fake)}")

    # Split using List_of_testing_videos.txt (official test split)
    list_path = cdf_root / "List_of_testing_videos.txt"
    test_ids = set()
    if list_path.exists():
        with open(list_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                fname = parts[-1].split("/")[-1].rsplit(".", 1)[0]
                test_ids.add(fname)
        print(f"Test ids from official list: {len(test_ids)}")
    else:
        print("WARNING: List_of_testing_videos.txt not found — using all data as test.")

    def add(label_key, vid, frames, mode):
        block.setdefault(label_key, {}).setdefault("train", {})
        block.setdefault(label_key, {}).setdefault("val", {})
        block.setdefault(label_key, {}).setdefault("test", {})
        block[label_key][mode][vid] = {"label": label_key, "frames": frames}

    block = {}
    for vid, (sub, frames) in all_real.items():
        mode = "test" if vid in test_ids else "train"
        add("CelebDFv2_real", vid, frames, mode)
        if mode == "test":
            add("CelebDFv2_real", vid, frames, "val")
    for vid, (sub, frames) in all_fake.items():
        mode = "test" if vid in test_ids else "train"
        add("CelebDFv2_fake", vid, frames, mode)
        if mode == "test":
            add("CelebDFv2_fake", vid, frames, "val")

    out = {"Celeb-DF-v2": block}
    with open(output_dir / "Celeb-DF-v2.json", "w") as f:
        json.dump(out, f)
    print("Wrote Celeb-DF-v2.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_root", required=True, help="Root that contains FaceForensics++/ and/or Celeb-DF-v2/")
    ap.add_argument("--output_dir", required=True, help="Where to write *.json (typically preprocessing/dataset_json_v3)")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--splits_dir", default="preprocessing/ffpp_splits", help="Where to cache FF++ split JSONs")
    ap.add_argument("--datasets", nargs="+", default=["FaceForensics++", "Celeb-DF-v2"])
    args = ap.parse_args()

    if "FaceForensics++" in args.datasets:
        download_ffpp_splits(args.splits_dir)
        build_ffpp(args.rgb_root, args.splits_dir, args.compression, args.output_dir)
    if "Celeb-DF-v2" in args.datasets:
        build_celebdfv2(args.rgb_root, args.output_dir)


if __name__ == "__main__":
    main()
