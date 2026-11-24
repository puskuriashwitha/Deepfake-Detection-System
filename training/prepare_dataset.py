from pathlib import Path
import csv
import random
import shutil
import re

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "dataset" / "frames"
SPLITS = ROOT / "dataset" / "splits"
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def video_key(path):
    # extract_frames.py names images like 1_frame_000000.jpg
    match = re.match(r"(.+?)_frame_\d+$", path.stem)
    if not match:
        raise ValueError(f"Unexpected frame filename (cannot identify source video): {path.name}")
    return match.group(1)

def main():
    rng = random.Random(SEED)
    all_rows = []
    for label in ("real", "fake"):
        folder = FRAMES / label
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")
        grouped = {}
        for image in sorted(folder.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                grouped.setdefault(video_key(image), []).append(image)
        if len(grouped) < 3:
            raise SystemExit(f"{label}: need at least 3 distinct source videos; found {len(grouped)}")
        video_ids = sorted(grouped)
        rng.shuffle(video_ids)
        # With five videos/class: 3 train, 1 val, 1 test.
        n = len(video_ids)
        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20))
        if n - n_test - n_val < 1:
            raise SystemExit(f"{label}: not enough videos for train/val/test.")
        assignments = {}
        for vid in video_ids[:n_test]:
            assignments[vid] = "test"
        for vid in video_ids[n_test:n_test+n_val]:
            assignments[vid] = "val"
        for vid in video_ids[n_test+n_val:]:
            assignments[vid] = "train"
        for vid, images in grouped.items():
            split = assignments[vid]
            for image in images:
                all_rows.append({
                    "split": split, "label": label, "video_id": vid,
                    "source": str(image.relative_to(ROOT)).replace("\\", "/"),
                })

    # Rebuild only the generated split folder.
    if SPLITS.exists():
        shutil.rmtree(SPLITS)
    for row in all_rows:
        source = ROOT / row["source"]
        destination = SPLITS / row["split"] / row["label"] / f'{row["video_id"]}_{source.name}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest = ROOT / "dataset" / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "video_id", "source"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("Video-level split completed (seed=42).")
    for split in ("train", "val", "test"):
        for label in ("real", "fake"):
            rows = [r for r in all_rows if r["split"] == split and r["label"] == label]
            videos = sorted({r["video_id"] for r in rows})
            print(f"{split:5s} | {label:4s} | {len(videos)} videos | {len(rows)} frames")
    print(f"Manifest: {manifest}")
    print("Important: frames from a source video are kept in exactly one split.")

if __name__ == "__main__":
    main()
