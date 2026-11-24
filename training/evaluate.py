from pathlib import Path
from collections import defaultdict
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from model import DeepfakeCNN

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "dataset" / "splits"
MODEL_PATH = ROOT / "models" / "deepfake_model.pth"
IMAGE_SIZE = 128
LABELS = {"real": 0, "fake": 1}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class TestDataset(Dataset):
    def __init__(self):
        self.items = []
        for label, target in LABELS.items():
            folder = SPLITS / "test" / label
            if folder.exists():
                self.items.extend((p, target, label) for p in sorted(folder.glob("*.jpg")))
        if not self.items:
            raise RuntimeError("No test frames. Run prepare_dataset.py first.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, target, label_name = self.items[index]
        image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - MEAN) / STD
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        video_id = path.name.split("_frame_")[0]
        return tensor, target, video_id, label_name

def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Trained model not found: {MODEL_PATH}. Run python training\\train.py first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = DeepfakeCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(TestDataset(), batch_size=16, shuffle=False, num_workers=0)

    true_labels, predictions = [], []
    video_scores = defaultdict(list)
    video_truth = {}
    with torch.no_grad():
        for images, labels, video_ids, label_names in loader:
            probabilities = torch.softmax(model(images), dim=1)[:, 1].numpy()
            preds = (probabilities >= 0.5).astype(int)
            true_labels.extend(labels.numpy().tolist())
            predictions.extend(preds.tolist())
            for vid, truth, score, label_name in zip(video_ids, labels.numpy().tolist(), probabilities.tolist(), label_names):
                key = f"{label_name}/{vid}"
                video_scores[key].append(score)
                video_truth[key] = truth

    print("FRAME-LEVEL TEST METRICS")
    print(f"Accuracy : {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(true_labels, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(true_labels, predictions, zero_division=0):.4f}")
    print("Confusion matrix [rows=true real,fake; columns=pred real,fake]:")
    print(confusion_matrix(true_labels, predictions, labels=[0, 1]))
    print()
    print("VIDEO-LEVEL RESULTS (mean fake probability per source video)")
    video_true, video_pred = [], []
    rows = []
    for key in sorted(video_scores):
        score = float(np.mean(video_scores[key]))
        truth = video_truth[key]
        pred = int(score >= 0.5)
        video_true.append(truth)
        video_pred.append(pred)
        rows.append({"video": key, "true_label": "fake" if truth else "real",
                     "mean_fake_probability": f"{score:.4f}",
                     "predicted_label": "fake" if pred else "real"})
        print(f"{key:12s} | mean fake probability={score:.4f} | predicted={'fake' if pred else 'real'}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report_path = out / "evaluation_video_results.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "true_label", "mean_fake_probability", "predicted_label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nVideo-level results saved to: {report_path}")
    print(f"Test set contains only {len(video_scores)} source videos. Treat these metrics as a smoke test, not a reliable estimate of generalization.")

if __name__ == "__main__":
    main()
