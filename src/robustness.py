import json
import time
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from .config import (
    PROJECT_ROOT,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    SELECTED_CLASSES,
    SPLIT_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .models import build_finetuned_resnet50, build_finetuned_resnet152
from .transforms import get_validation_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REAL-WORLD IMAGE PERTURBATION ENGINE (PURE PIL)
# ============================================================

def apply_perturbation(image: Image.Image, perturbation_type: str, severity: float = 1.0) -> Image.Image:
    """
    Applies controlled realistic environmental perturbations to a PIL Image.

    Supported perturbation_types:
    - 'original': Unmodified clean image.
    - 'brightness_low': Reduced lighting / shadows (factor = 0.5).
    - 'brightness_high': Overexposure / harsh sunlight (factor = 1.5).
    - 'contrast_low': Foggy / hazy conditions (factor = 0.5).
    - 'contrast_high': Harsh glare / vivid contrast (factor = 1.5).
    - 'blur_gaussian': Out-of-focus camera lens (radius = 2.5).
    - 'noise_gaussian': High ISO digital sensor noise.
    - 'rotation': Slight camera tilt (+15 degrees).
    - 'scale_crop': Distance variation / partial subject zoom (center crop 80%).
    - 'jpeg_compression': Low bandwidth / heavy JPEG compression (quality = 15).
    - 'occlusion': Obstruction by twigs/leaves (black patch covering 30% area).
    """
    img = image.convert("RGB").copy()

    if perturbation_type == "original":
        return img

    elif perturbation_type == "brightness_low":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.5)

    elif perturbation_type == "brightness_high":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.5)

    elif perturbation_type == "contrast_low":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.5)

    elif perturbation_type == "contrast_high":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.5)

    elif perturbation_type == "blur_gaussian":
        radius = 2.5 * severity
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif perturbation_type == "noise_gaussian":
        arr = np.array(img, dtype=np.float32)
        sigma = 25.0 * severity
        noise = np.random.normal(0, sigma, arr.shape)
        noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_arr)

    elif perturbation_type == "rotation":
        angle = 15.0 * severity
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    elif perturbation_type == "scale_crop":
        w, h = img.size
        crop_ratio = 0.8
        cw, ch = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - cw) // 2
        top = (h - ch) // 2
        cropped = img.crop((left, top, left + cw, top + ch))
        return cropped.resize((w, h), Image.Resampling.BILINEAR)

    elif perturbation_type == "jpeg_compression":
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=15)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif perturbation_type == "occlusion":
        arr = np.array(img).copy()
        h, w, _ = arr.shape
        patch_h, patch_w = int(h * 0.3), int(w * 0.3)
        top = (h - patch_h) // 2
        left = (w - patch_w) // 2
        arr[top : top + patch_h, left : left + patch_w, :] = 0
        return Image.fromarray(arr)

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


# ============================================================
# PERTURBED DATASET WRAPPER
# ============================================================

class PerturbedBirdDataset(Dataset):
    def __init__(self, samples, perturbation_type="original", transform=None):
        self.samples = samples
        self.perturbation_type = perturbation_type
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = apply_perturbation(image, self.perturbation_type)

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# ROBUSTNESS EVALUATION PIPELINE
# ============================================================

def load_validation_samples(csv_path: Path):
    """Load sample file paths and integer class labels from validation CSV."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            image_path = Path(parts[0])
            label = int(parts[1])
            samples.append((image_path, label))
    return samples


def evaluate_robustness(
    experiment_name: str = "finetuned_resnet50",
    max_samples: int = None,
    output_dir: Path = None,
):
    """
    Evaluates the specified trained model against 10 real-world image perturbation conditions.
    Computes Accuracy, Precision, Recall, F1-Score for each condition.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "experiments" / "robustness_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_csv = SPLIT_DIR / "validation.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation split file not found at {val_csv}")

    samples = load_validation_samples(val_csv)
    if max_samples and max_samples < len(samples):
        samples = samples[:max_samples]

    print("=" * 70)
    print(f"RUNNING ROBUSTNESS EVALUATION FOR MODEL: {experiment_name}")
    print(f"Total validation samples: {len(samples)}")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load Model
    if experiment_name == "finetuned_resnet152":
        model = build_finetuned_resnet152(NUM_CLASSES, 30)
    else:
        model = build_finetuned_resnet50(NUM_CLASSES, 30)

    checkpoint_path = CHECKPOINT_DIR / experiment_name / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    perturbation_conditions = [
        ("original", "Clean Original"),
        ("brightness_low", "Low Brightness"),
        ("brightness_high", "High Brightness"),
        ("contrast_low", "Low Contrast"),
        ("contrast_high", "High Contrast"),
        ("blur_gaussian", "Gaussian Blur"),
        ("noise_gaussian", "Sensor Noise"),
        ("rotation", "15° Rotation"),
        ("scale_crop", "Scale / Crop"),
        ("jpeg_compression", "JPEG Compression"),
        ("occlusion", "30% Occlusion"),
    ]

    transform = get_validation_transform()
    results = {}

    for p_key, p_label in perturbation_conditions:
        print(f"\n[Evaluating] Condition: {p_label} ({p_key})...")
        dataset = PerturbedBirdDataset(samples, perturbation_type=p_key, transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds = []
        all_labels = []

        start_time = time.time()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        elapsed = time.time() - start_time
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        results[p_key] = {
            "label": p_label,
            "accuracy": float(acc),
            "accuracy_percent": float(acc * 100),
            "precision_percent": float(prec * 100),
            "recall_percent": float(rec * 100),
            "f1_percent": float(f1 * 100),
            "evaluation_time_sec": float(elapsed),
        }

        print(f" -> Accuracy: {acc * 100:.2f}% | F1: {f1 * 100:.2f}% | Time: {elapsed:.2f}s")

    # Save JSON metrics report
    json_path = output_dir / f"robustness_metrics_{experiment_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Detailed metrics saved to: {json_path}")

    # Generate Plot Comparison Chart
    plot_robustness_bar_chart(results, output_dir / f"robustness_chart_{experiment_name}.png")

    return results


def plot_robustness_bar_chart(results: dict, save_path: Path):
    """Generates clean high-resolution bar chart comparing accuracy across environmental conditions."""
    labels = [v["label"] for v in results.values()]
    accuracies = [v["accuracy_percent"] for v in results.values()]

    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar(labels, accuracies, color="#38bdf8", edgecolor="#0284c7", width=0.6)

    # Highlight clean baseline in green
    bars[0].set_color("#22c55e")
    bars[0].set_edgecolor("#15803d")

    plt.title("Model Robustness Under Real-World Image Perturbations", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Environmental Condition / Image Perturbation", fontsize=11, labelpad=10)
    plt.ylabel("Classification Accuracy (%)", fontsize=11, labelpad=10)
    plt.ylim(0, 105)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] Robustness chart saved to: {save_path}")


if __name__ == "__main__":
    evaluate_robustness()
