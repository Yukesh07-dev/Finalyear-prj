import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    NUM_CLASSES,
    RESULTS_DIR,
    SELECTED_CLASSES,
    SPLIT_DIR,
)
from .dataset import BirdDataset
from .models import (
    build_finetuned_resnet152,
    build_finetuned_resnet50,
    build_resnet50,
)
from .transforms import get_validation_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_samples(csv_path):
    samples = []

    with open(csv_path, "r", encoding="utf-8") as f:
        next(f)

        for line in f:
            image_path, label, class_name = line.rstrip("\n").split(",", 2)
            samples.append((Path(image_path), int(label)))

    return samples


def build_model(experiment):
    if experiment == "frozen_resnet50":
        return build_resnet50(NUM_CLASSES)

    if experiment == "finetuned_resnet50":
        return build_finetuned_resnet50(NUM_CLASSES, 30)

    if experiment == "finetuned_resnet152":
        return build_finetuned_resnet152(NUM_CLASSES, 30)

    raise ValueError(f"Unknown experiment: {experiment}")


def evaluate_model(experiment, loader):
    print("\n" + "=" * 70)
    print(f"EVALUATING: {experiment}")
    print("=" * 70)

    model = build_model(experiment)

    checkpoint_path = CHECKPOINT_DIR / experiment / "best_model.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    avg_loss = total_loss / len(loader.dataset)

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=SELECTED_CLASSES,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
    )

    print(f"Loss:      {avg_loss:.4f}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(report)

    result_dir = RESULTS_DIR / experiment
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": experiment,
        "loss": avg_loss,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "precision": precision,
        "precision_percent": precision * 100,
        "recall": recall,
        "recall_percent": recall * 100,
        "f1_score": f1,
        "f1_score_percent": f1 * 100,
    }

    with open(
        result_dir / "evaluation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=4)

    with open(
        result_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    np.savetxt(
        result_dir / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {experiment}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.colorbar()

    plt.xticks(
        range(NUM_CLASSES),
        SELECTED_CLASSES,
        rotation=90,
    )

    plt.yticks(
        range(NUM_CLASSES),
        SELECTED_CLASSES,
    )

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(
        result_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    return metrics


def plot_training_history(experiment):
    history_path = RESULTS_DIR / experiment / "training_history.json"

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_accuracy"]) + 1)

    result_dir = RESULTS_DIR / experiment

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        history["val_accuracy"],
        marker="o",
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curve - {experiment}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        result_dir / "accuracy_curve.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {experiment}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        result_dir / "loss_curve.png",
        dpi=200,
    )
    plt.close()


def create_comparison(metrics_list):
    comparison_dir = RESULTS_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = comparison_dir / "model_comparison.json"

    with open(
        comparison_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics_list, f, indent=4)

    experiments = [m["experiment"] for m in metrics_list]

    accuracy = [m["accuracy_percent"] for m in metrics_list]
    precision = [m["precision_percent"] for m in metrics_list]
    recall = [m["recall_percent"] for m in metrics_list]
    f1 = [m["f1_score_percent"] for m in metrics_list]

    x = np.arange(len(experiments))
    width = 0.2

    plt.figure(figsize=(12, 7))

    plt.bar(
        x - 1.5 * width,
        accuracy,
        width,
        label="Accuracy",
    )

    plt.bar(
        x - 0.5 * width,
        precision,
        width,
        label="Precision",
    )

    plt.bar(
        x + 0.5 * width,
        recall,
        width,
        label="Recall",
    )

    plt.bar(
        x + 1.5 * width,
        f1,
        width,
        label="F1-score",
    )

    plt.ylabel("Score (%)")
    plt.xlabel("Model")
    plt.title("Comparison of ResNet Models")
    plt.xticks(x, experiments, rotation=15)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        comparison_dir / "model_comparison.png",
        dpi=200,
    )
    plt.close()

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for metrics in metrics_list:
        print(
            f"{metrics['experiment']:25s} "
            f"Accuracy: {metrics['accuracy_percent']:.2f}%  "
            f"Precision: {metrics['precision_percent']:.2f}%  "
            f"Recall: {metrics['recall_percent']:.2f}%  "
            f"F1: {metrics['f1_score_percent']:.2f}%"
        )


def main():
    print("=" * 70)
    print("BIRD SPECIES CLASSIFICATION - MODEL EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    validation_csv = SPLIT_DIR / "validation.csv"

    samples = load_samples(validation_csv)

    dataset = BirdDataset(
        samples,
        transform=get_validation_transform(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )

    experiments = [
        "frozen_resnet50",
        "finetuned_resnet50",
        "finetuned_resnet152",
    ]

    all_metrics = []

    for experiment in experiments:
        metrics = evaluate_model(
            experiment,
            loader,
        )

        plot_training_history(experiment)

        all_metrics.append(metrics)

    create_comparison(all_metrics)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()