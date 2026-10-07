import csv
import random
from pathlib import Path

from .config import (
    RAW_DATA_DIR,
    SPLIT_DIR,
    SELECTED_CLASSES,
    TRAIN_RATIO,
    RANDOM_SEED,
)
from .dataset import discover_images


def main():
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("PREPARING 15-CLASS BIRD DATASET")
    print("=" * 60)

    print(f"Dataset directory: {RAW_DATA_DIR}")

    samples = discover_images(RAW_DATA_DIR)

    print(f"Total selected images: {len(samples)}")

    class_counts = {}

    for path, label in samples:
        class_name = SELECTED_CLASSES[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    print("\nClass distribution:")

    for class_name in SELECTED_CLASSES:
        print(
            f"{class_name}: "
            f"{class_counts.get(class_name, 0)}"
        )

    random.shuffle(samples)

    split_index = int(len(samples) * TRAIN_RATIO)

    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = SPLIT_DIR / "train.csv"
    val_file = SPLIT_DIR / "validation.csv"

    with open(train_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])

        for path, label in train_samples:
            writer.writerow([
                str(path.resolve()),
                label,
                SELECTED_CLASSES[label],
            ])

    with open(val_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "class_name"])

        for path, label in val_samples:
            writer.writerow([
                str(path.resolve()),
                label,
                SELECTED_CLASSES[label],
            ])

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"Training images:   {len(train_samples)}")
    print(f"Validation images: {len(val_samples)}")
    print(f"Training CSV:      {train_file}")
    print(f"Validation CSV:    {val_file}")


if __name__ == "__main__":
    main()