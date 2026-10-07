import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    NUM_CLASSES,
    FROZEN_LR,
    FINETUNE_LR,
    LAST_CONV_LAYERS,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    SPLIT_DIR,
    SELECTED_CLASSES,
    RANDOM_SEED,
)

from .dataset import BirdDataset
from .transforms import (
    get_train_transform,
    get_validation_transform,
)

from .models import (
    build_resnet50,
    build_finetuned_resnet50,
    build_finetuned_resnet152,
    trainable_parameters,
    print_model_summary,
)

from .trainer import Trainer


def set_global_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_split(csv_file):

    samples = []

    with open(
        csv_file,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            samples.append(
                (
                    Path(row["image_path"]),
                    int(row["label"]),
                )
            )

    return samples


def build_model(experiment):

    if experiment == "frozen_resnet50":

        return build_resnet50(
            NUM_CLASSES
        )

    if experiment == "finetuned_resnet50":

        return build_finetuned_resnet50(
            NUM_CLASSES,
            LAST_CONV_LAYERS,
        )

    if experiment == "finetuned_resnet152":

        return build_finetuned_resnet152(
            NUM_CLASSES,
            LAST_CONV_LAYERS,
        )

    raise ValueError(
        f"Unknown experiment: {experiment}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "frozen_resnet50",
            "finetuned_resnet50",
            "finetuned_resnet152",
        ],
    )

    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)

    experiment = args.experiment

    print("=" * 70)
    print(
        "DEEP LEARNING-BASED BIRD SPECIES CLASSIFICATION"
    )
    print("Experiment:", experiment)
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_samples = read_split(
        SPLIT_DIR / "train.csv"
    )

    val_samples = read_split(
        SPLIT_DIR / "validation.csv"
    )

    augmentation = (
        experiment != "frozen_resnet50"
    )

    train_dataset = BirdDataset(
        train_samples,
        transform=get_train_transform(
            augmentation=augmentation
        ),
    )

    val_dataset = BirdDataset(
        val_samples,
        transform=get_validation_transform(),
    )

    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print()
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Augmentation:", augmentation)

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Loading ImageNet-pretrained model...")

    model = build_model(experiment)

    model = model.to(device)

    print_model_summary(model)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    if experiment == "frozen_resnet50":
        learning_rate = FROZEN_LR
    else:
        learning_rate = FINETUNE_LR

    optimizer = torch.optim.Adam(
        trainable_parameters(model),
        lr=learning_rate,
    )

    print("Learning rate:", learning_rate)
    print("Optimizer: Adam")
    print("Loss: CrossEntropyLoss")

    # --------------------------------------------------------
    # OUTPUT DIRECTORIES
    # --------------------------------------------------------

    checkpoint_dir = (
        CHECKPOINT_DIR / experiment
    )

    result_dir = (
        RESULTS_DIR / experiment
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        class_names=SELECTED_CLASSES,
    )

    trainer.fit()


if __name__ == "__main__":
    main()