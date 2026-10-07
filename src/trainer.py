import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        num_epochs,
        checkpoint_dir,
        result_dir,
        class_names,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

        self.checkpoint_dir = Path(checkpoint_dir)
        self.result_dir = Path(result_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.result_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.class_names = class_names

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        )

        for images, labels in progress:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            progress.set_postfix(
                loss=loss.item()
            )

        epoch_loss = running_loss / total
        epoch_accuracy = (
            100.0 * correct / total
        )

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = (
            100.0 * correct / total
        )

        return epoch_loss, epoch_accuracy

    def save_checkpoint(
        self,
        epoch,
        val_accuracy,
        filename="best_model.pth",
    ):

        path = (
            self.checkpoint_dir /
            filename
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    self.model.state_dict(),
                "optimizer_state_dict":
                    self.optimizer.state_dict(),
                "val_accuracy":
                    val_accuracy,
                "class_names":
                    self.class_names,
            },
            path,
        )

        return path

    def save_history(self):

        path = (
            self.result_dir /
            "training_history.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

    def fit(self):

        best_val_accuracy = 0.0

        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)

        print(
            f"Device: {self.device}"
        )

        for epoch in range(
            self.num_epochs
        ):

            start_time = time.time()

            print()
            print(
                f"Epoch "
                f"{epoch + 1}/"
                f"{self.num_epochs}"
            )

            train_loss, train_accuracy = (
                self.train_one_epoch()
            )

            val_loss, val_accuracy = (
                self.validate()
            )

            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "train_accuracy"
            ].append(train_accuracy)

            self.history[
                "val_loss"
            ].append(val_loss)

            self.history[
                "val_accuracy"
            ].append(val_accuracy)

            elapsed = (
                time.time() - start_time
            )

            print()
            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy: "
                f"{train_accuracy:.2f}%"
            )

            print(
                f"Validation Loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy: "
                f"{val_accuracy:.2f}%"
            )

            print(
                f"Epoch Time: "
                f"{elapsed / 60:.2f} minutes"
            )

            if (
                val_accuracy >
                best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )

                checkpoint = (
                    self.save_checkpoint(
                        epoch + 1,
                        val_accuracy,
                    )
                )

                print(
                    f"Best model saved: "
                    f"{checkpoint}"
                )

        self.save_history()

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.2f}%"
        )

        return self.history