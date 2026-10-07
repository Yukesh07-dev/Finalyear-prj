import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .config import SELECTED_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)


def discover_images(root_dir: Path):
    """
    Discover images from the selected 15 species.
    Returns a list of (image_path, class_index).
    """

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(SELECTED_CLASSES)
    }

    samples = []

    for class_name in SELECTED_CLASSES:
        class_dir = root_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(
                    (
                        image_path,
                        class_to_idx[class_name],
                    )
                )

    return samples


class BirdDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label