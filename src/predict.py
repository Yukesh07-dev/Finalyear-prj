import sys
from pathlib import Path

import torch
from PIL import Image

from .config import NUM_CLASSES, SELECTED_CLASSES
from .models import build_finetuned_resnet50
from .transforms import get_validation_transform


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    project_root = Path(__file__).resolve().parent.parent
    checkpoint_path = (
        project_root
        / "checkpoints"
        / "finetuned_resnet50"
        / "best_model.pth"
    )

    model = build_finetuned_resnet50(
        num_classes=NUM_CLASSES,
        last_conv_layers=30,
    )

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

    return model


def predict_image(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    transform = get_validation_transform()
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = SELECTED_CLASSES[predicted_index.item()]
    confidence_percent = confidence.item() * 100

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(5, NUM_CLASSES),
        dim=1,
    )

    print("\n" + "=" * 60)
    print("BIRD SPECIES PREDICTION")
    print("=" * 60)

    print(f"Image: {image_path}")
    print(f"Predicted species: {predicted_class}")
    print(f"Confidence: {confidence_percent:.2f}%")

    print("\nTop predictions:")

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0],
    ):
        name = SELECTED_CLASSES[index.item()]
        percentage = probability.item() * 100

        print(f"{name:30s} {percentage:6.2f}%")

    print("=" * 60)

    return predicted_class, confidence_percent


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("python -m src.predict path_to_image.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    predict_image(image_path)


if __name__ == "__main__":
    main()