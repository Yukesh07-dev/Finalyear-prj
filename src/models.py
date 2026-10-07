import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    resnet152,
    ResNet50_Weights,
    ResNet152_Weights,
)


def build_resnet50(num_classes=15):
    """
    ImageNet-pretrained ResNet-50.
    All convolutional layers are frozen.
    Only the final classification layer is trainable.
    """

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze every parameter
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


def build_finetuned_resnet50(
    num_classes=15,
    last_conv_layers=30,
):
    """
    ImageNet-pretrained ResNet-50.

    Early layers remain frozen.
    The final 30 convolutional layers are unfrozen,
    following the methodology described in the paper.
    """

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Initially freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Find convolutional layers
    conv_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]

    # Unfreeze final N convolutional layers
    for layer in conv_layers[-last_conv_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Classification head
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # FC must always be trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def build_finetuned_resnet152(
    num_classes=15,
    last_conv_layers=30,
):
    """
    ImageNet-pretrained ResNet-152.

    The same partial fine-tuning strategy is used:
    final 30 convolutional layers are trainable.
    """

    model = resnet152(weights=ResNet152_Weights.DEFAULT)

    # Initially freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Find convolutional layers
    conv_layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]

    # Unfreeze final 30 convolutional layers
    for layer in conv_layers[-last_conv_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Classification head
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    # FC must always be trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def trainable_parameters(model):
    """Return only parameters that require gradients."""
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def print_model_summary(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(
        f"Frozen parameters:    "
        f"{total - trainable:,}"
    )
    print("=" * 60)