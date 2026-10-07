import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageEnhance, ImageFilter


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for PyTorch models.
    Pure PyTorch & PIL implementation (no external C++ DLL dependencies).
    Supports ResNet architectures (ResNet-50, ResNet-152) and custom CNN backbones.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        Args:
            model: PyTorch model in evaluation mode.
            target_layer: Module to extract activation and gradients from.
                          If None, automatically selects the final conv layer.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = self._find_target_layer()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register forward and backward hooks
        self.handles = []
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Automatically find the final convolutional layer of the model."""
        if hasattr(self.model, "layer4"):
            return self.model.layer4[-1]

        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("No Conv2d layer found in the model for Grad-CAM.")

        return last_conv

    def _register_hooks(self):
        """Register hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles.extend([h1, h2])

    def generate(self, input_tensor: torch.Tensor, target_class: int = None):
        """
        Generate raw Grad-CAM heatmap normalized to [0, 1].

        Args:
            input_tensor: Preprocessed tensor of shape (1, C, H, W).
            target_class: Integer index of target class. If None, uses top predicted class.

        Returns:
            heatmap: 2D numpy array of size (H, W) with values in range [0, 1].
            target_class: Integer class index used.
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Target class score
        score = output[0, target_class]

        # Backward pass
        score.backward()

        activations = self.activations[0]
        gradients = self.gradients[0]

        # Neuron importance weights alpha via Global Average Pooling
        alpha = torch.mean(gradients, dim=(1, 2), keepdim=True)

        # Weighted sum of activation maps
        cam = torch.sum(alpha * activations, dim=0)

        # Apply ReLU
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        h_input, w_input = input_tensor.shape[2], input_tensor.shape[3]
        cam_pil = Image.fromarray(cam_np).resize((w_input, h_input), Image.Resampling.BILINEAR)
        heatmap = np.array(cam_pil)

        return heatmap, target_class

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        alpha: float = 0.5,
        colormap_name: str = "jet",
    ):
        """
        Overlay Grad-CAM heatmap onto the original PIL image using Matplotlib colormaps.

        Args:
            heatmap: 2D numpy array normalized to [0, 1].
            original_image: PIL Image object.
            alpha: Blending weight between image and heatmap (0 to 1).
            colormap_name: Matplotlib colormap string ('jet', 'viridis', 'inferno').

        Returns:
            heatmap_colored_pil: Colorized heatmap as PIL Image.
            overlay_pil: Blended overlay image as PIL Image.
        """
        orig_pil = original_image.convert("RGB")
        w_orig, h_orig = orig_pil.size

        # Resize heatmap to match original image dimensions
        heatmap_resized = np.array(
            Image.fromarray(heatmap).resize((w_orig, h_orig), Image.Resampling.BILINEAR)
        )

        # Apply Matplotlib colormap (returns RGBA floats [0, 1])
        import matplotlib
        cmap = matplotlib.colormaps[colormap_name]
        heatmap_rgba = cmap(heatmap_resized)  # Shape (H, W, 4)
        heatmap_rgb = np.uint8(255 * heatmap_rgba[:, :, :3])

        # Blend with original image
        orig_np = np.array(orig_pil)
        blended_np = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * orig_np)

        return Image.fromarray(heatmap_rgb), Image.fromarray(blended_np)

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks in repeated Streamlit calls."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
