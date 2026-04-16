from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import v2


@dataclass(frozen=True)
class DemoModel:
    model: torch.nn.Module
    checkpoint: dict[str, Any]
    device: torch.device

    @property
    def img_mode(self) -> str:
        return self.checkpoint["data"]["img_mode"]

    @property
    def img_size(self) -> int:
        return int(self.checkpoint["data"]["img_size"])

    @property
    def class_names(self) -> list[str]:
        return list(self.checkpoint["data"]["class_names"])

    @property
    def threshold(self) -> float:
        return float(self.checkpoint["evaluation"]["threshold"])


@dataclass(frozen=True)
class Prediction:
    label_idx: int
    label: str
    prob_ok: float
    prob_defect: float
    logits: torch.Tensor
    probs: torch.Tensor


def get_inference_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_demo_model(checkpoint_path: str | Path, device: torch.device | None = None) -> DemoModel:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = device or get_inference_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = OmegaConf.create(checkpoint["model"])
    model = instantiate(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return DemoModel(model=model, checkpoint=checkpoint, device=device)


def build_inference_transform(img_mode: str, img_size: int):
    if img_mode not in {"RGB", "L"}:
        raise ValueError(f"Unsupported image mode: {img_mode}")

    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if img_mode == "RGB" else ((0.5,), (0.5,))
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ]
    )


def preprocess_image(image: Image.Image, img_mode: str, img_size: int) -> torch.Tensor:
    transform = build_inference_transform(img_mode, img_size)
    return transform(image.convert(img_mode))


def predict_tensor(demo_model: DemoModel, x: torch.Tensor, threshold: float) -> Prediction:
    with torch.no_grad():
        xb = x.unsqueeze(0).to(demo_model.device)
        logits = demo_model.model(xb)
        probs = torch.softmax(logits, dim=-1).squeeze(0).detach().cpu()

    prob_ok = float(probs[0])
    prob_defect = float(probs[1])
    label_idx = int(prob_defect >= threshold)
    label = demo_model.class_names[label_idx]
    return Prediction(
        label_idx=label_idx,
        label=label,
        prob_ok=prob_ok,
        prob_defect=prob_defect,
        logits=logits.squeeze(0).detach().cpu(),
        probs=probs,
    )


def predict_image(demo_model: DemoModel, image: Image.Image, threshold: float) -> tuple[Prediction, torch.Tensor]:
    x = preprocess_image(image, demo_model.img_mode, demo_model.img_size)
    return predict_tensor(demo_model, x, threshold), x


def tensor_to_display_image(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu()
    arr = arr * 0.5 + 0.5
    arr = arr.clamp(0, 1)
    if arr.shape[0] == 1:
        return arr.squeeze(0).numpy()
    return arr.permute(1, 2, 0).numpy()


def make_gradcam_overlay(
    demo_model: DemoModel,
    x: torch.Tensor,
    class_idx: int,
    alpha: float = 0.45,
) -> np.ndarray:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None
    target_layer = demo_model.model.conv5

    def save_activation(_module, _inputs, output):
        nonlocal activations
        activations = output

    def save_gradient(_module, _grad_inputs, grad_outputs):
        nonlocal gradients
        gradients = grad_outputs[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        demo_model.model.zero_grad(set_to_none=True)
        xb = x.unsqueeze(0).to(demo_model.device)
        logits = demo_model.model(xb)
        logits[:, class_idx].sum().backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(demo_model.img_size, demo_model.img_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu()
        cam_min, cam_max = cam.min(), cam.max()
        if float(cam_max - cam_min) > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        base = tensor_to_display_image(x)
        if base.ndim == 2:
            base_rgb = np.repeat(base[..., None], 3, axis=2)
        else:
            base_rgb = base

        heat = np.zeros_like(base_rgb)
        heat[..., 0] = cam.numpy()
        heat[..., 1] = np.clip(1.0 - np.abs(cam.numpy() - 0.5) * 2.0, 0.0, 1.0)
        overlay = (1.0 - alpha) * base_rgb + alpha * heat
        return np.clip(overlay, 0.0, 1.0)
    finally:
        forward_handle.remove()
        backward_handle.remove()
