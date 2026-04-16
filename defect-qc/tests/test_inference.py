from PIL import Image
import torch

from defect_qc.inference import DemoModel, predict_tensor, preprocess_image


class FixedLogitModel(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 2)
        logits[:, 1] = 1.0
        return logits


def test_preprocess_image_grayscale_shape():
    image = Image.new("RGB", (320, 240), color=(128, 128, 128))

    x = preprocess_image(image, img_mode="L", img_size=224)

    assert tuple(x.shape) == (1, 224, 224)
    assert x.dtype == torch.float32


def test_predict_tensor_uses_threshold():
    demo_model = DemoModel(
        model=FixedLogitModel().eval(),
        checkpoint={
            "data": {
                "img_mode": "L",
                "img_size": 224,
                "class_names": ["ok", "defect"],
            },
            "evaluation": {"threshold": 0.5},
        },
        device=torch.device("cpu"),
    )
    x = torch.zeros(1, 224, 224)

    low_threshold = predict_tensor(demo_model, x, threshold=0.5)
    high_threshold = predict_tensor(demo_model, x, threshold=0.9)

    assert low_threshold.label == "defect"
    assert high_threshold.label == "ok"
