import torch
from torch.utils.data import DataLoader, TensorDataset
from defect_qc.eval import evaluate_binary_classification

class DummyModel(torch.nn.Module):
    def forward(self, x):
        # x: [B, ...] -> logits [B,2]
        B = x.shape[0]
        logits = torch.zeros(B, 2)
        logits[:, 1] = 1.0  # always predict class 1 with higher logit
        return logits

def test_evaluate_returns_keys():
    device = torch.device("cpu")
    model = DummyModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Fake data: 8 samples
    x = torch.randn(8, 3, 64, 64)
    y = torch.zeros(8, dtype=torch.long)  # all class 0
    dl = DataLoader(TensorDataset(x, y), batch_size=4)

    out = evaluate_binary_classification(model, dl, criterion, device, threshold=0.5, compute_cm=True)

    assert "loss" in out and isinstance(out["loss"], float)
    assert "acc" in out and isinstance(out["acc"], float)
    assert "f1" in out and isinstance(out["f1"], float)
    assert "precision" in out and isinstance(out["precision"], float)
    assert "recall" in out and isinstance(out["recall"], float)

    assert "cm" in out
    assert tuple(out["cm"].shape) == (2, 2)