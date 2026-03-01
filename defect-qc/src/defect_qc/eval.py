import torch
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np


def evaluate_binary_classification(model, dataloader, criterion, device, threshold: float, compute_cm: bool = False):
    
    f1_metric = BinaryF1Score().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)

    if compute_cm:
        cm_metric = BinaryConfusionMatrix(threshold = threshold).to(device)

    model.eval()
    with torch.no_grad():
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            batch_size = yb.numel()
            loss_sum += loss.item() * batch_size
            prob = torch.softmax(logits, dim=-1)[:,1]
            preds = (prob >= threshold).long()
            correct += (preds == yb).sum().item()
            total += batch_size

            # Update metrics
            f1_metric.update(preds, yb)
            precision_metric.update(preds, yb)
            recall_metric.update(preds, yb)

            if compute_cm:
                cm_metric.update(prob, yb)

        out = {
                "loss": loss_sum / total,
                "acc": correct / total,
                "f1": f1_metric.compute().item(),
                "precision": precision_metric.compute().item(),
                "recall": recall_metric.compute().item(),
            }
        
        if compute_cm:
            out["cm"] = cm_metric.compute().detach().cpu()

    return out


def save_confusion_matrix(cm, path, class_names):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Convert to numpy safely
    if isinstance(cm, torch.Tensor):
        cm_np = cm.detach().cpu().numpy()
    else:
        cm_np = np.asarray(cm)

    # Row-wise normalization (percent per true class)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_percent = cm_np / row_sums * 100

    # Use normalized values for coloring (0–100 scale)
    im = ax.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)

    # Ticks and labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Rotate x labels for better readability
    plt.setp(ax.get_xticklabels(), ha="right")

    # Add annotations (percent + count)
    for i in range(cm_np.shape[0]):
        for j in range(cm_np.shape[1]):
            text = f"{cm_percent[i, j]:.1f}%\n({cm_np[i, j]})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_np[i, j] > cm_np.max() / 2 else "black",
            )

    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)