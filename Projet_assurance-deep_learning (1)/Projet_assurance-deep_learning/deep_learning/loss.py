from __future__ import annotations
import torch
from config import NUM_CLASSES


def soft_qwk_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    temperature: float = 1.0,
) -> torch.Tensor:
    classes = torch.arange(1, num_classes + 1, device=preds.device, dtype=preds.dtype)

    # QWK weight matrix: W[i,j] = (i-j)^2 / (K-1)^2
    W = ((classes.unsqueeze(0) - classes.unsqueeze(1)) ** 2) / (num_classes - 1) ** 2

    # Soft prediction distribution via softmax over squared distances to each class centre
    dists = -((preds.unsqueeze(1) - classes.unsqueeze(0)) ** 2) / temperature
    P = torch.softmax(dists, dim=1)  # (N, K)

    # One-hot true distribution
    Q = torch.zeros(len(targets), num_classes, device=preds.device, dtype=preds.dtype)
    Q.scatter_(1, (targets.long() - 1).unsqueeze(1), 1.0)  # (N, K)

    N = preds.shape[0]
    O = P.T @ Q / N                                                # soft confusion (K, K)
    E = P.sum(0).unsqueeze(1) @ Q.sum(0).unsqueeze(0) / N ** 2   # expected matrix (K, K)

    # Minimising this ratio is equivalent to maximising QWK
    return (W * O).sum() / ((W * E).sum() + 1e-8)
