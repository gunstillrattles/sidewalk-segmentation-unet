import torch

from classes import IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    """Накопитель матрицы ошибок; пиксели с IGNORE_INDEX не учитываются."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.n = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, pred, target):
        pred, target = pred.flatten().cpu(), target.flatten().cpu()
        keep = target != IGNORE_INDEX
        idx = target[keep] * self.n + pred[keep]
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        tp = self.mat.diag().float()
        union = self.mat.sum(0).float() + self.mat.sum(1).float() - tp
        return torch.where(union > 0, tp / union.clamp(min=1), torch.full_like(tp, float("nan")))

    def miou(self):
        v = self.iou()
        return float(v[~torch.isnan(v)].mean()) if (~torch.isnan(v)).any() else 0.0

    def pixel_accuracy(self):
        total = self.mat.sum().item()
        return self.mat.diag().sum().item() / total if total else 0.0
