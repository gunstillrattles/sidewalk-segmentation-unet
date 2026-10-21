"""Оценка лучшей модели на тестовой части: python src/evaluate.py --ckpt checkpoints/best.pt"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from classes import CLASSES, IGNORE_INDEX, NUM_CLASSES, SIDEWALK
from data import MEAN, STD, SegDataset, load_cache, split_indices
from metrics import ConfusionMatrix
from model import build_model
from synthetic import make_synthetic


def palette():
    base = [plt.get_cmap("tab20")(i)[:3] for i in range(20)] + [plt.get_cmap("tab20b")(i)[:3] for i in range(20)]
    pal = (np.array(base[:NUM_CLASSES]) * 255).astype(np.uint8)
    pal[SIDEWALK] = (255, 0, 160)       # тротуар всегда ярко-розовый
    return pal


def colorize(mask, pal):
    out = np.full(mask.shape + (3,), 30, np.uint8)       # «unlabeled» = тёмно-серый, чтобы не путать с классом
    valid = mask != IGNORE_INDEX
    out[valid] = pal[mask[valid]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--data", default="hf", choices=["hf", "synthetic"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    size = tuple(ck["size"])
    images, masks = make_synthetic(n=64, size=size) if args.data == "synthetic" else load_cache(args.cache, size)
    idx = split_indices(len(images), ck["seed"])["test"]          # то же разбиение, что при обучении
    model = build_model(ck["model"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    ds = SegDataset(images, masks, idx, train=False)
    cm, preds = ConfusionMatrix(), []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=8):
            p = model(x.to(device)).argmax(1).cpu()
            cm.update(p, y); preds.append(p)
    preds = torch.cat(preds).numpy()
    iou = cm.iou().numpy()
    metrics = {"model": ck["model"], "n_test": len(idx), "mIoU": round(cm.miou() * 100, 2),
               "pixel_accuracy": round(cm.pixel_accuracy() * 100, 2),
               "sidewalk_IoU": None if np.isnan(iou[SIDEWALK]) else round(float(iou[SIDEWALK]) * 100, 2),
               "per_class_IoU": {c: (None if np.isnan(v) else round(float(v) * 100, 2)) for c, v in zip(CLASSES, iou)}}
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_IoU"}, indent=2))

    res = Path(args.results); res.mkdir(exist_ok=True)
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    rows = ["| class | IoU, % |", "|---|---|"] + [f"| {c} | {'n/a' if v is None else v} |" for c, v in metrics["per_class_IoU"].items()]
    (res / "per_class_iou.md").write_text("\n".join(rows))

    # IoU по классам
    order = [i for i in np.argsort(-np.nan_to_num(iou, nan=-1)) if not np.isnan(iou[i])]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([CLASSES[i] for i in order][::-1], [iou[i] * 100 for i in order][::-1],
            color=["#ff00a0" if i == SIDEWALK else "#4c78a8" for i in order][::-1])
    ax.set(xlabel="IoU, %", title=f"Per-class IoU (test), mIoU = {metrics['mIoU']}%"); ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(res / "per_class_iou.png", dpi=120); plt.close(fig)

    # примеры: первые N тестовых картинок подряд (без отбора)
    pal = palette(); n = min(args.samples, len(idx))
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.4 * n))
    axes = np.atleast_2d(axes)
    for r in range(n):
        raw, gt = images[idx[r]], masks[idx[r]]
        for c, (img, title) in enumerate(((raw, "image"), (colorize(gt, pal), "ground truth"), (colorize(preds[r], pal), "prediction"))):
            axes[r, c].imshow(img); axes[r, c].axis("off")
            if r == 0: axes[r, c].set_title(title)
    fig.tight_layout(); fig.savefig(res / "sample_predictions.png", dpi=110)


if __name__ == "__main__":
    main()
