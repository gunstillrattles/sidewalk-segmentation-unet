"""Обучение. Примеры:
    python src/train.py --model resnet34_unet --epochs 40
    python src/train.py --data synthetic --epochs 2        # проверка кода без загрузки датасета
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classes import IGNORE_INDEX
from data import SegDataset, load_cache, split_indices
from metrics import ConfusionMatrix
from model import build_model
from synthetic import make_synthetic


def get_data(args):
    size = (args.height, args.width)
    if args.data == "synthetic":
        return make_synthetic(n=64, size=size)
    return load_cache(args.cache, size)


def evaluate(model, loader, device):
    model.eval()
    cm = ConfusionMatrix()
    loss_sum, n = 0.0, 0
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item() * x.size(0); n += x.size(0)
            cm.update(out.argmax(1), y)
    return loss_sum / n, cm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet34_unet", choices=["unet", "resnet34_unet"])
    ap.add_argument("--data", default="hf", choices=["hf", "synthetic"])
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="по умолчанию: 3e-4 (предобученный энкодер), 1e-3 (с нуля)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, masks = get_data(args)
    idx = split_indices(len(images), args.seed)
    mk = lambda split, shuffle: DataLoader(SegDataset(images, masks, idx[split], train=(split == "train")),  # noqa: E731
                                           batch_size=args.batch_size, shuffle=shuffle, num_workers=2, drop_last=(split == "train"))
    train_dl, val_dl = mk("train", True), mk("val", False)
    print(f"device={device}  train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}")

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    lr = args.lr or (1e-3 if args.model == "unet" or args.no_pretrained else 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.epochs * len(train_dl))
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out = Path(args.out); out.mkdir(exist_ok=True)
    res = Path(args.results); res.mkdir(exist_ok=True)
    hist = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pixel_acc": []}
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); tl = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        vl, cm = evaluate(model, val_dl, device)
        hist["train_loss"].append(tl / len(train_dl)); hist["val_loss"].append(vl)
        hist["val_miou"].append(cm.miou()); hist["val_pixel_acc"].append(cm.pixel_accuracy())
        flag = ""
        if cm.miou() > best:
            best = cm.miou()
            torch.save({"model": args.model, "state_dict": model.state_dict(), "size": (args.height, args.width),
                        "seed": args.seed, "val_miou": best}, out / "best.pt")
            flag = "  <- best"
        print(f"epoch {ep:3d}/{args.epochs}  train_loss {hist['train_loss'][-1]:.4f}  val_loss {vl:.4f}  "
              f"val_mIoU {cm.miou():.4f}  val_pix_acc {cm.pixel_accuracy():.4f}{flag}")

    (res / "history.json").write_text(json.dumps(hist, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); xs = range(1, args.epochs + 1)
    ax[0].plot(xs, hist["train_loss"], label="train"); ax[0].plot(xs, hist["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(xs, hist["val_miou"], label="mIoU"); ax[1].plot(xs, hist["val_pixel_acc"], label="pixel acc.")
    ax[1].set(title="Validation", xlabel="epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(res / "training_curves.png", dpi=120)
    return best


if __name__ == "__main__":
    main()
