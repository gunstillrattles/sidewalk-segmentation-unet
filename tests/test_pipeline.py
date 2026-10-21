import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import data  # noqa: E402
import evaluate as ev  # noqa: E402
import train as tr  # noqa: E402
from classes import IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from metrics import ConfusionMatrix  # noqa: E402
from model import ResNetUNet, UNet  # noqa: E402
from synthetic import make_synthetic  # noqa: E402


def test_mask_resize_keeps_class_ids_and_shifts_labels():
    m = np.zeros((40, 80), np.uint8); m[:, :40] = 2; m[:, 40:] = 30      # метки датасета: 2=sidewalk, 30=sky
    _, out = data.resize_pair(Image.new("RGB", (80, 40)), Image.fromarray(m), (16, 32))
    assert set(np.unique(out)) == {1, 29}                                # без «смешанных» значений между классами
    m[0, 0] = 0                                                          # unlabeled -> игнорируется
    _, out = data.resize_pair(Image.new("RGB", (80, 40)), Image.fromarray(m), (40, 80))
    assert out[0, 0] == IGNORE_INDEX


def test_split_is_disjoint_and_stable():
    a, b = data.split_indices(100, 7), data.split_indices(100, 7)
    allidx = np.concatenate([a["train"], a["val"], a["test"]])
    assert sorted(allidx) == list(range(100)) and all((a[k] == b[k]).all() for k in a)


def test_confusion_matrix_iou_and_ignore():
    cm = ConfusionMatrix(3)
    target = torch.tensor([0, 0, 1, 1, 2, IGNORE_INDEX]); pred = torch.tensor([0, 1, 1, 1, 0, 2])
    cm.update(pred, target)
    iou = cm.iou()
    assert abs(iou[0].item() - 1 / 3) < 1e-6          # tp=1, fp=1(из класса 2), fn=1
    assert abs(iou[1].item() - 2 / 3) < 1e-6
    assert iou[2].item() == 0.0
    assert cm.mat.sum() == 5                          # игнорируемый пиксель не посчитан


def test_models_output_shapes():
    x = torch.zeros(2, 3, 64, 128)
    for m in (UNet(), ResNetUNet(pretrained=False)):
        assert m.eval()(x).shape == (2, NUM_CLASSES, 64, 128)


def test_full_cycle_on_synthetic_data(tmp_path):
    """Обучение должно реально учиться: mIoU на синтетике растёт, а evaluate отрабатывает до конца."""
    argv = ["--data", "synthetic", "--model", "unet", "--epochs", "12", "--batch-size", "8",
            "--height", "64", "--width", "128", "--out", str(tmp_path / "ck"), "--results", str(tmp_path / "res")]
    best = tr.main(argv)
    assert best > 0.5, f"модель не научилась даже на синтетике: mIoU={best:.3f}"
    ev.main(["--ckpt", str(tmp_path / "ck" / "best.pt"), "--data", "synthetic", "--results", str(tmp_path / "res")])
    for f in ("metrics.json", "per_class_iou.png", "sample_predictions.png", "training_curves.png"):
        assert (tmp_path / "res" / f).exists()
