"""Данные: загрузка segments/sidewalk-semantic, кэш уменьшенных копий, разбиение, аугментация.

Главное исправление относительно первой версии проекта: маски содержат НОМЕРА КЛАССОВ, поэтому их
нельзя масштабировать билинейно и делить на 255. Здесь маска масштабируется методом ближайшего соседа
и хранится как целые числа."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from classes import IGNORE_INDEX, NUM_CLASSES

HF_NAME = "segments/sidewalk-semantic"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet (нужно для предобученного энкодера)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def shift_labels(mask: np.ndarray) -> np.ndarray:
    """Метки датасета 1..34 -> 0..33, 0 (unlabeled) -> IGNORE_INDEX."""
    out = mask.astype(np.int16) - 1
    out[out < 0] = IGNORE_INDEX
    return out.astype(np.uint8)


def resize_pair(image: Image.Image, mask: Image.Image, size):
    h, w = size
    img = image.convert("RGB").resize((w, h), Image.BILINEAR)
    msk = mask.resize((w, h), Image.NEAREST)           # NEAREST: иначе номера классов смешиваются
    return np.asarray(img, dtype=np.uint8), shift_labels(np.asarray(msk))


def load_cache(cache_dir="cache", size=(256, 512), hf_name=HF_NAME):
    """Один раз читает датасет с Hugging Face и сохраняет уменьшенные массивы в cache/*.npz."""
    path = Path(cache_dir) / f"sidewalk_{size[0]}x{size[1]}.npz"
    if path.exists():
        z = np.load(path)
        return z["images"], z["masks"]
    from datasets import load_dataset
    ds = load_dataset(hf_name, split="train")
    images, masks = [], []
    for i in range(len(ds)):
        im, mk = resize_pair(ds[i]["pixel_values"], ds[i]["label"], size)
        images.append(im); masks.append(mk)
    images, masks = np.stack(images), np.stack(masks)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, masks=masks)
    return images, masks


def split_indices(n, seed=42, fractions=(0.8, 0.1, 0.1)):
    perm = np.random.RandomState(seed).permutation(n)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return {"train": perm[:a], "val": perm[a:b], "test": perm[b:]}


class SegDataset(Dataset):
    def __init__(self, images, masks, indices, train=False):
        self.images, self.masks, self.indices, self.train = images, masks, indices, train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        img = self.images[i].astype(np.float32) / 255.0
        msk = self.masks[i].astype(np.int64)
        if self.train:
            if np.random.rand() < 0.5:                        # горизонтальный разворот: тот же для маски
                img, msk = img[:, ::-1], msk[:, ::-1]
            img = np.clip((img - 0.5) * np.random.uniform(0.8, 1.2) + 0.5 + np.random.uniform(-0.1, 0.1), 0, 1)
        img = (img - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(msk))
