"""Синтетический набор для проверки кода без загрузки датасета: цветные прямоугольники, цвет = класс."""
import numpy as np

from classes import NUM_CLASSES


def make_synthetic(n=32, size=(64, 128), seed=0, n_classes=6):
    rs = np.random.RandomState(seed)
    palette = rs.randint(40, 255, size=(NUM_CLASSES, 3))
    h, w = size
    images = np.zeros((n, h, w, 3), np.uint8)
    masks = np.zeros((n, h, w), np.uint8)
    for k in range(n):
        masks[k] = rs.randint(0, n_classes)
        for _ in range(4):
            c = rs.randint(0, n_classes)
            y0, x0 = rs.randint(0, h // 2), rs.randint(0, w // 2)
            y1, x1 = y0 + rs.randint(h // 4, h // 2), x0 + rs.randint(w // 4, w // 2)
            masks[k, y0:y1, x0:x1] = c
        masks[k, :2, :2] = 255                                  # немного «unlabeled», чтобы проверять игнорирование
        images[k] = palette[np.where(masks[k] == 255, 0, masks[k])]
        images[k] = np.clip(images[k] + rs.randint(-12, 12, images[k].shape), 0, 255)
    return images, masks
