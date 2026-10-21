# Sidewalk semantic segmentation (U-Net, PyTorch)

Semantic segmentation of street-level photos into 34 classes (road, **sidewalk**, crosswalk, curb, car, person, building,
vegetation, ...) on the [segments/sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic) dataset.
Two models are compared: a **U-Net written from scratch** and a **U-Net with a pretrained ResNet-34 encoder**.

<!-- RESULTS -->

## Dataset

1,000 photos of sidewalks and roads (the dataset ships a single split, so I split it 80/10/10 with a fixed seed:
800 train / 100 val / 100 test). Labels are per-pixel class IDs; ID 0 (`unlabeled`) is ignored in the loss and in the metrics.
The dataset is gated and licensed **CC BY-NC 4.0** (non-commercial); it is not included here.

## Method

- **Input:** images resized to 256x512 (bilinear), ImageNet normalisation. Cached once as `.npz` so epochs are fast.
- **Masks:** resized with **nearest-neighbour** and kept as integer class IDs. (Resizing them bilinearly and dividing by 255,
  as in my first attempt, mixes class IDs into meaningless values: the model then cannot learn anything useful.)
- **Augmentation (train only):** horizontal flip (applied to image and mask together), random brightness/contrast.
- **Models:** `unet`: classic U-Net (4 down/up blocks, BatchNorm); `resnet34_unet`: ResNet-34 (ImageNet) encoder with a U-Net decoder.
- **Loss / optimiser:** cross-entropy with `ignore_index`, AdamW, one-cycle LR schedule, mixed precision on GPU.
- **Metrics:** mean IoU over classes present in the split, pixel accuracy, IoU of the `sidewalk` class. The checkpoint with the
  best **validation** mIoU is evaluated once on the **test** split.

## Run it

Easiest: open `notebooks/train_colab.ipynb` in Google Colab (free GPU). Locally:

```bash
pip install -r requirements.txt
huggingface-cli login                       # accept the dataset conditions on Hugging Face first
python src/train.py --model resnet34_unet --epochs 40
python src/evaluate.py --ckpt checkpoints/best.pt
pytest                                      # checks the pipeline on synthetic data, no dataset needed
```

`python src/train.py --data synthetic --epochs 2` runs the whole training loop on generated data to check the code.

## Layout

```
src/classes.py    class names, ignore index
src/data.py       loading, caching, split, augmentation
src/model.py      UNet, ResNetUNet
src/metrics.py    confusion matrix, IoU, pixel accuracy
src/train.py      training loop, checkpointing, curves
src/evaluate.py   test metrics, per-class IoU chart, example predictions
tests/            tests of mask handling, metrics, shapes and a full train/evaluate cycle
notebooks/        Colab notebook
```
