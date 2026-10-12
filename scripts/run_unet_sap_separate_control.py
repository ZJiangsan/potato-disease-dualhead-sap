# ADDITIONAL CONTROL: ResNet-34 U-Net + SAP, 5-fold on CLEAN 421 dataset.
# Standalone Spyder file. Requires completed unet_r34_s0_f0.pt ... f4.pt
# and folds.csv in ~/potato_revision_skips_CLEAN_421.
# Does NOT retrain baseline, regenerate folds, or overwrite old predictions.
# Retains same U-Net, SAP mechanism, training loss, augmentation, evaluation and fine-tuning settings.
# Default 5 folds x 1 seed; change EXTRA_SEEDS only if matching U-Net baseline checkpoints exist.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:43:33 2026

@author: nibio
"""





# %% Revision analyses for PBT-D-26-00881 (potato tuber disease, dual-head CNN-Transformer + SAP)
# Flat cell script, run top to bottom. Finished runs are skipped on re-run, so it can be stopped and resumed.
#
# Reviewer comment -> where it is handled
#   5  interpretability      Grad-CAM lesion attention ratio + lesion/background occlusion test
#   6  Attention U-Net       model "att_unet"
#   7  segmentation metrics  pixel precision, recall, Dice (= F1), IoU
#   8  generalisation        image-level 5-fold CV, augmentation on training images only (fixes the leakage);
#                            optional cross-source test (public -> in-house)
#   9  uncertainty           test-time augmentation: SD of lesion area, predictive entropy
#   10 complexity            parameters, FLOPs, GPU and CPU latency
#   11 significance          McNemar, paired Wilcoxon, bootstrap 95% CI, repeated seeds
#
# Data layout (from the screenshots): ~/potato_diseaseN/<Class>/<n>.jpg + <n>.json (AnyLabeling polygons)
#
# The model cell is a re-implementation from the manuscript text. If your own classes differ
# (Transformer depth, decoder, SAP normalisation), paste them into that cell; the only contract is
# forward(x) -> (class_logits [B,C], seg_logits [B,1,H,W]) and, for Grad-CAM, an attribute `cam_layer`.

import os
GPU_ID = "0"                # index of the 32 GB V100 as listed by nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # same numbering as nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID         # must be set before torch is imported

import json, random, time, warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (accuracy_score, f1_score, r2_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)
cv2.setNumThreads(0)        # OpenCV threads inside DataLoader workers only slow things down
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| albumentations", A.__version__)

# %% Config
DATA_DIR = Path.home() / "potato_diseaseN"
OUT_DIR = Path.home() / "potato_revision_skips_CLEAN_421"

# folder -> name used in tables. CHECK DR_S / Dry_Rot against Table 1 (Surface = 55, Severe = 56 images).
CLASS_NAMES = {
    "Black_Scurf": "Black Scurf",
    "Common_Scab": "Common Scab",
    "DR_I": "Dry Rot (Internal)",
    "DR_S": "Dry Rot (Surface)",
    "Dry_Rot": "Dry Rot (Severe)",
    "Pink_Rot": "Pink Rot",
    "Powdery_Scab": "Powdery Scab",
    "Silver_Scurf": "Silver Scurf",
}
LESION_LABELS = None        # None = every annotated shape is lesion; or a set of label names to keep

IMG_SIZE = 256              # not stated in the manuscript: set to what you trained with (multiple of 32)
PRETRAINED = True
N_FOLDS = 5
VAL_SPLITS = 8              # 1/8 of each training fold is held out for checkpoint selection
SPLIT_SEED = 42
DUP_HAMMING = 6             # retained in config for comparability; duplicate search disabled on cleaned data

MODELS = ["hybrid", "hybrid_sap", "cnn_only", "unet_r34", "att_unet", "trans_only", "vit_pre"]
SEEDS = {"hybrid": [0, 1, 2], "hybrid_sap": [0, 1, 2]}     # other models: seed 0 only
BASE, PROP = "hybrid", "hybrid_sap"                       # the pair that is tested against each other

# training, as in Section 2.5
LR = 1e-4
BATCH = 4
EPOCHS = 100
PATIENCE = 40               # early stopping on validation Dice
LAMBDA_CLS, LAMBDA_SEG = 1.0, 1.0
SAP_FROM_BASELINE = True    # manuscript: SAP model is fine-tuned from the baseline hybrid
SAP_EPOCHS = 50
SAP_DETACH_MASK = False     # True = no classification gradient into the segmentation head
HYBRID_SKIPS = True        # True = U-Net style skip connections in the hybrid decoder
TR_LAYERS, TR_HEADS = 4, 8

N_TTA = 20                  # stochastic passes for uncertainty
N_BOOT = 2000
SOURCE_CSV = DATA_DIR / "source.csv"   # optional: columns id (e.g. Black_Scurf/12), source (inhouse|public)

SMOKE = False               # True = 2-epoch dry run of the whole script (about 20 min) in a separate folder
if SMOKE:
    OUT_DIR = Path.home() / "potato_revision_smoke"
    EPOCHS, SAP_EPOCHS, SEEDS, N_TTA, N_BOOT = 2, 1, {}, 4, 100

NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = DEVICE == "cuda"
if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")

RUN_DIR, CKPT_DIR, FIG_DIR = OUT_DIR / "runs", OUT_DIR / "ckpt", OUT_DIR / "figures"
for d in (RUN_DIR, CKPT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
assert IMG_SIZE % 32 == 0

# finished runs are reused on re-run, so refuse to mix runs made with different settings
cfg = dict(IMG_SIZE=IMG_SIZE, PRETRAINED=PRETRAINED, N_FOLDS=N_FOLDS, VAL_SPLITS=VAL_SPLITS, SPLIT_SEED=SPLIT_SEED,
           DUP_HAMMING=DUP_HAMMING, LR=LR, BATCH=BATCH, EPOCHS=EPOCHS, PATIENCE=PATIENCE, LAMBDA_CLS=LAMBDA_CLS,
           LAMBDA_SEG=LAMBDA_SEG, SAP_FROM_BASELINE=SAP_FROM_BASELINE, SAP_EPOCHS=SAP_EPOCHS,
           SAP_DETACH_MASK=SAP_DETACH_MASK, HYBRID_SKIPS=HYBRID_SKIPS, TR_LAYERS=TR_LAYERS, TR_HEADS=TR_HEADS,
           N_TTA=N_TTA, LESION_LABELS=sorted(LESION_LABELS) if LESION_LABELS else None)
cfg_file = OUT_DIR / "config.json"
if cfg_file.exists():
    old = json.load(open(cfg_file))
    changed = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
    assert not changed, f"settings changed since the runs in {OUT_DIR} were made (old, new): {changed}. Use a new OUT_DIR or delete the old one."
else:
    json.dump(cfg, open(cfg_file, "w"), indent=1)

# %% Index the dataset
classes = [c for c in CLASS_NAMES if (DATA_DIR / c).is_dir()]
assert classes, f"no class folders found in {DATA_DIR}"
rows = []
for ci, c in enumerate(classes):
    for p in sorted((DATA_DIR / c).iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        j = p.with_suffix(".json")
        if not j.exists():
            print("no annotation, skipped:", p)
            continue
        rows.append(dict(id=f"{c}/{p.stem}", cls=c, y=ci, img=str(p), ann=str(j)))
meta = pd.DataFrame(rows)
assert meta.id.is_unique, "two image files share a name within a class folder (e.g. 1.jpg and 1.png)"
# All four verified duplicates have already been deleted manually from DATA_DIR.
# Never delete any images in this script. Fail clearly if the old dataset is used.
REMOVED_IDS = [
    "Common_Scab/29",
    "Dry_Rot/11",
    "Pink_Rot/29",
    "Silver_Scurf/IMG_20250114_090213",  # folder spelling corrected (not Silver_Scruf)
]
assert not (set(REMOVED_IDS) & set(meta.id)), (
    "Some manually removed images are still present: " + str(sorted(set(REMOVED_IDS) & set(meta.id)))
)
assert len(meta) == 421, (
    f"Expected 421 annotated images after removing four from 425; found {len(meta)}. "
    "Confirm DATA_DIR and that each remaining image has its JSON annotation."
)
meta[["id", "cls", "img", "ann"]].to_csv(OUT_DIR / "clean_dataset_manifest.csv", index=False)
N_CLS = len(classes)
print(meta.cls.value_counts().reindex(classes).rename(CLASS_NAMES).to_string(), "\ntotal", len(meta))

# %% Load images and rasterise the polygons at IMG_SIZE (kept in RAM)
IMGS, MASKS, labels_seen, n_rot, n_size = [], [], {}, 0, 0
for r in meta.itertuples():
    buf = np.fromfile(r.img, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img is not None, f"cannot read {r.img}"
    ann = json.load(open(r.ann, encoding="utf-8"))
    h, w = img.shape[:2]
    jw, jh = ann.get("imageWidth") or w, ann.get("imageHeight") or h
    if (jw, jh) == (h, w) and h != w:       # annotated without the EXIF rotation that OpenCV applies
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        n_rot += 1
    elif (jw, jh) != (w, h):
        n_size += 1
    w, h = jw, jh                           # polygon coordinates are in the annotation's image size
    mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    for s in ann.get("shapes", []):
        labels_seen[s.get("label")] = labels_seen.get(s.get("label"), 0) + 1
        if LESION_LABELS is not None and s.get("label") not in LESION_LABELS:
            continue
        pts = np.array(s["points"], np.float32) * [IMG_SIZE / w, IMG_SIZE / h]
        kind = s.get("shape_type", "polygon")
        if kind == "polygon" and len(pts) >= 3:
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        elif kind == "rectangle" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 1, -1)
        else:
            print("shape ignored:", r.id, kind)
    IMGS.append(cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    MASKS.append(mask)
meta["lesion_pct"] = [100 * m.mean() for m in MASKS]
print("annotation labels:", labels_seen)        # anything here that is not a lesion -> set LESION_LABELS
print("images with empty mask:", int((meta.lesion_pct == 0).sum()))
print(f"EXIF rotation undone for {n_rot} images; image size differs from the annotation for {n_size} images")

# %% LOOK AT THIS BEFORE TRAINING: four images per class with the lesion outline -> figures/check_masks.png
rng = np.random.default_rng(0)
fig, axes = plt.subplots(N_CLS, 4, figsize=(10, 2.6 * N_CLS), squeeze=False)
for ci in range(N_CLS):
    members = np.where(meta.y.values == ci)[0]
    for ax, i in zip(axes[ci], rng.choice(members, min(4, len(members)), replace=False)):
        im = IMGS[i].copy()
        cv2.drawContours(im, cv2.findContours(MASKS[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 0), 1)
        ax.imshow(im)
        ax.set_title(f"{meta.id[i]}  {meta.lesion_pct[i]:.1f}%", fontsize=8)
for ax in axes.ravel():
    ax.axis("off")
fig.savefig(FIG_DIR / "check_masks.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% Cleaned data: the user visually verified and manually removed duplicate images.
# One grouping identifier per remaining image, assuming no repeated tuber photographs remain.
meta["group"] = np.arange(len(meta))
print("Cleaned-data analysis: no automatic duplicate removal or near-duplicate detection")

# %% Folds: split the ORIGINAL images first, stratified by class, duplicates kept together
fold_file = OUT_DIR / "folds.csv"
if fold_file.exists():
    meta = meta.drop(columns="fold", errors="ignore").merge(pd.read_csv(fold_file), on="id", how="left")
    assert meta.fold.notna().all(), "folds.csv does not match the dataset: delete it and the runs/ folder"
else:
    meta["fold"] = -1
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SPLIT_SEED)
    for f, (_, te) in enumerate(sgkf.split(meta, meta.y, meta.group)):
        meta.loc[te, "fold"] = f
    meta[["id", "fold"]].to_csv(fold_file, index=False)
meta["fold"] = meta.fold.astype(int)
print(pd.crosstab(meta.cls, meta.fold).reindex(classes).to_string())


def inner_split(tr_idx, seed=SPLIT_SEED):
    """Hold out part of a training fold for checkpoint selection (originals, never augmented copies)."""
    sub = meta.loc[tr_idx]
    sg = StratifiedGroupKFold(VAL_SPLITS, shuffle=True, random_state=seed)
    a, b = next(sg.split(sub, sub.y, sub.group))
    return tr_idx[a], tr_idx[b]


# %% Augmentation (Table 2), applied on the fly to training images only
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Affine(scale=(0.85, 1.15), rotate=(-25, 25), translate_percent=(-0.0625, 0.0625), p=0.7),
    A.OneOf([A.ElasticTransform(alpha=40, sigma=8), A.GridDistortion()], p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.CLAHE(clip_limit=3.0, p=0.3),
    A.OneOf([A.RandomShadow(), A.RandomFog(), A.RandomRain()], p=0.5),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5)), A.MedianBlur(blur_limit=5)], p=0.4),
])


class TuberDS(Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = list(idx), train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        img, mask = IMGS[i], MASKS[i]
        if self.train:
            o = train_tf(image=img, mask=mask)
            img, mask = o["image"], o["mask"]
        x = torch.from_numpy(((img / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy()).float()
        return x, torch.from_numpy(mask.copy()).float()[None], int(meta.y[i]), i


def seed_worker(worker_id):
    """Give every worker, every epoch, its own augmentation random stream (otherwise they all repeat one)."""
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)
    if hasattr(train_tf, "set_random_seed"):
        train_tf.set_random_seed(s)


def loader(idx, train):
    return DataLoader(TuberDS(idx, train), batch_size=BATCH if train else 16, shuffle=train, drop_last=train,
                      num_workers=NUM_WORKERS, pin_memory=DEVICE == "cuda", worker_init_fn=seed_worker)


# %% Models
def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TBlock(nn.Module):
    """Pre-norm Transformer encoder block written with plain Linear layers (so FLOP counters see it)."""

    def __init__(self, dim, heads, mlp_ratio=2, drop=0.1):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, dim * 3), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(dim * mlp_ratio, dim), nn.Dropout(drop))

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(b, n, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-2, -1)) * (c // self.h) ** -0.5).softmax(-1)
        x = x + self.proj((a @ v).transpose(1, 2).reshape(b, n, c))
        return x + self.mlp(self.n2(x))


class AttGate(nn.Module):
    """Additive attention gate of Attention U-Net (Oktay et al., 2018)."""

    def __init__(self, cg, cx):
        super().__init__()
        ci = max(cx // 2, 16)
        self.wg = nn.Sequential(nn.Conv2d(cg, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.wx = nn.Sequential(nn.Conv2d(cx, ci, 1, bias=False), nn.BatchNorm2d(ci))
        self.psi = nn.Sequential(nn.Conv2d(ci, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        return x * self.psi(F.relu(self.wg(g) + self.wx(x)))


class DualHead(nn.Module):
    """ResNet-34 encoder, optional Transformer bottleneck, deconvolution decoder (optional skips /
    attention gates), classification from global average pooling or Segmentation-Aware Pooling."""

    def __init__(self, n_cls, transformer=False, skips=False, att=False, sap=False):
        super().__init__()
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1" if PRETRAINED else None)
        self.encoder = nn.Sequential(*list(r.children())[:-2])      # same indexing as in Section 2.7
        self.cam_layer = self.encoder[-1][-2]
        self.transformer, self.skips, self.att, self.sap = transformer, skips, att, sap
        if transformer:
            self.pos = nn.Parameter(torch.zeros(1, (IMG_SIZE // 32) ** 2, 512))
            nn.init.trunc_normal_(self.pos, std=0.02)
            self.tr = nn.Sequential(*[TBlock(512, TR_HEADS) for _ in range(TR_LAYERS)])
        cin, cout = [512, 256, 128, 64, 64], [256, 128, 64, 64, 32]
        cskip = [256, 128, 64, 64, 0] if skips else [0] * 5
        self.up = nn.ModuleList(nn.ConvTranspose2d(i, o, 2, stride=2) for i, o in zip(cin, cout))
        self.dec = nn.ModuleList(conv_block(o + s, o) for o, s in zip(cout, cskip))
        self.gates = nn.ModuleList(AttGate(o, s) for o, s in zip(cout[:4], cskip[:4])) if att else None
        self.seg = nn.Conv2d(32, 1, 1)
        self.cls = nn.Linear(512, n_cls)

    def forward(self, x):
        feats, f = [], x
        for i, layer in enumerate(self.encoder):
            f = layer(f)
            if i in (2, 4, 5, 6):                                   # 1/2, 1/4, 1/8, 1/16
                feats.append(f)
        if self.transformer:
            b, c, h, w = f.shape
            pos = self.pos
            if pos.shape[1] != h * w:
                s = int(pos.shape[1] ** 0.5)
                pos = F.interpolate(pos.transpose(1, 2).reshape(1, c, s, s), (h, w), mode="bilinear")
                pos = pos.flatten(2).transpose(1, 2)
            f = self.tr(f.flatten(2).transpose(1, 2) + pos).transpose(1, 2).reshape(b, c, h, w)
        d = f
        for i in range(5):
            d = self.up[i](d)
            if self.skips and i < 4:
                s = feats[3 - i]
                d = torch.cat([d, self.gates[i](d, s) if self.att else s], 1)
            d = self.dec[i](d)
        seg = self.seg(d)
        if self.sap:                                                # v = GAP(F * Norm(M)), Eq. 1
            m = F.adaptive_avg_pool2d(torch.sigmoid(seg.float()), f.shape[-2:])
            if SAP_DETACH_MASK:
                m = m.detach()
            v = (f.float() * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6)
        else:
            v = f.float().mean((2, 3))
        return self.cls(v.to(self.cls.weight.dtype)), seg


class TransOnly(nn.Module):
    """Transformer-only baseline: patch embedding, encoder trained from scratch, upsampling decoder."""

    def __init__(self, n_cls, dim=384, depth=6, heads=6, patch=16):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, (IMG_SIZE // patch) ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.tr = nn.Sequential(*[TBlock(dim, heads, 4) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        ch = [dim, 192, 96, 48, 24]
        self.dec = nn.Sequential(*[nn.Sequential(nn.ConvTranspose2d(a, b, 2, stride=2), nn.BatchNorm2d(b), nn.ReLU(True))
                                   for a, b in zip(ch[:-1], ch[1:])], nn.Conv2d(24, 1, 1))
        self.cls = nn.Linear(dim, n_cls)

    def forward(self, x):
        t = self.embed(x)
        b, c, h, w = t.shape
        t = self.norm(self.tr(t.flatten(2).transpose(1, 2) + self.pos))
        return self.cls(t.mean(1)), self.dec(t.transpose(1, 2).reshape(b, c, h, w))


class ViTPre(nn.Module):
    """ImageNet-pretrained ViT-B/16 with a classification head and a plain upsampling decoder."""

    def __init__(self, n_cls):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1" if PRETRAINED else None)
        self.vit.heads = nn.Identity()
        ch = [768, 256, 128, 64, 32]
        self.dec = nn.Sequential(*[nn.Sequential(nn.ConvTranspose2d(a, b, 2, stride=2), nn.BatchNorm2d(b), nn.ReLU(True))
                                   for a, b in zip(ch[:-1], ch[1:])], nn.Conv2d(32, 1, 1))
        self.cls = nn.Linear(768, n_cls)

    def forward(self, x):
        size = x.shape[-2:]
        t = self.vit._process_input(F.interpolate(x, (224, 224), mode="bilinear", align_corners=False))
        t = self.vit.encoder(torch.cat([self.vit.class_token.expand(len(t), -1, -1), t], 1))
        seg = self.dec(t[:, 1:].transpose(1, 2).reshape(len(t), 768, 14, 14))
        return self.cls(t[:, 0]), F.interpolate(seg, size, mode="bilinear", align_corners=False)


def build(name):
    if name == "hybrid":
        return DualHead(N_CLS, transformer=True, skips=HYBRID_SKIPS)
    if name == "hybrid_sap":
        return DualHead(N_CLS, transformer=True, skips=HYBRID_SKIPS, sap=True)
    if name == "cnn_only":
        return DualHead(N_CLS)
    if name == "unet_r34":
        return DualHead(N_CLS, skips=True)
    if name == "unet_sap":
        return DualHead(N_CLS, skips=True, sap=True)
    if name == "att_unet":
        return DualHead(N_CLS, skips=True, att=True)
    if name == "trans_only":
        return TransOnly(N_CLS)
    if name == "vit_pre":
        return ViTPre(N_CLS)
    raise ValueError(name)


MODEL_LABELS = {"cnn_only": "CNN-only", "unet_r34": "U-Net (ResNet-34)", "att_unet": "Attention U-Net (ResNet-34)",
                "trans_only": "Transformer-only", "vit_pre": "Pretrained ViT-B/16",
                "hybrid": "Hybrid CNN-Transformer", "hybrid_sap": "Hybrid CNN-Transformer + SAP"}

# %% Loss and training (Adam 1e-4, ReduceLROnPlateau on val loss, checkpoint on val Dice)
def total_loss(logits, seg, y, m):
    p = torch.sigmoid(seg)
    inter, den = (p * m).sum((1, 2, 3)), p.sum((1, 2, 3)) + m.sum((1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return LAMBDA_CLS * F.cross_entropy(logits, y) + LAMBDA_SEG * (dice + F.binary_cross_entropy_with_logits(seg, m))


def fit(model, tr_idx, va_idx, epochs, ckpt, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(train_tf, "set_random_seed"):
        train_tf.set_random_seed(seed)
    dl_tr, dl_va = loader(tr_idx, True), loader(va_idx, False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.9, patience=5)
    scaler = torch.amp.GradScaler(DEVICE, enabled=AMP) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=AMP)
    best, wait, hist = -1.0, 0, []
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, m, y, _ in dl_tr:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE, enabled=AMP):
                logits, seg = model(x)
            loss = total_loss(logits.float(), seg.float(), y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(x)
        model.eval()
        vl, dices, ious, hit = 0.0, [], [], 0
        with torch.no_grad():
            for x, m, y, _ in dl_va:
                x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                logits, seg = model(x)
                vl += total_loss(logits, seg, y, m).item() * len(x)
                p = (seg > 0).float()
                inter, a, b = (p * m).sum((1, 2, 3)), p.sum((1, 2, 3)), m.sum((1, 2, 3))
                dices += ((2 * inter + 1e-6) / (a + b + 1e-6)).tolist()
                ious += ((inter + 1e-6) / (a + b - inter + 1e-6)).tolist()
                hit += (logits.argmax(1) == y).sum().item()
        vl, vd = vl / len(va_idx), float(np.mean(dices))
        sched.step(vl)
        hist.append(dict(epoch=ep + 1, train_loss=tl / (len(dl_tr) * BATCH), val_loss=vl, val_dice=vd,
                         val_iou=float(np.mean(ious)), val_acc=hit / len(va_idx), lr=opt.param_groups[0]["lr"]))
        if vd > best:
            best, wait = vd, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return pd.DataFrame(hist)


# %% Evaluation of one trained model on held-out originals: metrics, Grad-CAM, uncertainty, occlusion
def gradcam(model, x):
    """Grad-CAM of the predicted class at model.cam_layer, upsampled to the input size."""
    store = {}

    def keep(mod, inp, out):
        out.retain_grad()
        store["a"] = out

    hook = model.cam_layer.register_forward_hook(keep)
    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        logits, _ = model(x)
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
    hook.remove()
    a, g = store["a"], store["a"].grad
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    return F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False).detach()


def tta_batch(x, gen):
    """N_TTA random views of one image: flips, rotation within +-25 deg, mild brightness/contrast."""
    out = []
    for _ in range(N_TTA):
        r = torch.rand(5, generator=gen).tolist()
        v = x.flip(2) if r[0] < 0.5 else x
        v = v.flip(1) if r[1] < 0.5 else v
        v = TF.rotate(v, (2 * r[2] - 1) * 25)
        out.append(v * (1 + (2 * r[3] - 1) * 0.2) + (2 * r[4] - 1) * 0.2)
    return torch.stack(out)


def evaluate(model, idx):
    model.eval()
    gen = torch.Generator().manual_seed(0)
    recs = []
    for x, m, y, ids in loader(idx, False):
        x, m = x.to(DEVICE), m.to(DEVICE)
        with torch.no_grad():
            logits, seg = model(x)
            prob = logits.softmax(1)
            p = (seg > 0).float()
            # occlusion: fill value 0 is the ImageNet mean after normalisation
            pred_les = model(x * m)[0].argmax(1)            # lesion pixels only
            pred_bg = model(x * (1 - m))[0].argmax(1)       # background only, lesion removed
        cam = gradcam(model, x) if hasattr(model, "cam_layer") else None
        for k in range(len(x)):
            tp, fp, fn = (p[k] * m[k]).sum().item(), (p[k] * (1 - m[k])).sum().item(), ((1 - p[k]) * m[k]).sum().item()
            with torch.no_grad():
                lt, st = model(tta_batch(x[k].cpu(), gen).to(DEVICE))
            pt = lt.softmax(1).mean(0)
            ratio = np.nan
            if cam is not None and cam[k].sum() > 0:
                ratio = ((cam[k] * m[k]).sum() / cam[k].sum()).item()
            recs.append(dict(
                i=int(ids[k]), y_true=int(y[k]), y_pred=int(prob[k].argmax()), p_max=prob[k].max().item(),
                tp=tp, fp=fp, fn=fn,
                dice=(2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6), iou=(tp + 1e-6) / (tp + fp + fn + 1e-6),
                pix_prec=tp / (tp + fp) if tp + fp else np.nan, pix_rec=tp / (tp + fn) if tp + fn else np.nan,
                area_gt=100 * m[k].mean().item(), area_pred=100 * p[k].mean().item(),
                tta_pred=int(pt.argmax()), tta_entropy=-(pt * (pt + 1e-12).log()).sum().item(),
                tta_area_sd=100 * (st > 0).float().mean((1, 2, 3)).std().item(),
                cam_ratio=ratio, pred_lesion_only=int(pred_les[k]), pred_bg_only=int(pred_bg[k])))
    d = pd.DataFrame(recs)
    d.insert(0, "id", meta.id[d.i].values)
    return d.drop(columns="i")


def train_and_eval(name, seed, tr_idx, te_idx, tag, init=None):
    """Train `name` on tr_idx (inner validation split for checkpoint selection), evaluate on te_idx."""
    out, ckpt = RUN_DIR / f"{tag}.csv", CKPT_DIR / f"{tag}.pt"
    if out.exists():
        return pd.read_csv(out)
    tr, va = inner_split(tr_idx)
    assert not set(meta.group[tr]) & set(meta.group[te_idx]) and not set(meta.group[va]) & set(meta.group[te_idx])
    torch.manual_seed(seed)
    model = build(name).to(DEVICE)
    epochs = EPOCHS
    if init is not None:
        model.load_state_dict(torch.load(init, map_location=DEVICE))
        epochs = SAP_EPOCHS
    t0 = time.time()
    hist = fit(model, tr, va, epochs, ckpt, seed)
    hist.to_csv(RUN_DIR / f"{tag}_history.csv", index=False)
    d = evaluate(model, te_idx)
    d.to_csv(out, index=False)
    print(f"{tag}: {len(hist)} epochs, {(time.time() - t0) / 60:.1f} min, "
          f"acc {100 * (d.y_true == d.y_pred).mean():.1f}, Dice {d.dice.mean():.3f}")
    return d


# %% Additional U-Net + SAP control: only new runs
# The original 7-model output directory and original summary CSVs are left untouched.
EXTRA_SEEDS = [0]            # change to [0,1,2] only if unet_r34_s1/s2 baseline ckpts exist
NEW_NAME = "unet_sap"
NEW_RUN_DIR = OUT_DIR / "unet_sap_control"
NEW_CKPT_DIR = NEW_RUN_DIR / "ckpt"
NEW_RUN_DIR.mkdir(parents=True, exist_ok=True)
NEW_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Strictly verify exact training/evaluation split and original baseline availability
assert len(meta) == 421 and meta.id.is_unique
assert len(meta.fold.unique()) == N_FOLDS and N_FOLDS == 5
original_folds = pd.read_csv(OUT_DIR / "folds.csv")
assert dict(zip(original_folds.id, original_folds.fold)) == dict(zip(meta.id, meta.fold)), "Fold mismatch"

# Check first; fail before any GPU work if missing baseline weights.
missing = [str(CKPT_DIR / f"unet_r34_s{s}_f{f}.pt")
           for s in EXTRA_SEEDS for f in range(N_FOLDS)
           if not (CKPT_DIR / f"unet_r34_s{s}_f{f}.pt").is_file()]
if missing:
    raise FileNotFoundError("Required trained U-Net checkpoints missing; cannot perform fair fine-tuning:\n" + "\n".join(missing))

all_idx = np.arange(len(meta))
records = []
for seed in EXTRA_SEEDS:
    for fold in range(N_FOLDS):
        tag = f"unet_sap_s{seed}_f{fold}"
        output_csv = NEW_RUN_DIR / f"{tag}.csv"
        output_ckpt = NEW_CKPT_DIR / f"{tag}.pt"
        base_ckpt = CKPT_DIR / f"unet_r34_s{seed}_f{fold}.pt"
        test_idx = all_idx[meta.fold.to_numpy() == fold]
        dev_idx = all_idx[meta.fold.to_numpy() != fold]
        tr_idx, va_idx = inner_split(dev_idx)
        assert not set(meta.group.iloc[tr_idx]) & set(meta.group.iloc[test_idx])
        assert not set(meta.group.iloc[va_idx]) & set(meta.group.iloc[test_idx])
        if output_csv.exists():
            print("SKIP existing:", tag)
            d = pd.read_csv(output_csv)
        else:
            print("Start:", tag, "baseline:", base_ckpt)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = build(NEW_NAME).to(DEVICE)
            try:
                state = torch.load(base_ckpt, map_location=DEVICE, weights_only=True)
            except TypeError:
                state = torch.load(base_ckpt, map_location=DEVICE)
            model.load_state_dict(state, strict=True)
            hist = fit(model, tr_idx, va_idx, SAP_EPOCHS, output_ckpt, seed)
            hist.to_csv(NEW_RUN_DIR / f"{tag}_history.csv", index=False)
            d = evaluate(model, test_idx)
            # Save only after a complete successful evaluation
            d.to_csv(output_csv, index=False)
            del model
            if DEVICE == "cuda": torch.cuda.empty_cache()
            print(tag, "accuracy", round(100*(d.y_true==d.y_pred).mean(), 2), "Dice", round(d.dice.mean(), 4))
        assert set(d.id) == set(meta.id.iloc[test_idx]), f"Missing or duplicate test images in {tag}"
        records.append(d.assign(model=NEW_NAME, seed=seed, fold=fold))

new = pd.concat(records, ignore_index=True)
new["correct"] = (new.y_true == new.y_pred).astype(int)
new["area_err"] = (new.area_pred - new.area_gt).abs()
new.to_csv(NEW_RUN_DIR / "unet_sap_all_oof.csv", index=False)

# Comparison from exact same image and fold, using saved original U-Net baseline predictions.
from sklearn.metrics import f1_score, r2_score
summary = []
comparisons = []
for seed in EXTRA_SEEDS:
    new_seed = new[new.seed == seed].set_index("id").sort_index()
    old_pieces = []
    for fold in range(N_FOLDS):
        old_path = RUN_DIR / f"unet_r34_s{seed}_f{fold}.csv"
        if not old_path.is_file():
            raise FileNotFoundError(str(old_path))
        old_pieces.append(pd.read_csv(old_path))
    baseline = pd.concat(old_pieces, ignore_index=True).set_index("id").sort_index()
    assert baseline.index.is_unique and new_seed.index.is_unique
    assert baseline.index.equals(new_seed.index), "Image IDs differ between U-Net and U-Net+SAP"
    assert np.array_equal(baseline.y_true.to_numpy(),new_seed.y_true.to_numpy())

    for label, d in [("U-Net", baseline), ("U-Net + SAP", new_seed)]:
        correct = (d.y_true == d.y_pred)
        summary.append(dict(seed=seed, model=label, n=len(d), accuracy_pct=100*correct.mean(),
           macro_f1=f1_score(d.y_true,d.y_pred,labels=list(range(N_CLS)),average="macro",zero_division=0),
           dice=d.dice.mean(), iou=d.iou.mean(), pixel_precision=d.pix_prec.mean(),
           pixel_recall=d.pix_rec.mean(), area_MAE_pp=(d.area_pred-d.area_gt).abs().mean(),
           area_R2=r2_score(d.area_gt,d.area_pred),
           gradcam_lesion_mass=d.cam_ratio.mean(),
           lesion_only_accuracy_pct=100*(d.pred_lesion_only==d.y_true).mean(),
           lesion_removed_accuracy_pct=100*(d.pred_bg_only==d.y_true).mean()))

    a=(baseline.y_pred==baseline.y_true).to_numpy()
    b=(new_seed.y_pred==new_seed.y_true).to_numpy()
    n01=int((~a & b).sum()); n10=int((a & ~b).sum())
    comparisons.append(dict(seed=seed, measure="classification_accuracy", baseline=100*a.mean(),
                            sap=100*b.mean(), delta=100*(b.mean()-a.mean()), test="exact_McNemar",
                            p=stats.binomtest(n01,n01+n10,0.5).pvalue if n01+n10 else 1.0,
                            only_sap_correct=n01, only_baseline_correct=n10))
    for col in ("dice", "iou", "cam_ratio"):
        delta=(new_seed[col]-baseline[col]).to_numpy(dtype=float)
        delta=delta[np.isfinite(delta)]
        p=(stats.wilcoxon(delta).pvalue if len(delta) and np.any(delta!=0) else np.nan)
        comparisons.append(dict(seed=seed,measure=col,baseline=baseline[col].mean(),
                                sap=new_seed[col].mean(),delta=new_seed[col].mean()-baseline[col].mean(),
                                test="paired_Wilcoxon",p=p))
    diff=((new_seed.area_pred-new_seed.area_gt).abs()-(baseline.area_pred-baseline.area_gt).abs()).to_numpy()
    comparisons.append(dict(seed=seed,measure="lesion_area_abs_error_pp",baseline=np.abs(baseline.area_pred-baseline.area_gt).mean(),
                            sap=np.abs(new_seed.area_pred-new_seed.area_gt).mean(),delta=np.mean(diff),
                            test="paired_Wilcoxon",p=stats.wilcoxon(diff).pvalue if np.any(diff!=0) else np.nan))

pd.DataFrame(summary).to_csv(NEW_RUN_DIR / "comparison_unet_vs_unet_sap.csv",index=False)
pd.DataFrame(comparisons).to_csv(NEW_RUN_DIR / "paired_tests_unet_vs_unet_sap.csv",index=False)
print("\nFINAL COMPARISON")
print(pd.DataFrame(summary).round(4).to_string(index=False))
print("Paired tests ->",NEW_RUN_DIR / "paired_tests_unet_vs_unet_sap.csv")
print("DONE ->",NEW_RUN_DIR)