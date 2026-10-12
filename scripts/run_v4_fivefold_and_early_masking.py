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

import json, random, re, time, warnings
from datetime import datetime
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
OUT_DIR = Path.home() / "potato_revision_v4"

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
# images left out of every analysis (checked by hand: duplicates / repeated shots of one tuber)
EXCLUDE = ["Common_Scab/29", "Dry_Rot/11", "Pink_Rot/29", "Silver_Scurf/IMG_20250114_090213"]
LESION_LABELS = None        # None = every annotated shape is lesion; or a set of label names to keep

IMG_SIZE = 256              # not stated in the manuscript: set to what you trained with (multiple of 32)
PRETRAINED = True
N_FOLDS = 5
VAL_SPLITS = 8              # 1/8 of each training fold is held out for checkpoint selection
SPLIT_SEED = 42
DUP_HAMMING = 6             # dHash distance <= this -> near-duplicates, kept in the same fold
DROP_HAMMING = -1           # automatic removal of identical images is off (EXCLUDE does it); 3 turns it back on
SAME_TUBER_S = 15            # camera photos of one class taken within this many seconds = same tuber, same fold

MODELS = ["hybrid", "hybrid_sap", "cnn_only", "unet_r34", "att_unet", "trans_only", "vit_pre", "hybrid_early"]
SEEDS = {"hybrid": [0, 1, 2], "hybrid_sap": [0, 1, 2], "hybrid_early": [0, 1, 2]}     # other models: seed 0 only
BASE, PROP = "hybrid", "hybrid_sap"                       # the pair that is tested against each other

# training, as in Section 2.5
LR = 1e-4
BATCH = 4
EPOCHS = 150
PATIENCE = 40               # early stopping on validation Dice
LAMBDA_CLS, LAMBDA_SEG = 1.0, 1.0
SAP_FROM_BASELINE = True    # manuscript: SAP model is fine-tuned from the baseline hybrid
SAP_EPOCHS = 50
SAP_DETACH_MASK = False     # True = no classification gradient into the segmentation head
HYBRID_SKIPS = True         # U-Net style skip connections in the hybrid decoder (False = original deconvolution decoder)
TR_LAYERS, TR_HEADS = 4, 8

# early lesion masking ("hybrid_early"): the trained hybrid segments, a second ResNet-34 classifies only the
# predicted lesion pixels. Not part of the settings guard: if you change these, delete runs/hybrid_early_* first.
MASK_DILATE = 5             # px at IMG_SIZE: lesion plus a thin margin is shown to the classifier
MASK_JITTER = 4             # random +-px on the dilation during training (robustness to segmentation errors)
P_TRUE_MASK = 0.5           # share of training batches that use the annotated mask instead of the predicted one

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
           EXCLUDE=sorted(EXCLUDE), DUP_HAMMING=DUP_HAMMING, DROP_HAMMING=DROP_HAMMING, SAME_TUBER_S=SAME_TUBER_S, LR=LR, BATCH=BATCH, EPOCHS=EPOCHS, PATIENCE=PATIENCE, LAMBDA_CLS=LAMBDA_CLS,
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
rows, excluded = [], []
for ci, c in enumerate(classes):
    for p in sorted((DATA_DIR / c).iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if f"{c}/{p.stem}" in EXCLUDE:
            excluded.append(f"{c}/{p.stem}")
            continue
        j = p.with_suffix(".json")
        if not j.exists():
            print("no annotation, skipped:", p)
            continue
        rows.append(dict(id=f"{c}/{p.stem}", cls=c, y=ci, img=str(p), ann=str(j)))
assert set(excluded) == set(EXCLUDE), f"EXCLUDE entries not found in the data folder: {set(EXCLUDE) - set(excluded)}"
print("excluded by hand:", excluded)
meta = pd.DataFrame(rows)
assert meta.id.is_unique, "two image files share a name within a class folder (e.g. 1.jpg and 1.png)"
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

# %% Groups that must stay in one fold: near-duplicate images, and photos of the same tuber taken seconds apart
hashes = []
for im in IMGS:
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    hashes.append((g[:, 1:] > g[:, :-1]).flatten())
hashes = np.array(hashes)
dist = (hashes[:, None, :] != hashes[None, :, :]).sum(-1)

# the same image stored twice (distance <= DROP_HAMMING, same class): keep the first, leave the other out.
# Nothing is deleted on disk; the excluded files are listed in removed_duplicates.csv.
drop, removed = set(), []
for i, k in zip(*np.where(np.triu(dist <= DROP_HAMMING, 1))):
    if meta.cls[i] == meta.cls[k] and i not in drop and k not in drop:
        drop.add(k)
        removed.append(dict(kept=meta.id[i], removed=meta.id[k], hamming=int(dist[i, k])))
if removed:
    keep = [i for i in range(len(meta)) if i not in drop]
    IMGS, MASKS = [IMGS[i] for i in keep], [MASKS[i] for i in keep]
    meta, dist = meta.iloc[keep].reset_index(drop=True), dist[np.ix_(keep, keep)]
    pd.DataFrame(removed).to_csv(OUT_DIR / "removed_duplicates.csv", index=False)
    print(f"{len(removed)} duplicate images left out -> removed_duplicates.csv; {len(meta)} images remain")
    print(meta.cls.value_counts().reindex(classes).rename(CLASS_NAMES).to_string())

parent = list(range(len(meta)))


def root(i):
    while parent[i] != i:
        i = parent[i]
    return i


dups = []
for i, k in zip(*np.where(np.triu(dist <= DUP_HAMMING, 1))):
    dups.append(dict(a=meta.id[i], b=meta.id[k], hamming=int(dist[i, k]), same_class=meta.cls[i] == meta.cls[k]))
    ri, rk = root(i), root(k)
    parent[max(ri, rk)] = min(ri, rk)

# camera files named ..._YYYYMMDD_HHMMSS: consecutive shots within SAME_TUBER_S seconds are treated as one tuber
stamp = {}
for i, name in enumerate(meta.id):
    hit = re.search(r"(\d{8})_(\d{6})", name)
    if hit:
        stamp[i] = datetime.strptime(hit.group(1) + hit.group(2), "%Y%m%d%H%M%S")
order = sorted(stamp, key=lambda i: (meta.cls[i], stamp[i]))
n_time = 0
for i, k in zip(order[:-1], order[1:]):
    if meta.cls[i] == meta.cls[k] and (stamp[k] - stamp[i]).total_seconds() <= SAME_TUBER_S:
        ri, rk = root(i), root(k)
        parent[max(ri, rk)] = min(ri, rk)
        n_time += 1

meta["group"] = [root(i) for i in range(len(parent))]
pd.DataFrame(dups).to_csv(OUT_DIR / "near_duplicates.csv", index=False)
multi = meta[meta.group.map(meta.group.value_counts()) > 1].sort_values("group")
multi[["group", "id"]].to_csv(OUT_DIR / "groups.csv", index=False)
print(f"{len(dups)} near-duplicate pairs -> near_duplicates.csv (cross-class pairs are label problems)")
print(f"{len(stamp)} time-stamped images, {n_time} consecutive pairs within {SAME_TUBER_S} s linked as the same tuber")
print(f"{meta.group.nunique()} groups for {len(meta)} images; {len(multi)} images share a group -> groups.csv; "
      f"largest group: {int(meta.group.value_counts().max())} images")

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


class EarlyMask(nn.Module):
    """Segment first, then classify only the predicted lesion pixels; everything else is set to the ImageNet mean.
    Segmentation branch: the trained hybrid of the same seed and fold, frozen. Classification branch: its own ResNet-34
    with mask-weighted pooling. The classifier therefore cannot see background by construction."""
    early = True

    def __init__(self, n_cls):
        super().__init__()
        self.seg_net = DualHead(n_cls, transformer=True, skips=HYBRID_SKIPS)
        for prm in self.seg_net.parameters():
            prm.requires_grad_(False)
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1" if PRETRAINED else None)
        self.cls_enc = nn.Sequential(*list(r.children())[:-2])
        self.cam_layer = self.cls_enc[-1][-2]
        self.cls = nn.Linear(512, n_cls)

    def train(self, mode=True):
        super().train(mode)
        self.seg_net.eval()                                         # frozen: no BatchNorm updates either
        return self

    @staticmethod
    def morph(m, r):
        return F.max_pool2d(m, 2 * r + 1, 1, r) if r > 0 else m

    def classify(self, x, mask):
        f = self.cls_enc(x * mask)
        w = F.adaptive_avg_pool2d(mask, f.shape[-2:])
        v = (f.float() * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-6)
        return self.cls(v.to(self.cls.weight.dtype))

    def forward(self, x, true_mask=None):
        with torch.no_grad():
            seg = self.seg_net(x)[1]
        mask = (seg > 0).float()
        if self.training:
            if true_mask is not None and random.random() < P_TRUE_MASK:
                mask = true_mask
            mask = self.morph(mask, MASK_DILATE + random.randint(-MASK_JITTER, MASK_JITTER))
        else:
            mask = self.morph(mask, MASK_DILATE)
        return self.classify(x, mask), seg


def build(name):
    if name == "hybrid":
        return DualHead(N_CLS, transformer=True, skips=HYBRID_SKIPS)
    if name == "hybrid_sap":
        return DualHead(N_CLS, transformer=True, skips=HYBRID_SKIPS, sap=True)
    if name == "cnn_only":
        return DualHead(N_CLS)
    if name == "unet_r34":
        return DualHead(N_CLS, skips=True)
    if name == "att_unet":
        return DualHead(N_CLS, skips=True, att=True)
    if name == "trans_only":
        return TransOnly(N_CLS)
    if name == "vit_pre":
        return ViTPre(N_CLS)
    if name == "hybrid_early":
        return EarlyMask(N_CLS)
    raise ValueError(name)


MODEL_LABELS = {"cnn_only": "CNN-only", "unet_r34": "U-Net (ResNet-34)", "att_unet": "Attention U-Net (ResNet-34)",
                "trans_only": "Transformer-only", "vit_pre": "Pretrained ViT-B/16",
                "hybrid": "Hybrid CNN-Transformer", "hybrid_sap": "Hybrid CNN-Transformer + SAP",
                "hybrid_early": "Hybrid CNN-Transformer, early lesion masking"}

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
    early = getattr(model, "early", False)
    opt = torch.optim.Adam([prm for prm in model.parameters() if prm.requires_grad], lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.9, patience=5)
    scaler = torch.amp.GradScaler(DEVICE, enabled=AMP) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=AMP)
    best, wait, hist = -np.inf, 0, []
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for x, m, y, _ in dl_tr:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE, enabled=AMP):
                logits, seg = model(x, m) if early else model(x)
            loss = total_loss(logits.float(), seg.float(), y, m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(x)
        model.eval()
        vl, vce, dices, ious, hit = 0.0, 0.0, [], [], 0
        with torch.no_grad():
            for x, m, y, _ in dl_va:
                x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                logits, seg = model(x)
                vl += total_loss(logits, seg, y, m).item() * len(x)
                vce += F.cross_entropy(logits.float(), y, reduction="sum").item()
                p = (seg > 0).float()
                inter, a, b = (p * m).sum((1, 2, 3)), p.sum((1, 2, 3)), m.sum((1, 2, 3))
                dices += ((2 * inter + 1e-6) / (a + b + 1e-6)).tolist()
                ious += ((inter + 1e-6) / (a + b - inter + 1e-6)).tolist()
                hit += (logits.argmax(1) == y).sum().item()
        vl, vd, vce = vl / len(va_idx), float(np.mean(dices)), vce / len(va_idx)
        score = -vce if early else vd   # the early model's segmentation is frozen, so select on classification
        sched.step(vl)
        hist.append(dict(epoch=ep + 1, train_loss=tl / (len(dl_tr) * BATCH), val_loss=vl, val_ce=vce, val_dice=vd,
                         val_iou=float(np.mean(ious)), val_acc=hit / len(va_idx), lr=opt.param_groups[0]["lr"]))
        if (ep + 1) % 10 == 0:
            print(f"  {ckpt.stem} epoch {ep + 1}/{epochs}: val Dice {vd:.3f}, val acc {hit / len(va_idx):.2f}", flush=True)
        if score > best:
            best, wait = score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
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
            oracle = model.classify(x, model.morph(m, MASK_DILATE)).argmax(1) if getattr(model, "early", False) else None
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
                cam_ratio=ratio, pred_lesion_only=int(pred_les[k]), pred_bg_only=int(pred_bg[k]),
                pred_true_mask=int(oracle[k]) if oracle is not None else -1))
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
        assert Path(init).exists(), f"{init} is missing: the hybrid runs must finish (and keep their ckpt) first"
        state = torch.load(init, map_location=DEVICE, weights_only=True)
        if getattr(model, "early", False):
            model.seg_net.load_state_dict(state)                     # trained hybrid segments; classifier is new
        else:
            model.load_state_dict(state)
            epochs = SAP_EPOCHS
    t0 = time.time()
    hist = fit(model, tr, va, epochs, ckpt, seed)
    hist.to_csv(RUN_DIR / f"{tag}_history.csv", index=False)
    d = evaluate(model, te_idx)
    d.to_csv(out, index=False)
    print(f"{tag}: {len(hist)} epochs, {(time.time() - t0) / 60:.1f} min, "
          f"acc {100 * (d.y_true == d.y_pred).mean():.1f}, Dice {d.dice.mean():.3f}")
    return d


# %% Run everything: model x seed x fold
all_idx = np.arange(len(meta))
for name in MODELS:
    for seed in SEEDS.get(name, [0]):
        for fold in range(N_FOLDS):
            te_idx, tr_idx = all_idx[meta.fold == fold], all_idx[meta.fold != fold]
            init = CKPT_DIR / f"{BASE}_s{seed}_f{fold}.pt" if (name == PROP and SAP_FROM_BASELINE) or name == "hybrid_early" else None
            train_and_eval(name, seed, tr_idx, te_idx, f"{name}_s{seed}_f{fold}", init)

# %% Collect out-of-fold predictions
res = []
for name in MODELS:
    for seed in SEEDS.get(name, [0]):
        for fold in range(N_FOLDS):
            d = pd.read_csv(RUN_DIR / f"{name}_s{seed}_f{fold}.csv")
            res.append(d.assign(model=name, seed=seed, fold=fold))
res = pd.concat(res, ignore_index=True)
res["correct"] = (res.y_true == res.y_pred).astype(int)
res["area_err"] = (res.area_pred - res.area_gt).abs()
res.to_csv(OUT_DIR / "per_image_results.csv", index=False)

# %% Main table: every model, pooled out-of-fold, bootstrap 95% CI, fold mean +- SD (comments 6, 7, 8, 11)
METRICS = ["Accuracy (%)", "Macro-F1", "Dice (F1)", "IoU", "Pixel precision", "Pixel recall",
           "Lesion-area R2", "Lesion-area MAE (pp)"]


def metric_vec(d):
    return np.array([100 * d.correct.mean(),
                     f1_score(d.y_true, d.y_pred, average="macro", labels=range(N_CLS), zero_division=0),
                     d.dice.mean(), d.iou.mean(), d.pix_prec.mean(), d.pix_rec.mean(),
                     r2_score(d.area_gt, d.area_pred), d.area_err.mean()])


def boot_ci(d, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([metric_vec(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(N_BOOT)])
    return np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


tab = []
for name in MODELS:
    d = res[(res.model == name) & (res.seed == 0)]
    v, (lo, hi) = metric_vec(d), boot_ci(d)
    fm = np.array([metric_vec(g) for _, g in d.groupby("fold")])
    row = {"Model": MODEL_LABELS[name]}
    for k, mname in enumerate(METRICS):
        f = ".1f" if "%" in mname else ".2f" if "pp" in mname else ".3f"
        row[mname] = f"{v[k]:{f}} [{lo[k]:{f}}, {hi[k]:{f}}]"
        row[mname + " fold mean+-SD"] = f"{fm[:, k].mean():{f}} +- {fm[:, k].std(ddof=1):{f}}"
    tp, fp, fn = d.tp.sum(), d.fp.sum(), d.fn.sum()     # pooled over all pixels (large lesions dominate)
    row["Dice, pooled pixels"] = f"{2 * tp / (2 * tp + fp + fn):.3f}"
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(OUT_DIR / "table_models.csv", index=False)
print(tab[["Model"] + METRICS].to_string(index=False))

# %% Per-class results of the proposed model (the two small classes get honest intervals here)
d = res[(res.model == PROP) & (res.seed == 0)]
pr, rc, f1, sup = precision_recall_fscore_support(d.y_true, d.y_pred, labels=range(N_CLS), zero_division=0)
pc = pd.DataFrame(dict(Class=[CLASS_NAMES[c] for c in classes], n=sup, Precision=pr, Recall=rc, F1=f1))
pc["Recall 95% CI (Wilson)"] = [
    "[{:.2f}, {:.2f}]".format(*stats.binomtest(int(round(r * n)), int(n)).proportion_ci(method="wilson")) if n else ""
    for r, n in zip(rc, sup)]
seg_pc = d.groupby("y_true")[["dice", "iou", "pix_prec", "pix_rec"]].mean().reindex(range(N_CLS))
pc = pd.concat([pc, seg_pc.reset_index(drop=True)], axis=1)
pc.to_csv(OUT_DIR / "table_per_class.csv", index=False)
print(pc.round(3).to_string(index=False))

# %% SAP vs baseline: paired tests on the same held-out images (comments 5 and 11)
def wilcoxon_p(a, b):
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[~np.isnan(diff)]
    return np.nan if len(diff) == 0 or np.all(diff == 0) else stats.wilcoxon(diff).pvalue


def mcnemar_p(a, b):
    """Exact McNemar test on paired 0/1 outcomes."""
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    return (1.0 if n01 + n10 == 0 else stats.binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10


tests = []
for seed in sorted(set(SEEDS.get(BASE, [0])) & set(SEEDS.get(PROP, [0]))):
    a = res[(res.model == BASE) & (res.seed == seed)].set_index("id")
    b = res[(res.model == PROP) & (res.seed == seed)].set_index("id").loc[a.index]
    p, n01, n10 = mcnemar_p(a.correct.values, b.correct.values)
    tests.append(dict(seed=seed, quantity="Classification accuracy (%)", baseline=100 * a.correct.mean(),
                      sap=100 * b.correct.mean(), test="McNemar exact", p=p,
                      note=f"only SAP correct {n01}, only baseline correct {n10}"))
    for col, label in [("dice", "Dice"), ("iou", "IoU"), ("area_err", "Lesion-area abs. error (pp)"),
                       ("cam_ratio", "Grad-CAM mass inside lesion mask")]:
        tests.append(dict(seed=seed, quantity=label, baseline=a[col].mean(), sap=b[col].mean(),
                          test="Wilcoxon signed-rank", p=wilcoxon_p(b[col], a[col]), note=""))
    tests.append(dict(seed=seed, quantity="Lesion-area R2", baseline=r2_score(a.area_gt, a.area_pred),
                      sap=r2_score(b.area_gt, b.area_pred), test="", p=np.nan, note="see bootstrap CI in table_models"))
    # occlusion: a model that reads the lesion should fail without it and survive without the background
    for col, label in [("pred_lesion_only", "Accuracy, background removed (%)"),
                       ("pred_bg_only", "Accuracy, lesion removed (%)")]:
        ca, cb = (a[col] == a.y_true).astype(int).values, (b[col] == b.y_true).astype(int).values
        p, n01, n10 = mcnemar_p(ca, cb)
        tests.append(dict(seed=seed, quantity=label, baseline=100 * ca.mean(), sap=100 * cb.mean(),
                          test="McNemar exact", p=p, note=f"only SAP correct {n01}, only baseline correct {n10}"))
    tests.append(dict(seed=seed, quantity="Mean lesion area fraction = chance level for the Grad-CAM ratio",
                      baseline=a.area_gt.mean() / 100, sap=b.area_gt.mean() / 100, test="", p=np.nan, note=""))
tests = pd.DataFrame(tests)
tests.to_csv(OUT_DIR / "table_sap_vs_baseline.csv", index=False)
print(tests.round(4).to_string(index=False))

# mean +- SD over seeds for the pair
seed_tab = []
for name in (BASE, PROP):
    v = np.array([metric_vec(res[(res.model == name) & (res.seed == s)]) for s in SEEDS.get(name, [0])])
    seed_tab.append({"Model": MODEL_LABELS[name], "seeds": len(v),
                     **{m: f"{v[:, k].mean():.3f} +- {v[:, k].std(ddof=1) if len(v) > 1 else 0:.3f}"
                        for k, m in enumerate(METRICS)}})
pd.DataFrame(seed_tab).to_csv(OUT_DIR / "table_seeds.csv", index=False)
print(pd.DataFrame(seed_tab).to_string(index=False))

# %% Early lesion masking: does classification hold up when the classifier can only see the lesion?
EARLY = "hybrid_early"
if EARLY in MODELS:
    rows_e = []
    for name in [BASE, PROP, EARLY, "unet_r34"]:
        for seed in SEEDS.get(name, [0]) if name in MODELS else []:
            d = res[(res.model == name) & (res.seed == seed)]
            true_mask = d.pred_true_mask if "pred_true_mask" in d else pd.Series(-1, index=d.index)
            rows_e.append({"Model": MODEL_LABELS[name], "seed": seed,
                           "Accuracy (%)": 100 * d.correct.mean(),
                           "Macro-F1": f1_score(d.y_true, d.y_pred, average="macro", labels=range(N_CLS), zero_division=0),
                           "Accuracy, annotated mask (%)": 100 * (true_mask == d.y_true).mean() if true_mask.ge(0).all() else np.nan,
                           "Accuracy, background removed (%)": 100 * (d.pred_lesion_only == d.y_true).mean(),
                           "Accuracy, lesion removed (%)": 100 * (d.pred_bg_only == d.y_true).mean(),
                           "Grad-CAM inside lesion": d.cam_ratio.mean()})
    early_tab = pd.DataFrame(rows_e)
    early_sum = early_tab.drop(columns="seed").groupby("Model", sort=False).agg(["mean", "std"]).round(3)
    early_sum.to_csv(OUT_DIR / "table_early_masking.csv")
    print(early_sum.to_string())
    et = []
    for other in (BASE, PROP):
        for seed in sorted(set(SEEDS.get(other, [0])) & set(SEEDS.get(EARLY, [0]))):
            a = res[(res.model == other) & (res.seed == seed)].set_index("id")
            b = res[(res.model == EARLY) & (res.seed == seed)].set_index("id").loc[a.index]
            p, n01, n10 = mcnemar_p(a.correct.values, b.correct.values)
            et.append(dict(comparison=f"early masking vs {MODEL_LABELS[other]}", seed=seed, acc_other=100 * a.correct.mean(),
                           acc_early=100 * b.correct.mean(), mcnemar_p=p, only_early_correct=n01, only_other_correct=n10))
    pd.DataFrame(et).to_csv(OUT_DIR / "table_early_vs_hybrid.csv", index=False)
    print(pd.DataFrame(et).round(4).to_string(index=False))

# %% Uncertainty (comment 9): does it flag the errors?
unc = []
for name in (BASE, PROP):
    d = res[(res.model == name) & (res.seed == 0)]
    wrong = 1 - d.correct
    unc.append({
        "Model": MODEL_LABELS[name],
        "Entropy, correct": d.tta_entropy[wrong == 0].mean(),
        "Entropy, misclassified": d.tta_entropy[wrong == 1].mean(),
        "Mann-Whitney p": stats.mannwhitneyu(d.tta_entropy[wrong == 1], d.tta_entropy[wrong == 0],
                                             alternative="greater").pvalue if 0 < wrong.sum() < len(d) else np.nan,
        "AUROC entropy -> error": roc_auc_score(wrong, d.tta_entropy) if 0 < wrong.sum() < len(d) else np.nan,
        "Lesion-area SD (pp), mean": d.tta_area_sd.mean(),
        "Lesion-area SD (pp), median": d.tta_area_sd.median(),
        "Spearman area SD vs abs. area error": stats.spearmanr(d.tta_area_sd, d.area_err)[0],
        "Accuracy with TTA (%)": 100 * (d.tta_pred == d.y_true).mean(),
    })
unc = pd.DataFrame(unc)
unc.to_csv(OUT_DIR / "table_uncertainty.csv", index=False)
print(unc.round(4).T.to_string())

# %% Complexity (comment 10): parameters, FLOPs, latency at batch size 1
def latency_ms(model, dev, n):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=dev)
    model = model.to(dev).eval()
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


def attention_macs(model, x):
    """Multiply-accumulates thop does not count: attention products, and all of nn.MultiheadAttention (ViT)."""
    extra = [0.0]

    def tblock(mod, inp, out):
        n, c = inp[0].shape[1:]
        extra[0] += 2 * n * n * c

    def mha(mod, inp, out):
        n, e = inp[0].shape[1], mod.embed_dim
        extra[0] += 4 * n * e * e + 2 * n * n * e

    hooks = [m.register_forward_hook(tblock if isinstance(m, TBlock) else mha)
             for m in model.modules() if isinstance(m, (TBlock, nn.MultiheadAttention))]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return extra[0]


cx = []
for name in MODELS:
    model = build(name).eval()
    row = {"Model": MODEL_LABELS[name], "Params (M)": sum(p.numel() for p in model.parameters()) / 1e6}
    try:
        from thop import profile
        x1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        row["GMACs"] = (profile(build(name).eval(), inputs=(x1,), verbose=False)[0] + attention_macs(model, x1)) / 1e9
    except Exception as e:
        row["GMACs"] = np.nan
        print("MAC count skipped (pip install thop):", e)
    row["CPU ms/image"] = latency_ms(model, "cpu", 20)
    if DEVICE == "cuda":
        row["GPU ms/image"] = latency_ms(model, "cuda", 200)
        row["GPU images/s"] = 1000 / row["GPU ms/image"]
    cx.append(row)
cx = pd.DataFrame(cx)
cx.to_csv(OUT_DIR / "table_complexity.csv", index=False)
print(cx.round(2).to_string(index=False))
print("CPU threads:", torch.get_num_threads(), "| GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none")

# %% Optional cross-source test (comment 8): train on public images, test on in-house images
# Needs source.csv with columns id, source. Only classes present in both sources can be scored.
if SOURCE_CSV.exists():
    src = meta.merge(pd.read_csv(SOURCE_CSV), on="id", how="left").source.str.lower()
    assert src.notna().all(), "source.csv does not cover every image"
    pub, inh = all_idx[(src == "public").values], all_idx[(src == "inhouse").values]
    shared = sorted(set(meta.y[pub]) & set(meta.y[inh]))
    te = inh[np.isin(meta.y[inh], shared) & ~np.isin(meta.group[inh], meta.group[pub])]
    xs = []
    for name in (BASE, PROP):
        init = CKPT_DIR / f"xsrc_{BASE}.pt" if name == PROP and SAP_FROM_BASELINE else None
        d = train_and_eval(name, 0, pub, te, f"xsrc_{name}", init)
        xs.append({"Model": MODEL_LABELS[name], "n test": len(d), "Classes": ", ".join(CLASS_NAMES[classes[c]] for c in shared),
                   "Accuracy (%)": 100 * (d.y_true == d.y_pred).mean(),
                   "Macro-F1": f1_score(d.y_true, d.y_pred, average="macro", labels=shared, zero_division=0),
                   "Dice": d.dice.mean(), "IoU": d.iou.mean(), "Lesion-area R2": r2_score(d.area_gt, d.area_pred)})
    xs = pd.DataFrame(xs)
    xs.to_csv(OUT_DIR / "table_cross_source.csv", index=False)
    print(xs.round(3).to_string(index=False))
else:
    print("no source.csv -> cross-source test skipped")

# %% Figures (bold text throughout, comment 4)
plt.rcParams.update({"font.size": 11, "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
C_BASE, C_SAP = "#7a7a7a", "#1f5fa8"
short = [CLASS_NAMES[c] for c in classes]

# confusion matrix, proposed model, pooled out-of-fold
d = res[(res.model == PROP) & (res.seed == 0)]
cm = confusion_matrix(d.y_true, d.y_pred, labels=range(N_CLS))
cmn = cm / cm.sum(1, keepdims=True).clip(1)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(N_CLS):
    for k in range(N_CLS):
        if cm[i, k]:
            ax.text(k, i, f"{cmn[i, k]:.2f}\n({cm[i, k]})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, k] > 0.5 else "black")
ax.set_xticks(range(N_CLS), short, rotation=45, ha="right")
ax.set_yticks(range(N_CLS), short)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.spines[["top", "right"]].set_visible(True)
fig.colorbar(im, ax=ax, fraction=0.046, label="Recall")
fig.savefig(FIG_DIR / "confusion_matrix_sap.png")
plt.close(fig)

# lesion area: predicted vs annotated, baseline and SAP
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
for ax, name, col in zip(axes, (BASE, PROP), (C_BASE, C_SAP)):
    d = res[(res.model == name) & (res.seed == 0)]
    lim = max(d.area_gt.max(), d.area_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
    ax.scatter(d.area_gt, d.area_pred, s=14, color=col, alpha=0.6, linewidths=0)
    ax.set_title(f"{MODEL_LABELS[name]}\nR\u00b2 = {r2_score(d.area_gt, d.area_pred):.3f}, "
                 f"MAE = {d.area_err.mean():.2f} pp", fontsize=10)
    ax.set_xlabel("Annotated lesion area (% of image)")
    ax.set_aspect("equal")
axes[0].set_ylabel("Predicted lesion area (% of image)")
fig.savefig(FIG_DIR / "lesion_area_scatter.png")
plt.close(fig)

# Grad-CAM mass inside the lesion mask, baseline vs SAP, same images
a = res[(res.model == BASE) & (res.seed == 0)].set_index("id")
b = res[(res.model == PROP) & (res.seed == 0)].set_index("id").loc[a.index]
ok = a.cam_ratio.notna() & b.cam_ratio.notna()
fig, ax = plt.subplots(figsize=(4.6, 4.8))
bp = ax.boxplot([a.cam_ratio[ok], b.cam_ratio[ok]], widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", lw=1.5))
for patch, col in zip(bp["boxes"], (C_BASE, C_SAP)):
    patch.set(facecolor=col, alpha=0.55)
ax.axhline(a.area_gt[ok].mean() / 100, color="black", lw=1, ls=":")
ax.text(2.45, a.area_gt[ok].mean() / 100, " chance", va="center", fontsize=9)
ax.set_xticks([1, 2], ["Baseline", "SAP"])
ax.set_ylabel("Grad-CAM mass inside lesion mask")
ax.set_ylim(0, 1)
ax.set_title(f"Wilcoxon p = {wilcoxon_p(b.cam_ratio[ok], a.cam_ratio[ok]):.2g}", fontsize=10)
fig.savefig(FIG_DIR / "gradcam_lesion_ratio.png")
plt.close(fig)

# training curves of the hybrid model, fold 0 (replacement for Fig. 5)
hist_file = RUN_DIR / f"{BASE}_s0_f0_history.csv"
if hist_file.exists():
    h = pd.read_csv(hist_file)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(h.epoch, h.train_loss, color=C_BASE, lw=2, label="Training")
    axes[0].plot(h.epoch, h.val_loss, color=C_SAP, lw=2, label="Validation")
    axes[0].set_ylabel("Loss")
    for col, lab, colr in [("val_dice", "Dice", C_SAP), ("val_iou", "IoU", "#c8102e"), ("val_acc", "Accuracy", C_BASE)]:
        axes[1].plot(h.epoch, h[col], lw=2, color=colr, label=lab)
    axes[1].set_ylabel("Validation metric")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "training_curves_fold0.png")
    plt.close(fig)

print("done ->", OUT_DIR)
