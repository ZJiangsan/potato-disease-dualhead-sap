# Dataset and evaluation scope

The revised experiments concern eight potato tuber disease classes and 421 annotated images. Four visually confirmed redundant photographs were excluded from the original 425-image collection: `Common_Scab/29`, `Dry_Rot/11`, `Pink_Rot/29`, and `Silver_Scurf/IMG_20250114_090213`.

Labels originate from per-image polygon/rectangle annotation JSON files. Lesion-area percentage is computed relative to the **image area**, not independently segmented tuber-surface area. Source/physical-tuber independence cannot be established from near-duplicate hashes alone. See the associated manuscript for constraints on reuse and interpretation.

Image and annotation files are not distributed in this repository. Reproduction of training requires lawful access to the originals, the exact saved fold assignments, and compatible preprocessing software.
