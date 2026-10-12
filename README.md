# Dual-Head CNN–Transformer and Segmentation-Aware Pooling for Potato Tuber Disease Phenotyping

**Research code and archived results for manuscript PBT-D-26-00881, *Postharvest Biology and Technology*.**

This repository documents an eight-class potato tuber image classification and lesion segmentation study. The primary experiment is a stratified group-aware **five-fold cross-validation** on **421 annotated images**. Seven baseline/model variants and a separate early-lesion-masking experiment are evaluated; the Hybrid and SAP variants are repeated over three seeds. The code also records quantitative Grad-CAM localization, lesion/background occlusion, test-time augmentation sensitivity, paired tests, and inference complexity estimates.

## Key interpretation

SAP is a **mask-normalized, segmentation-guided feature-pooling** operation. It improves Grad-CAM lesion localization for the Hybrid in the archived experiments, **but does not consistently improve classification or segmentation performance**. High accuracy when annotated lesions are removed demonstrates the importance of non-lesion information. Results must not be presented as proof that SAP removes background dependence. Lesion-area percentages refer to **image area**, not whole tuber surface area.

## Repository layout

- `scripts/run_v4_fivefold_and_early_masking.py`: **exact user-supplied v4 script** corresponding to the primary archived results (not retroactively edited).
- `results/v4_primary/`: archived v4 configuration, five-fold assignments, predictions, model-level and per-class tables, significance tests, figure files and run histories.
- `scripts/run_unet_sap_separate_control.py`: source of the additional U-Net + SAP fine-tuning evaluation.
- `results/unet_sap_separate_control/`: separately archived control predictions, tables and histories.
- `docs/REPRODUCIBILITY.md`: experiment provenance, known limitations, and instructions to verify any rerun.
- `docs/DATASET.md`: image/annotation structure and exclusions.

**Do not pool the primary v4 results with the U-Net + SAP control as if they used identical folds.** The standalone control was evaluated using a different earlier fold configuration. Treat it as an exploratory control within its own matched comparison.

## Setup

The experimental environment reported Python 3.8, PyTorch 2.1.2+cu118 and CUDA. A GPU is strongly recommended. Install a compatible PyTorch/torchvision build for your device, then install dependencies from `requirements.txt`. The primary script is a flat script designed for **Spyder** (no `main()` or command-line arguments).

## Primary analysis

1. Organize image/annotation pairs as `~/potato_diseaseN/<class>/<image>.jpg` and corresponding AnyLabeling `.json` files; see `docs/DATASET.md`.
2. Open `scripts/run_v4_fivefold_and_early_masking.py` in Spyder.
3. Review the constants at the top, especially `DATA_DIR`, `OUT_DIR`, `EXCLUDE`, `N_FOLDS`, `SPLIT_SEED`, `SAME_TUBER_S`, and image/group rules.
4. Run top-to-bottom. **Use a new output directory for any changed data or settings.** CSV results from completed runs are reused, but interrupted epochs within a run are not resumed.
5. Compare the newly generated `folds.csv`, `config.json`, and `table_models.csv` against the archive before claiming a reproduction.

The repository does **not** contain the original 421 images, JSON annotations, or `.pt` model checkpoints. Providing these requires appropriate sharing rights, privacy/source assessment, and size/hosting choices. Consequently the repository is an **analysis-code and result archive**, not yet a fully executable reproduction package without the dataset and checkpoints. Some pretrained model weights may be downloaded by torchvision on a first run.

## Source-specific and statistical caveats

- Original images were split before augmentation. Group heuristics reduce, but do not prove elimination of, physical-tuber/source dependence.
- Four visually confirmed repeated photographs were excluded from the original 425-image annotated collection; see `docs/DATASET.md`.
- The primary pipeline uses an optional source CSV for a cross-source evaluation; absent confirmed provenance it is **not an independent external validation**.
- Grad-CAM lesion mass is an attribution-localization measurement, not a causal faithfulness guarantee. The separate early-masking model is a frozen-segmenter + ResNet34 classifier, **not** simply SAP moved to an earlier Transformer layer.
- Uncertainty from random test-time augmentations is not a calibrated predictive interval. MAC estimates may be missing if a profiling package was unavailable.
- The saved data correspond to a **re-implementation** documented by the script and revised manuscript, not a proven byte-for-byte recovery of unavailable original training code.

## Citation and publication status

Manuscript **PBT-D-26-00881** is under revision. No DOI, publication acceptance, public code release, or open data access should be inferred from this archive. A final publication-specific citation, release tag/commit, license, and data-access procedure can be added once confirmed.

## Authors and correspondence

Author order, institutional affiliations, manuscript correspondence (Wenzhong Wang), and journal submission correspondence (handled by Jiangsan Zhao) should follow the final journal-approved manuscript. Submission-system contact does not by itself replace the manuscript corresponding-author designation.