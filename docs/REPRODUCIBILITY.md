# Reproducibility notes

- Primary study: `scripts/run_v4_fivefold_and_early_masking.py` (v4 analysis) with stratified group five-fold cross-validation, original-image splitting, and training-only augmentation.
- SAP is evaluated by fine-tuning the matching baseline network. The early-lesion-masking experiment instead uses a **frozen segmenter** and a separate classifier; it is not an earlier insertion of the same SAP operation.
- The U-Net + SAP control was trained and assessed separately. Its fold configuration must not be silently merged with the primary v4 predictions.
- Original images, annotations, checkpoints, and some GPU environment details are not included here.
- Retain and compare `folds.csv`, `config.json`, source IDs and code SHA when interpreting prediction tables. Different folds or exclusions imply different experiments.
- The class-wise severity proxy is percentage of lesion-covered **image pixels**. Do not interpret it as diseased tuber-surface fraction without tuber ROI masks.
- Saved predictions can support the manuscript's analyses, but they do not by themselves establish external-source validation.
