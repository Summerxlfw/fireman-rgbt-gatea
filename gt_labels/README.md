# Ground-truth labels — not redistributed here

The per-image ground-truth annotation files used for evaluation are derived from the
third-party datasets (RGBT-3M for fold 1, FireMan-UAV-RGBT for fold 2) and are **not
redistributed** in this repository, because their distribution terms are set by the original
dataset providers (RGBT-3M's distribution license is unspecified; obtain it from the authors).

To reproduce the evaluation and the bootstrap confidence intervals:

1. Obtain the datasets from their original sources (see ../README.md → Datasets).
2. Use the evaluation scripts in ../scripts/ to regenerate the per-image GT label files in the
   expected layout (class-id + box, unified ids 0=smoke, 1=fire, 2=person).
3. The per-image **model predictions** are provided under ../predictions/, so once the GT is in
   place the paired bootstrap (n_boot=200) reproduces the reported intervals without retraining.

The split image-id lists (../splits/) identify exactly which frames belong to each fold.
