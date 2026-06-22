# Cross-Dataset Reliability of Thermal Fusion in UAV Wildfire Detection — code & results release

Accompanies the manuscript *"Cross-Dataset Reliability of Thermal Fusion in UAV Wildfire Detection:
a Negative-Transfer-Safe Admission Baseline and an Annotation-Granularity Benchmark Finding"*
(Computers and Electronics in Agriculture, under review).

This release includes the **per-image model predictions**, so the GateA rule, the evaluation, and
the paired bootstrap (n_boot=200) can be re-run **without retraining or a GPU**. The per-image
ground-truth labels are derived from third-party datasets and are **not redistributed** here; they
are regenerated from the datasets with the provided evaluation scripts (see `gt_labels/README.md`).

## Layout

```
scripts/      authoritative code
  run_f6_evidence_package_20260613.py   GateA core: apply_gateA_add_only() (tau_overlap=0.7, tau_dual=0.05),
                                        bootstrap_ap50 / compute_delta_bootstrap (n_boot=200)
  run_f7a_reliability_gate_*.py         F7 learned gating-route ablation
  run_f5_*/ run_formal_*/ run_g1a_*     LOCO / formal-run drivers
  export_fig5_realframes_*.py           Fig 5 real-frame overlay export
  build_f6_evidence_tables.py           regenerate the manuscript tables from results/ CSVs
  render_*.py                           Fig 2 / Fig 3 / Fig 5 matplotlib renderers (fonttype 42)
splits/       LOCO per-fold image-id lists + per-fold SHA-256
  loco_fold1/ loco_fold2_nirfree/       the two folds evaluated in the paper (train.txt, val.txt, sha256.txt)
  fold0_pilot_only_paired/              fold0 RGB-only pilot split (yaml + id lists); see fold0 note below
predictions/  per-image prediction JSONL (fold1/2 x seeds x {rgb,dual} + formal seed2024)
gt_labels/    NOT redistributed (third-party dataset labels); README explains how to regenerate from datasets
runs/         raw run output dirs (f6 evidence package, formal p0/p1, f5 g1a) incl. FOLD0_GATEA_BLOCKED.md
results/      manuscript-ready table CSVs (the numbers in Tables 1-6 / S1-S3)
protocol/     locked study protocol, metric definition, baseline & experiment registries (G1)
envs/         conda environment exports (topic2_main.yml, yolov11_rgbt.yml)
PROVENANCE.md           git commits of external baseline repos + collection provenance
split_checksums.txt     SHA-256 of the split files
server_part_manifest.txt SHA-256 manifest of the server-collected part
SHA256SUMS.txt          SHA-256 of every file in this bundle
CITATION.cff  LICENSE  MANIFEST.md
```

## Reproducing the numbers

```
# evidence tables + GateA + bootstrap CIs from per-image predictions (+ GT regenerated from datasets)
python scripts/run_f6_evidence_package_20260613.py        # GateA add-only rule + n_boot=200 bootstrap
python scripts/build_f6_evidence_tables.py                # manuscript tables from results/ CSVs
python scripts/render_evidence_figures.py                 # Fig 2 / Fig 3 from CSV
python scripts/render_fig5_qualitative.py                 # Fig 5 from real-frame overlays
```

Seeds: 42, 1337, 2024. GateA thresholds locked at `tau_overlap=0.7`, `tau_dual=0.05` (chosen on a
2-seed fold-2 development sensitivity grid; seed 2024 held out of the grid; applied unchanged to
fold 1 — see manuscript Methods + Table S1).

### fold0 note (honest scope)
The paper evaluates GateA on fold 1 (retention) and fold 2 (cross-dataset stress). Fold 0 has **no
dual checkpoint** and GateA is **not** evaluated on it (see `runs/.../FOLD0_GATEA_BLOCKED.md` and the
manuscript's explicit "no fold-0 retention claim"). The fold0 RGB-only **pilot** split is included
under `splits/fold0_pilot_only_paired/` for completeness; the leakage check (train ∩ val = ∅) and
SHA-256 that back the reported results are for `loco_fold1/` and `loco_fold2_nirfree/`.

## Datasets (third-party — NOT redistributed)

Obtained from the original sources under their own terms; this package does not redistribute any
dataset or its labels. Only the per-image **model prediction** JSONL (the authors' own outputs) is included here;
the ground-truth labels are regenerated from the datasets with the evaluation scripts, after which
the evaluation and bootstrap re-run end to end.

| Dataset | Source DOI | Note |
|---|---|---|
| FireMan-UAV-RGBT | data 10.5281/zenodo.13732947 ; paper 10.1109/ETFA61755.2024.10710657 | Kularatne et al., 2024 IEEE ETFA |
| RGBT-3M | 10.3390/rs17152593 | Zhang, Rui, Song, *Remote Sensing* 2025 |
| JAG2023 | 10.1016/j.jag.2023.103554 | Rui et al., *IJAEOG* 2023; RGB + near-infrared (not long-wave thermal) |

## Integrity & provenance

Verify with `shasum -a 256 -c SHA256SUMS.txt`. External baseline code provenance (the
`/mnt/topic2_workspace` collection host is not a git repo): YOLOv11-RGBT commit `9cc2e208a3`,
M2D-LIF commit `210c7ca22b` (see `PROVENANCE.md`). No retraining or number regeneration was done
during collection. Third-party ground-truth labels are not redistributed (see `gt_labels/README.md`).

## Citation

See `CITATION.cff`. The Zenodo deposit DOI and the final paper DOI will be added once minted /
accepted.
