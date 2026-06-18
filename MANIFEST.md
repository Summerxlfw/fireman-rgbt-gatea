# MANIFEST — code & results release (complete)

Assembled 2026-06-18. This bundle is **complete**: the earlier "pending server artifacts" are now
included (collected from the GPU host and merged). Per-file SHA-256 for the entire bundle is in
`SHA256SUMS.txt`; verify with `shasum -a 256 -c SHA256SUMS.txt`.

## Provenance

- **Local part** (assembled on macOS from the project): `protocol/`, `results/` (manuscript-ready
  table CSVs), `scripts/build_f6_evidence_tables.py`, `scripts/render_*.py`, `README.md`, `LICENSE`,
  `CITATION.cff`.
- **Server part** (collected on `summer-3080:/mnt/topic2_workspace/`, no retraining / no number
  regeneration): `scripts/run_*.py` + `export_fig5_*`, `splits/`, `predictions/`, `runs/`, `envs/`,
  `PROVENANCE.md`, `split_checksums.txt`, `server_part_manifest.txt`. (Third-party `gt_labels/` were
  collected on the server but are **excluded from this public release**; see `gt_labels/README.md`.)
  - Server tarball verified at handback: `code_release_server_part_20260618.tar.gz`
    SHA-256 `43ab4ad18e6d437681c036aa2a5b464e11fa1a0f995a3b3309a00ee23ff348c7` (MATCH);
    `server_part_manifest.txt` SHA-256
    `eeca3a575432e406e205bb1d48d40326aeddbf3b06f740293eee55267bcb1800` (MATCH).
  - The server archive contained 0 symlinks. Third-party GT label files were collected for the
    authors' records but are excluded from this public release (license-gated; regenerable from
    the datasets via the eval scripts).

## Authoritative code

`scripts/run_f6_evidence_package_20260613.py` is the authoritative GateA implementation:
`apply_gateA_add_only()` (skip if conf < tau_dual; reject if same-class IoU >= tau_overlap; else
admit) with locked `TAU_OVERLAP=0.7` / `TAU_DUAL=0.05`, plus `bootstrap_ap50` /
`compute_delta_bootstrap` at `N_BOOT=200`.

## Reproducibility scope

- Bootstrap CIs (n_boot=200), GateA results, and evidence tables are reproducible from the
  included `predictions/` once the per-image GT is regenerated from the datasets (eval scripts);
  no retraining, checkpoints, or GPU are needed.
- External baseline code provenance (collection host not a git repo): YOLOv11-RGBT `9cc2e208a3`,
  M2D-LIF `210c7ca22b` (see `PROVENANCE.md`).

## Designed-out (not omissions)

- **fold0 dual checkpoint**: the paper does not evaluate GateA on fold 0 (see
  `runs/.../FOLD0_GATEA_BLOCKED.md` and the manuscript's "no fold-0 retention claim"). The fold0
  RGB-only pilot split is included for completeness; reported results rest on fold1/fold2.
- **per-image matching intermediate table**: not persisted by the original f6 run; the
  `predictions/` plus dataset-regenerated GT provide equivalent re-computation.
- **third-party GT labels**: excluded from the public release (license-gated); see
  `gt_labels/README.md` for regeneration.

## Integrity files

- `SHA256SUMS.txt` — every file in this bundle.
- `split_checksums.txt` + per-fold `splits/*/sha256.txt` — split-file checksums.
- `server_part_manifest.txt` — SHA-256 manifest of the server-collected part (475 KB).
