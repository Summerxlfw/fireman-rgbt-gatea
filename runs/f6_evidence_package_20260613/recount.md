# F6 Evidence Package Recount
# 日期: 2026-06-13
# 总耗时: 4294.1s (71.6min)

## Subtask Status

| Subtask | Status | Notes |
|---------|--------|-------|
| Artifact Inventory | DONE | 34 artifacts cataloged |
| Fold0 GateA Retention | BLOCKED | No fold0 dual checkpoint exists |
| Bootstrap CI | DONE | 9 targets computed |
| Qualitative Cases | DONE | selected cases cataloged |

## Fold0 GateA: BLOCKED

原因: 服务器无 `phase_f2_dual_fold0_*` checkpoint。F-1 pilot dual 训练在 pilot split，不可用于 formal fold0 GateA。
详见: `FOLD0_GATEA_BLOCKED.md`

## Bootstrap CI Results

### RGB_vs_Dual Fold2

| Seed | Arm A | Arm B | Point Δ | Boot Mean Δ | CI 2.5% | CI 97.5% | n_boot |
|------|-------|-------|---------|-------------|---------|----------|--------|
| 42 | RGB | Dual | +0.0880 | +0.0879 | +0.0831 | +0.0938 | 200 |
| 1337 | RGB | Dual | +0.0530 | +0.0534 | +0.0484 | +0.0588 | 200 |
| 2024 | RGB | Dual | +0.0952 | +0.0949 | +0.0908 | +0.0988 | 200 |

### RGB_vs_GateA Fold2

| Seed | Arm A | Arm B | Point Δ | Boot Mean Δ | CI 2.5% | CI 97.5% | n_boot |
|------|-------|-------|---------|-------------|---------|----------|--------|
| 42 | GateA | RGB | +0.0098 | +0.0099 | +0.0080 | +0.0117 | 200 |
| 1337 | GateA | RGB | +0.0009 | +0.0008 | +0.0000 | +0.0016 | 200 |
| 2024 | GateA | RGB | -0.0001 | -0.0001 | -0.0003 | -0.0000 | 200 |

### RGB_vs_GateA Fold1

| Seed | Arm A | Arm B | Point Δ | Boot Mean Δ | CI 2.5% | CI 97.5% | n_boot |
|------|-------|-------|---------|-------------|---------|----------|--------|
| 42 | GateA | RGB | +0.1080 | +0.1072 | +0.0964 | +0.1183 | 200 |
| 1337 | GateA | RGB | +0.0543 | +0.0540 | +0.0459 | +0.0635 | 200 |
| 2024 | GateA | RGB | +0.1270 | +0.1271 | +0.1178 | +0.1369 | 200 |

## Qualitative Cases

Primary seed: 42
Total selected: 11

- rgb_miss_rescued_by_gateA: 3/3
- dual_fp_rejected_by_gateA: 3/3
- harmful_thermal_collapse: 2/2
- thermal_helpful_retained: 3/3

## Sanity Check

Point estimates vs recount.md: ALL PASS

## Source Paths

Prediction JSONL:
- Fold2 seed=42: /mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607/predictions_fold2_seed42_rgb.jsonl
- Fold1 seed=42: /mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607/predictions_fold1_seed42_rgb.jsonl
- Fold2 seed=1337: /mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607/predictions_fold2_seed1337_rgb.jsonl
- Fold1 seed=1337: /mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607/predictions_fold1_seed1337_rgb.jsonl
- Fold2 seed=2024: /mnt/topic2_workspace/runs/formal_p0_p1_targeted_20260612/g1a_predictions/predictions_fold2_seed2024_rgb.jsonl
- Fold1 seed=2024: /mnt/topic2_workspace/runs/formal_p0_p1_targeted_20260612/g1a_predictions/predictions_fold1_seed2024_rgb.jsonl

GT labels:
- Fold1: /mnt/topic2_datasets/fire_loco_fold1/labels/val
- Fold2: /mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val

Bootstrap parameters:
- n_boot=200 (handoff: '200 if slow and label as such')
- method: paired image-level resampling with replacement
- RNG seed=42

GateA locked params:
- tau_overlap=0.7, tau_dual=0.05, mode=add-only
- dual prefilter conf >= 0.01

## Completion

- `statistical_significance_audit.md` 建议: `revise` → `conditional_pass`
  (fold1/fold2 bootstrap CI 可用，fold0 retention 永久 blocked)
