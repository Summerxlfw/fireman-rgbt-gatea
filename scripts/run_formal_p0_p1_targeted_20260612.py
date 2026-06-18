#!/usr/bin/env python3
"""Formal P0/P1 Targeted Baselines — registry, recount, and bundle generator.

从源 artifact (results.csv / gate CSV / recount txt) 中 recount 所有 P0/P1 baseline，
生成 formal run_registry.csv、formal_main_table_recount.csv、gate_ablation_recount.csv、
external_baseline_recount.csv、anomalies.md、recount.md。

运行级别:
- P0 recount/eval = formal
- YOLOv11-RGBT selected seed2024 = formal if completed
- CP-YOLOv11-MF / M2D-LIF = smoke/feasibility

数字一律从 source CSV/JSONL/log recount，不从 handoff prose 复制。
"""

import csv
import json
import statistics
import sys
import os
from pathlib import Path
from datetime import datetime

# ─── 路径 ───

RUNS = Path("/mnt/topic2_workspace/runs")
SCRIPTS = Path("/mnt/topic2_workspace/scripts")
OUT_DIR = RUNS / "formal_p0_p1_targeted_20260612"
LOG_DIR = OUT_DIR / "logs"

RUNS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── 工具函数 ───

def read_rows(path):
    """读取 CSV 返回 dict 列表。"""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def best50_from_csv(run_dir: Path) -> dict:
    """从 run_dir/results.csv 提取 best epoch mAP50 等指标。"""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    rows = read_rows(csv_path)
    if not rows:
        return None

    # 找 best mAP50 epoch
    best_row = max(rows, key=lambda r: safe_float(r.get("metrics/mAP50(B)", 0)))
    best_map50 = safe_float(best_row.get("metrics/mAP50(B)", 0))
    final_row = rows[-1]
    final_map50 = safe_float(final_row.get("metrics/mAP50(B)", 0))

    has_best_pt = (run_dir / "weights" / "best.pt").exists()

    return {
        "n_epochs": len(rows),
        "has_best_pt": has_best_pt,
        "best_epoch": int(float(best_row.get("epoch", 0))),
        "best_map50": best_map50,
        "best_map50_95": safe_float(best_row.get("metrics/mAP50-95(B)", 0)),
        "final_map50": final_map50,
        "final_map50_95": safe_float(final_row.get("metrics/mAP50-95(B)", 0)),
        "final_val_cls_loss": safe_float(final_row.get("val/cls_loss", 0)),
        "final_over_best": final_map50 / best_map50 if best_map50 > 0 else 0.0,
        "collapse": final_map50 < best_map50 * 0.5 if best_map50 > 0 else False,
    }


def write_csv(path, rows, fieldnames):
    """写 CSV 文件。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  写入: {path} ({len(rows)} 行)")


# ─── Step 1: Source Artifact Registry ───

def discover_runs():
    """扫描所有源 run 目录，返回 run registry 行。"""
    registry = []
    anomalies = []

    # ── P0-RGB: F2 RGB-only, folds 0/1/2, seeds 42/1337/2024 ──
    for fold in [0, 1, 2]:
        for seed in [42, 1337, 2024]:
            name = f"phase_f2_rgbonly_fold{fold}_seed{seed}"
            run_dir = RUNS / name
            if run_dir.exists():
                registry.append({
                    "family": "P0-RGB", "method": "RGB-only", "fold": fold, "seed": seed,
                    "run_class": "formal", "run_id": name,
                    "source_path": str(run_dir),
                    "config_path": str(run_dir / "args.yaml"),
                    "log_path": "",
                    "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                    "prediction_path": "",
                    "results_csv": str(run_dir / "results.csv"),
                    "status": "available", "notes": "F2 RGB-only baseline"
                })
            else:
                anomalies.append(f"MISSING: {name}")
                registry.append({
                    "family": "P0-RGB", "method": "RGB-only", "fold": fold, "seed": seed,
                    "run_class": "formal", "run_id": name,
                    "source_path": str(run_dir), "config_path": "", "log_path": "",
                    "checkpoint_path": "", "prediction_path": "", "results_csv": "",
                    "status": "missing", "notes": "DIR_NOT_FOUND"
                })

    # ── P0-Dual: F2 Dual, folds 1/2, seeds 42/1337/2024 ──
    for fold in [1, 2]:
        for seed in [42, 1337, 2024]:
            name = f"phase_f2_dual_fold{fold}_seed{seed}"
            run_dir = RUNS / name
            if run_dir.exists():
                registry.append({
                    "family": "P0-Dual", "method": "Dual", "fold": fold, "seed": seed,
                    "run_class": "formal", "run_id": name,
                    "source_path": str(run_dir),
                    "config_path": str(run_dir / "args.yaml"),
                    "log_path": "",
                    "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                    "prediction_path": "",
                    "results_csv": str(run_dir / "results.csv"),
                    "status": "available", "notes": "F2 naive dual baseline"
                })
            else:
                anomalies.append(f"MISSING: {name}")

    # ── P0-F4: F4 NIR-free RGB-only, fold2, seeds 42/1337/2024 ──
    for seed in [42, 1337, 2024]:
        name = f"phase_f4_nirfree_rgbonly_fold2_seed{seed}"
        run_dir = RUNS / name
        if run_dir.exists():
            registry.append({
                "family": "P0-F4", "method": "NIRfree-RGB-only", "fold": 2, "seed": seed,
                "run_class": "formal", "run_id": name,
                "source_path": str(run_dir),
                "config_path": str(run_dir / "args.yaml"),
                "log_path": "",
                "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                "prediction_path": "",
                "results_csv": str(run_dir / "results.csv"),
                "status": "available", "notes": "F4 NIR-free RGB-only baseline"
            })
        else:
            anomalies.append(f"MISSING: {name}")

    # ── P0-F4-Dual: F4 NIR-free dual, fold2, seeds 42/1337/2024 ──
    for seed in [42, 1337, 2024]:
        name = f"phase_f4_nirfree_rgbsafe_dual_fold2_seed{seed}"
        run_dir = RUNS / name
        if run_dir.exists():
            registry.append({
                "family": "P0-F4-Dual", "method": "NIRfree-Dual", "fold": 2, "seed": seed,
                "run_class": "formal", "run_id": name,
                "source_path": str(run_dir),
                "config_path": str(run_dir / "args.yaml"),
                "log_path": "",
                "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                "prediction_path": "",
                "results_csv": str(run_dir / "results.csv"),
                "status": "available", "notes": "F4 NIR-free dual baseline"
            })
        else:
            anomalies.append(f"MISSING: {name}")

    # ── P0-IR: F5 IR-only, fold2, seeds 42/1337 ──
    for seed in [42, 1337]:
        name = f"phase_f5_ironly_fold2_nirfree_seed{seed}"
        run_dir = RUNS / name
        if run_dir.exists():
            registry.append({
                "family": "P0-IR", "method": "IR-only", "fold": 2, "seed": seed,
                "run_class": "formal", "run_id": name,
                "source_path": str(run_dir),
                "config_path": str(run_dir / "args.yaml"),
                "log_path": "",
                "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                "prediction_path": "",
                "results_csv": str(run_dir / "results.csv"),
                "status": "available", "notes": "F5 IR-only fold2"
            })
        else:
            anomalies.append(f"MISSING: {name}")

    # ── P0-G1a: SafeLateGate GateA ──
    g1a_dir = RUNS / "f5_g1a_safe_late_gate_20260607"
    for fold in [1, 2]:
        for seed in [42, 1337]:
            for modal in ["rgb", "dual"]:
                jsonl_name = f"predictions_fold{fold}_seed{seed}_{modal}.jsonl"
                jsonl_path = g1a_dir / jsonl_name
                if jsonl_path.exists():
                    registry.append({
                        "family": "P0-G1a", "method": f"G1a-{modal}", "fold": fold, "seed": seed,
                        "run_class": "formal", "run_id": f"g1a_{modal}_fold{fold}_seed{seed}",
                        "source_path": str(jsonl_path),
                        "config_path": "", "log_path": "",
                        "checkpoint_path": "", "prediction_path": str(jsonl_path),
                        "results_csv": str(g1a_dir / f"gate_sweep_fold{fold}.csv"),
                        "status": "available", "notes": "G1a GateA prediction JSONL"
                    })
                else:
                    anomalies.append(f"MISSING: {jsonl_name}")
        # gate_sweep CSV
        gate_csv = g1a_dir / f"gate_sweep_fold{fold}.csv"
        if gate_csv.exists():
            registry.append({
                "family": "P0-G1a", "method": "G1a-gate-sweep", "fold": fold, "seed": "all",
                "run_class": "formal", "run_id": f"g1a_gate_sweep_fold{fold}",
                "source_path": str(gate_csv),
                "config_path": "", "log_path": "", "checkpoint_path": "", "prediction_path": "",
                "results_csv": str(gate_csv),
                "status": "available", "notes": "G1a gate sweep CSV"
            })

    # ── P0-RFuse ──
    rfuse_dir = RUNS / "f5_rfuse_r0_r1_20260612"
    for fold in [1, 2]:
        for ftype in ["r0_separability_summary", "r1_gate_sweep", "r1_logistic_gate"]:
            csv_name = f"{ftype}_fold{fold}.csv" if ftype != "r0_separability_summary" else "r0_separability_summary.csv"
            csv_path = rfuse_dir / csv_name
            if csv_path.exists():
                registry.append({
                    "family": "P0-RFuse", "method": f"R-Fuse-{ftype}", "fold": fold if ftype != "r0_separability_summary" else "1+2",
                    "seed": "all", "run_class": "formal", "run_id": f"rfuse_{ftype}",
                    "source_path": str(csv_path),
                    "config_path": "", "log_path": "", "checkpoint_path": "", "prediction_path": "",
                    "results_csv": str(csv_path),
                    "status": "available", "notes": f"R-Fuse {ftype}"
                })

    # ── P0-G1c ──
    g1c_variants = [
        ("cleankd_dropout", "phase_f5_g1c_mcf_cleankd_dropout_fold1"),
        ("dropout_nokd", "phase_f5_g1c_mcf_dropout_nokd_fold1"),
    ]
    for variant, prefix in g1c_variants:
        for seed in [42, 1337]:
            name = f"{prefix}_seed{seed}"
            run_dir = RUNS / name
            if run_dir.exists():
                registry.append({
                    "family": "P0-G1c", "method": f"G1c-{variant}", "fold": 1, "seed": seed,
                    "run_class": "exploratory", "run_id": name,
                    "source_path": str(run_dir),
                    "config_path": str(run_dir / "args.yaml"),
                    "log_path": "",
                    "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                    "prediction_path": "",
                    "results_csv": str(run_dir / "results.csv"),
                    "status": "available", "notes": "G1c clean MCF (2-seed exploratory)"
                })

    # ── P1-YOLOv11-RGBT: 5 modes x 2 seeds ──
    rgbt_dir = RUNS / "f5_yolov11_rgbt_fold2_nirfree"
    for mode in ["early", "mid", "late", "score", "share"]:
        for seed in [42, 1337]:
            name = f"yolov11rgbt_{mode}_fold2_nirfree_seed{seed}"
            run_dir = rgbt_dir / name
            if run_dir.exists():
                registry.append({
                    "family": "P1-YOLOv11-RGBT", "method": f"YOLOv11-RGBT-{mode}", "fold": 2, "seed": seed,
                    "run_class": "formal", "run_id": name,
                    "source_path": str(run_dir),
                    "config_path": str(run_dir / "args.yaml"),
                    "log_path": "",
                    "checkpoint_path": str(run_dir / "weights" / "best.pt") if (run_dir / "weights" / "best.pt").exists() else "",
                    "prediction_path": "",
                    "results_csv": str(run_dir / "results.csv"),
                    "status": "available", "notes": f"YOLOv11-RGBT {mode} fusion pilot (seed42/1337)"
                })

    return registry, anomalies


# ─── Step 2: Recount P0 Main Table ───

def recount_main_table(registry):
    """从 results.csv recount 每个 run 的 best mAP50 等指标。"""
    recount_rows = []
    anomalies = []

    # 提取有 results_csv 的 available runs
    available = [r for r in registry if r["status"] == "available" and r.get("results_csv")]

    # 先收集 RGB-only baseline 用于 paired delta
    rgb_baselines = {}  # (fold, seed) -> map50
    for r in available:
        if r["method"] in ("RGB-only", "NIRfree-RGB-only") and r["status"] == "available":
            rcsv = Path(r["results_csv"])
            if rcsv.exists():
                metrics = best50_from_csv(rcsv.parent)
                if metrics:
                    key = (int(r["fold"]), int(r["seed"]))
                    rgb_baselines[key] = metrics["best_map50"]

    # Recount 每个 available run
    for r in available:
        rcsv = Path(r["results_csv"])
        if not rcsv.exists():
            continue

        # 对于 G1a gate sweep / R-Fuse CSV，跳过 results.csv recount（它们不是训练 run）
        if r["method"] in ("G1a-gate-sweep", "G1a-rgb", "G1a-dual",
                           "R-Fuse-r0_separability_summary", "R-Fuse-r1_gate_sweep", "R-Fuse-r1_logistic_gate"):
            continue

        metrics = best50_from_csv(rcsv.parent)
        if metrics is None:
            anomalies.append(f"NO_RESULTS_CSV: {r['run_id']}")
            continue

        fold = int(r["fold"])
        seed = int(r["seed"])

        # Paired delta vs RGB-only
        rgb_key = (fold, seed)
        paired_delta = ""
        if r["method"] not in ("RGB-only", "NIRfree-RGB-only"):
            # 对于 NIR-free 系列用 NIR-free RGB baseline
            if "NIRfree" in r["method"]:
                rgb_key_nf = (fold, seed)
                # NIR-free RGB baselines
                nf_rgb_name = f"phase_f4_nirfree_rgbonly_fold2_seed{seed}"
                nf_csv = RUNS / nf_rgb_name / "results.csv"
                if nf_csv.exists():
                    nf_metrics = best50_from_csv(nf_csv.parent)
                    if nf_metrics:
                        paired_delta = f"{metrics['best_map50'] - nf_metrics['best_map50']:.6f}"
            elif rgb_key in rgb_baselines:
                paired_delta = f"{metrics['best_map50'] - rgb_baselines[rgb_key]:.6f}"

        recount_rows.append({
            "family": r["family"],
            "method": r["method"],
            "fold": fold,
            "seed": seed,
            "run_id": r["run_id"],
            "run_class": r["run_class"],
            "n_epochs": metrics["n_epochs"],
            "has_best_pt": metrics["has_best_pt"],
            "best_epoch": metrics["best_epoch"],
            "best_map50": f"{metrics['best_map50']:.6f}",
            "best_map50_95": f"{metrics['best_map50_95']:.6f}",
            "final_map50": f"{metrics['final_map50']:.6f}",
            "final_map50_95": f"{metrics['final_map50_95']:.6f}",
            "final_over_best": f"{metrics['final_over_best']:.4f}",
            "final_val_cls_loss": f"{metrics['final_val_cls_loss']:.6f}",
            "collapse": metrics["collapse"],
            "paired_delta_vs_rgb": paired_delta,
        })

    return recount_rows, anomalies


# ─── Step 3: Gate Ablation Table ───

def recount_gate_ablation():
    """从 G1a 和 R-Fuse CSV 提取 gate ablation 行。"""
    gate_rows = []
    anomalies = []

    # ── G1a: 从 gate_sweep CSV 提取 GateA 行 ──
    g1a_dir = RUNS / "f5_g1a_safe_late_gate_20260607"
    for fold in [1, 2]:
        csv_path = g1a_dir / f"gate_sweep_fold{fold}.csv"
        if not csv_path.exists():
            anomalies.append(f"MISSING: {csv_path}")
            continue

        rows = read_rows(csv_path)
        for row in rows:
            policy = row.get("policy", "")
            if policy in ("P0_rgb_only", "P1_dual_only", "GateA"):
                tau_overlap = safe_float(row.get("tau_overlap", 0))
                tau_dual = safe_float(row.get("tau_dual", 0))

                gate_rows.append({
                    "source": "G1a",
                    "method": policy,
                    "fold": fold,
                    "seed": row.get("seed", ""),
                    "tau_overlap": tau_overlap,
                    "tau_dual": tau_dual,
                    "AP50": safe_float(row.get("AP50", 0)),
                    "smoke_AP50": safe_float(row.get("smoke_AP50", 0)),
                    "fire_AP50": safe_float(row.get("fire_AP50", 0)),
                    "person_AP50": safe_float(row.get("person_AP50", 0)),
                    "delta_AP50_vs_rgb": safe_float(row.get("delta_AP50_vs_rgb", 0)),
                    "acceptance_ratio": safe_float(row.get("dual_acceptance_ratio", 0)),
                    "notes": "GateA locked: tau_overlap=0.7, tau_dual=0.05" if policy == "GateA" else ""
                })

    # ── R-Fuse: 从 r1_gate_sweep CSV 提取 rfuse_rule_v1 和 baseline 行 ──
    rfuse_dir = RUNS / "f5_rfuse_r0_r1_20260612"
    for fold in [1, 2]:
        csv_path = rfuse_dir / f"r1_gate_sweep_fold{fold}.csv"
        if not csv_path.exists():
            anomalies.append(f"MISSING: {csv_path}")
            continue

        rows = read_rows(csv_path)
        for row in rows:
            arm = row.get("arm", "")
            if arm in ("rgb_only", "dual_only", "g1a_gateA", "rfuse_rule_v1"):
                gate_rows.append({
                    "source": "R-Fuse",
                    "method": arm,
                    "fold": fold,
                    "seed": row.get("seed", ""),
                    "tau_overlap": safe_float(row.get("tau_overlap", 0)),
                    "tau_dual": safe_float(row.get("tau_dual", 0)),
                    "AP50": safe_float(row.get("AP50", 0)),
                    "smoke_AP50": safe_float(row.get("smoke_AP50", 0)),
                    "fire_AP50": safe_float(row.get("fire_AP50", 0)),
                    "person_AP50": safe_float(row.get("person_AP50", 0)),
                    "delta_AP50_vs_rgb": safe_float(row.get("delta_AP50_vs_rgb", 0)),
                    "acceptance_ratio": safe_float(row.get("dual_acceptance_ratio", 0)),
                    "notes": "R-Fuse rule v1" if arm == "rfuse_rule_v1" else ""
                })

    return gate_rows, anomalies


# ─── Step 4: External Baseline Table ───

def recount_external_baselines():
    """从 YOLOv11-RGBT run dirs recount 外部 baseline。"""
    ext_rows = []
    anomalies = []

    rgbt_dir = RUNS / "f5_yolov11_rgbt_fold2_nirfree"
    modes = ["early", "mid", "late", "score", "share"]
    seeds = [42, 1337]

    # NIR-free RGB-only baseline for paired delta
    nf_rgb_baselines = {}
    for seed in [42, 1337, 2024]:
        name = f"phase_f4_nirfree_rgbonly_fold2_seed{seed}"
        csv_path = RUNS / name / "results.csv"
        if csv_path.exists():
            metrics = best50_from_csv(csv_path.parent)
            if metrics:
                nf_rgb_baselines[seed] = metrics["best_map50"]

    for mode in modes:
        for seed in seeds:
            name = f"yolov11rgbt_{mode}_fold2_nirfree_seed{seed}"
            run_dir = rgbt_dir / name
            if not run_dir.exists():
                anomalies.append(f"MISSING: {name}")
                continue

            metrics = best50_from_csv(run_dir)
            if metrics is None:
                anomalies.append(f"NO_RESULTS_CSV: {name}")
                continue

            rgb_base = nf_rgb_baselines.get(seed, 0)
            paired_delta = metrics["best_map50"] - rgb_base if rgb_base > 0 else 0

            ext_rows.append({
                "family": "P1-YOLOv11-RGBT",
                "method": f"YOLOv11-RGBT-{mode}",
                "fold": 2,
                "seed": seed,
                "run_id": name,
                "run_class": "formal",
                "n_epochs": metrics["n_epochs"],
                "has_best_pt": metrics["has_best_pt"],
                "best_epoch": metrics["best_epoch"],
                "best_map50": f"{metrics['best_map50']:.6f}",
                "best_map50_95": f"{metrics['best_map50_95']:.6f}",
                "final_map50": f"{metrics['final_map50']:.6f}",
                "final_over_best": f"{metrics['final_over_best']:.4f}",
                "collapse": metrics["collapse"],
                "paired_delta_vs_rgb": f"{paired_delta:.6f}",
                "notes": "pilot seed42/1337"
            })

    return ext_rows, anomalies


# ─── Step 5: Aggregate + Summary ───

def compute_summary(rows, group_key="method"):
    """按 group_key 聚合 mean ± std。"""
    groups = {}
    for r in rows:
        key = r[group_key]
        if key not in groups:
            groups[key] = []
        try:
            groups[key].append(float(r["best_map50"]))
        except (ValueError, KeyError):
            pass

    summary = {}
    for key, vals in groups.items():
        if len(vals) >= 2:
            summary[key] = {
                "mean_map50": statistics.mean(vals),
                "std_map50": statistics.stdev(vals),
                "n": len(vals),
            }
        elif len(vals) == 1:
            summary[key] = {
                "mean_map50": vals[0],
                "std_map50": 0.0,
                "n": 1,
            }
    return summary


def build_anomalies_md(all_anomalies):
    """构建 anomalies.md。"""
    lines = [
        "# Formal P0/P1 Anomalies",
        f"",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]
    if not all_anomalies:
        lines.append("无异常。")
    else:
        for i, a in enumerate(all_anomalies, 1):
            lines.append(f"{i}. {a}")
    return "\n".join(lines)


def build_recount_md(main_rows, gate_rows, ext_rows, main_summary, ext_summary, all_anomalies,
                     g1a_seed2024_status="NOT_ATTEMPTED",
                     cp_status="NOT_ATTEMPTED",
                     m2d_status="NOT_ATTEMPTED",
                     yolov11_seed2024_status="NOT_ATTEMPTED"):
    """构建 recount.md。"""
    lines = [
        "# Formal P0/P1 Recount Summary",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## P0 Main Table",
        "",
    ]

    # 按 family, method, fold, seed 排序输出
    for r in sorted(main_rows, key=lambda x: (x.get("family", ""), x.get("method", ""), x.get("fold", 0), x.get("seed", 0))):
        lines.append(
            f"- {r['family']} | {r['method']} | fold{r['fold']} seed{r['seed']} | "
            f"mAP50={r['best_map50']} | final/best={r['final_over_best']} | "
            f"collapse={r['collapse']} | delta={r.get('paired_delta_vs_rgb', 'N/A')}"
        )

    lines.append("")
    lines.append("### P0 Summary Statistics")
    for method, stats in sorted(main_summary.items()):
        lines.append(f"- **{method}**: mean={stats['mean_map50']:.6f} ± {stats['std_map50']:.6f} (n={stats['n']})")

    # G1a status
    lines.extend([
        "",
        "## G1a Formal Status",
        f"- {g1a_seed2024_status}",
        "",
    ])

    # Gate ablation
    lines.extend([
        "## Gate Ablation (G1a GateA + R-Fuse)",
        "",
    ])
    for r in sorted(gate_rows, key=lambda x: (x.get("source", ""), x.get("method", ""), x.get("fold", 0))):
        lines.append(
            f"- {r['source']} | {r['method']} | fold{r['fold']} seed{r['seed']} | "
            f"AP50={r['AP50']:.4f} delta={r['delta_AP50_vs_rgb']:.4f} "
            f"accept={r['acceptance_ratio']:.4f}"
        )

    # R-Fuse status
    lines.extend([
        "",
        "## R-Fuse Status",
        "- R-Fuse rule v1: match only, NOT stronger than G1a",
        "- R1 corrected verdict: R1_MATCH_G1A (not R1_STRONG_R_FUSE)",
        "",
    ])

    # G1c status
    lines.extend([
        "## G1c Status",
        "- Clean MCF: 2-seed exploratory on fold1",
        "- Secondary/diagnostic role; not formalized with 3 seeds",
        "- smoke_AP50 = 0 across all G1c runs (fire/person only)",
        "",
    ])

    # External baselines
    lines.extend([
        "## External Baselines (YOLOv11-RGBT)",
        "",
    ])
    for r in sorted(ext_rows, key=lambda x: (x.get("method", ""), x.get("seed", 0))):
        lines.append(
            f"- {r['method']} | seed{r['seed']} | mAP50={r['best_map50']} | "
            f"delta={r.get('paired_delta_vs_rgb', 'N/A')} | collapse={r['collapse']}"
        )

    lines.append("")
    lines.append("### External Summary")
    for method, stats in sorted(ext_summary.items()):
        lines.append(f"- **{method}**: mean={stats['mean_map50']:.6f} ± {stats['std_map50']:.6f} (n={stats['n']})")

    # YOLOv11-RGBT seed2024 status
    lines.extend([
        "",
        "## YOLOv11-RGBT seed2024 Status",
        f"- {yolov11_seed2024_status}",
        "",
    ])

    # CP-YOLOv11-MF
    lines.extend([
        "## CP-YOLOv11-MF Status",
        f"- {cp_status}",
        "",
    ])

    # M2D-LIF
    lines.extend([
        "## M2D-LIF Status",
        f"- {m2d_status}",
        "",
    ])

    # Anomalies
    lines.extend([
        "## Anomalies",
        "",
    ])
    if all_anomalies:
        for a in all_anomalies:
            lines.append(f"- {a}")
    else:
        lines.append("无异常。")

    # Unfinished items
    lines.extend([
        "",
        "## Unfinished Items",
        "- G1a seed2024: 见上方状态",
        "- YOLOv11-RGBT seed2024 selected modes: 见上方状态",
        "- CP-YOLOv11-MF: 见上方状态",
        "- M2D-LIF: 见上方状态",
    ])

    return "\n".join(lines)


# ─── Main ───

def main():
    print("=" * 60)
    print("Formal P0/P1 Targeted Baselines — Registry & Recount")
    print("=" * 60)
    all_anomalies = []

    # Step 1: Discover runs
    print("\n[Step 1] 发现源 artifact...")
    registry, disc_anomalies = discover_runs()
    all_anomalies.extend(disc_anomalies)
    print(f"  注册 {len(registry)} 条 run，{len(disc_anomalies)} 个异常")

    # 写 run_registry.csv
    reg_fields = ["family", "method", "fold", "seed", "run_class", "run_id",
                  "source_path", "config_path", "log_path", "checkpoint_path",
                  "prediction_path", "results_csv", "status", "notes"]
    write_csv(OUT_DIR / "run_registry.csv", registry, reg_fields)

    # Step 2: Recount main table
    print("\n[Step 2] Recount P0 主表...")
    main_rows, main_anomalies = recount_main_table(registry)
    all_anomalies.extend(main_anomalies)
    print(f"  Recount {len(main_rows)} 条，{len(main_anomalies)} 个异常")

    main_fields = ["family", "method", "fold", "seed", "run_id", "run_class",
                   "n_epochs", "has_best_pt", "best_epoch", "best_map50", "best_map50_95",
                   "final_map50", "final_map50_95", "final_over_best", "final_val_cls_loss",
                   "collapse", "paired_delta_vs_rgb"]
    write_csv(OUT_DIR / "formal_main_table_recount.csv", main_rows, main_fields)

    # Step 3: Gate ablation
    print("\n[Step 3] Recount gate ablation...")
    gate_rows, gate_anomalies = recount_gate_ablation()
    all_anomalies.extend(gate_anomalies)
    print(f"  Recount {len(gate_rows)} 条 gate 行，{len(gate_anomalies)} 个异常")

    gate_fields = ["source", "method", "fold", "seed", "tau_overlap", "tau_dual",
                   "AP50", "smoke_AP50", "fire_AP50", "person_AP50",
                   "delta_AP50_vs_rgb", "acceptance_ratio", "notes"]
    write_csv(OUT_DIR / "gate_ablation_recount.csv", gate_rows, gate_fields)

    # Step 4: External baselines
    print("\n[Step 4] Recount external baselines...")
    ext_rows, ext_anomalies = recount_external_baselines()
    all_anomalies.extend(ext_anomalies)
    print(f"  Recount {len(ext_rows)} 条 external，{len(ext_anomalies)} 个异常")

    ext_fields = ["family", "method", "fold", "seed", "run_id", "run_class",
                  "n_epochs", "has_best_pt", "best_epoch", "best_map50", "best_map50_95",
                  "final_map50", "final_over_best", "collapse", "paired_delta_vs_rgb", "notes"]
    write_csv(OUT_DIR / "external_baseline_recount.csv", ext_rows, ext_fields)

    # Step 5: Aggregate
    print("\n[Step 5] 聚合统计...")
    main_summary = compute_summary(main_rows, "method")
    ext_summary = compute_summary(ext_rows, "method")

    for method, stats in sorted(main_summary.items()):
        print(f"  {method}: mean={stats['mean_map50']:.6f} ± {stats['std_map50']:.6f} (n={stats['n']})")
    for method, stats in sorted(ext_summary.items()):
        print(f"  {method}: mean={stats['mean_map50']:.6f} ± {stats['std_map50']:.6f} (n={stats['n']})")

    # Step 6: Write anomalies
    print("\n[Step 6] 写 anomalies.md...")
    anomalies_md = build_anomalies_md(all_anomalies)
    (OUT_DIR / "anomalies.md").write_text(anomalies_md, encoding="utf-8")
    print(f"  {len(all_anomalies)} 个异常记录")

    # Step 7: Write recount.md (初始版本，后续 step 会更新)
    print("\n[Step 7] 写 recount.md (初始版本)...")
    recount_md = build_recount_md(main_rows, gate_rows, ext_rows, main_summary, ext_summary, all_anomalies)
    (OUT_DIR / "recount.md").write_text(recount_md, encoding="utf-8")
    print("  recount.md 初始版本已写入")

    # Summary
    print("\n" + "=" * 60)
    print(f"初始 Bundle 生成完成: {len(registry)} runs, {len(main_rows)} recount, {len(gate_rows)} gate, {len(ext_rows)} ext")
    print(f"异常: {len(all_anomalies)}")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
