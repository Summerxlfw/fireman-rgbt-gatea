#!/usr/bin/env python3
"""从 formal source CSV 重算 F6 证据表。

本脚本只读取 source bundle，不读取 handoff 摘要，避免论文数字漂移。
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "02_experiments" / "formal_p0_p1_targeted_20260612_sources"
OUT = PROJECT / "03_evidence" / "f6_tables_20260613"
SERVER_F6 = PROJECT / "03_evidence" / "f6_server_20260613"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_delta(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def summarize_rows(
    rows: list[dict[str, str]],
    metric_col: str,
    delta_col: str | None = None,
    collapse_col: str | None = None,
) -> dict[str, object]:
    vals = [f(r.get(metric_col)) for r in rows]
    vals = [v for v in vals if v is not None]
    deltas: list[float] = []
    if delta_col:
        deltas = [f(r.get(delta_col)) for r in rows]
        deltas = [v for v in deltas if v is not None]
    collapse = ""
    if collapse_col:
        n_collapse = sum(str(r.get(collapse_col, "")).lower() == "true" for r in rows)
        collapse = f"{n_collapse}/{len(rows)}"
    return {
        "seeds": ",".join(r.get("seed", "") for r in rows),
        "n": len(rows),
        "mean": mean(vals),
        "std": pstdev(vals),
        "mean_delta": mean(deltas) if deltas else None,
        "positive_seeds": f"{sum(d > 0 for d in deltas)}/{len(deltas)}" if deltas else "-",
        "collapse": collapse or "-",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(SOURCE / "formal_main_table_recount.csv")
    external_rows = read_csv(SOURCE / "external_baseline_recount.csv")
    gate_rows = read_csv(SOURCE / "gate_ablation_recount.csv")
    bootstrap_rows = read_csv(SERVER_F6 / "bootstrap_ci.csv") if (SERVER_F6 / "bootstrap_ci.csv").exists() else []
    qualitative_rows = read_csv(SERVER_F6 / "qualitative_cases.csv") if (SERVER_F6 / "qualitative_cases.csv").exists() else []

    # Fold2 stress table: mix training-run baselines, GateA prediction-level rows,
    # and final 3-seed YOLOv11-RGBT rows from the external recount.
    fold2_specs = [
        ("NIRfree RGB-only", [r for r in main_rows if r["family"] == "P0-F4"], "best_map50", None, "collapse"),
        ("NIRfree Dual", [r for r in main_rows if r["family"] == "P0-F4-Dual"], "best_map50", "paired_delta_vs_rgb", "collapse"),
        ("IR-only", [r for r in main_rows if r["family"] == "P0-IR"], "best_map50", "paired_delta_vs_rgb", "collapse"),
        (
            "GateA (G1a locked)",
            [
                r
                for r in gate_rows
                if r["source"] == "G1a"
                and r["method"] == "GateA"
                and r["fold"] == "2"
                and r["tau_overlap"] == "0.7"
                and r["tau_dual"] == "0.05"
            ],
            "AP50",
            "delta_AP50_vs_rgb",
            None,
        ),
        ("YOLOv11-RGBT-score", [r for r in external_rows if r["method"] == "YOLOv11-RGBT-score"], "best_map50", "paired_delta_vs_rgb", "collapse"),
        ("YOLOv11-RGBT-share", [r for r in external_rows if r["method"] == "YOLOv11-RGBT-share"], "best_map50", "paired_delta_vs_rgb", "collapse"),
        ("YOLOv11-RGBT-mid", [r for r in external_rows if r["method"] == "YOLOv11-RGBT-mid"], "best_map50", "paired_delta_vs_rgb", "collapse"),
    ]
    fold2_table: list[dict[str, object]] = []
    for method, rows, metric, delta, collapse in fold2_specs:
        s = summarize_rows(rows, metric, delta, collapse)
        fold2_table.append(
            {
                "method": method,
                "seeds": s["seeds"],
                "n": s["n"],
                "mean_mAP50": fmt(s["mean"]),
                "std": fmt(s["std"]),
                "mean_delta_vs_rgb": fmt_delta(s["mean_delta"]),
                "positive_seeds": s["positive_seeds"],
                "collapse": s["collapse"],
            }
        )

    write_csv(
        OUT / "fold2_stress_table.csv",
        fold2_table,
        ["method", "seeds", "n", "mean_mAP50", "std", "mean_delta_vs_rgb", "positive_seeds", "collapse"],
    )

    # Fold1 retention from locked GateA rows.
    by_seed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in gate_rows:
        if r["source"] == "G1a" and r["fold"] == "1":
            by_seed[r["seed"]][r["method"]] = r

    fold1_rows: list[dict[str, object]] = []
    retentions: list[float] = []
    gate_deltas: list[float] = []
    dual_deltas: list[float] = []
    for seed in sorted(by_seed, key=lambda x: int(x)):
        rgb = by_seed[seed].get("P0_rgb_only")
        dual = by_seed[seed].get("P1_dual_only")
        gate = by_seed[seed].get("GateA")
        if not (rgb and dual and gate):
            continue
        dual_delta = f(dual["delta_AP50_vs_rgb"]) or 0.0
        gate_delta = f(gate["delta_AP50_vs_rgb"]) or 0.0
        retention = gate_delta / dual_delta if dual_delta else None
        if retention is not None:
            retentions.append(retention)
        gate_deltas.append(gate_delta)
        dual_deltas.append(dual_delta)
        fold1_rows.append(
            {
                "seed": seed,
                "rgb_mAP50": fmt(f(rgb["AP50"])),
                "dual_mAP50": fmt(f(dual["AP50"])),
                "gateA_mAP50": fmt(f(gate["AP50"])),
                "dual_delta": fmt_delta(dual_delta),
                "gate_delta": fmt_delta(gate_delta),
                "retention": fmt(retention, 3),
                "acceptance": fmt(f(gate["acceptance_ratio"]), 3),
            }
        )
    fold1_rows.append(
        {
            "seed": "mean",
            "rgb_mAP50": "-",
            "dual_mAP50": "-",
            "gateA_mAP50": "-",
            "dual_delta": fmt_delta(mean(dual_deltas)),
            "gate_delta": fmt_delta(mean(gate_deltas)),
            "retention": fmt(mean(retentions), 3),
            "acceptance": "-",
        }
    )
    write_csv(
        OUT / "fold1_retention_table.csv",
        fold1_rows,
        ["seed", "rgb_mAP50", "dual_mAP50", "gateA_mAP50", "dual_delta", "gate_delta", "retention", "acceptance"],
    )

    # External baseline table.
    external_table: list[dict[str, object]] = []
    for method in sorted({r["method"] for r in external_rows}):
        rows = [r for r in external_rows if r["method"] == method]
        s = summarize_rows(rows, "best_map50", "paired_delta_vs_rgb", "collapse")
        external_table.append(
            {
                "method": method,
                "seeds": s["seeds"],
                "n": s["n"],
                "mean_mAP50": fmt(s["mean"]),
                "std": fmt(s["std"]),
                "mean_delta_vs_rgb": fmt_delta(s["mean_delta"]),
                "positive_seeds": s["positive_seeds"],
                "collapse": s["collapse"],
            }
        )
    write_csv(
        OUT / "external_yolov11_rgbt_table.csv",
        external_table,
        ["method", "seeds", "n", "mean_mAP50", "std", "mean_delta_vs_rgb", "positive_seeds", "collapse"],
    )

    # Gate sensitivity: fold2 sweep only has seed42/1337 for the grid; seed2024
    # exists for the locked policy. Keep this explicitly exploratory.
    grid: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in gate_rows:
        if r["source"] == "G1a" and r["method"] == "GateA" and r["fold"] == "2" and r["seed"] in {"42", "1337"}:
            grid[(r["tau_overlap"], r["tau_dual"])].append(r)
    sensitivity: list[dict[str, object]] = []
    for (tau_overlap, tau_dual), rows in sorted(grid.items(), key=lambda kv: (float(kv[0][0]), float(kv[0][1]))):
        deltas = [f(r["delta_AP50_vs_rgb"]) or 0.0 for r in rows]
        accepts = [f(r["acceptance_ratio"]) or 0.0 for r in rows]
        smoke = [f(r["smoke_AP50"]) or 0.0 for r in rows]
        sensitivity.append(
            {
                "tau_overlap": tau_overlap,
                "tau_dual": tau_dual,
                "n_seeds": len(rows),
                "mean_delta": fmt_delta(mean(deltas)),
                "std_delta": fmt(pstdev(deltas)),
                "mean_acceptance": fmt(mean(accepts), 3),
                "mean_smoke_AP50": fmt(mean(smoke), 4),
                "locked_policy": "yes" if (tau_overlap, tau_dual) == ("0.7", "0.05") else "no",
            }
        )
    write_csv(
        OUT / "gate_sensitivity_fold2_2seed.csv",
        sensitivity,
        ["tau_overlap", "tau_dual", "n_seeds", "mean_delta", "std_delta", "mean_acceptance", "mean_smoke_AP50", "locked_policy"],
    )

    if bootstrap_rows:
        bootstrap_table: list[dict[str, object]] = []
        for row in bootstrap_rows:
            bootstrap_table.append(
                {
                    "target": row["target"],
                    "fold": row["fold"],
                    "seed": row["seed"],
                    "contrast": f"{row['arm_a']} - {row['arm_b']}",
                    "point_delta": fmt_delta(f(row["point_delta"])),
                    "boot_mean_delta": fmt_delta(f(row["boot_mean_delta"])),
                    "ci_2.5": fmt_delta(f(row["ci_2.5"])),
                    "ci_97.5": fmt_delta(f(row["ci_97.5"])),
                    "n_images": row["n_images"],
                    "n_boot": row["n_boot"],
                }
            )
        write_csv(
            OUT / "bootstrap_ci_table.csv",
            bootstrap_table,
            ["target", "fold", "seed", "contrast", "point_delta", "boot_mean_delta", "ci_2.5", "ci_97.5", "n_images", "n_boot"],
        )

    if qualitative_rows:
        qualitative_table: list[dict[str, object]] = []
        for row in qualitative_rows:
            qualitative_table.append(
                {
                    "case_id": row["case_id"],
                    "case_type": row["case_type"],
                    "fold": row["fold"],
                    "seed": row["seed"],
                    "image_id": row["image_id"],
                    "rgb_fn": row["rgb_fn"],
                    "dual_fp": row["dual_fp"],
                    "gate_fp": row["gate_fp"],
                    "gate_fn": row["gate_fn"],
                    "why_selected": row["why_selected"],
                    "image_path": row["image_path"],
                }
            )
        write_csv(
            OUT / "qualitative_case_catalog.csv",
            qualitative_table,
            ["case_id", "case_type", "fold", "seed", "image_id", "rgb_fn", "dual_fp", "gate_fp", "gate_fn", "why_selected", "image_path"],
        )

    md = OUT / "formal_tables_20260613.md"
    with md.open("w") as fmd:
        fmd.write("# F6 Evidence Tables · 2026-06-13\n\n")
        fmd.write("Generated by `03_evidence/build_f6_evidence_tables.py` from source CSV files in `02_experiments/formal_p0_p1_targeted_20260612_sources/`.\n\n")
        fmd.write("## SOTA / Best-Under-Protocol Boundary\n\n")
        fmd.write("- Raw fold2 mAP50 best is not GateA; NIR-free RGB-only and YOLOv11-RGBT-share are around 0.266.\n")
        fmd.write("- GateA is best under the safe-admission objective: paired fold2 delta is non-negative, fold1 retention is 68.1%, and selected external fusion modes do not exceed it by the protocol upgrade margin.\n")
        fmd.write("- Therefore manuscript wording should be `strongest verified negative-transfer-safe baseline under our LOCO protocol`, not `global SOTA multimodal object detector`.\n\n")

        def write_table(title: str, rows: list[dict[str, object]], cols: list[str]) -> None:
            fmd.write(f"## {title}\n\n")
            fmd.write("| " + " | ".join(cols) + " |\n")
            fmd.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for row in rows:
                fmd.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")
            fmd.write("\n")

        write_table(
            "Fold2 NIR-Free Stress Table",
            fold2_table,
            ["method", "seeds", "n", "mean_mAP50", "std", "mean_delta_vs_rgb", "positive_seeds", "collapse"],
        )
        write_table(
            "Fold1 GateA Retention Table",
            fold1_rows,
            ["seed", "rgb_mAP50", "dual_mAP50", "gateA_mAP50", "dual_delta", "gate_delta", "retention", "acceptance"],
        )
        write_table(
            "External YOLOv11-RGBT Table",
            external_table,
            ["method", "seeds", "n", "mean_mAP50", "std", "mean_delta_vs_rgb", "positive_seeds", "collapse"],
        )
        write_table(
            "Gate Sensitivity Fold2, 2-Seed Exploratory",
            sensitivity,
            ["tau_overlap", "tau_dual", "n_seeds", "mean_delta", "std_delta", "mean_acceptance", "mean_smoke_AP50", "locked_policy"],
        )

        if bootstrap_rows:
            write_table(
                "Bootstrap CI, Paired Image-Level Resampling",
                bootstrap_table,
                ["target", "fold", "seed", "contrast", "point_delta", "boot_mean_delta", "ci_2.5", "ci_97.5", "n_images", "n_boot"],
            )
            fmd.write("Notes: `RGB - Dual` is harm magnitude for unconditional dual fusion; `GateA - RGB` is gate gain/safety. Bootstrap uses fixed predictions and GT, paired image-level resampling, n_boot=200.\n\n")

        if qualitative_rows:
            counts: dict[str, int] = defaultdict(int)
            for row in qualitative_rows:
                counts[row["case_type"]] += 1
            fmd.write("## Qualitative Case Catalog\n\n")
            for case_type, count in sorted(counts.items()):
                fmd.write(f"- `{case_type}`: {count}\n")
            fmd.write("\n")
            fmd.write("Full catalog: `03_evidence/f6_tables_20260613/qualitative_case_catalog.csv`.\n\n")

        fmd.write("## Remaining Evidence Limits\n\n")
        fmd.write("1. Fold0 GateA retention is blocked: no formal fold0 dual checkpoint exists; F-1 pilot dual uses a different split and must not be substituted.\n")
        fmd.write("2. Bootstrap CI is available for fold1/fold2 with n_boot=200; use as minimum acceptable uncertainty evidence, not high-precision inference.\n")
        fmd.write("3. Qualitative cases are cataloged as illustrative evidence; visual exports are not present in the local bundle yet.\n")
        fmd.write("4. Gate sensitivity with seed2024 across the full grid remains unavailable; seed2024 exists only for locked GateA.\n")


if __name__ == "__main__":
    main()
