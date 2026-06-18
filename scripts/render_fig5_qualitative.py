#!/usr/bin/env python3
# 渲染 Fig5 定性图:只使用服务器回传的真实帧/真实框 overlay,不生成或补造图像内容。

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
from PIL import Image


BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(BASE, ".."))
CATALOG = os.path.join(PROJECT, "03_evidence", "f6_tables_20260613", "qualitative_case_catalog.csv")
EXPORT_ROOT = os.path.join(BASE, "figure_outputs", "server_export_20260617")
EXPORT_MANIFEST = os.path.join(EXPORT_ROOT, "export_manifest.csv")
FIG5_DIR = os.path.join(EXPORT_ROOT, "fig5_export")
OUT = os.path.join(BASE, "figure_outputs")
ALIGNMENT_CSV = os.path.join(OUT, "fig5_case_alignment_20260618.csv")

GROUPS = [
    ("rgb_miss_rescued_by_gateA", "RGB miss rescued\nby GateA"),
    ("dual_fp_rejected_by_gateA", "Dual false positives\nrejected by GateA"),
    ("harmful_thermal_collapse", "Harmful thermal\ncollapse guarded"),
    ("thermal_helpful_retained", "Helpful thermal\nretained by GateA"),
]


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_cases():
    catalog_rows = read_csv(CATALOG)
    manifest_rows = read_csv(EXPORT_MANIFEST)

    boxes_by_case = {}
    raw_counts = defaultdict(set)
    for row in manifest_rows:
        if row["group"] != "fig5":
            continue
        case_id = row["case_id_or_tile"]
        if row["box_source"] == "gt+rgb+dual+gate":
            boxes_by_case[case_id] = os.path.join(FIG5_DIR, row["export_png"])
        elif row["modality"] in {"RGB", "thermal"}:
            raw_counts[case_id].add(row["modality"])

    catalog_ids = [row["case_id"] for row in catalog_rows]
    missing_box = [cid for cid in catalog_ids if cid not in boxes_by_case]
    missing_raw = [cid for cid in catalog_ids if raw_counts[cid] != {"RGB", "thermal"}]
    extra_box = sorted(set(boxes_by_case) - set(catalog_ids))
    if missing_box or missing_raw or extra_box:
        raise SystemExit(
            "Fig5 export/catalog mismatch: "
            f"missing_box={missing_box}, missing_raw={missing_raw}, extra_box={extra_box}"
        )

    for cid, path in boxes_by_case.items():
        if not os.path.exists(path):
            raise SystemExit(f"Missing exported overlay for {cid}: {path}")

    with open(ALIGNMENT_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case_id", "case_type", "fold", "seed", "image_id", "box_png", "why_selected"])
        for row in catalog_rows:
            writer.writerow(
                [
                    row["case_id"],
                    row["case_type"],
                    row["fold"],
                    row["seed"],
                    row["image_id"],
                    os.path.relpath(boxes_by_case[row["case_id"]], PROJECT),
                    row["why_selected"],
                ]
            )

    grouped = defaultdict(list)
    for row in catalog_rows:
        grouped[row["case_type"]].append(row)
    return grouped, boxes_by_case


def render():
    grouped, boxes_by_case = load_cases()
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(16.2, 11.2))
    fig.patch.set_facecolor("white")

    for col, (case_type, title) in enumerate(GROUPS):
        rows = grouped.get(case_type, [])
        for row_i in range(3):
            ax = axes[row_i][col]
            ax.set_axis_off()
            if row_i >= len(rows):
                continue
            row = rows[row_i]
            case_id = row["case_id"]
            img = Image.open(boxes_by_case[case_id]).convert("RGB")
            ax.imshow(img)
            ax.text(
                0.01,
                0.99,
                f"{chr(ord('A') + col)}{row_i + 1} | fold {row['fold']} | {row['why_selected']}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", edgecolor="0.25", linewidth=0.35, alpha=0.88, pad=2),
            )
        axes[0][col].set_title(title, fontsize=12, fontweight="bold", pad=10)

    fig.suptitle(
        "Qualitative GateA behavior on the 11 pre-cataloged cases (illustration only)",
        fontsize=14,
        fontweight="bold",
        y=0.987,
    )
    fig.text(
        0.5,
        0.012,
        "Each tile is a server-exported real-frame overlay: green=GT, blue=RGB-only prediction, red=unconditional dual prediction, yellow=GateA output. "
        "Cases illustrate rescue, false-positive rejection, harmful-collapse guarding, and helpful-thermal retention; they are not statistical evidence.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.05, wspace=0.045, hspace=0.08)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig5.{ext}"), dpi=300)
    plt.close(fig)

    counts = {case_type: len(grouped.get(case_type, [])) for case_type, _ in GROUPS}
    print("[Fig5] cases:", counts)
    print("[Fig5] alignment:", ALIGNMENT_CSV)
    print("[Fig5] outputs:", os.path.join(OUT, "fig5.pdf"), os.path.join(OUT, "fig5.png"))


if __name__ == "__main__":
    render()
