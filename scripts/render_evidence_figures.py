#!/usr/bin/env python3
# 渲染森林火灾 RGB-T 论文证据图 Fig2 / Fig3。
# 数据一律从 03_evidence/f6_tables_20260613/*.csv 读取,不硬编码数值(反 source)。
# 输出 PDF(投稿稿)+ PNG(目视核对)。matplotlib fonttype=42 避 IEEE Type-3 坑。
# 口径纪律见 05_manuscript/table_packet_20260615.md:fold2 Δ 基线不统一,故 Panel 用 raw mAP50 + bootstrap CI,
# 不画混基线 Δ bar。

import csv
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE, "..", "03_evidence", "f6_tables_20260613")
OUT = os.path.join(BASE, "figure_outputs")
os.makedirs(OUT, exist_ok=True)


def rd(name):
    with open(os.path.join(CSV_DIR, name)) as fh:
        return list(csv.DictReader(fh))


def fnum(s):
    return float(str(s).replace("+", ""))


# ---------------- Fig 2: fold2 stress (Panel A raw mAP50, Panel B bootstrap CI) ----------------
def fig2():
    stress = rd("fold2_stress_table.csv")
    boot = rd("bootstrap_ci_table.csv")

    # Panel A: raw mAP50 mean +/- std, per method
    labels, means, stds = [], [], []
    disp = {
        "NIRfree RGB-only": "RGB-only",
        "NIRfree Dual": "Dual\n(uncond.)",
        "IR-only": "IR-only",
        "GateA (G1a locked)": "GateA†",
        "YOLOv11-RGBT-score": "Y11-score",
        "YOLOv11-RGBT-share": "Y11-share",
        "YOLOv11-RGBT-mid": "Y11-mid",
    }
    order = ["NIRfree RGB-only", "NIRfree Dual", "IR-only", "GateA (G1a locked)",
             "YOLOv11-RGBT-score", "YOLOv11-RGBT-share", "YOLOv11-RGBT-mid"]
    rowmap = {r["method"]: r for r in stress}
    for m in order:
        labels.append(disp[m])
        means.append(fnum(rowmap[m]["mean_mAP50"]))
        stds.append(fnum(rowmap[m]["std"]))
    colors = ["#4C72B0", "#C44E52", "#8C8C8C", "#55A868",
              "#DD8452", "#DD8452", "#DD8452"]

    rgb_base = fnum(rowmap["NIRfree RGB-only"]["mean_mAP50"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1]})

    x = np.arange(len(labels))
    axA.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    axA.axhline(rgb_base, ls="--", lw=0.8, color="#4C72B0", alpha=0.7)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")
    axA.set_ylabel("fold2 mAP50", fontsize=8)
    axA.set_title("(a) Raw fold2 mAP50 (mean ± std over seeds)", fontsize=8)
    axA.tick_params(axis="y", labelsize=7)
    axA.set_ylim(0, 0.32)
    axA.text(0.02, rgb_base + 0.004, "RGB-only", fontsize=7, color="#4C72B0")

    # Panel B: bootstrap CI, harm (RGB-Dual) and safety (GateA-RGB) on fold2
    harm = [r for r in boot if r["fold"] == "2" and "Dual" in r["contrast"]]
    safe = [r for r in boot if r["fold"] == "2" and "GateA" in r["contrast"]]
    harm.sort(key=lambda r: int(r["seed"]))
    safe.sort(key=lambda r: int(r["seed"]))
    seeds = [r["seed"] for r in harm]
    xb = np.arange(len(seeds))
    w = 0.34

    def pts(rows):
        pe = [fnum(r["point_delta"]) for r in rows]
        lo = [fnum(r["point_delta"]) - fnum(r["ci_2.5"]) for r in rows]
        hi = [fnum(r["ci_97.5"]) - fnum(r["point_delta"]) for r in rows]
        return pe, [lo, hi]

    he, herr = pts(harm)
    se, serr = pts(safe)
    axB.errorbar(xb - w / 2, he, yerr=herr, fmt="o", color="#C44E52", capsize=3,
                 ms=4, label="RGB−Dual (harm)")
    axB.errorbar(xb + w / 2, se, yerr=serr, fmt="s", color="#55A868", capsize=3,
                 ms=4, label="GateA−RGB (safety)")
    axB.axhline(0, color="black", lw=0.6)
    axB.set_xticks(xb)
    axB.set_xticklabels(seeds, fontsize=7)
    axB.set_xlabel("seed", fontsize=8)
    axB.set_ylabel("paired Δ mAP50 (bootstrap CI)", fontsize=8)
    axB.set_title("(b) Fold2 paired bootstrap CI (n_boot=200)", fontsize=8)
    axB.tick_params(axis="y", labelsize=7)
    axB.set_ylim(-0.006, 0.112)
    axB.legend(fontsize=6.5, loc="upper center", framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[Fig2] PanelA means:", dict(zip([l.replace("\n", " ") for l in labels], means)))
    print("[Fig2] PanelB harm point:", he, "safety point:", se)


# ---------------- Fig 3: fold1 retention ----------------
def fig3():
    ret = rd("fold1_retention_table.csv")
    rows = [r for r in ret if r["seed"] != "mean"]
    seeds = [r["seed"] for r in rows]
    rgb = [fnum(r["rgb_mAP50"]) for r in rows]
    dual = [fnum(r["dual_mAP50"]) for r in rows]
    gate = [fnum(r["gateA_mAP50"]) for r in rows]
    retent = [fnum(r["retention"]) for r in rows]
    mean_ret = fnum([r for r in ret if r["seed"] == "mean"][0]["retention"])

    x = np.arange(len(seeds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.bar(x - w, rgb, w, label="RGB-only", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x, dual, w, label="Dual", color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.bar(x + w, gate, w, label="GateA", color="#55A868", edgecolor="black", linewidth=0.5)
    for i in range(len(seeds)):
        ax.text(x[i] + w, gate[i] + 0.004, f"ret {retent[i]:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=7)
    ax.set_xlabel("seed", fontsize=8)
    ax.set_ylabel("fold1 mAP50", fontsize=8)
    ax.set_title(f"Fold1 thermal-benefit retention (mean retention {mean_ret:.3f})", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, 0.26)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[Fig3] rgb:", rgb, "dual:", dual, "gate:", gate, "ret:", retent, "mean_ret:", mean_ret)


if __name__ == "__main__":
    fig2()
    fig3()
    print("OK ->", OUT)
