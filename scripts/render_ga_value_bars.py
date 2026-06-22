#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GA 证据 value bars 真数据图(终稿 GA 人矢量重画的数据底图)。

红线(见 04_figures/terminal_redraw_todo.md + memory feedback_figure_redraw_real_data_discipline):
- 证据 bar 必 Python + 真实 CSV,禁用 image2 蓝图 dummy / 禁回抄蓝图编的数值。
- 数值全部从 source CSV 现读,不 hardcode:
  Panel A 源 = 02_experiments/f12_server_20260616/f12_indomain_fire_summary.csv (E021, best.pt)
  Panel B 源 = 03_evidence/f6_tables_20260613/bootstrap_ci_table.csv (E011 per-seed delta+CI) +
               03_evidence/f6_tables_20260613/fold2_stress_table.csv (E003 GateA mean delta)
- Panel A: in-domain fire AP50 per-seed,motif="齐(dual) vs 崩(RGB-only 2/3)";不标 mean、不标裸增益(C016 caveat)。
- Panel B: cross-dataset fold2 GateA−RGB per-seed delta(smoke-scoped);seed2024 贴零/微负=near-RGB safety margin;不美化成全正大增益。
- 图例无 "with GateA"(Panel A 是 in-domain 普通 dual 臂 F12,非 GateA);全图无 SOTA 措辞。
- fonttype=42 防 IEEE Type-3(memory feedback_figure_matplotlib_type3_font_ieee)。
此图仅作 GA 真数据底图供人 PPT/矢量重画嵌入;image2 蓝图位图不入论文。
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Type-3 字体防御(IEEE PDF eXpress 拒 Type 3)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ = SCRIPT_DIR.parent
SEEDS = ["42", "1337", "2024"]

# palette: GA 蓝图 TWO accents = slate(RGB/视觉) + amber(thermal/dual)
SLATE = "#4C6079"
AMBER = "#E2932E"
NEUTRAL = "#6E7B8B"
ZERO_NEG = "#B9534B"


def read_panelA():
    """读 f12 in-domain summary,取 best.pt 的 per-seed fire RGB / dual AP50。"""
    path = PROJ / "02_experiments/f12_server_20260616/f12_indomain_fire_summary.csv"
    rgb, dual = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["selection"] != "best.pt":
                continue
            # fire_delta 行同时带 fire_rgb_ap50 / fire_dual_ap50
            if row["arm"].startswith("fire_delta") and row["fire_rgb_ap50"]:
                rgb[row["seed"]] = float(row["fire_rgb_ap50"])
                dual[row["seed"]] = float(row["fire_dual_ap50"])
    return [rgb[s] for s in SEEDS], [dual[s] for s in SEEDS], path


def read_panelB():
    """读 bootstrap_ci_table,取 fold2 'GateA - RGB' per-seed point_delta + CI;mean 来自 fold2_stress_table。"""
    ci_path = PROJ / "03_evidence/f6_tables_20260613/bootstrap_ci_table.csv"
    pts, lo, hi = {}, {}, {}
    with open(ci_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["fold"] == "2" and row["contrast"].strip() == "GateA - RGB":
                s = row["seed"]
                pts[s] = float(row["point_delta"])
                lo[s] = float(row["ci_2.5"])
                hi[s] = float(row["ci_97.5"])
    stress_path = PROJ / "03_evidence/f6_tables_20260613/fold2_stress_table.csv"
    mean_delta = None
    with open(stress_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"].startswith("GateA"):
                mean_delta = float(row["mean_delta_vs_rgb"])
    return ([pts[s] for s in SEEDS], [lo[s] for s in SEEDS], [hi[s] for s in SEEDS],
            mean_delta, ci_path)


def main():
    rgb, dual, pa_src = read_panelA()
    pts, lo, hi, mean_delta, pb_src = read_panelB()

    # 自反核:打印读到的真值供 stdout 对 CSV
    print(f"[Panel A src] {pa_src}")
    for s, r, d in zip(SEEDS, rgb, dual):
        print(f"  seed{s}: RGB fire AP50={r:.4f}  dual fire AP50={d:.4f}")
    print(f"[Panel B src] {pb_src}  mean_delta(E003)={mean_delta:+.4f}")
    for s, p, l, h in zip(SEEDS, pts, lo, hi):
        print(f"  seed{s}: GateA-RGB delta={p:+.4f}  CI[{l:+.4f},{h:+.4f}]")

    x = range(len(SEEDS))
    out_dir = PROJ / "04_figures/figure_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Panel A 独立图(嵌 GA 栏③ A 卡片);大字号适配缩放 =====
    figA, axA = plt.subplots(figsize=(4.2, 3.7))
    w = 0.40
    axA.bar([i - w / 2 for i in x], rgb, w, color=SLATE, label="RGB-only")
    axA.bar([i + w / 2 for i in x], dual, w, color=AMBER, label="Dual (RGB-T)")
    axA.set_xticks(list(x))
    axA.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=12)
    axA.tick_params(axis="y", labelsize=12)
    axA.set_ylim(0, 1.30)
    axA.set_ylabel("fire AP$_{50}$", fontsize=15)
    axA.set_title("In-domain fire detection\n(within-RGBT-3M, video-disjoint)", fontsize=12)
    # 图例放顶部空白横条横排,避开高耸的 dual bar
    axA.legend(frameon=False, fontsize=11, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=1.3, handletextpad=0.5)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    figA.tight_layout()
    for ext in ("pdf", "png"):
        figA.savefig(out_dir / f"ga_panelA.{ext}", dpi=200, bbox_inches="tight")

    # ===== Panel B 独立图(嵌 GA 栏③ B 卡片);大字号适配缩放 =====
    figB, axB = plt.subplots(figsize=(4.2, 3.7))
    colors = [NEUTRAL if p > 0.0005 else ZERO_NEG for p in pts]
    yerr_lo = [p - l for p, l in zip(pts, lo)]
    yerr_hi = [h - p for p, h in zip(pts, hi)]
    axB.bar(list(x), pts, 0.55, color=colors,
            yerr=[yerr_lo, yerr_hi], capsize=5, ecolor="#333333", error_kw={"lw": 1.2})
    axB.axhline(0, color="#333333", lw=1.0)
    axB.set_xticks(list(x))
    axB.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=12)
    axB.tick_params(axis="y", labelsize=11)
    axB.set_ylabel(r"$\Delta$ mAP$_{50}$ (GateA $-$ RGB)", fontsize=14)
    axB.set_title("Cross-dataset fold2 safe-admission\n(smoke-scoped fold2 comparison)", fontsize=12)
    # 注释放右侧空白区(seed1337/2024 列上方,delta≈0 留白多)分行,避开标题与 seed42 CI
    axB.text(0.60, 0.80,
             f"Near-neutral\n(mean {mean_delta:+.4f})\nnear RGB",
             transform=axB.transAxes, ha="left", va="top", fontsize=11, color="#333333")
    axB.set_ylim(min(lo) - 0.002, max(hi) + 0.005)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    figB.tight_layout()
    for ext in ("pdf", "png"):
        figB.savefig(out_dir / f"ga_panelB.{ext}", dpi=200, bbox_inches="tight")

    print(f"[saved] {out_dir}/ga_panelA.(pdf|png) + ga_panelB.(pdf|png)")


if __name__ == "__main__":
    main()
