#!/usr/bin/env python3
"""导出三图终稿真帧 — Fig5(11 case 带框)+ method Panel A(4 tile)+ GA anchor.

确定性后处理,零重训零重推,不触 G3 lock:
- box 全用已存 JSONL_F5 seed42 预测 + YOLO GT + apply_gateA_add_only(locked tau_overlap=0.7 / tau_dual=0.05,add-only,dual 预过滤 conf>=0.01)
- 复用 run_f6_evidence_package_20260613.py 的纯函数(复制,不 import —— 避免 catalog 脚本 module-level setup_logging() 的文件副作用)

硬红线(违反即 raise 停下):
1. 零重训零重推 —— 只用已存 JSONL 预测
2. 禁伪造 —— 缺帧 / 缺 box / img_id 在 JSONL 缺失 → raise
3. 模态别画反 —— thermal 全以 RGB 模式存(R≈G≈B 灰度),只能靠目录名区分,不靠 PIL mode
4. box 计数反核 catalog —— compute_per_image_metrics 重算 TP/FP/FN 逐项 == qualitative_cases.csv,不符 raise
"""
import json
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# fonttype=42 防 Type3(IEEE/MICCAI 投稿要求)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ─── 路径常量 ───
JSONL_F5 = Path("/mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607")
# 硬红线 3:模态靠目录名(thermal 存为 RGB 模式无法靠 PIL mode 区分)
#   fold1: images/(复数)=RGB,  image/(单数)=thermal
#   fold2: images/=RGB,         images_ir/=thermal
FOLD_RGB = {
    1: Path("/mnt/topic2_datasets/fire_loco_fold1/images/val"),
    2: Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val"),
}
FOLD_THERMAL = {
    1: Path("/mnt/topic2_datasets/fire_loco_fold1/image/val"),           # 单数 = thermal
    2: Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images_ir/val"),
}
FOLD_LBL = {
    1: Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val"),
    2: Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val"),
}
FOLD_CAT_IDS = {1: [0, 1, 2], 2: [0, 1]}  # fold2 无 person 类
QUAL_CSV = Path("/mnt/topic2_workspace/runs/f6_evidence_package_20260613/qualitative_cases.csv")

TAU_OVERLAP = 0.7
TAU_DUAL = 0.05
DUAL_PREFILTER = 0.01
SEED = 42

OUT_ROOT = Path("/mnt/topic2_workspace/04_figures/figure_outputs/server_export_20260617")
FIG5_DIR = OUT_ROOT / "fig5_export"
TILE_DIR = OUT_ROOT / "method_panelA_tiles"
GA_DIR = OUT_ROOT / "ga_anchor"

BOX_COLOR = {"GT": "#2ECC71", "RGB-only": "#3498DB", "Dual": "#E74C3C", "GateA": "#F1C40F"}

# ─── Task B/C 代表帧(Explore 已像素核确认存在)───
TILES = [
    # (dataset 名, RGB src, thermal src, thermal 模态标签)
    ("fireman",
     "/mnt/topic2_datasets/FireMan-UAV-RGBT/Binary/Binary/Multimodal/rgbt/val/Fire/dji_video_004/dji_video_004_rgb/frame_val_mm_fire_0031.jpg",
     "/mnt/topic2_datasets/FireMan-UAV-RGBT/Binary/Binary/Multimodal/rgbt/val/Fire/dji_video_004/dji_video_004_thermal/frame_val_mm_fire_0031.jpg",
     "LWIR"),
    ("rgbt3m",
     "/mnt/topic2_datasets/RGBT-3M/extracted/RGBTUAVwildfire/RGBT-3M-converted/images/train/video1_frame_01764.jpg",
     "/mnt/topic2_datasets/RGBT-3M/extracted/RGBTUAVwildfire/RGBT-3M-converted/image/train/video1_frame_01764.jpg",  # 单数=thermal
     "LWIR"),
    ("jag",
     "/mnt/topic2_datasets/JAG2023_RGBT_Wildfire/converted/images/train/291_gt.png",
     "/mnt/topic2_datasets/JAG2023_RGBT_Wildfire/converted/images_nir/train/291_gt.png",
     "NIR"),  # JAG 是 NIR 非 LWIR,caption 要精准
    ("flame3_sycan",
     "/mnt/topic2_datasets/FLAME3/CVSubset/FLAME 3 CV Dataset (Sycan Marsh)/Fire/RGB/Raw/00138.JPG",
     "/mnt/topic2_datasets/FLAME3/CVSubset/FLAME 3 CV Dataset (Sycan Marsh)/Fire/Thermal/Raw JPG/00138.JPG",
     "LWIR"),
]
GA_ANCHOR = (
    "fireman",
    "/mnt/topic2_datasets/FireMan-UAV-RGBT/Binary/Binary/Multimodal/rgbt/val/Fire/dji_video_004/dji_video_004_rgb/frame_val_mm_fire_0050.jpg",
    "/mnt/topic2_datasets/FireMan-UAV-RGBT/Binary/Binary/Multimodal/rgbt/val/Fire/dji_video_004/dji_video_004_thermal/frame_val_mm_fire_0050.jpg",
    "LWIR",
)


# ═══ 复制自 run_f6_evidence_package_20260613.py(locked params 不变,纯函数)═══

def load_predictions_jsonl(path):
    """加载 JSONL predictions,返回 {img_id: np.array(N,6)},6=xyxy+conf+cls。"""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["img_id"]] = (
                np.array(rec["boxes"], dtype=np.float64) if rec["boxes"] else np.zeros((0, 6))
            )
    return preds


def compute_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, :4].astype(np.float64)
    b = boxes_b[:, :4].astype(np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-10)


def apply_gateA_add_only(rgb_boxes, dual_boxes, tau_overlap=TAU_OVERLAP, tau_dual=TAU_DUAL):
    """GateA add-only:保留全部 RGB boxes,只追加不重叠(conf>=tau_dual 且 IoU<tau_overlap)的 dual box。
    返回 (merged(N,6), n_admitted, n_dual_total)。merged = [rgb_boxes..., admitted_duals...]。"""
    n_dual_total = len(dual_boxes)
    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total
    output = list(rgb_boxes)
    n_admitted = 0
    for d in dual_boxes:
        d_cls = int(d[5])
        d_conf = d[4]
        if d_conf < tau_dual:
            continue
        same_cls = [i for i in range(len(output)) if int(output[i][5]) == d_cls]
        if not same_cls:
            output.append(d)
            n_admitted += 1
            continue
        same_boxes = np.array([output[i][:4] for i in same_cls])
        ious = compute_iou_matrix(d[:4].reshape(1, 4), same_boxes)[0]
        if ious.max() < tau_overlap:
            output.append(d)
            n_admitted += 1
    return (np.array(output) if output else np.zeros((0, 6))), n_admitted, n_dual_total


def compute_per_image_metrics(preds, gts_by_img, img_ids, cat_ids):
    """对每张图算 TP/FP/FN(IoU=0.5,与 catalog 同口径)。返回 {img_id: {total_tp/fp/fn, n_pred, n_gt}}。"""
    per_img = {}
    for img_id in img_ids:
        gt_list = gts_by_img.get(img_id, [])
        boxes = preds.get(img_id, np.zeros((0, 6)))
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        matched = set()
        if len(boxes) > 0:
            order = np.argsort(-boxes[:, 4])
            boxes = boxes[order]
        for box in boxes:
            cls = int(box[5])
            px1, py1, px2, py2 = box[:4]
            best_iou = 0.5
            best_gi = -1
            for gi, gt in enumerate(gt_list):
                if gi in matched or gt["category_id"] != cls:
                    continue
                gx1, gy1, gx2, gy2 = gt["bbox_xyxy"]
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_p = (px2 - px1) * (py2 - py1)
                area_g = (gx2 - gx1) * (gy2 - gy1)
                union = area_p + area_g - inter
                iou = inter / max(union, 1e-10)
                if iou >= best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0:
                tp[cls] += 1
                matched.add(best_gi)
            else:
                fp[cls] += 1
        for gi, gt in enumerate(gt_list):
            if gi not in matched:
                fn[gt["category_id"]] += 1
        per_img[img_id] = {
            "total_tp": sum(tp.values()),
            "total_fp": sum(fp.values()),
            "total_fn": sum(fn.values()),
            "n_pred": len(boxes),
            "n_gt": len(gt_list),
        }
    return per_img


# ═══ 本脚本独有 ═══

def load_gt_single(img_id, lbl_dir, W, H):
    """解析单帧 YOLO txt → [{category_id, bbox_xyxy}]。缺文件返回 [](硬红线 2:缺 GT 会在反核时暴露)。"""
    lbl = lbl_dir / (img_id + ".txt")
    gts = []
    if not lbl.exists():
        return gts
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        p = line.split()
        cls = int(float(p[0]))
        xc, yc, nw, nh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        pw, ph = nw * W, nh * H
        px, py = (xc - nw / 2) * W, (yc - nh / 2) * H
        gts.append({"category_id": cls, "bbox_xyxy": [px, py, px + pw, py + ph]})
    return gts


def draw_boxes(ax, boxes, color, lw=1.0):
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=lw, edgecolor=color, facecolor="none"))


def render_boxes_png(rgb_arr, sets, titles, case_id, why, out_path):
    """2×2 RGB 四视图:GT / RGB-only pred / Dual pred / GateA output。GateA 子图标 admitted dual ⊕。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, name in zip(axes.flat, ["GT", "RGB-only", "Dual", "GateA"]):
        ax.imshow(rgb_arr)
        boxes = sets[name]
        lw = 0.7 if name == "Dual" else 1.2  # Dual FP 多,细线免糊
        draw_boxes(ax, boxes, BOX_COLOR[name], lw=lw)
        if name == "GateA" and len(sets["_admitted"]) > 0:
            adm = sets["_admitted"]
            cx = (adm[:, 0] + adm[:, 2]) / 2
            cy = (adm[:, 1] + adm[:, 3]) / 2
            ax.plot(cx, cy, "w+", markersize=9, mew=2)  # ⊕ 标被 admit 的 dual box
        ax.set_title(titles[name], fontsize=10, color=BOX_COLOR[name], fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{case_id}   |   {why}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_frame(src, dst, max_side=None):
    """导帧 PNG。max_side 给定时 thumbnail 缩(仅 tile/GA,fig5 保留原尺寸供 box 对齐)。"""
    im = Image.open(src).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(dst)


def main():
    for d in (FIG5_DIR, TILE_DIR, GA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ─── 加载 fold1/fold2 seed42 预测(零重推,硬红线 1)───
    preds = {}
    for fold in (1, 2):
        rgb_path = JSONL_F5 / f"predictions_fold{fold}_seed{SEED}_rgb.jsonl"
        dual_path = JSONL_F5 / f"predictions_fold{fold}_seed{SEED}_dual.jsonl"
        if not rgb_path.exists() or not dual_path.exists():
            raise RuntimeError(f"硬红线 2:预测 JSONL 缺失 rgb={rgb_path} dual={dual_path}")
        rgb_p = load_predictions_jsonl(rgb_path)
        dual_raw = load_predictions_jsonl(dual_path)
        # dual 预过滤 conf>=0.01(与 catalog 同口径)
        dual_p = {}
        for k, v in dual_raw.items():
            dual_p[k] = v[v[:, 4] >= DUAL_PREFILTER] if len(v) > 0 else v
        preds[fold] = {"rgb": rgb_p, "dual": dual_p}
        print(f"[load] fold{fold} seed{SEED}: rgb={len(rgb_p)} dual={len(dual_p)} imgs")

    # ─── 读 catalog 11 case(权威 box 计数源)───
    with open(QUAL_CSV) as f:
        cases = list(csv.DictReader(f))
    print(f"[catalog] {len(cases)} cases from {QUAL_CSV.name}")

    manifest = []
    failures = []
    exported_rgb_frames = set()  # img_id → 已导裸帧(frame_000296 重用避免重复导)

    # ═══ Task A — Fig5:11 case ═══
    print("\n=== Task A: Fig5 11 case ===")
    for c in cases:
        case_id = c["case_id"]
        fold = int(c["fold"])
        img_id = c["image_id"]
        why = c["why_selected"]
        cat_ids = FOLD_CAT_IDS[fold]

        # 取帧(硬红线 3:按目录名)
        rgb_src = FOLD_RGB[fold] / (img_id + ".jpg")
        th_src = FOLD_THERMAL[fold] / (img_id + ".jpg")
        if not rgb_src.exists():
            raise RuntimeError(f"硬红线 2:RGB 帧缺失 {rgb_src}")
        if not th_src.exists():
            raise RuntimeError(f"硬红线 3/2:thermal 帧缺失 {th_src}(模态目录:{FOLD_THERMAL[fold]})")

        with Image.open(rgb_src) as im:
            W, H = im.size
        rgb_arr = np.array(Image.open(rgb_src).convert("RGB"))

        # 取 box
        rgb_b = preds[fold]["rgb"].get(img_id, np.zeros((0, 6)))
        dual_b = preds[fold]["dual"].get(img_id, np.zeros((0, 6)))
        gt_list = load_gt_single(img_id, FOLD_LBL[fold], W, H)
        gate_merged, n_admit, _ = apply_gateA_add_only(rgb_b, dual_b)
        # admitted dual = merged 中 rgb 之后的(apply_gateA_add_only 先放全部 rgb 再追加 admit)
        n_rgb = len(rgb_b)
        admitted = gate_merged[n_rgb:] if n_rgb > 0 else gate_merged

        # ─── 反核(硬红线 4):重算 TP/FP/FN == catalog ───
        gts_by_img = {img_id: gt_list}
        m_rgb = compute_per_image_metrics({img_id: rgb_b}, gts_by_img, [img_id], cat_ids)[img_id]
        m_dual = compute_per_image_metrics({img_id: dual_b}, gts_by_img, [img_id], cat_ids)[img_id]
        m_gate = compute_per_image_metrics({img_id: gate_merged}, gts_by_img, [img_id], cat_ids)[img_id]

        checks = [
            ("rgb_tp", m_rgb["total_tp"], int(c["rgb_tp"])),
            ("rgb_fp", m_rgb["total_fp"], int(c["rgb_fp"])),
            ("rgb_fn", m_rgb["total_fn"], int(c["rgb_fn"])),
            ("dual_tp", m_dual["total_tp"], int(c["dual_tp"])),
            ("dual_fp", m_dual["total_fp"], int(c["dual_fp"])),
            ("dual_fn", m_dual["total_fn"], int(c["dual_fn"])),
            ("gate_tp", m_gate["total_tp"], int(c["gate_tp"])),
            ("gate_fp", m_gate["total_fp"], int(c["gate_fp"])),
            ("gate_fn", m_gate["total_fn"], int(c["gate_fn"])),
        ]
        mismatch = [(k, got, exp) for k, got, exp in checks if got != exp]
        status = "PASS" if not mismatch else "FAIL"
        print(f"  [{status}] {case_id}  fold{fold}  GT={len(gt_list)} rgb={len(rgb_b)} "
              f"dual={len(dual_b)} gate={len(gate_merged)} admit={n_admit}  | {why}")
        if mismatch:
            failures.append((case_id, mismatch))
            continue  # 反核不过不渲染该 case(硬红线 4)

        # 导裸帧(img_id 共享,000296 重用不重复导)
        rgb_png = FIG5_DIR / f"{img_id}__rgb.png"
        th_png = FIG5_DIR / f"{img_id}__thermal.png"
        if img_id not in exported_rgb_frames:
            save_frame(rgb_src, rgb_png)
            save_frame(th_src, th_png)
            exported_rgb_frames.add(img_id)

        # 渲染 boxes.png(case_id 命名,why 故事不同故每 case 独立)
        gt_arr = np.array([[*g["bbox_xyxy"], 1.0, g["category_id"]] for g in gt_list], dtype=float) if gt_list else np.zeros((0, 6))
        sets = {"GT": gt_arr, "RGB-only": rgb_b, "Dual": dual_b, "GateA": gate_merged, "_admitted": admitted}
        titles = {
            "GT": f"GT  n={len(gt_list)}",
            "RGB-only": f"RGB-only pred   TP={m_rgb['total_tp']} FP={m_rgb['total_fp']} FN={m_rgb['total_fn']}",
            "Dual": f"Dual pred   TP={m_dual['total_tp']} FP={m_dual['total_fp']} FN={m_dual['total_fn']}",
            "GateA": f"GateA output   TP={m_gate['total_tp']} FP={m_gate['total_fp']} FN={m_gate['total_fn']}   (+{n_admit} dual admit ⊕)",
        }
        boxes_png = FIG5_DIR / f"{case_id}__boxes.png"
        render_boxes_png(rgb_arr, sets, titles, case_id, why, boxes_png)

        # manifest 3 行(box 计数列填原始框数)
        box_counts = f"{len(gt_list)},{len(rgb_b)},{len(dual_b)},{len(gate_merged)}"
        manifest.append(f"fig5,{case_id},RGB,{rgb_src},{rgb_png.name},none,{box_counts}")
        manifest.append(f"fig5,{case_id},thermal,{th_src},{th_png.name},none,{box_counts}")
        manifest.append(f"fig5,{case_id},RGB,{rgb_src},{boxes_png.name},gt+rgb+dual+gate,{box_counts}")

    if failures:
        msg = "硬红线 4:box 反核与 catalog 不符(不擅改 catalog,停下回报):\n"
        for case_id, mm in failures:
            msg += f"  {case_id}: " + ", ".join(f"{k}(got={got},exp={exp})" for k, got, exp in mm) + "\n"
        raise RuntimeError(msg)

    # ═══ Task B — method Panel A:4 tile(裸帧,无 box)═══
    print("\n=== Task B: method Panel A 4 tile ===")
    for name, rgb_src, th_src, th_label in TILES:
        rgb_src, th_src = Path(rgb_src), Path(th_src)
        if not rgb_src.exists() or not th_src.exists():
            raise RuntimeError(f"硬红线 2:tile 帧缺失 {rgb_src} / {th_src}")
        rgb_png = TILE_DIR / f"{name}__rgb.png"
        th_png = TILE_DIR / f"{name}__thermal.png"
        save_frame(rgb_src, rgb_png, max_side=1280)
        save_frame(th_src, th_png, max_side=1280)
        manifest.append(f"method_panelA,{name},RGB,{rgb_src},{rgb_png.name},none,,,,")
        manifest.append(f"method_panelA,{name},{th_label},{th_src},{th_png.name},none,,,,")
        print(f"  [OK] {name} ({th_label})")

    # ═══ Task C — GA anchor:1 张(裸帧)═══
    print("\n=== Task C: GA anchor ===")
    name, rgb_src, th_src, th_label = GA_ANCHOR
    rgb_src, th_src = Path(rgb_src), Path(th_src)
    if not rgb_src.exists() or not th_src.exists():
        raise RuntimeError(f"硬红线 2:GA anchor 帧缺失 {rgb_src} / {th_src}")
    save_frame(rgb_src, GA_DIR / "ga_anchor__rgb.png", max_side=1280)
    save_frame(th_src, GA_DIR / "ga_anchor__thermal.png", max_side=1280)
    manifest.append(f"ga_anchor,{name},RGB,{rgb_src},ga_anchor__rgb.png,none,,,,")
    manifest.append(f"ga_anchor,{name},{th_label},{th_src},ga_anchor__thermal.png,none,,,,")
    print(f"  [OK] ga_anchor ({name}, {th_label})")

    # ─── manifest ───
    manifest_csv = OUT_ROOT / "export_manifest.csv"
    header = "group,case_id_or_tile,modality,src_path,export_png,box_source,gt_n,rgbonly_n,dual_n,gateA_n"
    with open(manifest_csv, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(manifest) + "\n")
    print(f"\n[manifest] {len(manifest)} rows → {manifest_csv}")

    # ─── 汇总 ───
    import subprocess
    n_fig5 = len(list(FIG5_DIR.glob("*.png")))
    n_tile = len(list(TILE_DIR.glob("*.png")))
    n_ga = len(list(GA_DIR.glob("*.png")))
    du = subprocess.run(["du", "-sh", str(OUT_ROOT)], capture_output=True, text=True).stdout.strip()
    print("\n" + "=" * 60)
    print(f"DONE. 反核 11 case 全 PASS.")
    print(f"fig5_export: {n_fig5} PNG | method_panelA_tiles: {n_tile} PNG | ga_anchor: {n_ga} PNG")
    print(f"manifest: {len(manifest)} rows")
    print(du)
    print("=" * 60)


if __name__ == "__main__":
    main()
