#!/usr/bin/env python3
"""G1a SafeLateGate seed2024 eval — formal P0/P1 补齐。

用 seed2024 的 RGB-only 和 dual checkpoint 生成 prediction JSONL，
应用 locked GateA (tau_overlap=0.7, tau_dual=0.05, add-only)，
用 fast AP50 评估，结果 append 到 gate_ablation_recount.csv。

不需要训练，只需推理 + gate sweep。
"""

import sys
import os
import json
import csv
import time
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# ─── 路径设置 ───
sys.path.insert(0, "/mnt/topic2_workspace/engineering_packs/MutilModel_199099010")
from ultralytics import YOLOMM

OUT_DIR = Path("/mnt/topic2_workspace/runs/formal_p0_p1_targeted_20260612")
LOG_DIR = OUT_DIR / "logs"
RUNS = Path("/mnt/topic2_workspace/runs")

# 数据路径
FOLD2_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val")
FOLD2_IR_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/image/val")
FOLD2_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val")

FOLD1_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/images/val")
FOLD1_IR_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/image/val")
FOLD1_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val")

# seed2024 checkpoint
SEED = 2024
CKPTS = {
    "fold2": {
        "rgb": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbonly_fold2_seed2024/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbsafe_dual_fold2_seed2024/weights/best.pt",
    },
    "fold1": {
        "rgb": "/mnt/topic2_workspace/runs/phase_f2_rgbonly_fold1_seed2024/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f2_dual_fold1_seed2024/weights/best.pt",
    },
}

# locked GateA 参数
GATEA_PARAMS = {"tau_overlap": 0.7, "tau_dual": 0.05}

CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
FOLD2_CAT_IDS = [0, 1]
FOLD1_CAT_IDS = [0, 1, 2]

# ─── 日志 ───

def setup_logging():
    handlers = [
        logging.FileHandler(LOG_DIR / "g1a_seed2024.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("g1a_seed2024")

log = setup_logging()

# ─── 数据加载 ───

def load_val_pairs(rgb_dir, ir_dir=None):
    pairs = []
    rgb_files = sorted([f for f in rgb_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        ir_path = None
        if ir_dir is not None:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = ir_dir / (stem + ext)
                if candidate.exists():
                    ir_path = candidate
                    break
        with Image.open(rgb_path) as img:
            W, H = img.size
        pairs.append({"img_id": stem, "rgb": rgb_path, "ir": ir_path, "W": W, "H": H})
    log.info(f"加载 {len(pairs)} 张 val 图片 ({rgb_dir.name})")
    return pairs


def load_ground_truths(lbl_dir, pairs, cat_ids):
    gts = []
    gts_by_img = defaultdict(list)
    for pair in pairs:
        lbl_path = lbl_dir / (pair["img_id"] + ".txt")
        if not lbl_path.exists():
            continue
        W, H = pair["W"], pair["H"]
        for line in lbl_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            xc, yc, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            pw, ph = nw * W, nh * H
            px, py = (xc - nw / 2) * W, (yc - nh / 2) * H
            gts.append({
                "image_id": pair["img_id"],
                "category_id": cls,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
            })
            gts_by_img[pair["img_id"]].append({
                "category_id": cls,
                "bbox_xyxy": [px, py, px + pw, py + ph],
            })
    log.info(f"GT: {len(gts)} boxes, {len(pairs)} imgs, cat_ids={cat_ids}")
    return gts, gts_by_img, [p["img_id"] for p in pairs], cat_ids


# ─── 预测 ───

def run_predictions(ckpt_path, pairs, is_rgb_only, label):
    log.info(f"[{label}] 加载模型: {ckpt_path}")
    model = YOLOMM(ckpt_path)
    preds_dict = {}
    n_zero = 0
    t0 = time.time()

    for i, pair in enumerate(pairs):
        rgb = str(pair["rgb"])
        x_src = rgb if is_rgb_only else (str(pair["ir"]) if pair["ir"] else None)
        if x_src is None:
            continue

        results = model.predict(rgb_source=rgb, x_source=x_src, conf=0.001, iou=0.7, max_det=300, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            preds_dict[pair["img_id"]] = results[0].boxes.copy()
        else:
            preds_dict[pair["img_id"]] = np.zeros((0, 6), dtype=np.float64)
            n_zero += 1

        if (i + 1) % 400 == 0:
            log.info(f"  [{label}] {i+1}/{len(pairs)} ({time.time()-t0:.1f}s)")

    log.info(f"[{label}] 完成: {len(preds_dict)} 张, {n_zero} 零检测, {time.time()-t0:.1f}s")
    return preds_dict


def save_predictions_jsonl(preds_dict, path):
    with open(path, "w") as f:
        for img_id in sorted(preds_dict.keys()):
            boxes = preds_dict[img_id]
            f.write(json.dumps({"img_id": img_id, "boxes": boxes.tolist() if len(boxes) > 0 else []}) + "\n")
    log.info(f"保存: {path} ({len(preds_dict)} 行)")


# ─── Fast AP50 ───

def fast_ap50_eval(preds_dict, gts_by_img, img_ids, cat_ids):
    per_class_ap50 = {}
    for cat_id in cat_ids:
        all_scores = []
        all_tp = []
        n_gt_total = 0

        for img_id in img_ids:
            gt_list = [g for g in gts_by_img.get(img_id, []) if g["category_id"] == cat_id]
            gt_xyxy = np.array([g["bbox_xyxy"] for g in gt_list], dtype=np.float64) if gt_list else np.zeros((0, 4))
            gt_matched = np.zeros(len(gt_list), dtype=bool)
            n_gt_total += len(gt_list)

            boxes = preds_dict.get(img_id, np.zeros((0, 6)))
            if len(boxes) == 0:
                continue
            cls_mask = boxes[:, 5].astype(int) == cat_id
            cls_boxes = boxes[cls_mask]
            if len(cls_boxes) == 0:
                continue

            order = np.argsort(-cls_boxes[:, 4])
            cls_boxes = cls_boxes[order]

            for pred in cls_boxes:
                px1, py1, px2, py2 = pred[:4]
                score = pred[4]

                if len(gt_xyxy) == 0:
                    all_scores.append(score)
                    all_tp.append(False)
                    continue

                ix1 = np.maximum(px1, gt_xyxy[:, 0])
                iy1 = np.maximum(py1, gt_xyxy[:, 1])
                ix2 = np.minimum(px2, gt_xyxy[:, 2])
                iy2 = np.minimum(py2, gt_xyxy[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                area_p = (px2 - px1) * (py2 - py1)
                area_g = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
                union = area_p + area_g - inter
                ious = inter / np.maximum(union, 1e-10)

                best_idx = -1
                best_iou = 0.5
                for gi in range(len(gt_matched)):
                    if gt_matched[gi]:
                        continue
                    if ious[gi] >= best_iou:
                        best_iou = ious[gi]
                        best_idx = gi

                if best_idx >= 0:
                    all_scores.append(score)
                    all_tp.append(True)
                    gt_matched[best_idx] = True
                else:
                    all_scores.append(score)
                    all_tp.append(False)

        if n_gt_total == 0:
            per_class_ap50[cat_id] = 0.0
            continue
        if not all_scores:
            per_class_ap50[cat_id] = 0.0
            continue

        scores = np.array(all_scores)
        tp = np.array(all_tp, dtype=np.float64)
        order = np.argsort(-scores, kind='mergesort')
        tp = tp[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / n_gt_total
        precision = tp_cum / (tp_cum + fp_cum)

        ap = 0.0
        for r_thresh in np.linspace(0, 1, 101):
            mask = recall >= r_thresh
            if mask.any():
                ap += precision[mask].max()
        ap /= 101
        per_class_ap50[cat_id] = float(ap)

    present = [c for c in cat_ids if c in per_class_ap50]
    per_class_ap50["mAP50"] = float(np.mean([per_class_ap50[c] for c in present])) if present else 0.0
    return per_class_ap50


# ─── Gate A (add-only) ───

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


def apply_gateA(rgb_boxes, dual_boxes, tau_overlap, tau_dual):
    """GateA add-only: dual box 加入当且仅当 conf >= tau_dual 且无同类 RGB box IoU >= tau_overlap。"""
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0

    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total

    output = list(rgb_boxes)
    is_original_rgb = [True] * len(rgb_boxes)

    for d in dual_boxes:
        d_cls = int(d[5])
        d_conf = d[4]

        if d_conf < tau_dual:
            continue

        same_cls_indices = [i for i in range(len(output))
                           if int(output[i][5]) == d_cls and is_original_rgb[i]]

        if not same_cls_indices:
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1
            continue

        same_cls_boxes = np.array([output[i][:4] for i in same_cls_indices])
        d_box = d[:4].reshape(1, 4)
        ious = compute_iou_matrix(d_box, same_cls_boxes)[0]
        max_iou = ious.max()

        if max_iou < tau_overlap:
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1

    return np.array(output) if output else np.zeros((0, 6)), n_dual_accepted, n_dual_total


# ─── Main ───

def main():
    t_start = time.time()
    log.info("=" * 60)
    log.info("G1a seed2024 eval — locked GateA (tau_overlap=0.7, tau_dual=0.05)")
    log.info("=" * 60)

    # 验证 checkpoint
    for fold, ckpts in CKPTS.items():
        for mod, path in ckpts.items():
            if not Path(path).exists():
                log.error(f"CHECKPOINT MISSING: {path}")
                log.error("G1A_SEED2024_BLOCKED_MISSING_INPUTS")
                # 写 blocker 到 anomalies
                with open(OUT_DIR / "anomalies.md", "a") as f:
                    f.write(f"\n\n## G1a seed2024 BLOCKED\n")
                    f.write(f"G1A_SEED2024_BLOCKED_MISSING_INPUTS: {path}\n")
                return
            log.info(f"  OK: {fold}/{mod} -> {path}")

    gate_results = []

    for fold_name, (rgb_dir, ir_dir, lbl_dir, cat_ids) in {
        "fold2": (FOLD2_RGB_DIR, FOLD2_IR_DIR, FOLD2_LBL_DIR, FOLD2_CAT_IDS),
        "fold1": (FOLD1_RGB_DIR, FOLD1_IR_DIR, FOLD1_LBL_DIR, FOLD1_CAT_IDS),
    }.items():
        log.info(f"\n{'='*40}")
        log.info(f"处理 {fold_name} seed={SEED}")
        log.info(f"{'='*40}")

        # 加载数据
        pairs = load_val_pairs(rgb_dir, ir_dir)
        gts, gts_by_img, img_ids, cat_ids = load_ground_truths(lbl_dir, pairs, cat_ids)

        # 预测 RGB-only
        rgb_preds = run_predictions(CKPTS[fold_name]["rgb"], pairs, True, f"{fold_name}-RGB")
        # 预测 Dual
        dual_preds = run_predictions(CKPTS[fold_name]["dual"], pairs, False, f"{fold_name}-Dual")

        # 保存 JSONL
        jsonl_dir = OUT_DIR / "g1a_predictions"
        jsonl_dir.mkdir(exist_ok=True)
        save_predictions_jsonl(rgb_preds, jsonl_dir / f"predictions_{fold_name}_seed{SEED}_rgb.jsonl")
        save_predictions_jsonl(dual_preds, jsonl_dir / f"predictions_{fold_name}_seed{SEED}_dual.jsonl")

        # RGB-only baseline (fast AP50)
        rgb_ap = fast_ap50_eval(rgb_preds, gts_by_img, img_ids, cat_ids)
        log.info(f"  RGB-only mAP50={rgb_ap['mAP50']:.6f}")

        gate_results.append({
            "source": "G1a", "method": "P0_rgb_only", "fold": int(fold_name[-1]),
            "seed": SEED, "tau_overlap": 0, "tau_dual": 0,
            "AP50": rgb_ap["mAP50"],
            "smoke_AP50": rgb_ap.get(0, 0),
            "fire_AP50": rgb_ap.get(1, 0),
            "person_AP50": rgb_ap.get(2, 0),
            "delta_AP50_vs_rgb": 0.0,
            "acceptance_ratio": 0.0,
            "notes": "RGB-only baseline"
        })

        # Dual-only
        dual_ap = fast_ap50_eval(dual_preds, gts_by_img, img_ids, cat_ids)
        log.info(f"  Dual-only mAP50={dual_ap['mAP50']:.6f} delta={dual_ap['mAP50'] - rgb_ap['mAP50']:+.6f}")

        gate_results.append({
            "source": "G1a", "method": "P1_dual_only", "fold": int(fold_name[-1]),
            "seed": SEED, "tau_overlap": 0, "tau_dual": 0,
            "AP50": dual_ap["mAP50"],
            "smoke_AP50": dual_ap.get(0, 0),
            "fire_AP50": dual_ap.get(1, 0),
            "person_AP50": dual_ap.get(2, 0),
            "delta_AP50_vs_rgb": dual_ap["mAP50"] - rgb_ap["mAP50"],
            "acceptance_ratio": 1.0,
            "notes": "Dual-only"
        })

        # GateA locked
        to = GATEA_PARAMS["tau_overlap"]
        td = GATEA_PARAMS["tau_dual"]
        merged_preds = {}
        total_accepted = 0
        total_dual_count = 0

        for img_id in sorted(img_ids):
            rgb_b = rgb_preds.get(img_id, np.zeros((0, 6)))
            dual_b = dual_preds.get(img_id, np.zeros((0, 6)))
            merged, n_acc, n_tot = apply_gateA(rgb_b, dual_b, to, td)
            merged_preds[img_id] = merged
            total_accepted += n_acc
            total_dual_count += n_tot

        gate_ap = fast_ap50_eval(merged_preds, gts_by_img, img_ids, cat_ids)
        accept_ratio = total_accepted / max(total_dual_count, 1)

        log.info(f"  GateA mAP50={gate_ap['mAP50']:.6f} delta={gate_ap['mAP50'] - rgb_ap['mAP50']:+.6f} "
                 f"smoke={gate_ap.get(0, 0):.4f} accept={accept_ratio:.4f}")

        gate_results.append({
            "source": "G1a", "method": "GateA", "fold": int(fold_name[-1]),
            "seed": SEED, "tau_overlap": to, "tau_dual": td,
            "AP50": gate_ap["mAP50"],
            "smoke_AP50": gate_ap.get(0, 0),
            "fire_AP50": gate_ap.get(1, 0),
            "person_AP50": gate_ap.get(2, 0),
            "delta_AP50_vs_rgb": gate_ap["mAP50"] - rgb_ap["mAP50"],
            "acceptance_ratio": accept_ratio,
            "notes": f"GateA locked: tau_overlap={to}, tau_dual={td}"
        })

    # 写结果到 gate_ablation_recount.csv (append)
    gate_csv = OUT_DIR / "gate_ablation_recount.csv"
    gate_fields = ["source", "method", "fold", "seed", "tau_overlap", "tau_dual",
                   "AP50", "smoke_AP50", "fire_AP50", "person_AP50",
                   "delta_AP50_vs_rgb", "acceptance_ratio", "notes"]

    # 读取已有行
    existing = []
    if gate_csv.exists():
        with open(gate_csv, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    # 追加 seed2024 行
    all_rows = existing + gate_results

    with open(gate_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=gate_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - t_start
    log.info(f"\n{'='*60}")
    log.info(f"G1a seed2024 eval 完成 ({elapsed:.1f}s)")
    for r in gate_results:
        log.info(f"  {r['method']} fold{r['fold']} seed{r['seed']}: "
                 f"AP50={r['AP50']:.6f} delta={r['delta_AP50_vs_rgb']:+.6f}")
    log.info(f"已 append {len(gate_results)} 行到 {gate_csv}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
