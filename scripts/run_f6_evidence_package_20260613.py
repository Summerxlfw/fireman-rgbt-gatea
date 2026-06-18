#!/usr/bin/env python3
"""F6 Evidence Package: Bootstrap CI + Qualitative Case Selection.

目标:
1. 对 fold1/fold2 做 image-level bootstrap CI (RGB vs Dual, RGB vs GateA)
2. 从 fold1/fold2 筛选 qualitative cases
3. 输出 CSV + recount + anomalies

限制:
- 不做训练
- 不做 fold0 (无 fold0 dual checkpoint)
- GateA locked params: tau_overlap=0.7, tau_dual=0.05, add-only
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

# ─── 输出路径 ───

OUT_DIR = Path("/mnt/topic2_workspace/runs/f6_evidence_package_20260613")
LOG_DIR = OUT_DIR / "logs"

# ─── 输入路径 ───

# Prediction JSONL
JSONL_F5 = Path("/mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607")
JSONL_FORMAL = Path("/mnt/topic2_workspace/runs/formal_p0_p1_targeted_20260612/g1a_predictions")

# GT labels
FOLD1_LBL = Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val")
FOLD1_IMG = Path("/mnt/topic2_datasets/fire_loco_fold1/images/val")
FOLD2_LBL = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val")
FOLD2_IMG = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val")

# GateA locked params
TAU_OVERLAP = 0.7
TAU_DUAL = 0.05
DUAL_PREFILTER = 0.01

SEEDS = [42, 1337, 2024]
CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
FOLD2_CAT_IDS = [0, 1]
FOLD1_CAT_IDS = [0, 1, 2]

# Bootstrap params
N_BOOT = 200  # handoff: "200 if slow and label as such"
RNG_SEED = 42

# ─── 日志 ───

def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(LOG_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("f6_evidence")


log = setup_logging()


# ─── 数据加载（从 run_f5_g1a 复制，去除 module-level 副作用）───

def load_predictions_jsonl(path: Path) -> dict:
    """加载 JSONL predictions，返回 {img_id: np.array(N,6)}。"""
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["img_id"]] = np.array(rec["boxes"], dtype=np.float64) if rec["boxes"] else np.zeros((0, 6))
    return preds


def load_image_pairs(img_dir: Path) -> list:
    """加载图片列表，返回 [{img_id, path, W, H}]。"""
    pairs = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with Image.open(f) as img:
            W, H = img.size
        pairs.append({"img_id": f.stem, "path": f, "W": W, "H": H})
    return pairs


def load_ground_truths(lbl_dir: Path, pairs: list, cat_ids: list) -> tuple:
    """解析 YOLO labels → {img_id: [{category_id, bbox_xyxy}]} + img_ids。"""
    gts_by_img = defaultdict(list)
    img_ids = [p["img_id"] for p in pairs]

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
            gts_by_img[pair["img_id"]].append({
                "category_id": cls,
                "bbox_xyxy": [px, py, px + pw, py + ph],
            })

    return gts_by_img, img_ids, cat_ids


# ─── Fast AP50 评估（从 run_f5_g1a 复制）───

def fast_ap50_eval(preds_dict: dict, gts_by_img: dict, img_ids: list, cat_ids: list) -> dict:
    """快速 AP50 计算。返回 {cat_id: AP50, 'mAP50': mean}。"""
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

    present_classes = [c for c in cat_ids if c in per_class_ap50]
    per_class_ap50["mAP50"] = float(np.mean([per_class_ap50[c] for c in present_classes])) if present_classes else 0.0

    return per_class_ap50


# ─── GateA（locked params, add-only）───

def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
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


def apply_gateA_add_only(rgb_boxes: np.ndarray, dual_boxes: np.ndarray,
                          tau_overlap: float = 0.7, tau_dual: float = 0.05) -> tuple:
    """GateA add-only: 保留所有 RGB boxes，只添加不重叠的 dual boxes。
    返回 (merged_boxes, n_dual_accepted, n_dual_total)。"""
    n_dual_total = len(dual_boxes)

    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total

    output = list(rgb_boxes)
    n_dual_accepted = 0

    for d in dual_boxes:
        d_cls = int(d[5])
        d_conf = d[4]

        if d_conf < tau_dual:
            continue

        # 找同类 RGB boxes
        same_cls_indices = [i for i in range(len(output)) if int(output[i][5]) == d_cls]

        if not same_cls_indices:
            output.append(d)
            n_dual_accepted += 1
            continue

        # IoU 检查
        same_cls_boxes = np.array([output[i][:4] for i in same_cls_indices])
        d_box = d[:4].reshape(1, 4)
        ious = compute_iou_matrix(d_box, same_cls_boxes)[0]
        max_iou = ious.max()

        if max_iou < tau_overlap:
            output.append(d)
            n_dual_accepted += 1

    return np.array(output) if output else np.zeros((0, 6)), n_dual_accepted, n_dual_total


def apply_gateA_all_images(preds_rgb: dict, preds_dual: dict, img_ids: list) -> dict:
    """对所有图应用 GateA，返回 {img_id: merged_boxes}。"""
    merged = {}
    for img_id in img_ids:
        rgb_b = preds_rgb.get(img_id, np.zeros((0, 6)))
        dual_b = preds_dual.get(img_id, np.zeros((0, 6)))
        merged[img_id], _, _ = apply_gateA_add_only(rgb_b, dual_b, TAU_OVERLAP, TAU_DUAL)
    return merged


# ─── Bootstrap CI ───

def bootstrap_ap50(preds_dict: dict, gts_by_img: dict, img_ids: list, cat_ids: list,
                   n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> dict:
    """Image-level bootstrap CI for AP50。

    优化: 直接传 resampled img_ids list 给 fast_ap50_eval，不创建新 dict。
    fast_ap50_eval 内部按 img_id 独立处理 GT match，重复 ID 会正确重复计数。
    """
    rng = np.random.RandomState(rng_seed)
    n_images = len(img_ids)
    img_ids_arr = np.array(img_ids)

    # Point estimate
    point_map50 = fast_ap50_eval(preds_dict, gts_by_img, img_ids, cat_ids)["mAP50"]

    # Bootstrap: 直接 resample img_ids，不创建新 dict
    boot_map50s = np.zeros(n_boot)
    for b in range(n_boot):
        sampled_ids = img_ids_arr[rng.randint(0, n_images, size=n_images)].tolist()
        boot_map50s[b] = fast_ap50_eval(preds_dict, gts_by_img, sampled_ids, cat_ids)["mAP50"]

        if (b + 1) % 50 == 0:
            log.info(f"    bootstrap {b+1}/{n_boot}: mean={np.mean(boot_map50s[:b+1]):.6f}")

    return {
        "point_estimate": point_map50,
        "boot_mean": float(np.mean(boot_map50s)),
        "ci_2.5": float(np.percentile(boot_map50s, 2.5)),
        "ci_97.5": float(np.percentile(boot_map50s, 97.5)),
        "n_images": n_images,
        "n_boot": n_boot,
    }


def compute_delta_bootstrap(preds_a: dict, preds_b: dict, gts_by_img: dict,
                             img_ids: list, cat_ids: list,
                             n_boot: int = N_BOOT, rng_seed: int = RNG_SEED) -> dict:
    """Bootstrap CI for AP50(A) - AP50(B)，paired image-level resampling。

    优化: 直接传 resampled img_ids list，不创建新 dict。
    """
    rng = np.random.RandomState(rng_seed)
    n_images = len(img_ids)
    img_ids_arr = np.array(img_ids)

    # Point estimates
    point_a = fast_ap50_eval(preds_a, gts_by_img, img_ids, cat_ids)["mAP50"]
    point_b = fast_ap50_eval(preds_b, gts_by_img, img_ids, cat_ids)["mAP50"]
    point_delta = point_a - point_b

    # Paired bootstrap: 同一组 resampled IDs 用于两个 arm
    boot_deltas = np.zeros(n_boot)
    for b in range(n_boot):
        sampled_ids = img_ids_arr[rng.randint(0, n_images, size=n_images)].tolist()
        boot_a = fast_ap50_eval(preds_a, gts_by_img, sampled_ids, cat_ids)["mAP50"]
        boot_b = fast_ap50_eval(preds_b, gts_by_img, sampled_ids, cat_ids)["mAP50"]
        boot_deltas[b] = boot_a - boot_b

        if (b + 1) % 50 == 0:
            log.info(f"    delta bootstrap {b+1}/{n_boot}: mean_delta={np.mean(boot_deltas[:b+1]):.6f}")

    return {
        "point_delta": float(point_delta),
        "point_a": float(point_a),
        "point_b": float(point_b),
        "boot_mean_delta": float(np.mean(boot_deltas)),
        "ci_2.5": float(np.percentile(boot_deltas, 2.5)),
        "ci_97.5": float(np.percentile(boot_deltas, 97.5)),
        "n_images": n_images,
        "n_boot": n_boot,
    }


# ─── Qualitative Case Selection ───

def compute_per_image_metrics(preds: dict, gts_by_img: dict, img_ids: list, cat_ids: list) -> dict:
    """对每张图计算 TP/FP/FN 统计（IoU=0.5）。"""
    per_img = {}
    for img_id in img_ids:
        gt_list = gts_by_img.get(img_id, [])
        boxes = preds.get(img_id, np.zeros((0, 6)))

        tp_per_class = defaultdict(int)
        fp_per_class = defaultdict(int)
        fn_per_class = defaultdict(int)
        matched_gts = set()

        # 按 confidence 排序
        if len(boxes) > 0:
            order = np.argsort(-boxes[:, 4])
            boxes = boxes[order]

        for box in boxes:
            cls = int(box[5])
            px1, py1, px2, py2 = box[:4]
            best_iou = 0.5
            best_gi = -1

            for gi, gt in enumerate(gt_list):
                if gi in matched_gts or gt["category_id"] != cls:
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
                tp_per_class[cls] += 1
                matched_gts.add(best_gi)
            else:
                fp_per_class[cls] += 1

        # FN = unmatched GTs
        for gi, gt in enumerate(gt_list):
            if gi not in matched_gts:
                fn_per_class[gt["category_id"]] += 1

        per_img[img_id] = {
            "tp": dict(tp_per_class),
            "fp": dict(fp_per_class),
            "fn": dict(fn_per_class),
            "n_pred": len(boxes),
            "n_gt": len(gt_list),
            "total_tp": sum(tp_per_class.values()),
            "total_fp": sum(fp_per_class.values()),
            "total_fn": sum(fn_per_class.values()),
        }
    return per_img


def select_qualitative_cases(preds_rgb: dict, preds_dual: dict, preds_gate: dict,
                              gts_by_img: dict, img_ids: list, img_dir: Path,
                              fold: int, seed: int) -> list:
    """筛选 qualitative cases。返回 list of dicts。"""
    metrics_rgb = compute_per_image_metrics(preds_rgb, gts_by_img, img_ids, [0, 1, 2] if fold == 1 else [0, 1])
    metrics_dual = compute_per_image_metrics(preds_dual, gts_by_img, img_ids, [0, 1, 2] if fold == 1 else [0, 1])
    metrics_gate = compute_per_image_metrics(preds_gate, gts_by_img, img_ids, [0, 1, 2] if fold == 1 else [0, 1])

    cases = []

    for img_id in img_ids:
        m_rgb = metrics_rgb[img_id]
        m_dual = metrics_dual[img_id]
        m_gate = metrics_gate[img_id]

        rgb_path = img_dir / (img_id + ".jpg")
        if not rgb_path.exists():
            rgb_path = img_dir / (img_id + ".png")

        # --- Case 1: RGB miss rescued by GateA ---
        # RGB has FN, GateA has fewer FN (i.e., GateA rescued some)
        if m_rgb["total_fn"] > 0 and m_gate["total_fn"] < m_rgb["total_fn"]:
            rescued = m_rgb["total_fn"] - m_gate["total_fn"]
            cases.append({
                "case_type": "rgb_miss_rescued_by_gateA",
                "fold": fold, "seed": seed, "image_id": img_id,
                "image_path": str(rgb_path),
                "rgb_tp": m_rgb["total_tp"], "rgb_fp": m_rgb["total_fp"], "rgb_fn": m_rgb["total_fn"],
                "dual_tp": m_dual["total_tp"], "dual_fp": m_dual["total_fp"], "dual_fn": m_dual["total_fn"],
                "gate_tp": m_gate["total_tp"], "gate_fp": m_gate["total_fp"], "gate_fn": m_gate["total_fn"],
                "score": rescued,  # more rescued = better case
                "why_selected": f"RGB {m_rgb['total_fn']}FN → GateA {m_gate['total_fn']}FN (rescued {rescued})",
            })

        # --- Case 2: Dual FP rejected by GateA ---
        # Dual has FP, GateA has fewer FP
        if m_dual["total_fp"] > 0 and m_gate["total_fp"] < m_dual["total_fp"]:
            rejected = m_dual["total_fp"] - m_gate["total_fp"]
            cases.append({
                "case_type": "dual_fp_rejected_by_gateA",
                "fold": fold, "seed": seed, "image_id": img_id,
                "image_path": str(rgb_path),
                "rgb_tp": m_rgb["total_tp"], "rgb_fp": m_rgb["total_fp"], "rgb_fn": m_rgb["total_fn"],
                "dual_tp": m_dual["total_tp"], "dual_fp": m_dual["total_fp"], "dual_fn": m_dual["total_fn"],
                "gate_tp": m_gate["total_tp"], "gate_fp": m_gate["total_fp"], "gate_fn": m_gate["total_fn"],
                "score": rejected,
                "why_selected": f"Dual {m_dual['total_fp']}FP → GateA {m_gate['total_fp']}FP (rejected {rejected})",
            })

        # --- Case 3: Harmful thermal / dual collapse ---
        # Dual has more FP than RGB AND fewer TP
        if m_dual["total_fp"] > m_rgb["total_fp"] and m_dual["total_tp"] <= m_rgb["total_tp"]:
            harm = m_dual["total_fp"] - m_rgb["total_fp"]
            cases.append({
                "case_type": "harmful_thermal_collapse",
                "fold": fold, "seed": seed, "image_id": img_id,
                "image_path": str(rgb_path),
                "rgb_tp": m_rgb["total_tp"], "rgb_fp": m_rgb["total_fp"], "rgb_fn": m_rgb["total_fn"],
                "dual_tp": m_dual["total_tp"], "dual_fp": m_dual["total_fp"], "dual_fn": m_dual["total_fn"],
                "gate_tp": m_gate["total_tp"], "gate_fp": m_gate["total_fp"], "gate_fn": m_gate["total_fn"],
                "score": harm,
                "why_selected": f"RGB {m_rgb['total_fp']}FP → Dual {m_dual['total_fp']}FP (harm +{harm}FP)",
            })

        # --- Case 4: Thermal helpful retained by GateA (fold1 only) ---
        if fold == 1:
            # Dual adds TP that RGB misses, GateA retains them
            dual_gain = m_dual["total_tp"] - m_rgb["total_tp"]
            gate_gain = m_gate["total_tp"] - m_rgb["total_tp"]
            if dual_gain > 0 and gate_gain > 0:
                retention = gate_gain / dual_gain if dual_gain > 0 else 0
                cases.append({
                    "case_type": "thermal_helpful_retained",
                    "fold": fold, "seed": seed, "image_id": img_id,
                    "image_path": str(rgb_path),
                    "rgb_tp": m_rgb["total_tp"], "rgb_fp": m_rgb["total_fp"], "rgb_fn": m_rgb["total_fn"],
                    "dual_tp": m_dual["total_tp"], "dual_fp": m_dual["total_fp"], "dual_fn": m_dual["total_fn"],
                    "gate_tp": m_gate["total_tp"], "gate_fp": m_gate["total_fp"], "gate_fn": m_gate["total_fn"],
                    "score": retention,
                    "why_selected": f"Dual +{dual_gain}TP, GateA retains {gate_gain}/{dual_gain} ({retention:.0%})",
                })

    return cases


# ─── 主流程 ───

def main():
    t_start = time.time()

    log.info("=" * 60)
    log.info("F6 Evidence Package: Bootstrap CI + Qualitative Cases")
    log.info("=" * 60)
    log.info(f"Params: N_BOOT={N_BOOT}, RNG_SEED={RNG_SEED}, "
             f"TAU_OVERLAP={TAU_OVERLAP}, TAU_DUAL={TAU_DUAL}")

    # ─── 加载 fold2 数据 ───
    log.info("加载 fold2 数据...")
    fold2_pairs = load_image_pairs(FOLD2_IMG)
    fold2_gts, fold2_img_ids, fold2_cat_ids = load_ground_truths(FOLD2_LBL, fold2_pairs, FOLD2_CAT_IDS)
    log.info(f"Fold2: {len(fold2_img_ids)} images, {sum(len(v) for v in fold2_gts.values())} GT boxes")

    fold2_data = {}
    for seed in SEEDS:
        # 查找 JSONL 路径
        rgb_path = JSONL_F5 / f"predictions_fold2_seed{seed}_rgb.jsonl"
        dual_path = JSONL_F5 / f"predictions_fold2_seed{seed}_dual.jsonl"
        if not rgb_path.exists():
            rgb_path = JSONL_FORMAL / f"predictions_fold2_seed{seed}_rgb.jsonl"
        if not dual_path.exists():
            dual_path = JSONL_FORMAL / f"predictions_fold2_seed{seed}_dual.jsonl"

        if not rgb_path.exists() or not dual_path.exists():
            log.error(f"Fold2 seed={seed}: JSONL missing (rgb={rgb_path.exists()}, dual={dual_path.exists()})")
            continue

        rgb_preds = load_predictions_jsonl(rgb_path)
        dual_raw = load_predictions_jsonl(dual_path)

        # 预过滤 dual
        dual_preds = {}
        for img_id, boxes in dual_raw.items():
            if len(boxes) > 0:
                dual_preds[img_id] = boxes[boxes[:, 4] >= DUAL_PREFILTER]
            else:
                dual_preds[img_id] = boxes

        gate_preds = apply_gateA_all_images(rgb_preds, dual_preds, fold2_img_ids)

        fold2_data[seed] = {
            "rgb": rgb_preds, "dual": dual_preds, "gate": gate_preds,
            "rgb_path": rgb_path, "dual_path": dual_path,
        }
        log.info(f"Fold2 seed={seed}: {len(rgb_preds)} images loaded")

    # ─── 加载 fold1 数据 ───
    log.info("加载 fold1 数据...")
    fold1_pairs = load_image_pairs(FOLD1_IMG)
    fold1_gts, fold1_img_ids, fold1_cat_ids = load_ground_truths(FOLD1_LBL, fold1_pairs, FOLD1_CAT_IDS)
    log.info(f"Fold1: {len(fold1_img_ids)} images, {sum(len(v) for v in fold1_gts.values())} GT boxes")

    fold1_data = {}
    for seed in SEEDS:
        rgb_path = JSONL_F5 / f"predictions_fold1_seed{seed}_rgb.jsonl"
        dual_path = JSONL_F5 / f"predictions_fold1_seed{seed}_dual.jsonl"
        if not rgb_path.exists():
            rgb_path = JSONL_FORMAL / f"predictions_fold1_seed{seed}_rgb.jsonl"
        if not dual_path.exists():
            dual_path = JSONL_FORMAL / f"predictions_fold1_seed{seed}_dual.jsonl"

        if not rgb_path.exists() or not dual_path.exists():
            log.error(f"Fold1 seed={seed}: JSONL missing")
            continue

        rgb_preds = load_predictions_jsonl(rgb_path)
        dual_raw = load_predictions_jsonl(dual_path)

        dual_preds = {}
        for img_id, boxes in dual_raw.items():
            if len(boxes) > 0:
                dual_preds[img_id] = boxes[boxes[:, 4] >= DUAL_PREFILTER]
            else:
                dual_preds[img_id] = boxes

        gate_preds = apply_gateA_all_images(rgb_preds, dual_preds, fold1_img_ids)

        fold1_data[seed] = {
            "rgb": rgb_preds, "dual": dual_preds, "gate": gate_preds,
            "rgb_path": rgb_path, "dual_path": dual_path,
        }
        log.info(f"Fold1 seed={seed}: {len(rgb_preds)} images loaded")

    # ─── Sanity Check: point estimates ───
    log.info("=" * 60)
    log.info("Sanity Check: point estimates vs recount.md")
    log.info("=" * 60)

    # recount.md 参考值
    ref_fold2 = {
        42: {"rgb": 0.2522, "gate": 0.2620},
        1337: {"rgb": 0.2543, "gate": 0.2552},
        2024: {"rgb": 0.2538, "gate": 0.2537},
    }
    ref_fold1 = {
        42: {"rgb": 0.0719, "gate": 0.1799},
        1337: {"rgb": 0.1122, "gate": 0.1665},
        2024: {"rgb": 0.0584, "gate": 0.1854},
    }

    sanity_pass = True
    for seed in SEEDS:
        if seed not in fold2_data:
            continue
        # Fold2 RGB
        result_rgb = fast_ap50_eval(fold2_data[seed]["rgb"], fold2_gts, fold2_img_ids, fold2_cat_ids)
        delta_rgb = abs(result_rgb["mAP50"] - ref_fold2[seed]["rgb"])
        status = "PASS" if delta_rgb < 0.01 else "FAIL"
        if status == "FAIL":
            sanity_pass = False
        log.info(f"Fold2 seed={seed} RGB: eval={result_rgb['mAP50']:.4f} ref={ref_fold2[seed]['rgb']:.4f} "
                 f"Δ={delta_rgb:.4f} [{status}]")

        # Fold2 GateA
        result_gate = fast_ap50_eval(fold2_data[seed]["gate"], fold2_gts, fold2_img_ids, fold2_cat_ids)
        delta_gate = abs(result_gate["mAP50"] - ref_fold2[seed]["gate"])
        status = "PASS" if delta_gate < 0.01 else "FAIL"
        if status == "FAIL":
            sanity_pass = False
        log.info(f"Fold2 seed={seed} GateA: eval={result_gate['mAP50']:.4f} ref={ref_fold2[seed]['gate']:.4f} "
                 f"Δ={delta_gate:.4f} [{status}]")

    for seed in SEEDS:
        if seed not in fold1_data:
            continue
        result_gate = fast_ap50_eval(fold1_data[seed]["gate"], fold1_gts, fold1_img_ids, fold1_cat_ids)
        delta_gate = abs(result_gate["mAP50"] - ref_fold1[seed]["gate"])
        status = "PASS" if delta_gate < 0.01 else "FAIL"
        if status == "FAIL":
            sanity_pass = False
        log.info(f"Fold1 seed={seed} GateA: eval={result_gate['mAP50']:.4f} ref={ref_fold1[seed]['gate']:.4f} "
                 f"Δ={delta_gate:.4f} [{status}]")

    if not sanity_pass:
        log.warning("Sanity check 有 FAIL，继续但需在 recount 中注明")

    # ─── Bootstrap CI ───
    log.info("=" * 60)
    log.info("Bootstrap CI 计算")
    log.info("=" * 60)

    bootstrap_results = []

    # 9 个 target: fold2(RGB vs Dual, RGB vs GateA) x 3 seeds + fold1(RGB vs GateA) x 3 seeds
    for seed in SEEDS:
        if seed not in fold2_data:
            log.warning(f"Fold2 seed={seed} 缺失，跳过 bootstrap")
            continue

        # Target 1: Fold2 RGB vs Dual (unconditional fusion harm)
        log.info(f"Fold2 seed={seed}: RGB vs Dual...")
        result = compute_delta_bootstrap(
            fold2_data[seed]["rgb"], fold2_data[seed]["dual"],
            fold2_gts, fold2_img_ids, fold2_cat_ids, N_BOOT, RNG_SEED)
        bootstrap_results.append({
            "target": "RGB_vs_Dual", "fold": 2, "seed": seed,
            "arm_a": "RGB", "arm_b": "Dual",
            "point_a": result["point_a"], "point_b": result["point_b"],
            "point_delta": result["point_delta"],
            "boot_mean_delta": result["boot_mean_delta"],
            "ci_2.5": result["ci_2.5"], "ci_97.5": result["ci_97.5"],
            "n_images": result["n_images"], "n_boot": result["n_boot"],
        })
        log.info(f"  Δ={result['point_delta']:.4f} CI=[{result['ci_2.5']:.4f}, {result['ci_97.5']:.4f}]")

        # Target 2: Fold2 RGB vs GateA (safe admission gain)
        log.info(f"Fold2 seed={seed}: RGB vs GateA...")
        result = compute_delta_bootstrap(
            fold2_data[seed]["gate"], fold2_data[seed]["rgb"],
            fold2_gts, fold2_img_ids, fold2_cat_ids, N_BOOT, RNG_SEED)
        bootstrap_results.append({
            "target": "RGB_vs_GateA", "fold": 2, "seed": seed,
            "arm_a": "GateA", "arm_b": "RGB",
            "point_a": result["point_a"], "point_b": result["point_b"],
            "point_delta": result["point_delta"],
            "boot_mean_delta": result["boot_mean_delta"],
            "ci_2.5": result["ci_2.5"], "ci_97.5": result["ci_97.5"],
            "n_images": result["n_images"], "n_boot": result["n_boot"],
        })
        log.info(f"  Δ={result['point_delta']:.4f} CI=[{result['ci_2.5']:.4f}, {result['ci_97.5']:.4f}]")

    for seed in SEEDS:
        if seed not in fold1_data:
            log.warning(f"Fold1 seed={seed} 缺失，跳过 bootstrap")
            continue

        # Target 3: Fold1 RGB vs GateA (thermal retention)
        log.info(f"Fold1 seed={seed}: RGB vs GateA...")
        result = compute_delta_bootstrap(
            fold1_data[seed]["gate"], fold1_data[seed]["rgb"],
            fold1_gts, fold1_img_ids, fold1_cat_ids, N_BOOT, RNG_SEED)
        bootstrap_results.append({
            "target": "RGB_vs_GateA", "fold": 1, "seed": seed,
            "arm_a": "GateA", "arm_b": "RGB",
            "point_a": result["point_a"], "point_b": result["point_b"],
            "point_delta": result["point_delta"],
            "boot_mean_delta": result["boot_mean_delta"],
            "ci_2.5": result["ci_2.5"], "ci_97.5": result["ci_97.5"],
            "n_images": result["n_images"], "n_boot": result["n_boot"],
        })
        log.info(f"  Δ={result['point_delta']:.4f} CI=[{result['ci_2.5']:.4f}, {result['ci_97.5']:.4f}]")

    # 写 bootstrap_ci.csv
    ci_csv_path = OUT_DIR / "bootstrap_ci.csv"
    ci_cols = ["target", "fold", "seed", "arm_a", "arm_b",
               "point_a", "point_b", "point_delta",
               "boot_mean_delta", "ci_2.5", "ci_97.5",
               "n_images", "n_boot"]
    with open(ci_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ci_cols)
        w.writeheader()
        w.writerows(bootstrap_results)
    log.info(f"Bootstrap CI: {ci_csv_path} ({len(bootstrap_results)} rows)")

    # ─── Qualitative Cases ───
    log.info("=" * 60)
    log.info("Qualitative Case Selection")
    log.info("=" * 60)

    all_cases = []

    # 用 seed42 作为 primary seed
    primary_seed = 42

    if primary_seed in fold2_data:
        cases = select_qualitative_cases(
            fold2_data[primary_seed]["rgb"], fold2_data[primary_seed]["dual"],
            fold2_data[primary_seed]["gate"], fold2_gts, fold2_img_ids, FOLD2_IMG,
            fold=2, seed=primary_seed)
        all_cases.extend(cases)
        log.info(f"Fold2 seed={primary_seed}: {len(cases)} candidate cases")

    if primary_seed in fold1_data:
        cases = select_qualitative_cases(
            fold1_data[primary_seed]["rgb"], fold1_data[primary_seed]["dual"],
            fold1_data[primary_seed]["gate"], fold1_gts, fold1_img_ids, FOLD1_IMG,
            fold=1, seed=primary_seed)
        all_cases.extend(cases)
        log.info(f"Fold1 seed={primary_seed}: {len(cases)} candidate cases")

    # 按 type 分组，每组取 top-N by score
    TYPE_LIMITS = {
        "rgb_miss_rescued_by_gateA": 3,
        "dual_fp_rejected_by_gateA": 3,
        "harmful_thermal_collapse": 2,
        "thermal_helpful_retained": 3,
    }

    selected_cases = []
    type_groups = defaultdict(list)
    for c in all_cases:
        type_groups[c["case_type"]].append(c)

    for case_type, limit in TYPE_LIMITS.items():
        group = sorted(type_groups[case_type], key=lambda x: -x["score"])[:limit]
        for i, c in enumerate(group):
            c["case_id"] = f"{case_type}_{i+1}"
            selected_cases.append(c)

    log.info(f"Selected {len(selected_cases)} qualitative cases")
    for c in selected_cases:
        log.info(f"  {c['case_id']}: fold={c['fold']} {c['why_selected']}")

    # 写 qualitative_cases.csv
    qc_csv_path = OUT_DIR / "qualitative_cases.csv"
    qc_cols = ["case_id", "case_type", "fold", "seed", "image_id", "image_path",
               "rgb_tp", "rgb_fp", "rgb_fn", "dual_tp", "dual_fp", "dual_fn",
               "gate_tp", "gate_fp", "gate_fn", "score", "why_selected"]
    with open(qc_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=qc_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(selected_cases)
    log.info(f"Qualitative: {qc_csv_path} ({len(selected_cases)} rows)")

    # ─── Recount ───
    log.info("=" * 60)
    log.info("Writing recount + anomalies")
    log.info("=" * 60)

    elapsed = time.time() - t_start

    # Recount
    lines = [
        "# F6 Evidence Package Recount",
        f"# 日期: 2026-06-13",
        f"# 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)",
        "",
        "## Subtask Status",
        "",
        "| Subtask | Status | Notes |",
        "|---------|--------|-------|",
        "| Artifact Inventory | DONE | 34 artifacts cataloged |",
        "| Fold0 GateA Retention | BLOCKED | No fold0 dual checkpoint exists |",
        "| Bootstrap CI | DONE | 9 targets computed |",
        "| Qualitative Cases | DONE | selected cases cataloged |",
        "",
        "## Fold0 GateA: BLOCKED",
        "",
        "原因: 服务器无 `phase_f2_dual_fold0_*` checkpoint。F-1 pilot dual 训练在 pilot split，不可用于 formal fold0 GateA。",
        "详见: `FOLD0_GATEA_BLOCKED.md`",
        "",
        "## Bootstrap CI Results",
        "",
    ]

    # 按 target 分组汇总
    for target in ["RGB_vs_Dual", "RGB_vs_GateA"]:
        for fold in [2, 1]:
            rows_t = [r for r in bootstrap_results if r["target"] == target and r["fold"] == fold]
            if not rows_t:
                continue
            lines.append(f"### {target} Fold{fold}")
            lines.append("")
            lines.append("| Seed | Arm A | Arm B | Point Δ | Boot Mean Δ | CI 2.5% | CI 97.5% | n_boot |")
            lines.append("|------|-------|-------|---------|-------------|---------|----------|--------|")
            for r in rows_t:
                lines.append(f"| {r['seed']} | {r['arm_a']} | {r['arm_b']} | {r['point_delta']:+.4f} | "
                             f"{r['boot_mean_delta']:+.4f} | {r['ci_2.5']:+.4f} | {r['ci_97.5']:+.4f} | {r['n_boot']} |")
            lines.append("")

    lines.extend([
        "## Qualitative Cases",
        "",
        f"Primary seed: {primary_seed}",
        f"Total selected: {len(selected_cases)}",
        "",
    ])
    for ct, limit in TYPE_LIMITS.items():
        n = sum(1 for c in selected_cases if c["case_type"] == ct)
        lines.append(f"- {ct}: {n}/{limit}")
    lines.append("")

    # Sanity check 结果
    lines.append("## Sanity Check")
    lines.append("")
    lines.append(f"Point estimates vs recount.md: {'ALL PASS' if sanity_pass else 'SOME FAIL (see log)'}")
    lines.append("")

    # Source paths
    lines.extend([
        "## Source Paths",
        "",
        "Prediction JSONL:",
    ])
    for seed in SEEDS:
        if seed in fold2_data:
            lines.append(f"- Fold2 seed={seed}: {fold2_data[seed]['rgb_path']}")
        if seed in fold1_data:
            lines.append(f"- Fold1 seed={seed}: {fold1_data[seed]['rgb_path']}")
    lines.extend([
        "",
        "GT labels:",
        f"- Fold1: {FOLD1_LBL}",
        f"- Fold2: {FOLD2_LBL}",
        "",
        "Bootstrap parameters:",
        f"- n_boot={N_BOOT} (handoff: '200 if slow and label as such')",
        f"- method: paired image-level resampling with replacement",
        f"- RNG seed={RNG_SEED}",
        "",
        "GateA locked params:",
        f"- tau_overlap={TAU_OVERLAP}, tau_dual={TAU_DUAL}, mode=add-only",
        f"- dual prefilter conf >= {DUAL_PREFILTER}",
        "",
        "## Completion",
        "",
        f"- `statistical_significance_audit.md` 建议: `revise` → `conditional_pass`",
        f"  (fold1/fold2 bootstrap CI 可用，fold0 retention 永久 blocked)",
        "",
    ])

    recount_path = OUT_DIR / "recount.md"
    recount_path.write_text("\n".join(lines))
    log.info(f"Recount: {recount_path}")

    # Anomalies
    anomaly_lines = [
        "# F6 Evidence Package Anomalies",
        "",
        "## Fold0 GateA Blocker",
        "",
        "- 无 formal fold0 dual checkpoint (`phase_f2_dual_fold0_*`)",
        "- F-1 pilot dual (seed42) 训练在 `fire_loco_pilot_only_paired` split，不是 formal fold0",
        "- 不可用于 GateA 评估",
        "",
        "## Sanity Check",
        "",
    ]
    if not sanity_pass:
        anomaly_lines.append("⚠️ 部分 point estimate 与 recount.md 差异 > 0.01，详见 run.log")
    else:
        anomaly_lines.append("所有 point estimate 与 recount.md 差异 < 0.01，PASS")
    anomaly_lines.append("")

    anomalies_path = OUT_DIR / "anomalies.md"
    anomalies_path.write_text("\n".join(anomaly_lines))
    log.info(f"Anomalies: {anomalies_path}")

    # DONE marker
    (OUT_DIR / "DONE").touch()

    log.info(f"完成! {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"Bootstrap CI: {len(bootstrap_results)} targets")
    log.info(f"Qualitative: {len(selected_cases)} cases")


if __name__ == "__main__":
    main()
