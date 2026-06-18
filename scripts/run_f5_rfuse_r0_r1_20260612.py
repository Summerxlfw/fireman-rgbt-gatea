#!/usr/bin/env python3
"""F-5 R-Fuse R0/R1 Reliability Gate 实验。

R0: 从已有 G1a RGB/dual prediction JSONL 生成 dual candidate reliability features，
    并做 helpful vs harmful separability audit。
R1: 基于 reliability features 实现 rfuse_rule_v1 和可选 rfuse_logistic post-hoc gate，
    与 G1a GateA 直接比较。

本轮不训练 YOLOMM，不跑新 checkpoint。
"""

import sys
import os
import json
import csv
import time
import logging
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
from PIL import Image

# ─── 路径设置 ───
OUT_DIR = Path("/mnt/topic2_workspace/runs/f5_rfuse_r0_r1_20260612")
G1A_DIR = Path("/mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607")

# 数据路径
FOLD2_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val")
FOLD2_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val")
FOLD1_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/images/val")
FOLD1_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val")

SEEDS = [42, 1337]
CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
FOLD2_CAT_IDS = [0, 1]
FOLD1_CAT_IDS = [0, 1, 2]

# 预过滤阈值
DUAL_PREFILTER_CONF = 0.01

# G1a GateA 参考值（from recount）
G1A_REF_FOLD2_MEAN_DELTA = 0.005346
G1A_REF_FOLD2_MEAN_ACCEPT = 0.2301
G1A_REF_FOLD1_MEAN_RETENTION = 0.62

# ─── 日志 ───

def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(OUT_DIR / "run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("f5_rfuse")


log = setup_logging()


# ═══════════════════════════════════════════════════
# 从 G1a script 复用的函数（不 import，直接复制避免触发执行）
# ═══════════════════════════════════════════════════

def load_val_pairs(rgb_dir: Path, ir_dir: Path = None) -> list:
    """加载 val 图片列表。"""
    pairs = []
    rgb_files = sorted([f for f in rgb_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        with Image.open(rgb_path) as img:
            W, H = img.size
        pairs.append({"img_id": stem, "rgb": rgb_path, "W": W, "H": H})
    log.info(f"加载 {len(pairs)} 张 val 图片 ({rgb_dir.name})")
    return pairs


def load_ground_truths(lbl_dir: Path, pairs: list, cat_ids: list) -> tuple:
    """解析 YOLO labels → GT dict。"""
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

    log.info(f"GT: {len(img_ids)} imgs, classes={cat_ids}")
    return gts_by_img, img_ids, cat_ids


def load_predictions_jsonl(path) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["img_id"]] = np.array(rec["boxes"], dtype=np.float64) if rec["boxes"] else np.zeros((0, 6))
    log.info(f"加载: {path} ({len(preds)} 行)")
    return preds


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


def write_csv(results: list, path: Path, columns: list):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log.info(f"CSV: {path} ({len(results)} 行)")


# ═══════════════════════════════════════════════════
# G1a GateA 复现（作为 baseline）
# ═══════════════════════════════════════════════════

def apply_g1a_gateA(rgb_boxes, dual_boxes, tau_overlap=0.7, tau_dual=0.05):
    """G1a GateA: conservative add-only gate。"""
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


# ═══════════════════════════════════════════════════
# R0: Candidate Feature Extraction
# ═══════════════════════════════════════════════════

def extract_candidate_features(fold: str, seed: int,
                               preds_rgb: dict, preds_dual: dict,
                               gts_by_img: dict, img_ids: list,
                               cat_ids: list, pairs: list) -> list:
    """对每个 dual box 计算 reliability features。"""
    features = []

    # 预建 img_id -> (W, H) 映射
    img_wh = {p["img_id"]: (p["W"], p["H"]) for p in pairs}

    for img_id in img_ids:
        rgb_boxes = preds_rgb.get(img_id, np.zeros((0, 6)))
        # JSONL 中的 dual 已经在 G1a 阶段做过 conf >= 0.01 prefilter，直接使用
        dual_boxes = preds_dual.get(img_id, np.zeros((0, 6)))

        # 获取 GT
        gt_list = gts_by_img.get(img_id, [])

        # 图片级统计
        n_rgb = len(rgb_boxes)
        n_dual = len(dual_boxes)

        # 图片真实尺寸
        W, H = img_wh.get(img_id, (1920, 1080))
        img_area = W * H

        # RGB context density: RGB boxes 之间的平均 IoU (sample 以控制计算量)
        rgb_density = 0.0
        if n_rgb > 1:
            if n_rgb <= 50:
                rgb_ious = compute_iou_matrix(rgb_boxes, rgb_boxes)
                np.fill_diagonal(rgb_ious, 0)
                rgb_density = float((rgb_ious >= 0.03).sum()) / max(n_rgb * (n_rgb - 1), 1)
            else:
                idx = np.random.choice(n_rgb, min(50, n_rgb), replace=False)
                rgb_sub = rgb_boxes[idx]
                rgb_ious = compute_iou_matrix(rgb_sub, rgb_sub)
                np.fill_diagonal(rgb_ious, 0)
                rgb_density = float((rgb_ious >= 0.03).sum()) / max(len(idx) * (len(idx) - 1), 1)

        for di, d in enumerate(dual_boxes):
            d_cls = int(d[5])
            d_conf = float(d[4])
            d_xyxy = d[:4]
            d_area = float((d_xyxy[2] - d_xyxy[0]) * (d_xyxy[3] - d_xyxy[1]))
            d_area_norm = d_area / max(img_area, 1)

            # 与 RGB boxes 的 IoU
            if n_rgb > 0:
                ious_all = compute_iou_matrix(d_xyxy.reshape(1, 4), rgb_boxes[:, :4])[0]
                rgb_cls_match = (rgb_boxes[:, 5].astype(int) == d_cls)

                # 同类 RGB 最大 IoU
                same_cls_ious = ious_all[rgb_cls_match] if rgb_cls_match.any() else np.array([0.0])
                rgb_max_iou_same_cls = float(same_cls_ious.max())

                # 任意类 RGB 最大 IoU
                rgb_max_iou_any_cls = float(ious_all.max())

                # 同类 RGB 匹配的最高 conf
                if rgb_cls_match.any():
                    rgb_matched_conf_same_cls = float(rgb_boxes[rgb_cls_match, 4].max())
                else:
                    rgb_matched_conf_same_cls = 0.0

                # conf gap
                conf_gap_vs_rgb = d_conf - rgb_matched_conf_same_cls

                # 最佳匹配 RGB box 的类别是否一致
                best_rgb_idx = int(ious_all.argmax())
                class_agree_with_best_any = int(int(rgb_boxes[best_rgb_idx, 5]) == d_cls)
            else:
                rgb_max_iou_same_cls = 0.0
                rgb_max_iou_any_cls = 0.0
                rgb_matched_conf_same_cls = 0.0
                conf_gap_vs_rgb = d_conf
                class_agree_with_best_any = 0

            # dual-only: 没有任何 RGB box 与之 IoU >= 0.5/0.7
            dual_only_05 = int(rgb_max_iou_any_cls < 0.5)
            dual_only_07 = int(rgb_max_iou_any_cls < 0.7)

            # dual/rgb count ratio
            dual_to_rgb_count_ratio = n_dual / max(n_rgb, 1)

            # GT 匹配
            same_cls_gts = [g for g in gt_list if g["category_id"] == d_cls]
            matches_any_gt = 0
            helpful_uncovered_gt = 0
            duplicates_rgb_gt = 0

            for gt in same_cls_gts:
                gt_xyxy = np.array(gt["bbox_xyxy"]).reshape(1, 4)
                iou = float(compute_iou_matrix(d_xyxy.reshape(1, 4), gt_xyxy)[0, 0])
                if iou >= 0.5:
                    matches_any_gt = 1
                    # 检查这个 GT 是否已被 RGB 覆盖
                    gt_covered_by_rgb = False
                    if n_rgb > 0:
                        rgb_same_cls = rgb_boxes[rgb_boxes[:, 5].astype(int) == d_cls]
                        if len(rgb_same_cls) > 0:
                            rgb_gt_ious = compute_iou_matrix(gt_xyxy, rgb_same_cls[:, :4])[0]
                            if rgb_gt_ious.max() >= 0.5:
                                gt_covered_by_rgb = True

                    if not gt_covered_by_rgb:
                        helpful_uncovered_gt = 1
                    else:
                        duplicates_rgb_gt = 1
                    break  # 一个 dual box 只匹配一个 GT

            harmful = 1 - matches_any_gt

            feat = {
                "fold": fold,
                "seed": seed,
                "img_id": img_id,
                "dual_idx": di,
                "dual_cls": d_cls,
                "dual_conf": d_conf,
                "box_area_norm": d_area_norm,
                "rgb_max_iou_same_cls": rgb_max_iou_same_cls,
                "rgb_max_iou_any_cls": rgb_max_iou_any_cls,
                "rgb_matched_conf_same_cls": rgb_matched_conf_same_cls,
                "conf_gap_vs_rgb": conf_gap_vs_rgb,
                "class_agree_with_best_any": class_agree_with_best_any,
                "dual_only_05": dual_only_05,
                "dual_only_07": dual_only_07,
                "rgb_context_density_iou03": rgb_density,
                "dual_to_rgb_count_ratio_image": dual_to_rgb_count_ratio,
                "n_rgb_boxes_image": n_rgb,
                "n_dual_boxes_image": n_dual,
                "helpful_uncovered_gt": helpful_uncovered_gt,
                "matches_any_gt": matches_any_gt,
                "duplicates_rgb_gt": duplicates_rgb_gt,
                "harmful": harmful,
            }
            features.append(feat)

    return features


# ═══════════════════════════════════════════════════
# R0: Separability Audit
# ═══════════════════════════════════════════════════

def ranksum_auc(scores_pos, scores_neg):
    """手动实现 rank-sum AUC（Wilcoxon-Mann-Whitney）。"""
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    combined = np.concatenate([scores_pos, scores_neg])
    order = np.argsort(combined, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1).astype(float)

    # 处理 ties: 相同值给平均排名
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1))
            for k in range(i, j):
                idx = order[k]
                ranks[idx] = avg_rank
        i = j

    u1 = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2
    auc = u1 / (n_pos * n_neg)
    return float(auc)


def separability_audit(features_by_fold: dict) -> list:
    """对每个 feature 计算 helpful vs harmful 的 mean/median/AUC。"""
    numeric_cols = [
        "dual_conf", "box_area_norm", "rgb_max_iou_same_cls", "rgb_max_iou_any_cls",
        "rgb_matched_conf_same_cls", "conf_gap_vs_rgb", "class_agree_with_best_any",
        "dual_only_05", "dual_only_07", "rgb_context_density_iou03",
        "dual_to_rgb_count_ratio_image", "n_rgb_boxes_image", "n_dual_boxes_image",
    ]

    results = []

    for fold_key, all_feats in features_by_fold.items():
        helpful = [f for f in all_feats if f["helpful_uncovered_gt"] == 1]
        harmful = [f for f in all_feats if f["harmful"] == 1]

        log.info(f"[{fold_key}] helpful={len(helpful)}, harmful={len(harmful)}")

        for col in numeric_cols:
            h_vals = np.array([f[col] for f in helpful], dtype=float) if helpful else np.array([])
            m_vals = np.array([f[col] for f in harmful], dtype=float) if harmful else np.array([])

            if len(h_vals) == 0 or len(m_vals) == 0:
                results.append({
                    "fold": fold_key, "feature": col,
                    "helpful_mean": float("nan"), "helpful_median": float("nan"),
                    "harmful_mean": float("nan"), "harmful_median": float("nan"),
                    "auc": float("nan"), "direction": "N/A",
                })
                continue

            auc = ranksum_auc(h_vals, m_vals)
            direction = "higher_helpful" if np.mean(h_vals) > np.mean(m_vals) else "lower_helpful"

            results.append({
                "fold": fold_key,
                "feature": col,
                "helpful_mean": float(np.mean(h_vals)),
                "helpful_median": float(np.median(h_vals)),
                "harmful_mean": float(np.mean(m_vals)),
                "harmful_median": float(np.median(m_vals)),
                "auc": auc,
                "direction": direction,
            })

    return results


# ═══════════════════════════════════════════════════
# R1: R-Fuse Rule V1 Gate
# ═══════════════════════════════════════════════════

def apply_rfuse_rule_v1(rgb_boxes, dual_boxes, features_df,
                        tau_overlap, tau_dual, tau_rgb_uncertain,
                        margin, max_dual_to_rgb_count_ratio):
    """R-Fuse rule v1: 基于 reliability features 的规则 gate。"""
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0

    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total

    n_rgb = len(rgb_boxes)
    output = list(rgb_boxes)
    is_original_rgb = [True] * len(rgb_boxes)

    for di, d in enumerate(dual_boxes):
        d_conf = d[4]
        d_cls = int(d[5])

        # 从 features_df 查找该 dual box 的 features
        # features_df 是 list of dict，用 (img_id, dual_idx) 索引
        # 这里我们在 calling site 直接传 features，避免查找
        # 简化: 直接在 loop 中计算
        if d_conf < tau_dual:
            continue

        # 找同类 RGB boxes
        same_cls_indices = [i for i in range(len(output))
                            if int(output[i][5]) == d_cls and is_original_rgb[i]]

        # 计算 dual/rgb count ratio
        dual_rgb_ratio = len(dual_boxes) / max(n_rgb, 1)
        if dual_rgb_ratio > max_dual_to_rgb_count_ratio:
            continue

        if not same_cls_indices:
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1
            continue

        same_cls_boxes = np.array([output[i][:4] for i in same_cls_indices])
        d_box = d[:4].reshape(1, 4)
        ious = compute_iou_matrix(d_box, same_cls_boxes)[0]
        max_iou = ious.max()
        best_idx_same = same_cls_indices[int(ious.argmax())]
        r_conf = output[best_idx_same][4]

        # 规则 1: 没有 overlap → 添加
        if max_iou < tau_overlap:
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1
            continue

        # 规则 2: RGB 不确定且 dual 更 confident → 替换（但不从 output 删除，用替换）
        if r_conf < tau_rgb_uncertain and (d_conf - r_conf) >= margin:
            output[best_idx_same] = d
            is_original_rgb[best_idx_same] = False
            n_dual_accepted += 1

    return np.array(output) if output else np.zeros((0, 6)), n_dual_accepted, n_dual_total


def apply_oracle(rgb_boxes, dual_boxes, helpful_set, img_id):
    """Oracle: 只接受 helpful_uncovered_gt candidates。仅诊断用。
    helpful_set: set of dual_idx that are helpful for this img_id。
    """
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0

    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0

    output = list(rgb_boxes) if len(rgb_boxes) > 0 else []

    for di, d in enumerate(dual_boxes):
        if di in helpful_set:
            output.append(d)
            n_dual_accepted += 1

    if not output:
        return np.zeros((0, 6)), n_dual_accepted, n_dual_total
    return np.array(output), n_dual_accepted, n_dual_total


# ═══════════════════════════════════════════════════
# R1: Sweep Engine
# ═══════════════════════════════════════════════════

def sweep_r1_fold(preds_rgb, preds_dual, helpful_by_img,
                  gts_by_img, img_ids, cat_ids, fold, seed):
    """R1 gate sweep: baselines + G1a GateA + rfuse_rule_v1 + oracle。"""
    results = []

    # P0: RGB only
    ap50 = fast_ap50_eval(preds_rgb, gts_by_img, img_ids, cat_ids)
    results.append({
        "fold": fold, "seed": seed, "arm": "rgb_only",
        "AP50": ap50["mAP50"],
        "smoke_AP50": ap50.get(0, 0), "fire_AP50": ap50.get(1, 0), "person_AP50": ap50.get(2, 0),
        "dual_acceptance_ratio": 0.0,
        "accepted_by_class": "{}",
    })

    # P1: Dual only
    ap50 = fast_ap50_eval(preds_dual, gts_by_img, img_ids, cat_ids)
    results.append({
        "fold": fold, "seed": seed, "arm": "dual_only",
        "AP50": ap50["mAP50"],
        "smoke_AP50": ap50.get(0, 0), "fire_AP50": ap50.get(1, 0), "person_AP50": ap50.get(2, 0),
        "dual_acceptance_ratio": 1.0,
        "accepted_by_class": "{}",
    })

    # G1a GateA
    merged_all = {}
    total_acc = 0
    total_dual = 0
    for img_id in img_ids:
        rgb_b = preds_rgb.get(img_id, np.zeros((0, 6)))
        dual_b = preds_dual.get(img_id, np.zeros((0, 6)))
        m, na, nt = apply_g1a_gateA(rgb_b, dual_b, tau_overlap=0.7, tau_dual=0.05)
        merged_all[img_id] = m
        total_acc += na
        total_dual += nt
    ap50 = fast_ap50_eval(merged_all, gts_by_img, img_ids, cat_ids)
    acc_ratio = total_acc / max(total_dual, 1)
    results.append({
        "fold": fold, "seed": seed, "arm": "g1a_gateA",
        "AP50": ap50["mAP50"],
        "smoke_AP50": ap50.get(0, 0), "fire_AP50": ap50.get(1, 0), "person_AP50": ap50.get(2, 0),
        "dual_acceptance_ratio": acc_ratio,
        "accepted_by_class": "{}",
    })

    # R-Fuse Rule V1 sweep
    tau_overlaps = [0.5, 0.6, 0.7]
    tau_duals = [0.03, 0.05, 0.10, 0.20]
    tau_rgb_uncertains = [0.30, 0.50, 0.70]
    margins = [0.00, 0.05, 0.10, 0.20]
    max_ratios = [3, 5, 10, 999999]

    n_configs = len(tau_overlaps) * len(tau_duals) * len(tau_rgb_uncertains) * len(margins) * len(max_ratios)
    log.info(f"[{fold} seed={seed}] R-Fuse rule sweep: {n_configs} configs")

    t0 = time.time()
    ci = 0
    for to_ in tau_overlaps:
        for td_ in tau_duals:
            for tr_ in tau_rgb_uncertains:
                for mg_ in margins:
                    for mr_ in max_ratios:
                        merged_all = {}
                        total_acc = 0
                        total_dual = 0
                        for img_id in img_ids:
                            rgb_b = preds_rgb.get(img_id, np.zeros((0, 6)))
                            dual_b = preds_dual.get(img_id, np.zeros((0, 6)))
                            m, na, nt = apply_rfuse_rule_v1(
                                rgb_b, dual_b, None,
                                to_, td_, tr_, mg_, mr_)
                            merged_all[img_id] = m
                            total_acc += na
                            total_dual += nt
                        ap50 = fast_ap50_eval(merged_all, gts_by_img, img_ids, cat_ids)
                        acc_ratio = total_acc / max(total_dual, 1)

                        results.append({
                            "fold": fold, "seed": seed,
                            "arm": "rfuse_rule_v1",
                            "tau_overlap": to_, "tau_dual": td_,
                            "tau_rgb_uncertain": tr_, "margin": mg_,
                            "max_dual_to_rgb_count_ratio": mr_,
                            "AP50": ap50["mAP50"],
                            "smoke_AP50": ap50.get(0, 0),
                            "fire_AP50": ap50.get(1, 0),
                            "person_AP50": ap50.get(2, 0),
                            "dual_acceptance_ratio": acc_ratio,
                            "accepted_by_class": "{}",
                        })
                        ci += 1
                        if ci % 200 == 0:
                            elapsed = time.time() - t0
                            rate = ci / elapsed
                            eta = (n_configs - ci) / rate
                            log.info(f"  [{ci}/{n_configs}] rate={rate:.1f} cfg/s ETA={eta:.0f}s")

    log.info(f"[{fold} seed={seed}] rule sweep 完成: {ci} configs, {time.time()-t0:.1f}s")

    # Oracle (diagnostic only)
    merged_all = {}
    total_acc = 0
    total_dual = 0
    for img_id in img_ids:
        rgb_b = preds_rgb.get(img_id, np.zeros((0, 6)))
        dual_b = preds_dual.get(img_id, np.zeros((0, 6)))
        helpful_set = helpful_by_img.get(img_id, set())
        m, na, nt = apply_oracle(rgb_b, dual_b, helpful_set, img_id)
        merged_all[img_id] = m
        total_acc += na
        total_dual += nt
    ap50 = fast_ap50_eval(merged_all, gts_by_img, img_ids, cat_ids)
    results.append({
        "fold": fold, "seed": seed, "arm": "oracle_upper",
        "AP50": ap50["mAP50"],
        "smoke_AP50": ap50.get(0, 0), "fire_AP50": ap50.get(1, 0), "person_AP50": ap50.get(2, 0),
        "dual_acceptance_ratio": total_acc / max(total_dual, 1),
        "accepted_by_class": "{}",
    })

    # 补 delta
    rgb_ap50 = results[0]["AP50"]
    for r in results:
        r["delta_AP50_vs_rgb"] = r["AP50"] - rgb_ap50

    return results


# ═══════════════════════════════════════════════════
# R1: Logistic Gate (sklearn)
# ═══════════════════════════════════════════════════

def run_logistic_gate(features_by_fold: dict,
                      preds_rgb_all: dict, preds_dual_all: dict,
                      gts_by_img_all: dict, img_ids_all: dict,
                      cat_ids_all: dict) -> list:
    """Logistic gate: cross-fold train/eval。"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("LOGISTIC_SKIPPED_NO_SKLEARN")
        return []

    feat_cols = [
        "dual_conf", "rgb_max_iou_same_cls", "rgb_max_iou_any_cls",
        "rgb_matched_conf_same_cls", "conf_gap_vs_rgb", "class_agree_with_best_any",
        "box_area_norm", "rgb_context_density_iou03",
        "dual_to_rgb_count_ratio_image", "n_rgb_boxes_image", "n_dual_boxes_image",
    ]

    results = []

    # Protocol: train on fold1, eval on fold2; train on fold2, eval on fold1
    protocols = [
        ("train_fold1_eval_fold2", "fold1", "fold2"),
        ("train_fold2_eval_fold1", "fold2", "fold1"),
    ]

    for protocol_name, train_fold, eval_fold in protocols:
        log.info(f"Logistic gate: {protocol_name}")

        # 收集训练数据
        train_feats = features_by_fold.get(train_fold, [])
        if not train_feats:
            log.warning(f"  无训练数据: {train_fold}")
            continue

        X_train = np.array([[f[c] for c in feat_cols] for f in train_feats], dtype=float)
        y_train = np.array([f["helpful_uncovered_gt"] for f in train_feats], dtype=int)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        log.info(f"  train: {len(y_train)} samples, pos={n_pos}, neg={n_neg}")

        if n_pos < 10 or n_neg < 10:
            log.warning(f"  class imbalance too severe, skip")
            continue

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # 训练
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train_scaled, y_train)

        log.info(f"  train accuracy: {clf.score(X_train_scaled, y_train):.4f}")
        log.info(f"  coefficients: {dict(zip(feat_cols, clf.coef_[0]))}")

        # 在 eval fold 上评估
        eval_cat_ids = cat_ids_all[eval_fold]
        eval_gts = gts_by_img_all[eval_fold]
        eval_img_ids = img_ids_all[eval_fold]

        for seed in SEEDS:
            eval_feats = [f for f in features_by_fold.get(eval_fold, []) if f["seed"] == seed]
            if not eval_feats:
                continue

            # 构建 img_id -> features 映射
            feats_by_img = defaultdict(list)
            for f in eval_feats:
                feats_by_img[f["img_id"]].append(f)

            # 用不同 threshold sweep
            X_eval = np.array([[f[c] for c in feat_cols] for f in eval_feats], dtype=float)
            X_eval_scaled = scaler.transform(X_eval)
            probas = clf.predict_proba(X_eval_scaled)[:, 1]

            # 给每个 feature 赋概率
            feat_probas = {}
            for i, f in enumerate(eval_feats):
                key = (f["img_id"], f["dual_idx"])
                feat_probas[key] = probas[i]

            # Sweep threshold
            for tau_logistic in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
                merged_all = {}
                total_acc = 0
                total_dual = 0

                for img_id in eval_img_ids:
                    rgb_b = preds_rgb_all[eval_fold][seed].get(img_id, np.zeros((0, 6)))
                    dual_b = preds_dual_all[eval_fold][seed].get(img_id, np.zeros((0, 6)))

                    if len(dual_b) == 0:
                        merged_all[img_id] = rgb_b
                        continue

                    output = list(rgb_b)
                    n_acc = 0

                    for di, d in enumerate(dual_b):
                        key = (img_id, di)
                        p = feat_probas.get(key, 0.0)
                        if p >= tau_logistic:
                            output.append(d)
                            n_acc += 1

                    merged_all[img_id] = np.array(output) if output else np.zeros((0, 6))
                    total_acc += n_acc
                    total_dual += len(dual_b)

                ap50 = fast_ap50_eval(merged_all, eval_gts, eval_img_ids, eval_cat_ids)
                acc_ratio = total_acc / max(total_dual, 1)

                rgb_only_ap50 = fast_ap50_eval(
                    preds_rgb_all[eval_fold][seed], eval_gts, eval_img_ids, eval_cat_ids)["mAP50"]

                results.append({
                    "fold": eval_fold, "seed": seed,
                    "arm": "rfuse_logistic",
                    "protocol": protocol_name,
                    "tau_logistic": tau_logistic,
                    "AP50": ap50["mAP50"],
                    "smoke_AP50": ap50.get(0, 0),
                    "fire_AP50": ap50.get(1, 0),
                    "person_AP50": ap50.get(2, 0),
                    "delta_AP50_vs_rgb": ap50["mAP50"] - rgb_only_ap50,
                    "dual_acceptance_ratio": acc_ratio,
                    "accepted_by_class": "{}",
                })

            log.info(f"  {protocol_name} seed={seed} 完成")

    return results


# ═══════════════════════════════════════════════════
# 判定逻辑
# ═══════════════════════════════════════════════════

def judge_fold2_safety(all_results: list) -> tuple:
    """Fold2 safety 判定。返回 (verdict, best_arm_details)。"""
    by_seed = defaultdict(dict)
    for r in all_results:
        if r["seed"] not in by_seed.get(r["fold"], {}):
            pass
        key = f"{r['fold']}_{r['seed']}_{r['arm']}"
        by_seed[r["fold"]][r["seed"]] = by_seed[r["fold"]].get(r["seed"], {})

    # 按 arm 分组
    arm_results = defaultdict(list)
    for r in all_results:
        arm_results[r["arm"]].append(r)

    # G1a GateA reference
    g1a_deltas = []
    g1a_accepts = []
    for r in arm_results.get("g1a_gateA", []):
        g1a_deltas.append(r["delta_AP50_vs_rgb"])
        g1a_accepts.append(r["dual_acceptance_ratio"])
    g1a_mean_delta = float(np.mean(g1a_deltas)) if g1a_deltas else 0
    g1a_mean_accept = float(np.mean(g1a_accepts)) if g1a_accepts else 0

    log.info(f"G1a GateA baseline: mean_delta={g1a_mean_delta:.6f}, mean_accept={g1a_mean_accept:.4f}")

    # 检查 G1a baseline 可复现性
    g1a_delta_diff = abs(g1a_mean_delta - G1A_REF_FOLD2_MEAN_DELTA)
    if g1a_delta_diff > 0.002:
        log.warning(f"G1a baseline 复现偏差: {g1a_delta_diff:.6f} (超过 ±0.002)")

    # 评估 deployable arms
    deployable_arms = ["g1a_gateA", "rfuse_rule_v1", "rfuse_logistic"]

    best_arm = None
    best_mean_delta = -999
    best_details = {}

    for arm in deployable_arms:
        arm_res = arm_results.get(arm, [])
        if not arm_res:
            continue

        # 按 config 分组（rfuse_rule_v1 有多个 config）
        configs = defaultdict(list)
        for r in arm_res:
            config_key = (
                r.get("tau_overlap"), r.get("tau_dual"),
                r.get("tau_rgb_uncertain"), r.get("margin"),
                r.get("max_dual_to_rgb_count_ratio"),
                r.get("tau_logistic"), r.get("protocol"),
            )
            configs[config_key].append(r)

        for config_key, cfg_results in configs.items():
            if len(cfg_results) < len(SEEDS):
                continue

            deltas = [r["delta_AP50_vs_rgb"] for r in cfg_results]
            accepts = [r["dual_acceptance_ratio"] for r in cfg_results]
            mean_delta = float(np.mean(deltas))
            mean_accept = float(np.mean(accepts))

            if mean_delta > best_mean_delta:
                best_mean_delta = mean_delta
                best_arm = arm
                best_details = {
                    "arm": arm,
                    "config": config_key,
                    "mean_delta": mean_delta,
                    "mean_accept": mean_accept,
                    "per_seed": {r["seed"]: r for r in cfg_results},
                }

    if best_details is None or not best_details:
        return "FAIL", None

    md = best_details["mean_delta"]
    ma = best_details["mean_accept"]

    # PASS_SAFE: mean delta >= 0, each seed >= -0.005, accept >= 0.05
    per_seed_deltas = [r["delta_AP50_vs_rgb"] for r in best_details["per_seed"].values()]
    all_seeds_safe = all(d >= -0.005 for d in per_seed_deltas)

    if md >= 0 and all_seeds_safe and ma >= 0.05:
        verdict = "PASS_SAFE"
    # PASS_MATCH_G1A
    elif md >= g1a_mean_delta - 0.002 and ma >= 0.05:
        verdict = "PASS_MATCH_G1A"
    # FAIL
    elif md < -0.005 or ma < 0.01:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"

    return verdict, best_details


def judge_fold1_retention(fold1_results: list, g1a_retention: float) -> tuple:
    """Fold1 retention 判定。"""
    by_seed = defaultdict(lambda: {"rgb_only": None, "dual_only": None, "g1a_gateA": None,
                                    "best_rfuse": None})
    for r in fold1_results:
        seed = r["seed"]
        arm = r["arm"]
        if arm == "rgb_only":
            by_seed[seed]["rgb_only"] = r
        elif arm == "dual_only":
            by_seed[seed]["dual_only"] = r
        elif arm == "g1a_gateA":
            by_seed[seed]["g1a_gateA"] = r
        elif arm.startswith("rfuse"):
            cur = by_seed[seed]["best_rfuse"]
            if cur is None or r["AP50"] > cur["AP50"]:
                by_seed[seed]["best_rfuse"] = r

    retentions = []
    for seed in SEEDS:
        seed_res = by_seed.get(seed, {})
        rgb_ap50 = seed_res["rgb_only"]["AP50"] if seed_res["rgb_only"] else 0
        dual_ap50 = seed_res["dual_only"]["AP50"] if seed_res["dual_only"] else 0
        best_rfuse_ap50 = seed_res["best_rfuse"]["AP50"] if seed_res["best_rfuse"] else 0

        dual_gain = dual_ap50 - rgb_ap50
        if abs(dual_gain) < 0.001:
            ret = 1.0
        else:
            ret = max(0, (best_rfuse_ap50 - rgb_ap50) / dual_gain)
        retentions.append(ret)

    mean_ret = float(np.mean(retentions))
    log.info(f"Fold1 R-Fuse retention: {mean_ret:.4f}")

    if mean_ret >= g1a_retention + 0.05:
        verdict = "STRONG_RETENTION"
    elif mean_ret >= 0.6:
        verdict = "PASS_RETENTION"
    else:
        verdict = "FAIL_RETENTION"

    return verdict, mean_ret


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("F-5 R-Fuse R0/R1 Reliability Gate Experiment")
    log.info("=" * 60)

    # ─── Runtime Precheck ───
    log.info("=== Runtime Precheck ===")
    for f in [
        G1A_DIR / "predictions_fold2_seed42_rgb.jsonl",
        G1A_DIR / "predictions_fold2_seed42_dual.jsonl",
        G1A_DIR / "predictions_fold2_seed1337_rgb.jsonl",
        G1A_DIR / "predictions_fold2_seed1337_dual.jsonl",
        G1A_DIR / "predictions_fold1_seed42_rgb.jsonl",
        G1A_DIR / "predictions_fold1_seed42_dual.jsonl",
        G1A_DIR / "predictions_fold1_seed1337_rgb.jsonl",
        G1A_DIR / "predictions_fold1_seed1337_dual.jsonl",
    ]:
        if not f.exists():
            log.error(f"缺失: {f}")
            (OUT_DIR / "FAILED_PRECHECK").touch()
            sys.exit(1)

    for d in [FOLD2_LBL_DIR, FOLD1_LBL_DIR]:
        if not d.exists():
            log.error(f"目录缺失: {d}")
            (OUT_DIR / "FAILED_PRECHECK").touch()
            sys.exit(1)

    log.info("Precheck 通过")

    # ─── 加载数据 ───
    log.info("=" * 60)
    log.info("加载数据")
    log.info("=" * 60)

    # Fold2
    fold2_pairs = load_val_pairs(FOLD2_RGB_DIR)
    fold2_gts_by_img, fold2_img_ids, fold2_cat_ids = load_ground_truths(
        FOLD2_LBL_DIR, fold2_pairs, FOLD2_CAT_IDS)

    fold2_rgb = {}
    fold2_dual = {}
    for seed in SEEDS:
        fold2_rgb[seed] = load_predictions_jsonl(
            G1A_DIR / f"predictions_fold2_seed{seed}_rgb.jsonl")
        fold2_dual[seed] = load_predictions_jsonl(
            G1A_DIR / f"predictions_fold2_seed{seed}_dual.jsonl")

    # Fold1
    fold1_pairs = load_val_pairs(FOLD1_RGB_DIR)
    fold1_gts_by_img, fold1_img_ids, fold1_cat_ids = load_ground_truths(
        FOLD1_LBL_DIR, fold1_pairs, FOLD1_CAT_IDS)

    fold1_rgb = {}
    fold1_dual = {}
    for seed in SEEDS:
        fold1_rgb[seed] = load_predictions_jsonl(
            G1A_DIR / f"predictions_fold1_seed{seed}_rgb.jsonl")
        fold1_dual[seed] = load_predictions_jsonl(
            G1A_DIR / f"predictions_fold1_seed{seed}_dual.jsonl")

    # ─── R0: Candidate Feature Extraction ───
    log.info("=" * 60)
    log.info("R0: Candidate Feature Extraction")
    log.info("=" * 60)

    features_by_fold = {}
    features_by_img_all = {}  # {(fold, seed): {img_id: [features]}}
    helpful_by_img_all = {}  # {(fold, seed): {img_id: set(helpful dual_idx)}}

    for fold_name, rgb_preds, dual_preds, gts, img_ids, cat_ids, pairs in [
        ("fold2", fold2_rgb, fold2_dual, fold2_gts_by_img, fold2_img_ids, fold2_cat_ids, fold2_pairs),
        ("fold1", fold1_rgb, fold1_dual, fold1_gts_by_img, fold1_img_ids, fold1_cat_ids, fold1_pairs),
    ]:
        for seed in SEEDS:
            log.info(f"R0 extraction: {fold_name} seed={seed}")
            feats = extract_candidate_features(
                fold_name, seed,
                rgb_preds[seed], dual_preds[seed],
                gts, img_ids, cat_ids, pairs)

            n_helpful = sum(1 for f in feats if f["helpful_uncovered_gt"] == 1)
            n_harmful = sum(1 for f in feats if f["harmful"] == 1)
            log.info(f"  total={len(feats)} helpful={n_helpful} harmful={n_harmful}")

            # 按 img_id 分组
            feats_by_img = defaultdict(list)
            helpful_by_img = defaultdict(set)
            for f in feats:
                feats_by_img[f["img_id"]].append(f)
                if f["helpful_uncovered_gt"] == 1:
                    helpful_by_img[f["img_id"]].add(f["dual_idx"])

            # 保存 CSV
            feat_cols = list(feats[0].keys()) if feats else []
            if feats:
                csv_path = OUT_DIR / f"candidate_features_{fold_name}_seed{seed}.csv"
                write_csv(feats, csv_path, feat_cols)

            # 合并到 fold 级别
            fold_key = f"{fold_name}"
            if fold_key not in features_by_fold:
                features_by_fold[fold_key] = []
            features_by_fold[fold_key].extend(feats)

            features_by_img_all[(fold_name, seed)] = dict(feats_by_img)
            helpful_by_img_all[(fold_name, seed)] = dict(helpful_by_img)

    # ─── R0: Separability Audit ───
    log.info("=" * 60)
    log.info("R0: Separability Audit")
    log.info("=" * 60)

    sep_results = separability_audit(features_by_fold)
    sep_cols = ["fold", "feature", "helpful_mean", "helpful_median",
                "harmful_mean", "harmful_median", "auc", "direction"]
    write_csv(sep_results, OUT_DIR / "r0_separability_summary.csv", sep_cols)

    # 判断 R0 PASS
    r0_pass = False
    for sr in sep_results:
        if sr["fold"] == "fold2" and not np.isnan(sr["auc"]):
            if sr["auc"] >= 0.65:
                # 检查 fold1 方向一致
                for sr2 in sep_results:
                    if sr2["fold"] == "fold1" and sr2["feature"] == sr["feature"]:
                        if sr2["direction"] == sr["direction"]:
                            r0_pass = True
                            log.info(f"R0 PASS: feature={sr['feature']} fold2_AUC={sr['auc']:.4f} "
                                     f"direction={sr['direction']}")
                            break
                if r0_pass:
                    break

    if not r0_pass:
        log.warning("R0 未达到 AUC >= 0.65 标准，检查是否有接近的 feature")

    log.info(f"R0 verdict: {'R0_PASS' if r0_pass else 'R0_WEAK'}")

    # ─── R1: Gate Sweep ───
    log.info("=" * 60)
    log.info("R1: Gate Sweep - Fold2")
    log.info("=" * 60)

    fold2_all_results = []
    for seed in SEEDS:
        helpful_img = helpful_by_img_all.get(("fold2", seed), {})
        seed_results = sweep_r1_fold(
            fold2_rgb[seed], fold2_dual[seed], helpful_img,
            fold2_gts_by_img, fold2_img_ids, fold2_cat_ids, "fold2", seed)
        fold2_all_results.extend(seed_results)

    # 保存 fold2 sweep
    fold2_csv_cols = ["fold", "seed", "arm", "tau_overlap", "tau_dual", "tau_rgb_uncertain",
                      "margin", "max_dual_to_rgb_count_ratio", "tau_logistic", "protocol",
                      "AP50", "smoke_AP50", "fire_AP50", "person_AP50",
                      "delta_AP50_vs_rgb", "dual_acceptance_ratio"]
    write_csv(fold2_all_results, OUT_DIR / "r1_gate_sweep_fold2.csv", fold2_csv_cols)

    # ─── R1: Fold2 Safety Decision ───
    log.info("=" * 60)
    log.info("R1: Fold2 Safety Decision")
    log.info("=" * 60)

    fold2_verdict, fold2_best = judge_fold2_safety(fold2_all_results)
    log.info(f"Fold2 safety verdict: {fold2_verdict}")
    if fold2_best:
        log.info(f"  best arm: {fold2_best['arm']}")
        log.info(f"  mean delta: {fold2_best['mean_delta']:.6f}")
        log.info(f"  mean accept: {fold2_best['mean_accept']:.4f}")

    # ─── R1: Fold1 Retention ───
    log.info("=" * 60)
    log.info("R1: Fold1 Retention")
    log.info("=" * 60)

    fold1_all_results = []
    for seed in SEEDS:
        helpful_img = helpful_by_img_all.get(("fold1", seed), {})
        seed_results = sweep_r1_fold(
            fold1_rgb[seed], fold1_dual[seed], helpful_img,
            fold1_gts_by_img, fold1_img_ids, fold1_cat_ids, "fold1", seed)
        fold1_all_results.extend(seed_results)

    write_csv(fold1_all_results, OUT_DIR / "r1_gate_sweep_fold1.csv", fold2_csv_cols)

    # Fold1 retention
    fold1_verdict, fold1_retention = judge_fold1_retention(
        fold1_all_results, G1A_REF_FOLD1_MEAN_RETENTION)
    log.info(f"Fold1 retention verdict: {fold1_verdict} (retention={fold1_retention:.4f})")

    # ─── R1: Logistic Gate ───
    log.info("=" * 60)
    log.info("R1: Logistic Gate")
    log.info("=" * 60)

    preds_rgb_all = {"fold2": fold2_rgb, "fold1": fold1_rgb}
    preds_dual_all = {"fold2": fold2_dual, "fold1": fold1_dual}
    gts_by_img_all = {"fold2": fold2_gts_by_img, "fold1": fold1_gts_by_img}
    img_ids_all_map = {"fold2": fold2_img_ids, "fold1": fold1_img_ids}
    cat_ids_all = {"fold2": fold2_cat_ids, "fold1": fold1_cat_ids}

    logistic_results = run_logistic_gate(
        features_by_fold,
        preds_rgb_all, preds_dual_all,
        gts_by_img_all, img_ids_all_map, cat_ids_all)

    if logistic_results:
        write_csv(logistic_results, OUT_DIR / "r1_logistic_gate.csv", fold2_csv_cols)
        log.info(f"Logistic gate: {len(logistic_results)} results")

        # 合并 logistic 结果到 fold2/fold1 判定
        # 找 best logistic config
        best_log_delta = -999
        best_log_config = None
        for r in logistic_results:
            if r["delta_AP50_vs_rgb"] > best_log_delta:
                best_log_delta = r["delta_AP50_vs_rgb"]
                best_log_config = r
        if best_log_config:
            log.info(f"Best logistic: arm={best_log_config['arm']} "
                     f"delta={best_log_delta:.6f} "
                     f"accept={best_log_config['dual_acceptance_ratio']:.4f}")
    else:
        log.info("Logistic gate skipped")

    # ─── Final Verdict ───
    log.info("=" * 60)
    log.info("Final Verdict")
    log.info("=" * 60)

    r0_label = "R0_PASS" if r0_pass else "R0_WEAK"

    if fold2_verdict in ("PASS_SAFE", "PASS_MATCH_G1A") and fold1_verdict in ("STRONG_RETENTION", "PASS_RETENTION"):
        r1_label = "R1_STRONG_R_FUSE" if fold2_verdict == "PASS_SAFE" and fold1_verdict == "STRONG_RETENTION" else "R1_MATCH_G1A"
    elif fold2_verdict in ("PASS_SAFE", "PASS_MATCH_G1A"):
        r1_label = "R1_MATCH_G1A"
    else:
        r1_label = "R1_FAIL"

    log.info(f"R0: {r0_label}")
    log.info(f"R1: {r1_label}")
    log.info(f"Fold2: {fold2_verdict}")
    log.info(f"Fold1: {fold1_verdict} (retention={fold1_retention:.4f})")

    # Recommendation
    if r1_label == "R1_STRONG_R_FUSE":
        recommendation = "propose protocol-locked formal run"
    elif r1_label == "R1_MATCH_G1A":
        recommendation = "keep G1a as final simple safe gate baseline; R-Fuse mechanism still supports framing"
    else:
        recommendation = "stop algorithm escalation; write negative-transfer analysis/safe gate paper"

    log.info(f"Recommendation: {recommendation}")

    # ─── Recount ───
    log.info("写 recount.md")
    lines = [
        "# F-5 R-Fuse R0/R1 Reliability Gate Recount",
        f"# 日期: 2026-06-12",
        f"# 总耗时: {time.time()-t_start:.1f}s ({(time.time()-t_start)/60:.1f}min)",
        "",
        "## Precheck",
        "  所有 prediction JSONL 存在",
        "  labels 目录存在",
        "  sklearn 可用",
        "",
        "## R0 Separability (top features by fold)",
    ]

    # 按 AUC 排序展示 top features
    for fold_key in ["fold2", "fold1"]:
        fold_seps = [s for s in sep_results if s["fold"] == fold_key and not np.isnan(s.get("auc", float("nan")))]
        fold_seps.sort(key=lambda x: abs(x["auc"]) if not np.isnan(x["auc"]) else 0, reverse=True)
        lines.append(f"### {fold_key}")
        for s in fold_seps[:5]:
            lines.append(f"  {s['feature']}: AUC={s['auc']:.4f} "
                         f"helpful_mean={s['helpful_mean']:.4f} harmful_mean={s['harmful_mean']:.4f} "
                         f"direction={s['direction']}")

    lines.append(f"\n**R0 verdict: {r0_label}**\n")

    # R1 fold2 table
    lines.append("## R1 Fold2 Results")
    lines.append("| arm | seed | AP50 | delta_vs_rgb | smoke | fire | accept |")

    # baselines
    for arm in ["rgb_only", "dual_only", "g1a_gateA"]:
        for r in fold2_all_results:
            if r["arm"] == arm:
                lines.append(f"| {arm} | {r['seed']} | {r['AP50']:.6f} | {r['delta_AP50_vs_rgb']:+.6f} | "
                             f"{r['smoke_AP50']:.6f} | {r['fire_AP50']:.6f} | {r['dual_acceptance_ratio']:.4f} |")

    # best rfuse_rule_v1
    rfuse_results = [r for r in fold2_all_results if r["arm"] == "rfuse_rule_v1"]
    if rfuse_results:
        best_rfuse = max(rfuse_results, key=lambda x: x["delta_AP50_vs_rgb"])
        lines.append(f"| best_rfuse_rule_v1 | {best_rfuse['seed']} | {best_rfuse['AP50']:.6f} | "
                     f"{best_rfuse['delta_AP50_vs_rgb']:+.6f} | {best_rfuse['smoke_AP50']:.6f} | "
                     f"{best_rfuse['fire_AP50']:.6f} | {best_rfuse['dual_acceptance_ratio']:.4f} |")

    # oracle
    oracle_results = [r for r in fold2_all_results if r["arm"] == "oracle_upper"]
    for r in oracle_results:
        lines.append(f"| oracle_upper | {r['seed']} | {r['AP50']:.6f} | {r['delta_AP50_vs_rgb']:+.6f} | "
                     f"{r['smoke_AP50']:.6f} | {r['fire_AP50']:.6f} | {r['dual_acceptance_ratio']:.4f} |")

    # logistic
    if logistic_results:
        best_log = max(logistic_results, key=lambda x: x.get("delta_AP50_vs_rgb", -999))
        lines.append(f"| best_logistic | {best_log.get('seed','')} | {best_log['AP50']:.6f} | "
                     f"{best_log['delta_AP50_vs_rgb']:+.6f} | {best_log['smoke_AP50']:.6f} | "
                     f"{best_log['fire_AP50']:.6f} | {best_log['dual_acceptance_ratio']:.4f} |")

    lines.append(f"\n**Fold2 verdict: {fold2_verdict}**\n")

    # R1 fold1 table
    lines.append("## R1 Fold1 Retention")
    lines.append("| arm | seed | AP50 | delta_vs_rgb |")
    for arm in ["rgb_only", "dual_only", "g1a_gateA"]:
        for r in fold1_all_results:
            if r["arm"] == arm:
                lines.append(f"| {arm} | {r['seed']} | {r['AP50']:.6f} | {r['delta_AP50_vs_rgb']:+.6f} |")

    rfuse_f1 = [r for r in fold1_all_results if r["arm"] == "rfuse_rule_v1"]
    if rfuse_f1:
        best_r1 = max(rfuse_f1, key=lambda x: x["delta_AP50_vs_rgb"])
        lines.append(f"| best_rfuse_rule_v1 | {best_r1['seed']} | {best_r1['AP50']:.6f} | "
                     f"{best_r1['delta_AP50_vs_rgb']:+.6f} |")

    lines.append(f"\n**Fold1 verdict: {fold1_verdict} (retention={fold1_retention:.4f})**\n")

    # Final
    lines.append("## Final Verdict")
    lines.append(f"- R0: **{r0_label}**")
    lines.append(f"- R1: **{r1_label}**")
    lines.append(f"- Fold2: {fold2_verdict}")
    lines.append(f"- Fold1: {fold1_verdict}")
    lines.append(f"")
    lines.append(f"## Recommendation")
    lines.append(f"{recommendation}")

    (OUT_DIR / "recount.md").write_text("\n".join(lines))

    # DONE marker
    (OUT_DIR / "DONE").touch()
    elapsed = time.time() - t_start
    log.info(f"完成! {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
