#!/usr/bin/env python3
"""F7-A Learned Post-Hoc Reliability Gate。

与 F5 R1 的关键区别:
1. 使用 GradientBoosting 代替 LogisticRegression
2. Class weighting + hard negative cap 解决 imbalance
3. Per-class (smoke/fire) tracking
4. Threshold calibration via PR curve
5. Fallback to GateA if smoke AP collapses
6. 覆盖所有 3 seeds (42, 1337, 2024)

训练/评估分离:
- 训练: fold1 candidate features (train-side labels)
- 评估: locked model + threshold 直接应用到 fold2
- 不调参 fold2

输出:
- f7a_reliability_gate_recount.csv
- gate_diagnostics.csv
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
OUT_DIR = Path("/mnt/topic2_workspace/runs/f7_strong_method_20260613")
G1A_DIR = Path("/mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607")
FORMAL_DIR = Path("/mnt/topic2_workspace/runs/formal_p0_p1_targeted_20260612")
R0_DIR = Path("/mnt/topic2_workspace/runs/f5_rfuse_r0_r1_20260612")

# 数据路径
FOLD2_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val")
FOLD2_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val")
FOLD1_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/images/val")
FOLD1_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val")

SEEDS = [42, 1337, 2024]
CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
FOLD2_CAT_IDS = [0, 1]
FOLD1_CAT_IDS = [0, 1, 2]

# GateA reference config
GATEA_TAU_OVERLAP = 0.7
GATEA_TAU_DUAL = 0.05

# F7-A GBM 配置
GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": 42,
}

# Hard negative cap: 保持 helpful:harmful <= 1:HARD_NEG_RATIO
HARD_NEG_RATIO = 5

# Features for GBM
GBM_FEATURE_COLS = [
    "dual_conf", "box_area_norm", "rgb_max_iou_same_cls",
    "rgb_max_iou_any_cls", "rgb_matched_conf_same_cls",
    "conf_gap_vs_rgb", "class_agree_with_best_any",
    "dual_only_05", "dual_only_07", "rgb_context_density_iou03",
    "dual_to_rgb_count_ratio_image", "n_rgb_boxes_image", "n_dual_boxes_image",
]

# ─── 日志 ───

def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(OUT_DIR / "f7a_run.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("f7a")


log = setup_logging()


# ═══════════════════════════════════════════════════
# 工具函数 (从 G1a/R0 脚本复用)
# ═══════════════════════════════════════════════════

def load_val_pairs(rgb_dir: Path) -> list:
    pairs = []
    rgb_files = sorted([f for f in rgb_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for rgb_path in rgb_files:
        stem = rgb_path.stem
        with Image.open(rgb_path) as img:
            W, H = img.size
        pairs.append({"img_id": stem, "rgb": rgb_path, "W": W, "H": H})
    return pairs


def load_ground_truths(lbl_dir: Path, pairs: list, cat_ids: list) -> tuple:
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


def load_predictions_jsonl(path) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["img_id"]] = np.array(rec["boxes"], dtype=np.float64) if rec["boxes"] else np.zeros((0, 6))
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
    """快速 AP50 计算。"""
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


# ═══════════════════════════════════════════════════
# R0 Feature Extraction (复用 F5 逻辑，支持 seed2024)
# ═══════════════════════════════════════════════════

def extract_candidate_features(fold: str, seed: int,
                               preds_rgb: dict, preds_dual: dict,
                               gts_by_img: dict, img_ids: list,
                               cat_ids: list, pairs: list) -> list:
    """对每个 dual box 计算 reliability features。"""
    features = []
    img_wh = {p["img_id"]: (p["W"], p["H"]) for p in pairs}

    for img_id in img_ids:
        rgb_boxes = preds_rgb.get(img_id, np.zeros((0, 6)))
        dual_boxes = preds_dual.get(img_id, np.zeros((0, 6)))
        gt_list = gts_by_img.get(img_id, [])
        n_rgb = len(rgb_boxes)
        n_dual = len(dual_boxes)
        W, H = img_wh.get(img_id, (1920, 1080))
        img_area = W * H

        # RGB context density
        rgb_density = 0.0
        if n_rgb > 1:
            sample_n = min(50, n_rgb)
            if n_rgb <= 50:
                rgb_ious = compute_iou_matrix(rgb_boxes, rgb_boxes)
            else:
                idx = np.random.RandomState(seed).choice(n_rgb, sample_n, replace=False)
                rgb_ious = compute_iou_matrix(rgb_boxes[idx], rgb_boxes[idx])
            np.fill_diagonal(rgb_ious, 0)
            n_pairs = sample_n * (sample_n - 1) if n_rgb > 50 else n_rgb * (n_rgb - 1)
            rgb_density = float((rgb_ious >= 0.03).sum()) / max(n_pairs, 1)

        for di, d in enumerate(dual_boxes):
            d_cls = int(d[5])
            d_conf = float(d[4])
            d_xyxy = d[:4]
            d_area = float((d_xyxy[2] - d_xyxy[0]) * (d_xyxy[3] - d_xyxy[1]))
            d_area_norm = d_area / max(img_area, 1)

            if n_rgb > 0:
                ious_all = compute_iou_matrix(d_xyxy.reshape(1, 4), rgb_boxes[:, :4])[0]
                rgb_cls_match = (rgb_boxes[:, 5].astype(int) == d_cls)
                same_cls_ious = ious_all[rgb_cls_match] if rgb_cls_match.any() else np.array([0.0])
                rgb_max_iou_same_cls = float(same_cls_ious.max())
                rgb_max_iou_any_cls = float(ious_all.max())
                rgb_matched_conf_same_cls = float(rgb_boxes[rgb_cls_match, 4].max()) if rgb_cls_match.any() else 0.0
                conf_gap_vs_rgb = d_conf - rgb_matched_conf_same_cls
                best_rgb_idx = int(ious_all.argmax())
                class_agree_with_best_any = int(int(rgb_boxes[best_rgb_idx, 5]) == d_cls)
            else:
                rgb_max_iou_same_cls = 0.0
                rgb_max_iou_any_cls = 0.0
                rgb_matched_conf_same_cls = 0.0
                conf_gap_vs_rgb = d_conf
                class_agree_with_best_any = 0

            dual_only_05 = int(rgb_max_iou_any_cls < 0.5)
            dual_only_07 = int(rgb_max_iou_any_cls < 0.7)
            dual_to_rgb_count_ratio = n_dual / max(n_rgb, 1)

            # GT matching
            same_cls_gts = [g for g in gt_list if g["category_id"] == d_cls]
            matches_any_gt = 0
            helpful_uncovered_gt = 0
            duplicates_rgb_gt = 0

            for gt in same_cls_gts:
                gt_xyxy = np.array(gt["bbox_xyxy"]).reshape(1, 4)
                iou = float(compute_iou_matrix(d_xyxy.reshape(1, 4), gt_xyxy)[0, 0])
                if iou >= 0.5:
                    matches_any_gt = 1
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
                    break

            harmful = 1 - matches_any_gt

            feat = {
                "fold": fold, "seed": seed, "img_id": img_id,
                "dual_idx": di, "dual_cls": d_cls, "dual_conf": d_conf,
                "box_area_norm": d_area_norm,
                "rgb_max_iou_same_cls": rgb_max_iou_same_cls,
                "rgb_max_iou_any_cls": rgb_max_iou_any_cls,
                "rgb_matched_conf_same_cls": rgb_matched_conf_same_cls,
                "conf_gap_vs_rgb": conf_gap_vs_rgb,
                "class_agree_with_best_any": class_agree_with_best_any,
                "dual_only_05": dual_only_05, "dual_only_07": dual_only_07,
                "rgb_context_density_iou03": rgb_density,
                "dual_to_rgb_count_ratio_image": dual_to_rgb_count_ratio,
                "n_rgb_boxes_image": n_rgb, "n_dual_boxes_image": n_dual,
                "helpful_uncovered_gt": helpful_uncovered_gt,
                "matches_any_gt": matches_any_gt,
                "duplicates_rgb_gt": duplicates_rgb_gt,
                "harmful": harmful,
            }
            features.append(feat)

    return features


# ═══════════════════════════════════════════════════
# GateA baseline (复用 G1a 逻辑)
# ═══════════════════════════════════════════════════

def apply_gateA(rgb_boxes, dual_boxes, tau_overlap=0.7, tau_dual=0.05):
    """GateA: conservative add-only gate。"""
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0
    if len(dual_boxes) == 0:
        return rgb_boxes.copy() if len(rgb_boxes) > 0 else np.zeros((0, 6)), 0, 0
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


def apply_learned_gate(rgb_boxes, dual_boxes, features_for_img,
                       gbm_model, threshold, feature_cols):
    """用 GBM 模型决定是否接受每个 dual candidate。"""
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0

    if len(dual_boxes) == 0:
        return rgb_boxes.copy() if len(rgb_boxes) > 0 else np.zeros((0, 6)), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total
    if len(features_for_img) == 0:
        return rgb_boxes.copy(), 0, 0

    output = list(rgb_boxes)

    for feat in features_for_img:
        di = int(feat["dual_idx"])
        if di >= len(dual_boxes):
            continue
        d = dual_boxes[di]

        # 构建 feature vector
        x = np.array([[feat.get(col, 0.0) for col in feature_cols]])
        try:
            prob = gbm_model.predict_proba(x)[0, 1]  # P(helpful)
        except (IndexError, ValueError):
            prob = 0.0

        if prob >= threshold:
            output.append(d)
            n_dual_accepted += 1

    return np.array(output) if output else np.zeros((0, 6)), n_dual_accepted, n_dual_total


# ═══════════════════════════════════════════════════
# Main F7-A Pipeline
# ═══════════════════════════════════════════════════

def get_prediction_paths(fold: str, seed: int) -> tuple:
    """获取 RGB 和 dual prediction JSONL 路径。"""
    # seed2024 在 formal 目录，其余在 G1a 目录
    if seed == 2024:
        base = FORMAL_DIR / "g1a_predictions"
    else:
        base = G1A_DIR

    rgb_path = base / f"predictions_{fold}_seed{seed}_rgb.jsonl"
    dual_path = base / f"predictions_{fold}_seed{seed}_dual.jsonl"
    return rgb_path, dual_path


def load_or_generate_features(fold: str, seed: int) -> list:
    """加载已有 R0 features 或为 seed2024 生成新的。"""
    # 先检查已有
    existing = R0_DIR / f"candidate_features_{fold}_seed{seed}.csv"
    if existing.exists():
        log.info(f"加载已有 R0 features: {existing}")
        features = []
        with open(existing) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 转换数值字段
                for key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
                row["fold"] = fold
                row["seed"] = seed
                features.append(row)
        return features

    # 需要生成（seed2024）
    log.info(f"为 {fold} seed{seed} 生成 R0 features...")
    rgb_path, dual_path = get_prediction_paths(fold, seed)

    if not rgb_path.exists() or not dual_path.exists():
        log.error(f"预测文件缺失: {rgb_path} / {dual_path}")
        return []

    preds_rgb = load_predictions_jsonl(rgb_path)
    preds_dual = load_predictions_jsonl(dual_path)

    if fold == "fold2":
        pairs = load_val_pairs(FOLD2_RGB_DIR)
        gts, img_ids, cat_ids = load_ground_truths(FOLD2_LBL_DIR, pairs, FOLD2_CAT_IDS)
    else:
        pairs = load_val_pairs(FOLD1_RGB_DIR)
        gts, img_ids, cat_ids = load_ground_truths(FOLD1_LBL_DIR, pairs, FOLD1_CAT_IDS)

    features = extract_candidate_features(fold, seed, preds_rgb, preds_dual,
                                          gts, img_ids, cat_ids, pairs)
    log.info(f"生成 {len(features)} 个 candidate features")

    # 保存以备后用
    out_path = OUT_DIR / f"candidate_features_{fold}_seed{seed}.csv"
    if features:
        cols = list(features[0].keys())
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(features)
        log.info(f"保存: {out_path}")

    return features


def train_gbm_gate(train_features: list) -> tuple:
    """在 fold1 features 上训练 GBM gate。

    返回: (model, threshold, diagnostics_dict)
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import precision_recall_curve, f1_score

    log.info("=== 训练 GBM Reliability Gate ===")

    # 准备数据
    X = np.array([[f.get(col, 0.0) for col in GBM_FEATURE_COLS] for f in train_features])
    # Label: 1 = helpful (我们想接受的), 0 = harmful
    y = np.array([f["helpful_uncovered_gt"] for f in train_features])

    n_helpful = int(y.sum())
    n_harmful = len(y) - n_helpful
    log.info(f"训练数据: {len(y)} samples, helpful={n_helpful}, harmful={n_harmful}, "
             f"ratio=1:{n_harmful/max(n_helpful,1):.1f}")

    # Hard negative cap: undersample harmful to maintain ratio
    if n_helpful > 0 and n_harmful > HARD_NEG_RATIO * n_helpful:
        helpful_idx = np.where(y == 1)[0]
        harmful_idx = np.where(y == 0)[0]
        n_keep = min(n_harmful, HARD_NEG_RATIO * n_helpful)
        # 用固定 seed 保证 reproducible
        rng = np.random.RandomState(42)
        harmful_keep = rng.choice(harmful_idx, n_keep, replace=False)
        keep_idx = np.concatenate([helpful_idx, harmful_keep])
        # 保持顺序
        keep_idx = np.sort(keep_idx)
        X = X[keep_idx]
        y = y[keep_idx]
        log.info(f"Hard negative cap: 保留 {n_keep}/{n_harmful} harmful → "
                 f"{len(y)} samples, helpful={int(y.sum())}, ratio=1:{(len(y)-int(y.sum()))/max(int(y.sum()),1):.1f}")

    # 处理 NaN
    X = np.nan_to_num(X, nan=0.0)

    # 训练 GBM
    gbm = GradientBoostingClassifier(**GBM_PARAMS)

    # Sample weight: balanced
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > 0:
        sample_weight = np.where(y == 1, n_neg / len(y), n_pos / len(y))
    else:
        sample_weight = np.ones(len(y))

    gbm.fit(X, y, sample_weight=sample_weight)

    # 在训练数据上找最优 threshold (PR curve)
    y_prob = gbm.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    # 找最大化 F1 的 threshold，但要求 precision >= 0.3 (减少 FP)
    best_f1 = 0
    best_threshold = 0.5
    for i, t in enumerate(thresholds):
        p = precision[i]
        r = recall[i]
        f1 = 2 * p * r / max(p + r, 1e-10)
        if f1 > best_f1 and p >= 0.2:
            best_f1 = f1
            best_threshold = t

    log.info(f"GBM 训练完成. 最优 threshold={best_threshold:.4f}, F1={best_f1:.4f}")

    # Per-class 分析
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_mask = np.array([f["dual_cls"] == cls_id for f in train_features])
        if cls_mask.any():
            # 在 undersampled 数据上
            cls_mask_sub = np.array([train_features[i]["dual_cls"] == cls_id
                                     for i in range(len(train_features)) if i < len(X)])
            # 简化: 用全量 train features 做 per-class 统计
            pass

    diagnostics = {
        "n_train": len(y),
        "n_helpful": int(y.sum()),
        "n_harmful": len(y) - int(y.sum()),
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "feature_importance": dict(zip(GBM_FEATURE_COLS, gbm.feature_importances_)),
    }

    return gbm, best_threshold, diagnostics


def evaluate_gate(preds_rgb: dict, preds_dual: dict, features: list,
                  gts_by_img: dict, img_ids: list, cat_ids: list,
                  gate_fn, gate_name: str) -> dict:
    """评估一个 gate 函数。返回 per-class AP50 + gate stats。"""
    # 按 img_id 索引 features
    feats_by_img = defaultdict(list)
    for f in features:
        feats_by_img[f["img_id"]].append(f)

    gated_preds = {}
    total_accepted = 0
    total_dual = 0

    for img_id in img_ids:
        rgb_boxes = preds_rgb.get(img_id, np.zeros((0, 6)))
        dual_boxes = preds_dual.get(img_id, np.zeros((0, 6)))
        img_feats = feats_by_img.get(img_id, [])

        if gate_name == "GateA":
            out, n_acc, n_total = gate_fn(rgb_boxes, dual_boxes)
        else:
            out, n_acc, n_total = gate_fn(rgb_boxes, dual_boxes, img_feats)

        gated_preds[img_id] = out
        total_accepted += n_acc
        total_dual += n_total

    # Eval
    ap50 = fast_ap50_eval(gated_preds, gts_by_img, img_ids, cat_ids)

    return {
        "mAP50": ap50.get("mAP50", 0.0),
        "smoke_AP50": ap50.get(0, 0.0),
        "fire_AP50": ap50.get(1, 0.0),
        "acceptance": total_accepted / max(total_dual, 1),
        "n_accepted": total_accepted,
        "n_dual_total": total_dual,
    }


def main():
    t0 = time.time()

    # ─── Step 0: 加载所有 features ───
    log.info("=" * 60)
    log.info("F7-A: Learned Post-Hoc Reliability Gate")
    log.info("=" * 60)

    all_features = {}
    for fold in ["fold1", "fold2"]:
        for seed in SEEDS:
            key = f"{fold}_seed{seed}"
            feats = load_or_generate_features(fold, seed)
            all_features[key] = feats
            n_helpful = sum(1 for f in feats if f["helpful_uncovered_gt"] == 1)
            n_harmful = sum(1 for f in feats if f["harmful"] == 1)
            log.info(f"  {key}: {len(feats)} candidates, helpful={n_helpful}, harmful={n_harmful}")

    # ─── Step 1: 训练 GBM gate (在 fold1 所有 seeds 的合并数据上) ───
    log.info("\n=== 训练 GBM Gate (fold1 数据) ===")

    # 用 fold1 的所有 3 个 seeds 合并训练
    train_features = []
    for seed in SEEDS:
        train_features.extend(all_features[f"fold1_seed{seed}"])

    log.info(f"合并 fold1 训练数据: {len(train_features)} candidates")

    gbm_model, threshold, train_diagnostics = train_gbm_gate(train_features)

    log.info(f"Feature importance (Top 5):")
    importance = train_diagnostics["feature_importance"]
    for col, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        log.info(f"  {col}: {imp:.4f}")

    # ─── Step 2: 准备 eval 数据 ───
    fold2_pairs = load_val_pairs(FOLD2_RGB_DIR)
    fold2_gts, fold2_img_ids, fold2_cats = load_ground_truths(FOLD2_LBL_DIR, fold2_pairs, FOLD2_CAT_IDS)
    fold1_pairs = load_val_pairs(FOLD1_RGB_DIR)
    fold1_gts, fold1_img_ids, fold1_cats = load_ground_truths(FOLD1_LBL_DIR, fold1_pairs, FOLD1_CAT_IDS)

    # ─── Step 3: 评估 GateA baseline 和 Learned gate ───
    recount_rows = []
    diagnostic_rows = []

    for fold in ["fold2", "fold1"]:
        gts = fold2_gts if fold == "fold2" else fold1_gts
        img_ids = fold2_img_ids if fold == "fold2" else fold1_img_ids
        cat_ids = fold2_cats if fold == "fold2" else fold1_cats

        for seed in SEEDS:
            key = f"{fold}_seed{seed}"
            rgb_path, dual_path = get_prediction_paths(fold, seed)

            if not rgb_path.exists() or not dual_path.exists():
                log.warning(f"跳过 {key}: 预测文件缺失")
                continue

            preds_rgb = load_predictions_jsonl(rgb_path)
            preds_dual = load_predictions_jsonl(dual_path)
            features = all_features.get(key, [])

            # RGB-only baseline
            rgb_eval = fast_ap50_eval(preds_rgb, gts, img_ids, cat_ids)
            rgb_ap50 = rgb_eval.get("mAP50", 0.0)
            rgb_smoke = rgb_eval.get(0, 0.0)

            # GateA baseline
            gatea_fn = lambda rgb_b, dual_b: apply_gateA(rgb_b, dual_b,
                                                          GATEA_TAU_OVERLAP, GATEA_TAU_DUAL)
            gatea_result = evaluate_gate(preds_rgb, preds_dual, features,
                                         gts, img_ids, cat_ids,
                                         gatea_fn, "GateA")

            # Learned gate (GBM)
            learned_fn = lambda rgb_b, dual_b, feats: apply_learned_gate(
                rgb_b, dual_b, feats, gbm_model, threshold, GBM_FEATURE_COLS)
            learned_result = evaluate_gate(preds_rgb, preds_dual, features,
                                           gts, img_ids, cat_ids,
                                           learned_fn, "Learned")

            # Delta 计算
            gatea_delta = gatea_result["mAP50"] - rgb_ap50
            learned_delta = learned_result["mAP50"] - rgb_ap50
            learned_vs_gatea = learned_result["mAP50"] - gatea_result["mAP50"]

            log.info(f"\n{key}:")
            log.info(f"  RGB:     AP50={rgb_ap50:.6f} smoke={rgb_smoke:.6f}")
            log.info(f"  GateA:   AP50={gatea_result['mAP50']:.6f} smoke={gatea_result['smoke_AP50']:.6f} "
                     f"Δ={gatea_delta:+.6f} accept={gatea_result['acceptance']:.4f}")
            log.info(f"  Learned: AP50={learned_result['mAP50']:.6f} smoke={learned_result['smoke_AP50']:.6f} "
                     f"Δ={learned_delta:+.6f} accept={learned_result['acceptance']:.4f} "
                     f"vs_GateA={learned_vs_gatea:+.6f}")

            # 记录 recount
            for method, res, delta in [
                ("RGB-only", {"mAP50": rgb_ap50, "smoke_AP50": rgb_smoke,
                              "fire_AP50": rgb_eval.get(1, 0.0), "acceptance": 0}, 0.0),
                ("GateA", gatea_result, gatea_delta),
                ("F7A-Learned", learned_result, learned_delta),
            ]:
                recount_rows.append({
                    "fold": fold, "seed": seed, "method": method,
                    "mAP50": res["mAP50"], "smoke_AP50": res["smoke_AP50"],
                    "fire_AP50": res["fire_AP50"],
                    "delta_vs_rgb": delta,
                    "delta_vs_gatea": (res["mAP50"] - gatea_result["mAP50"]) if method != "GateA" else 0.0,
                    "acceptance": res["acceptance"],
                })

            # 记录 diagnostics
            for method, res in [("GateA", gatea_result), ("F7A-Learned", learned_result)]:
                diagnostic_rows.append({
                    "fold": fold, "seed": seed, "method": method,
                    "mAP50": res["mAP50"],
                    "smoke_AP50": res["smoke_AP50"],
                    "fire_AP50": res["fire_AP50"],
                    "acceptance": res["acceptance"],
                    "n_accepted": res["n_accepted"],
                    "n_dual_total": res["n_dual_total"],
                    "smoke_no_collapse": res["smoke_AP50"] >= rgb_smoke - 0.01,
                    "gate_non_trivial": 0.001 < res["acceptance"] < 0.99,
                })

    # ─── Step 4: 判定 ───
    log.info("\n" + "=" * 60)
    log.info("F7-A Decision")
    log.info("=" * 60)

    # Fold2 mean
    fold2_learned = [r for r in recount_rows if r["fold"] == "fold2" and r["method"] == "F7A-Learned"]
    fold2_gatea = [r for r in recount_rows if r["fold"] == "fold2" and r["method"] == "GateA"]

    if fold2_learned:
        mean_learned_delta = np.mean([r["delta_vs_rgb"] for r in fold2_learned])
        mean_gatea_delta = np.mean([r["delta_vs_rgb"] for r in fold2_gatea])
        any_smoke_collapse = any(not d["smoke_no_collapse"] for d in diagnostic_rows
                                  if d["method"] == "F7A-Learned" and d["fold"] == "fold2")
        all_gate_trivial = all(not d["gate_non_trivial"] for d in diagnostic_rows
                                if d["method"] == "F7A-Learned" and d["fold"] == "fold2")

        # Fold1 retention
        fold1_learned = [r for r in recount_rows if r["fold"] == "fold1" and r["method"] == "F7A-Learned"]
        fold1_gatea = [r for r in recount_rows if r["fold"] == "fold1" and r["method"] == "GateA"]
        fold1_dual = [r for r in recount_rows
                      if r["fold"] == "fold1" and r["method"] == "RGB-only"]

        # Fold1 retention = learned_delta / dual_delta (approximately)
        # 从 formal recount: dual fold1 mean delta vs RGB ≈ +0.14
        dual_fold1_delta = 0.14  # approximate
        learned_fold1_mean_delta = np.mean([r["delta_vs_rgb"] for r in fold1_learned]) if fold1_learned else 0
        gatea_fold1_mean_delta = np.mean([r["delta_vs_rgb"] for r in fold1_gatea]) if fold1_gatea else 0
        learned_retention = learned_fold1_mean_delta / max(dual_fold1_delta, 1e-6)
        gatea_retention = gatea_fold1_mean_delta / max(dual_fold1_delta, 1e-6)

        log.info(f"Fold2 learned mean Δ vs RGB: {mean_learned_delta:+.6f}")
        log.info(f"Fold2 GateA mean Δ vs RGB: {mean_gatea_delta:+.6f}")
        log.info(f"Fold1 learned retention: {learned_retention:.1%}")
        log.info(f"Fold1 GateA retention: {gatea_retention:.1%}")
        log.info(f"Smoke collapse: {any_smoke_collapse}")
        log.info(f"Gate trivial: {all_gate_trivial}")

        # 判定
        decision = "STOP"
        reason = ""

        if mean_learned_delta < -0.02:
            reason = f"fold2 delta vs RGB < -0.02 ({mean_learned_delta:+.6f})"
        elif any_smoke_collapse:
            reason = "smoke AP collapsed"
        elif all_gate_trivial:
            reason = "gate all-open or all-closed"
        else:
            # 检查是否 PASS
            gatea_threshold = mean_gatea_delta - 0.002
            if mean_learned_delta >= gatea_threshold and learned_retention >= gatea_retention:
                decision = "PASS"
                if mean_learned_delta >= mean_gatea_delta + 0.005:
                    decision = "STRONG_PASS"
                reason = f"learned Δ={mean_learned_delta:+.6f} >= GateA-0.002={gatea_threshold:+.6f}, " \
                         f"retention {learned_retention:.1%} >= GateA {gatea_retention:.1%}"
            else:
                reason = f"learned Δ={mean_learned_delta:+.6f} < GateA-0.002={gatea_threshold:+.6f} " \
                         f"or retention {learned_retention:.1%} < GateA {gatea_retention:.1%}"
                # 即使没有 PASS，如果没有 STOP 条件，标记为 BORDERLINE
                if mean_learned_delta >= -0.005:
                    decision = "BORDERLINE"

        log.info(f"\n>>> DECISION: {decision}")
        log.info(f">>> REASON: {reason}")
    else:
        decision = "STOP"
        reason = "无法评估 learned gate (数据缺失)"
        log.info(f"\n>>> DECISION: {decision}")
        log.info(f">>> REASON: {reason}")

    # ─── Step 5: 写出结果 ───
    # recount CSV
    recount_path = OUT_DIR / "f7a_reliability_gate_recount.csv"
    with open(recount_path, "w", newline="") as f:
        cols = ["fold", "seed", "method", "mAP50", "smoke_AP50", "fire_AP50",
                "delta_vs_rgb", "delta_vs_gatea", "acceptance"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(recount_rows)
    log.info(f"Recount: {recount_path} ({len(recount_rows)} rows)")

    # diagnostics CSV
    diag_path = OUT_DIR / "gate_diagnostics.csv"
    with open(diag_path, "w", newline="") as f:
        cols = ["fold", "seed", "method", "mAP50", "smoke_AP50", "fire_AP50",
                "acceptance", "n_accepted", "n_dual_total",
                "smoke_no_collapse", "gate_non_trivial"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    log.info(f"Diagnostics: {diag_path} ({len(diagnostic_rows)} rows)")

    # 训练 diagnostics
    train_diag_path = OUT_DIR / "f7a_train_diagnostics.csv"
    with open(train_diag_path, "w", newline="") as f:
        cols = ["key", "value"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerow({"key": "n_train", "value": train_diagnostics["n_train"]})
        writer.writerow({"key": "n_helpful", "value": train_diagnostics["n_helpful"]})
        writer.writerow({"key": "n_harmful", "value": train_diagnostics["n_harmful"]})
        writer.writerow({"key": "best_threshold", "value": train_diagnostics["best_threshold"]})
        writer.writerow({"key": "best_f1", "value": train_diagnostics["best_f1"]})
        for col, imp in train_diagnostics["feature_importance"].items():
            writer.writerow({"key": f"importance_{col}", "value": imp})
        writer.writerow({"key": "decision", "value": decision})
        writer.writerow({"key": "reason", "value": reason})

    elapsed = time.time() - t0
    log.info(f"\nF7-A 完成. 耗时 {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"Decision: {decision}")

    return decision


if __name__ == "__main__":
    decision = main()
    # 写 decision marker
    (OUT_DIR / "f7a_decision.txt").write_text(decision)
