#!/usr/bin/env python3
"""F-5-G1a SafeLateGate: prediction + post-hoc gate sweep 实验（优化版）。

核心优化:
1. 预过滤 dual boxes (conf >= 0.01)，从 253K 降至 ~10-20K
2. Sweep 用 fast AP50 evaluator（只算 IoU=0.5），跳过完整 COCO eval
3. 找到 best config 后再跑完整 COCO eval 验证

判定标准:
- PASS_NONTRIVIAL: fold2 mean Δ >= -0.005, smoke AP50 drop <= 0.01, acceptance >= 0.01
- TRIVIAL_SAFE_ONLY: RGB-only safe 但所有 non-trivial gate acceptance < 0.01
- FAIL_UNSAFE: best non-trivial gate mean Δ < -0.02 或 smoke AP50 drop > 0.03
- BLOCKED_EVAL: evaluator sanity 差距 > 0.05
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
sys.path.insert(0, "/mnt/topic2_workspace/engineering_packs/MutilModel_199099010")

from ultralytics import YOLOMM
from ultralytics.utils.coco_eval_bbox_mm import COCOevalBBoxMM

# ─── 常量 ───

OUT_DIR = Path("/mnt/topic2_workspace/runs/f5_g1a_safe_late_gate_20260607")

# 数据路径
FOLD2_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/images/val")
FOLD2_IR_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/image/val")
FOLD2_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold2_nirfree/labels/val")

FOLD1_RGB_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/images/val")
FOLD1_IR_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/image/val")
FOLD1_LBL_DIR = Path("/mnt/topic2_datasets/fire_loco_fold1/labels/val")

# Source baselines
SOURCE_RGB_MAP50 = {42: 0.269590, 1337: 0.264490}
SOURCE_DUAL_MAP50 = {42: 0.227580, 1337: 0.252130}

# Checkpoint 映射
FOLD2_CKPTS = {
    42: {
        "rgb": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbonly_fold2_seed42/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbsafe_dual_fold2_seed42/weights/best.pt",
    },
    1337: {
        "rgb": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbonly_fold2_seed1337/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f4_nirfree_rgbsafe_dual_fold2_seed1337/weights/best.pt",
    },
}

FOLD1_CKPTS = {
    42: {
        "rgb": "/mnt/topic2_workspace/runs/phase_f2_rgbonly_fold1_seed42/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f2_dual_fold1_seed42/weights/best.pt",
    },
    1337: {
        "rgb": "/mnt/topic2_workspace/runs/phase_f2_rgbonly_fold1_seed1337/weights/best.pt",
        "dual": "/mnt/topic2_workspace/runs/phase_f2_dual_fold1_seed1337/weights/best.pt",
    },
}

SEEDS = [42, 1337]
CLASS_NAMES = {0: "smoke", 1: "fire", 2: "person"}
FOLD2_CAT_IDS = [0, 1]
FOLD1_CAT_IDS = [0, 1, 2]

# Gate sweep 参数
TAU_OVERLAP = [0.5, 0.6, 0.7]
TAU_DUAL = [0.05, 0.10, 0.20, 0.30]
TAU_RGB_UNCERTAIN = [0.30, 0.50, 0.70]
MARGIN = [0.05, 0.10, 0.20]

# 预过滤阈值: 低于此值的 dual box 在任何 gate 中都不会被接受
DUAL_PREFILTER_CONF = 0.01

# 判定阈值
PASS_NONTRIVIAL_MEAN_DELTA = -0.005
PASS_NONTRIVIAL_SMOKE_DROP = 0.01
PASS_NONTRIVIAL_ACCEPT = 0.01
FAIL_UNSAFE_MEAN_DELTA = -0.02
FAIL_UNSAFE_SMOKE_DROP = 0.03

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
    return logging.getLogger("f5_g1a")


log = setup_logging()


# ─── 数据加载 ───

def load_val_pairs(rgb_dir: Path, ir_dir: Path = None) -> list:
    """加载 val 图片列表。"""
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


def load_ground_truths(lbl_dir: Path, pairs: list, cat_ids: list) -> tuple:
    """解析 YOLO labels → pixel-xywh GT 列表 + 按图片分组的 GT dict。"""
    gts = []
    gts_by_img = defaultdict(list)  # {img_id: [{category_id, bbox_xyxy}, ...]}
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
            gts.append({
                "image_id": pair["img_id"],
                "category_id": cls,
                "bbox": [px, py, pw, ph],
                "area": pw * ph,
            })
            # 同时存 xyxy 格式用于 fast AP50
            gts_by_img[pair["img_id"]].append({
                "category_id": cls,
                "bbox_xyxy": [px, py, px + pw, py + ph],
            })

    log.info(f"GT: {len(gts)} boxes, {len(img_ids)} imgs, classes={cat_ids}")
    return gts, gts_by_img, img_ids, cat_ids


# ─── 预测 ───

def run_predictions(ckpt_path: str, pairs: list, is_rgb_only: bool, label: str) -> dict:
    """运行推理，返回 {img_id: boxes(N,6)} in pixel-xyxy。"""
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


def save_predictions_jsonl(preds_dict: dict, path: Path):
    with open(path, "w") as f:
        for img_id in sorted(preds_dict.keys()):
            boxes = preds_dict[img_id]
            f.write(json.dumps({"img_id": img_id, "boxes": boxes.tolist() if len(boxes) > 0 else []}) + "\n")
    log.info(f"保存: {path} ({len(preds_dict)} 行)")


def load_predictions_jsonl(path: Path) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            preds[rec["img_id"]] = np.array(rec["boxes"], dtype=np.float64) if rec["boxes"] else np.zeros((0, 6))
    log.info(f"加载: {path} ({len(preds)} 行)")
    return preds


# ─── COCO 评估（完整版，只用于 sanity check 和 final verification）───

def boxes_to_coco_dt(img_id: str, boxes: np.ndarray) -> list:
    dts = []
    for i in range(len(boxes)):
        x1, y1, x2, y2, conf, cls = boxes[i]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        dts.append({
            "image_id": img_id, "category_id": int(cls),
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "score": float(conf), "area": float(w * h),
        })
    return dts


def full_coco_eval(gts: list, dts: list, img_ids: list, cat_ids: list) -> tuple:
    """完整 COCO 评估（10 IoU 阈值）。用于 sanity check 和最终验证。"""
    evaluator = COCOevalBBoxMM()
    evaluator.set_data(gts, dts, img_ids, cat_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    stats = evaluator.summarize()
    per_class = evaluator.compute_per_class_metrics()
    return stats, per_class


def full_coco_eval_preds(preds_dict: dict, gts: list, img_ids: list, cat_ids: list) -> tuple:
    all_dts = []
    for img_id in sorted(preds_dict.keys()):
        boxes = preds_dict[img_id]
        if len(boxes) > 0:
            all_dts.extend(boxes_to_coco_dt(img_id, boxes))
    return full_coco_eval(gts, all_dts, img_ids, cat_ids)


# ─── Fast AP50 评估（只算 IoU=0.5，用于 sweep）───

def fast_ap50_eval(preds_dict: dict, gts_by_img: dict, img_ids: list, cat_ids: list) -> dict:
    """快速 AP50 计算。返回 {cat_id: AP50, 'mAP50': mean}。"""
    per_class_ap50 = {}

    for cat_id in cat_ids:
        # 收集所有 predictions 和 GT matches
        all_scores = []
        all_tp = []
        n_gt_total = 0

        for img_id in img_ids:
            # GT boxes for this class
            gt_list = [g for g in gts_by_img.get(img_id, []) if g["category_id"] == cat_id]
            gt_xyxy = np.array([g["bbox_xyxy"] for g in gt_list], dtype=np.float64) if gt_list else np.zeros((0, 4))
            gt_matched = np.zeros(len(gt_list), dtype=bool)
            n_gt_total += len(gt_list)

            # Pred boxes for this class
            boxes = preds_dict.get(img_id, np.zeros((0, 6)))
            if len(boxes) == 0:
                continue
            cls_mask = boxes[:, 5].astype(int) == cat_id
            cls_boxes = boxes[cls_mask]
            if len(cls_boxes) == 0:
                continue

            # Sort by confidence descending
            order = np.argsort(-cls_boxes[:, 4])
            cls_boxes = cls_boxes[order]

            for pred in cls_boxes:
                px1, py1, px2, py2 = pred[:4]
                score = pred[4]

                if len(gt_xyxy) == 0:
                    all_scores.append(score)
                    all_tp.append(False)
                    continue

                # IoU with all GTs
                ix1 = np.maximum(px1, gt_xyxy[:, 0])
                iy1 = np.maximum(py1, gt_xyxy[:, 1])
                ix2 = np.minimum(px2, gt_xyxy[:, 2])
                iy2 = np.minimum(py2, gt_xyxy[:, 3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                area_p = (px2 - px1) * (py2 - py1)
                area_g = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
                union = area_p + area_g - inter
                ious = inter / np.maximum(union, 1e-10)

                # Best unmatched GT
                best_idx = -1
                best_iou = 0.5  # threshold
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

        # Sort by score desc
        scores = np.array(all_scores)
        tp = np.array(all_tp, dtype=np.float64)
        order = np.argsort(-scores, kind='mergesort')
        tp = tp[order]

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(1 - tp)
        recall = tp_cum / n_gt_total
        precision = tp_cum / (tp_cum + fp_cum)

        # 101-point interpolation
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 101):
            mask = recall >= r_thresh
            if mask.any():
                ap += precision[mask].max()
        ap /= 101
        per_class_ap50[cat_id] = float(ap)

    # mAP50
    present_classes = [c for c in cat_ids if c in per_class_ap50]
    per_class_ap50["mAP50"] = float(np.mean([per_class_ap50[c] for c in present_classes])) if present_classes else 0.0

    return per_class_ap50


# ─── Gate 策略（优化版：batch IoU）───

def apply_gate_to_image(policy: str, rgb_boxes: np.ndarray, dual_boxes: np.ndarray,
                        tau_overlap: float, tau_dual: float,
                        tau_rgb_uncertain: float, margin: float) -> tuple:
    """对单张图应用 gate。返回 (merged_boxes, n_dual_accepted, n_dual_total)。"""
    n_dual_total = len(dual_boxes)
    n_dual_accepted = 0

    if len(dual_boxes) == 0:
        return rgb_boxes.copy(), 0, 0
    if len(rgb_boxes) == 0:
        return dual_boxes.copy(), len(dual_boxes), n_dual_total

    if policy == "P0_rgb_only":
        return rgb_boxes.copy(), 0, n_dual_total

    if policy == "P1_dual_only":
        return dual_boxes.copy(), len(dual_boxes), n_dual_total

    # Gate A / B / C: 需要 per-class 处理
    output = list(rgb_boxes)
    # 追踪每个 output 槽位是否为原始 RGB
    is_original_rgb = [True] * len(rgb_boxes)

    for d in dual_boxes:
        d_cls = int(d[5])
        d_conf = d[4]

        # Gate C: per-class 有效阈值
        if policy == "GateC":
            if d_cls == 0:  # smoke: strict
                eff_tau = tau_dual * 1.5
                allow_replace = False
            elif d_cls == 1:  # fire: full
                eff_tau = tau_dual
                allow_replace = True
            else:  # person: strictest
                eff_tau = tau_dual * 2.0
                allow_replace = False
        else:
            eff_tau = tau_dual
            allow_replace = (policy == "GateB")

        if d_conf < eff_tau:
            continue

        # 找当前 output 中同类的原始 RGB boxes
        same_cls_indices = [i for i in range(len(output))
                            if int(output[i][5]) == d_cls and is_original_rgb[i]]

        if not same_cls_indices:
            # 无同类 RGB → 添加
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1
            continue

        # Batch IoU
        same_cls_boxes = np.array([output[i][:4] for i in same_cls_indices])
        d_box = d[:4].reshape(1, 4)
        ious = compute_iou_matrix(d_box, same_cls_boxes)[0]
        max_idx = np.argmax(ious)
        max_iou = ious[max_idx]
        best_out_idx = same_cls_indices[max_idx]

        if max_iou >= tau_overlap and allow_replace:
            # 替换条件
            r_conf = output[best_out_idx][4]
            if (d_conf - r_conf >= margin) and (r_conf < tau_rgb_uncertain):
                output[best_out_idx] = d
                is_original_rgb[best_out_idx] = False
                n_dual_accepted += 1
        elif max_iou < tau_overlap:
            # 添加
            output.append(d)
            is_original_rgb.append(False)
            n_dual_accepted += 1

    return np.array(output) if output else np.zeros((0, 6)), n_dual_accepted, n_dual_total


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


# ─── Sweep 引擎（优化版）───

def build_sweep_configs() -> list:
    configs = []
    configs.append({"policy": "P0_rgb_only", "tau_overlap": None, "tau_dual": None,
                     "tau_rgb_uncertain": None, "margin": None})
    configs.append({"policy": "P1_dual_only", "tau_overlap": None, "tau_dual": None,
                     "tau_rgb_uncertain": None, "margin": None})
    for to, td in product(TAU_OVERLAP, TAU_DUAL):
        configs.append({"policy": "GateA", "tau_overlap": to, "tau_dual": td,
                         "tau_rgb_uncertain": None, "margin": None})
    for to, td, tr, mg in product(TAU_OVERLAP, TAU_DUAL, TAU_RGB_UNCERTAIN, MARGIN):
        configs.append({"policy": "GateB", "tau_overlap": to, "tau_dual": td,
                         "tau_rgb_uncertain": tr, "margin": mg})
    for to, td, tr, mg in product(TAU_OVERLAP, TAU_DUAL, TAU_RGB_UNCERTAIN, MARGIN):
        configs.append({"policy": "GateC", "tau_overlap": to, "tau_dual": td,
                         "tau_rgb_uncertain": tr, "margin": mg})
    return configs


def sweep_fold(preds_rgb: dict, preds_dual: dict, gts_by_img: dict,
               img_ids: list, cat_ids: list, fold: str, seed: int) -> list:
    """用 fast AP50 做 gate sweep。"""
    configs = build_sweep_configs()
    results = []
    log.info(f"[{fold} seed={seed}] sweep: {len(configs)} configs")

    # 统计
    total_rgb = sum(len(v) for v in preds_rgb.values())
    total_dual = sum(len(v) for v in preds_dual.values())
    log.info(f"  RGB boxes: {total_rgb}, Dual boxes: {total_dual}")

    t0 = time.time()
    for ci, cfg in enumerate(configs):
        # 对每张图 apply gate → 得到 merged predictions
        merged_preds = {}
        total_accepted = 0
        total_dual_count = 0

        for img_id in sorted(img_ids):
            rgb_b = preds_rgb.get(img_id, np.zeros((0, 6)))
            dual_b = preds_dual.get(img_id, np.zeros((0, 6)))

            merged, n_acc, n_tot = apply_gate_to_image(
                cfg["policy"], rgb_b, dual_b,
                cfg.get("tau_overlap") or 0.5,
                cfg.get("tau_dual") or 0.1,
                cfg.get("tau_rgb_uncertain") or 0.5,
                cfg.get("margin") or 0.1,
            )
            merged_preds[img_id] = merged
            total_accepted += n_acc
            total_dual_count += n_tot

        # Fast AP50
        ap50_result = fast_ap50_eval(merged_preds, gts_by_img, img_ids, cat_ids)
        accept_ratio = total_accepted / max(total_dual_count, 1)

        result = {
            "fold": fold, "seed": seed,
            "policy": cfg["policy"],
            "tau_overlap": cfg["tau_overlap"], "tau_dual": cfg["tau_dual"],
            "tau_rgb_uncertain": cfg["tau_rgb_uncertain"], "margin": cfg["margin"],
            "AP50": ap50_result["mAP50"],
            "smoke_AP50": ap50_result.get(0, 0),
            "fire_AP50": ap50_result.get(1, 0),
            "person_AP50": ap50_result.get(2, 0),
            "dual_acceptance_ratio": accept_ratio,
            "n_rgb_boxes": total_rgb,
            "n_dual_boxes": total_dual_count,
            "n_output_boxes": total_rgb + total_accepted,
        }
        results.append(result)

        if (ci + 1) % 50 == 0 or ci < 5:
            elapsed = time.time() - t0
            rate = (ci + 1) / elapsed
            eta = (len(configs) - ci - 1) / rate
            log.info(f"  [{ci+1}/{len(configs)}] {cfg['policy']} AP50={result['AP50']:.4f} "
                     f"Δ={result['AP50'] - results[0]['AP50']:+.4f} accept={accept_ratio:.4f} "
                     f"({rate:.1f} cfg/s, ETA {eta:.0f}s)")

    # 补 delta
    rgb_ap50 = results[0]["AP50"]
    for r in results:
        r["delta_AP50_vs_rgb"] = r["AP50"] - rgb_ap50

    log.info(f"[{fold} seed={seed}] sweep 完成: {len(results)} configs, {time.time()-t0:.1f}s")
    return results


# ─── 判定逻辑 ───

def classify_fold2(sweep_results: list) -> tuple:
    """判定 fold2。返回 (verdict, best_policy_key, best_details)。"""
    by_seed = defaultdict(list)
    for r in sweep_results:
        by_seed[r["seed"]].append(r)

    rgb_ap50_by_seed = {}
    rgb_smoke_by_seed = {}
    for seed, results in by_seed.items():
        for r in results:
            if r["policy"] == "P0_rgb_only":
                rgb_ap50_by_seed[seed] = r["AP50"]
                rgb_smoke_by_seed[seed] = r["smoke_AP50"]

    # 所有 non-trivial policies
    non_trivial_keys = set()
    for r in sweep_results:
        if r["policy"] not in ("P0_rgb_only", "P1_dual_only"):
            non_trivial_keys.add((r["policy"], r["tau_overlap"], r["tau_dual"],
                                   r["tau_rgb_uncertain"], r["margin"]))

    best_key = None
    best_mean_delta = -999

    for pk in non_trivial_keys:
        deltas = []
        smoke_drops = []
        accepts = []
        for seed in SEEDS:
            for r in by_seed[seed]:
                if (r["policy"], r["tau_overlap"], r["tau_dual"],
                    r["tau_rgb_uncertain"], r["margin"]) == pk:
                    deltas.append(r["delta_AP50_vs_rgb"])
                    smoke_drops.append(rgb_smoke_by_seed.get(seed, 0) - r["smoke_AP50"])
                    accepts.append(r["dual_acceptance_ratio"])

        if len(deltas) < len(SEEDS):
            continue

        md = np.mean(deltas)
        if md > best_mean_delta:
            best_mean_delta = md
            best_key = pk
            best_details = {
                "mean_delta": md,
                "mean_smoke_drop": np.mean(smoke_drops),
                "mean_accept": np.mean(accepts),
            }

    if best_key is None:
        return "TRIVIAL_SAFE_ONLY", None, None

    log.info(f"Best non-trivial: {best_key[0]} to={best_key[1]} td={best_key[2]} "
             f"tr={best_key[3]} mg={best_key[4]}")
    log.info(f"  mean Δ={best_details['mean_delta']:.6f} "
             f"smoke drop={best_details['mean_smoke_drop']:.6f} "
             f"accept={best_details['mean_accept']:.4f}")

    md = best_details["mean_delta"]
    ms = best_details["mean_smoke_drop"]
    ma = best_details["mean_accept"]

    if md >= PASS_NONTRIVIAL_MEAN_DELTA and ms <= PASS_NONTRIVIAL_SMOKE_DROP and ma >= PASS_NONTRIVIAL_ACCEPT:
        return "PASS_NONTRIVIAL", best_key, best_details
    if md < FAIL_UNSAFE_MEAN_DELTA or ms > FAIL_UNSAFE_SMOKE_DROP:
        return "FAIL_UNSAFE", best_key, best_details
    return "TRIVIAL_SAFE_ONLY", best_key, best_details


def classify_fold1_retention(fold1_results: list, fold2_best_key: tuple) -> str:
    by_seed = defaultdict(list)
    for r in fold1_results:
        by_seed[r["seed"]].append(r)

    retentions = []
    for seed in SEEDS:
        rgb_ap50 = dual_ap50 = gate_ap50 = 0
        for r in by_seed[seed]:
            if r["policy"] == "P0_rgb_only":
                rgb_ap50 = r["AP50"]
            elif r["policy"] == "P1_dual_only":
                dual_ap50 = r["AP50"]
            elif (r["policy"], r["tau_overlap"], r["tau_dual"],
                  r["tau_rgb_uncertain"], r["margin"]) == fold2_best_key:
                gate_ap50 = r["AP50"]

        dual_gain = dual_ap50 - rgb_ap50
        if abs(dual_gain) < 0.001:
            retentions.append(1.0)
        else:
            retentions.append(max(0, (gate_ap50 - rgb_ap50) / dual_gain))

    mean_ret = np.mean(retentions)
    log.info(f"Fold1 retention: {mean_ret:.4f}")
    return "PASS_RETENTION" if mean_ret >= 0.5 else "FAIL_RETENTION"


# ─── CSV 输出 ───

CSV_COLS = ["fold", "seed", "policy", "tau_overlap", "tau_dual", "tau_rgb_uncertain", "margin",
            "AP50", "smoke_AP50", "fire_AP50", "person_AP50",
            "delta_AP50_vs_rgb", "dual_acceptance_ratio", "n_rgb_boxes", "n_dual_boxes", "n_output_boxes"]

# 完整版 CSV (含 full COCO metrics)
FULL_CSV_COLS = ["fold", "seed", "policy", "tau_overlap", "tau_dual", "tau_rgb_uncertain", "margin",
                 "AP", "AP50", "AP75", "smoke_AP50", "fire_AP50", "person_AP50",
                 "delta_AP50_vs_rgb", "dual_acceptance_ratio", "n_rgb_boxes", "n_dual_boxes", "n_output_boxes"]


def write_csv(results: list, path: Path, columns: list = None):
    cols = columns or CSV_COLS
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    log.info(f"CSV: {path} ({len(results)} 行)")


# ─── Recount ───

def write_recount(fold2_verdict: str, fold2_results: list, fold2_best_key: tuple,
                  fold2_best_details: dict,
                  fold1_verdict: str, fold1_results: list,
                  sanity_info: dict, final_coco: dict, elapsed: float):
    lines = [
        "# F-5-G1a SafeLateGate Experiment Recount",
        f"# 日期: 2026-06-07",
        f"# 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)",
        "",
        "## Sanity Check (full COCO eval)",
    ]
    for seed in SEEDS:
        info = sanity_info.get(seed, {})
        lines.append(f"  seed={seed}: eval_mAP50={info.get('eval_ap50', 0):.6f} "
                     f"source={info.get('source_ap50', 0):.6f} "
                     f"Δ={info.get('delta', 0):.6f} [{info.get('status', 'N/A')}]")
    lines.append("")

    # Fold2 baselines
    lines.append("## Fold2 Baselines (fast AP50)")
    for seed in SEEDS:
        for r in fold2_results:
            if r["seed"] == seed:
                if r["policy"] == "P0_rgb_only":
                    lines.append(f"  seed={seed} RGB-only: AP50={r['AP50']:.6f} smoke={r['smoke_AP50']:.6f} fire={r['fire_AP50']:.6f}")
                elif r["policy"] == "P1_dual_only":
                    lines.append(f"  seed={seed} Dual:     AP50={r['AP50']:.6f} Δ={r['delta_AP50_vs_rgb']:+.6f}")
    lines.append("")

    # Best non-trivial
    if fold2_best_key:
        policy, to, td, tr, mg = fold2_best_key
        lines.append(f"## Fold2 Best Non-Trivial: {policy}")
        lines.append(f"  params: tau_overlap={to} tau_dual={td} tau_rgb_uncertain={tr} margin={mg}")
        if fold2_best_details:
            lines.append(f"  mean Δ={fold2_best_details['mean_delta']:.6f} "
                         f"mean smoke drop={fold2_best_details['mean_smoke_drop']:.6f} "
                         f"mean accept={fold2_best_details['mean_accept']:.4f}")
        for seed in SEEDS:
            for r in fold2_results:
                if r["seed"] == seed and (r["policy"], r["tau_overlap"], r["tau_dual"],
                                           r["tau_rgb_uncertain"], r["margin"]) == fold2_best_key:
                    lines.append(f"  seed={seed}: AP50={r['AP50']:.6f} Δ={r['delta_AP50_vs_rgb']:+.6f} "
                                 f"smoke={r['smoke_AP50']:.6f} fire={r['fire_AP50']:.6f} "
                                 f"accept={r['dual_acceptance_ratio']:.4f}")
        lines.append("")

    # Final COCO verification
    if final_coco:
        lines.append("## Full COCO Verification (best config)")
        for seed, fc in final_coco.items():
            lines.append(f"  seed={seed}: AP={fc.get('AP',0):.6f} AP50={fc.get('AP50',0):.6f} AP75={fc.get('AP75',0):.6f} "
                         f"smoke={fc.get('smoke_AP50',0):.6f} fire={fc.get('fire_AP50',0):.6f}")
        lines.append("")

    # Verdicts
    lines.append(f"## Fold2 Verdict: {fold2_verdict}")
    if fold1_verdict:
        lines.append(f"## Fold1 Verdict: {fold1_verdict}")
        if fold1_results:
            for seed in SEEDS:
                for r in fold1_results:
                    if r["seed"] == seed:
                        if r["policy"] == "P0_rgb_only":
                            lines.append(f"  seed={seed} RGB: AP50={r['AP50']:.6f}")
                        elif r["policy"] == "P1_dual_only":
                            lines.append(f"  seed={seed} Dual: AP50={r['AP50']:.6f} Δ={r['delta_AP50_vs_rgb']:+.6f}")
                        elif fold2_best_key and (r["policy"], r["tau_overlap"], r["tau_dual"],
                                                  r["tau_rgb_uncertain"], r["margin"]) == fold2_best_key:
                            lines.append(f"  seed={seed} Gate: AP50={r['AP50']:.6f} Δ={r['delta_AP50_vs_rgb']:+.6f}")
    lines.append("")
    lines.append(f"## FINAL: fold2={fold2_verdict}" + (f" fold1={fold1_verdict}" if fold1_verdict else ""))

    (OUT_DIR / "recount.md").write_text("\n".join(lines))
    log.info(f"Recount: {OUT_DIR / 'recount.md'}")


# ─── Runtime Precheck ───

def runtime_precheck():
    log.info("=== Runtime Precheck ===")
    import torch
    if not torch.cuda.is_available():
        log.error("CUDA 不可用"); return False
    log.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}, "
             f"VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    for seed in SEEDS:
        for fold_name, ckpts in [("fold2", FOLD2_CKPTS), ("fold1", FOLD1_CKPTS)]:
            for role, path in ckpts[seed].items():
                if not Path(path).exists():
                    log.error(f"缺失: {path}"); return False

    for d in [FOLD2_RGB_DIR, FOLD2_IR_DIR, FOLD2_LBL_DIR, FOLD1_RGB_DIR, FOLD1_IR_DIR, FOLD1_LBL_DIR]:
        if not d.exists():
            log.error(f"目录缺失: {d}"); return False
    log.info("Precheck 通过")
    return True


# ─── 主流程 ───

def main():
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("F-5-G1a SafeLateGate Experiment (optimized)")
    log.info("=" * 60)

    if not runtime_precheck():
        (OUT_DIR / "FAILED_precheck").touch(); sys.exit(1)

    # ─── Phase A: Fold2 预测 ───
    log.info("=" * 60)
    log.info("Phase A: Fold2 预测导出")
    log.info("=" * 60)

    fold2_pairs = load_val_pairs(FOLD2_RGB_DIR, FOLD2_IR_DIR)
    fold2_gts, fold2_gts_by_img, fold2_img_ids, fold2_cat_ids = load_ground_truths(
        FOLD2_LBL_DIR, fold2_pairs, FOLD2_CAT_IDS)

    fold2_preds = {}
    for seed in SEEDS:
        rgb_path = OUT_DIR / f"predictions_fold2_seed{seed}_rgb.jsonl"
        dual_path = OUT_DIR / f"predictions_fold2_seed{seed}_dual.jsonl"

        rgb_preds = load_predictions_jsonl(rgb_path) if rgb_path.exists() else \
            (lambda p: (save_predictions_jsonl(p, rgb_path), p)[1])(
                run_predictions(FOLD2_CKPTS[seed]["rgb"], fold2_pairs, True, f"fold2_rgb_s{seed}"))

        dual_raw = load_predictions_jsonl(dual_path) if dual_path.exists() else \
            (lambda p: (save_predictions_jsonl(p, dual_path), p)[1])(
                run_predictions(FOLD2_CKPTS[seed]["dual"], fold2_pairs, False, f"fold2_dual_s{seed}"))

        # 预过滤 dual boxes: conf >= DUAL_PREFILTER_CONF
        dual_preds = {}
        n_before = sum(len(v) for v in dual_raw.values())
        for img_id, boxes in dual_raw.items():
            if len(boxes) > 0:
                mask = boxes[:, 4] >= DUAL_PREFILTER_CONF
                dual_preds[img_id] = boxes[mask]
            else:
                dual_preds[img_id] = boxes
        n_after = sum(len(v) for v in dual_preds.values())
        log.info(f"  seed={seed} dual prefilter: {n_before} → {n_after} boxes (conf >= {DUAL_PREFILTER_CONF})")

        fold2_preds[seed] = {"rgb": rgb_preds, "dual": dual_preds}

    # ─── Phase B: Sanity + Sweep ───
    log.info("=" * 60)
    log.info("Phase B: Sanity + Gate Sweep (fast AP50)")
    log.info("=" * 60)

    # Sanity check: 用完整 COCO eval 验证 RGB-only
    sanity_info = {}
    for seed in SEEDS:
        stats, per_class = full_coco_eval_preds(
            fold2_preds[seed]["rgb"], fold2_gts, fold2_img_ids, fold2_cat_ids)
        eval_ap50 = stats["AP50"]
        src_ap50 = SOURCE_RGB_MAP50[seed]
        delta = abs(eval_ap50 - src_ap50)
        status = "PASS" if delta < 0.05 else "FAIL"
        sanity_info[seed] = {"eval_ap50": eval_ap50, "source_ap50": src_ap50, "delta": delta, "status": status}
        log.info(f"  seed={seed}: eval={eval_ap50:.6f} source={src_ap50:.6f} Δ={delta:.6f} [{status}]")
        if status == "FAIL":
            log.error("BLOCKED_EVAL")
            write_recount("BLOCKED_EVAL", [], None, None, None, None, sanity_info, {}, time.time() - t_start)
            (OUT_DIR / "FAILED_evaluator_sanity").touch(); sys.exit(1)

    # Sweep
    fold2_all_results = []
    for seed in SEEDS:
        seed_results = sweep_fold(
            fold2_preds[seed]["rgb"], fold2_preds[seed]["dual"],
            fold2_gts_by_img, fold2_img_ids, fold2_cat_ids, "fold2", seed)
        fold2_all_results.extend(seed_results)

    write_csv(fold2_all_results, OUT_DIR / "gate_sweep_fold2.csv")

    # 判定
    fold2_verdict, fold2_best_key, fold2_best_details = classify_fold2(fold2_all_results)
    log.info(f"Fold2 verdict: {fold2_verdict}")

    # Full COCO eval 验证 best config
    final_coco = {}
    if fold2_best_key:
        for seed in SEEDS:
            # 用 best config apply gate，然后跑完整 COCO eval
            best_merged = {}
            for img_id in sorted(fold2_img_ids):
                rgb_b = fold2_preds[seed]["rgb"].get(img_id, np.zeros((0, 6)))
                dual_b = fold2_preds[seed]["dual"].get(img_id, np.zeros((0, 6)))
                merged, _, _ = apply_gate_to_image(
                    fold2_best_key[0], rgb_b, dual_b,
                    fold2_best_key[1] or 0.5, fold2_best_key[2] or 0.1,
                    fold2_best_key[3] or 0.5, fold2_best_key[4] or 0.1)
                best_merged[img_id] = merged

            stats, per_class = full_coco_eval_preds(best_merged, fold2_gts, fold2_img_ids, fold2_cat_ids)
            final_coco[seed] = {
                "AP": stats["AP"], "AP50": stats["AP50"], "AP75": stats["AP75"],
                "smoke_AP50": per_class.get(0, {}).get("AP50", 0),
                "fire_AP50": per_class.get(1, {}).get("AP50", 0),
            }
            log.info(f"  Full COCO seed={seed}: AP={stats['AP']:.6f} AP50={stats['AP50']:.6f} "
                     f"smoke={per_class.get(0, {}).get('AP50', 0):.6f}")

    # ─── Phase C: Fold1 Retention ───
    fold1_verdict = None
    fold1_all_results = None

    if fold2_best_key and fold2_verdict in ("PASS_NONTRIVIAL", "TRIVIAL_SAFE_ONLY"):
        log.info("=" * 60)
        log.info("Phase C: Fold1 Retention")
        log.info("=" * 60)

        fold1_pairs = load_val_pairs(FOLD1_RGB_DIR, FOLD1_IR_DIR)
        fold1_gts, fold1_gts_by_img, fold1_img_ids, fold1_cat_ids = load_ground_truths(
            FOLD1_LBL_DIR, fold1_pairs, FOLD1_CAT_IDS)

        fold1_preds = {}
        for seed in SEEDS:
            rgb_p = OUT_DIR / f"predictions_fold1_seed{seed}_rgb.jsonl"
            dual_p = OUT_DIR / f"predictions_fold1_seed{seed}_dual.jsonl"

            rgb_preds = load_predictions_jsonl(rgb_p) if rgb_p.exists() else \
                (lambda p: (save_predictions_jsonl(p, rgb_p), p)[1])(
                    run_predictions(FOLD1_CKPTS[seed]["rgb"], fold1_pairs, True, f"fold1_rgb_s{seed}"))

            dual_raw = load_predictions_jsonl(dual_p) if dual_p.exists() else \
                (lambda p: (save_predictions_jsonl(p, dual_p), p)[1])(
                    run_predictions(FOLD1_CKPTS[seed]["dual"], fold1_pairs, False, f"fold1_dual_s{seed}"))

            dual_preds = {}
            for img_id, boxes in dual_raw.items():
                if len(boxes) > 0:
                    dual_preds[img_id] = boxes[boxes[:, 4] >= DUAL_PREFILTER_CONF]
                else:
                    dual_preds[img_id] = boxes
            fold1_preds[seed] = {"rgb": rgb_preds, "dual": dual_preds}

        # 只跑 P0 + P1 + best policy
        fold1_all_results = []
        for seed in SEEDS:
            for cfg in [
                {"policy": "P0_rgb_only", "tau_overlap": None, "tau_dual": None, "tau_rgb_uncertain": None, "margin": None},
                {"policy": "P1_dual_only", "tau_overlap": None, "tau_dual": None, "tau_rgb_uncertain": None, "margin": None},
                {"policy": fold2_best_key[0], "tau_overlap": fold2_best_key[1],
                 "tau_dual": fold2_best_key[2], "tau_rgb_uncertain": fold2_best_key[3], "margin": fold2_best_key[4]},
            ]:
                merged_preds = {}
                for img_id in sorted(fold1_img_ids):
                    rgb_b = fold1_preds[seed]["rgb"].get(img_id, np.zeros((0, 6)))
                    dual_b = fold1_preds[seed]["dual"].get(img_id, np.zeros((0, 6)))
                    merged, n_acc, n_tot = apply_gate_to_image(
                        cfg["policy"], rgb_b, dual_b,
                        cfg.get("tau_overlap") or 0.5, cfg.get("tau_dual") or 0.1,
                        cfg.get("tau_rgb_uncertain") or 0.5, cfg.get("margin") or 0.1)
                    merged_preds[img_id] = merged

                ap50_res = fast_ap50_eval(merged_preds, fold1_gts_by_img, fold1_img_ids, fold1_cat_ids)
                n_dual = sum(len(fold1_preds[seed]["dual"].get(i, np.zeros((0,6)))) for i in fold1_img_ids)
                n_acc = sum(len(merged_preds.get(i, np.zeros((0,6)))) - len(fold1_preds[seed]["rgb"].get(i, np.zeros((0,6))))
                           for i in fold1_img_ids)

                fold1_all_results.append({
                    "fold": "fold1", "seed": seed,
                    "policy": cfg["policy"],
                    "tau_overlap": cfg["tau_overlap"], "tau_dual": cfg["tau_dual"],
                    "tau_rgb_uncertain": cfg["tau_rgb_uncertain"], "margin": cfg["margin"],
                    "AP50": ap50_res["mAP50"],
                    "smoke_AP50": ap50_res.get(0, 0), "fire_AP50": ap50_res.get(1, 0), "person_AP50": ap50_res.get(2, 0),
                    "dual_acceptance_ratio": n_acc / max(n_dual, 1),
                    "n_rgb_boxes": sum(len(fold1_preds[seed]["rgb"].get(i, np.zeros((0,6)))) for i in fold1_img_ids),
                    "n_dual_boxes": n_dual, "n_output_boxes": 0,
                })

        # 补 delta
        for seed in SEEDS:
            rgb_ap50 = next((r["AP50"] for r in fold1_all_results if r["seed"] == seed and r["policy"] == "P0_rgb_only"), 0)
            for r in fold1_all_results:
                if r["seed"] == seed:
                    r["delta_AP50_vs_rgb"] = r["AP50"] - rgb_ap50

        write_csv(fold1_all_results, OUT_DIR / "gate_sweep_fold1.csv")
        fold1_verdict = classify_fold1_retention(fold1_all_results, fold2_best_key)
        log.info(f"Fold1 verdict: {fold1_verdict}")

    # ─── 输出 ───
    elapsed = time.time() - t_start
    write_recount(fold2_verdict, fold2_all_results, fold2_best_key, fold2_best_details,
                  fold1_verdict, fold1_all_results, sanity_info, final_coco, elapsed)
    (OUT_DIR / "DONE").touch()
    log.info(f"完成! {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"FINAL: fold2={fold2_verdict}" + (f" fold1={fold1_verdict}" if fold1_verdict else ""))


if __name__ == "__main__":
    main()
