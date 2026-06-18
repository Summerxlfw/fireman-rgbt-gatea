# Metric Definition — forest_fire_rgbt (G1 lockdown 2026-05-30)

## Primary Metric

- **Name**: mAP50(COCO mAP @ IoU=0.5)。
- **Formula**: mean Average Precision over classes at IoU threshold 0.5。
- **Direction**: higher is better。
- **Epoch 口径(锁死,防漂移)**: 一律取 **best epoch(best.pt)**,**不用 final epoch**。
  > 教训(2026-05-30):服务器报告曾用 final-epoch 把 RGB-only mean 从 best 0.295 误报成 0.266、方差夸大一倍。下游一切 mAP 数字以 best.pt 对应 epoch 为准。
- **Aggregation**: per-fold per-seed → 3 seed mean±std → 跨 fold 分别报(不混 fold 求总均值)。
- **Inference 口径**: `half=False`(AMP off,per memory `feedback_amp_inference_default_off`)。

## Secondary Metrics

| Metric | Formula | Purpose | Direction |
|---|---|---|---|
| mAP50-95 | COCO mAP @ IoU 0.5:0.95 | box 紧致度 | higher |
| per-class AP50 | smoke / fire / person 各自 AP50 | 类间瓶颈(已知 fire 弱:DEIM ep7 fire 0.036) | higher |
| **thermal Δ** | (dual val mAP50) − (architecture-paired RGB-only val mAP50),**同 fold 同 epoch 同 split** | 主 claim 核心证据 | higher = thermal 有贡献 |
| Precision / Recall | 标准 | 瓶颈分解(recall 偏低) | higher |

## thermal Δ 口径(main claim 核心,硬锁)

- 定义: 同一 fold 的同一 val 上,`dual mAP50 − RGB-only mAP50`。RGB-only = architecture-paired(同模型,X=rgb_clone)。
- **必须同 split**: fold-0 = JAG val;fold-1 = RGBT-3M val;fold-2 = FireMan val。**严禁** dual(JAG)vs RGB-only(internal)跨 split。
- **已锁定值(fold-0)**: dual JAG mAP50=**0.2447** / mAP50-95=**0.0702**;RGB-only JAG 3seed mean mAP50=**0.0841** / mAP50-95=**0.0188** → **thermal Δ = +0.1606(mAP50) / +0.0514(mAP50-95)**。已设计端亲核 eval log 确认(`eval_suite_rgbonly_seed42_verify`,2 modality + 146 图同 GT)。
- 梯度参照(跨难度,非同 split,仅叙事): Sycan(easy,AUC)+0.009 → internal(域内 detection)+0.043 → JAG(跨域 detection)+0.1606。

## Exclusion Rules

- Failed inference / NaN 权重 checkpoint(如 DEIM ep11+)→ 排除,不入 metric。
- Missing label fold(JAG 无 person)→ test=JAG 时 person 不计入 mAP(只 fire/smoke)。

## Reporting Rules

- Decimal: mAP 报 4 位;表内 thermal Δ 报 +0.xxxx。
- Per-dataset/fold 分别报,不跨 fold 平均成单一数字。
- image-level(检测任务)。

## Metric Anti-cheating Checks

- No test-set threshold tuning;no metric switching after results。
- **No best-vs-final epoch 混用** —— 统一 best.pt(见 Primary)。
- thermal Δ 必同 split 同口径;F-1 锁定数字(0.2447/0.338/0.0702 等)不得改。
- No cherry-picking seed:报 3 seed mean±std,含 bad seed(seed2024)。
