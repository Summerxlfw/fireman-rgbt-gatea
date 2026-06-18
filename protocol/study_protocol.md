# Study Protocol — forest_fire_rgbt (G1 lockdown 2026-05-30)

> 锁定于切片 ③(2026-05-30),LOCO=方案 A 已 user 拍板。规格在 fold-1/2 runs 前锁定。
> 任何改动走 §Change Control,不得 silent 改 split/metric/baseline。

## Study Objective

- **Primary**: 量化 thermal(IR)模态相对 RGB-only 在 UAV 森林火灾检测上的**条件性增益** —— 主指标 = LOCO/cross-dataset val 上 dual vs architecture-paired RGB-only 的 mAP50 差(thermal Δ)。
- **Secondary**: 刻画该增益随任务难度/域移的梯度(Sycan easy → internal 域内 → JAG/LOCO 跨域)。

## Data Sources

| Dataset | Role | N(图) | Label source | Modality | Notes |
|---|---|---:|---|---|---|
| FireMan | train/test(LOCO) | — | 原标注 | RGB+IR | 含 smoke/fire/person |
| RGBT-3M | train/test(LOCO) | — | 原标注 | RGB+IR | 含 person |
| JAG2023 | train/test(LOCO) | 146(val) | 原标注 | RGB+IR | **无 person 类**(只 fire/smoke) |
| Sycan(辅评) | external eval | — | image-level | RGB+IR | easy 负向锚点(thermal Δ+0.009) |
| Hanna/m300(辅评) | external eval | — | — | RGB+IR | motivation-only(含混淆,不进主因果) |

## Split Plan — LOCO 方案 A(经典 cross-dataset 3-fold,user 拍板 2026-05-30)

- **Split unit**: dataset(leave-one-dataset-out)。
- **fold-0**: train FireMan+RGBT-3M → test **JAG** ✅ done(thermal Δ+0.1606 坐实)
- **fold-1**: train FireMan+JAG → test **RGBT-3M** (planned)
- **fold-2**: train RGBT-3M+JAG → test **FireMan_val(paired val, 1684 帧)** (planned)
  > 用 paired val(有 thermal 的子集),不用全 val(5624);否则 dual 只能 eval paired 子集、两边 eval 集不同 → thermal Δ 不可比。
- **Leakage risks**: 三 dataset 互不重叠(LOCO 严格);同一 video/场景不跨 fold。
- **Leakage checks**: split txt 文件生成后 **checksum 落盘锁定**(plan §12-4);fold 间 image-id 交集必须为空(服务器生成时验证)。
- **JAG 无 person 处理**: fold-1/2 train pool 仍含 FireMan/RGBT-3M 的 person 实例;test=RGBT-3M/FireMan 时正常评 3 类;论文 §dataset 注明 "JAG only contributes fire/smoke labels"。
- **⚠️ checksum 待回填**: fold-1/2 的 train/val txt + sha256 由服务器按本规格生成后回填本节(G1 完全 lock 需 checksum 到位)。

## Model / Method

- **Method under test**: YOLO11n-mm-mid dual-stream(RGB primary + IR as X)。
- **Architecture-paired RGB-only ablation**: 同 yolo11n-mm-mid,X 路径 = `rgb_clone_as_x`(RGB 克隆,无 thermal 新信息);**唯一差异 = X 通道 thermal vs RGB-clone** → thermal Δ 干净归因。
- **DEIM Plan B**(dual-stream): 可选 upside,当前 **hold**(eval pipeline 已验无 bug,但训练 loss_fgl 爆炸待修;timebox 已显示 ep7 无优势)。
- Pretraining: dual=dfine/COCO pretrain;YOLO=COCO。Init/frozen: 全 trainable。

## Baselines

See `baseline_registry.yaml`(YOLO dual / RGB-only / DEIM / top-8 SOTA from `refs/high_tier_sota_inventory_2026-05-28.md`)。

## Metrics

See `metric_definition.md`。主指标 = LOCO val mAP50(**best-epoch 口径**)+ thermal Δ。

## Statistical Plan

- **Primary endpoint**: 每 fold 的 dual / RGB-only val mAP50 + thermal Δ = dual − RGB-only(同 fold 同口径)。
- **Seed policy**: 3 seed {42, 1337, 2024},报 **mean ± std**(population std)。
- **方差监控**: 若某 fold seed std > 0.05 mAP50 → 加 seed 至 5(plan §11)。**已知 seed2024 是 bad seed**(过拟合,ep4 峰值),保留入 mean(真实 seed 方差)。
- CI: 3 seed 报 mean±std;n=3 不做正式显著性检验,描述性报告 + 标 std。

## Fairness Rules(architecture-paired 对称,硬约束)

- **Same data access**: dual 与 RGB-only 用同一份图、同一 split;唯一差异 = X 通道(IR vs rgb_clone)。
- **Same augmentation**: mosaic on + **close_mosaic=10**(dual 与 RGB-only 对称统一)。⚠️ fold-0 原 RGB-only 用 close=2(不对称瑕疵,2026-05-30 服务器问出)→ **补跑 fold-0 RGB-only close=10 修正**,三 fold 全对称。同 mixup policy。
- **Same schedule**: 同 50 epoch、同 batch=8、同 imgsz=640、同 optimizer/lr、同 early stopping(取 best.pt)。
- **Same post-processing**: 同 conf/NMS;eval `half=False`(AMP off,per memory)。
- **同 split 比较**: thermal Δ 只在**同一 fold 的同一 val** 上算;严禁 internal val vs JAG val 跨 split 比(F-1 避坑)。

## Change Control

| Date | Change | Reason | Affected results | Approved by |
|---|---|---|---|---|
| 2026-05-30 | G1 lockdown 建立, LOCO=方案 A | thermal Δ fold-0 坐实, 进 fold-1/2 前锁规格 | fold-1/2 | user(LOCO A) |
| 2026-05-30 | close_mosaic 统一=10(对称) + fold-0 RGB-only 补跑 close=10; fold-2 用 paired val(1684) | fold-0 dual10/RGB2 aug 不对称(服务器问出); thermal Δ fairness | fold-0 RGB-only(thermal Δ 微调, 预计仍 decisive) | 设计端 |
