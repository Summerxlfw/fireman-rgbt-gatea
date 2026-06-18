# Provenance — forest_fire_rgbt code/data deposit(server part, 2026-06-18)

> COMPAG 投稿公开 deposit 的服务器端产物包。本文件记录代码/环境/split 的来源与版本。

## 训练 / 基线代码仓库(git commit,基线代码 provenance)
| repo | commit | tag/desc | remote | 角色 |
|---|---|---|---|---|
| YOLOv11-RGBT | `9cc2e208a3d3e8452e15b5ef47e1c536766aa1f7` | v1.13.9-5-g9cc2e20 (2025-12-15 wandahangFY) | https://github.com/wandahangFY/YOLOv11-RGBT.git | P0/P1 训练框架(ultralytics 多模态 fork,mm_project 26.5.22) |
| M2D-LIF | `210c7ca22b7670e7d4629cca995427bf7e74653f` | 210c7ca (2026-03-24 TY) | https://github.com/Zhao-Tian-yi/M2D-LIF.git | F9 external baseline |

## 自研脚本(非版本控制)
`/mnt/topic2_workspace` **非 git 仓库**。GateA 推理 / recount / bootstrap 评估脚本位于 `scripts/`,无 git commit;以 `server_part_manifest.txt` 中的文件 SHA-256 + 文件 mtime 标识。

## conda 环境
- `envs/topic2_main.yml` — 评估 / recount / bootstrap 脚本运行环境(`conda env export --no-builds`)
- `envs/yolov11_rgbt.yml` — YOLOv11-RGBT 训练 / 推理环境(ultralytics 8.3.75)
- 注:训练框架代码 provenance 以 YOLOv11-RGBT fork commit 为准(上表);运行时依赖见对应 yml。

## GateA 锁定参数(add-only 安全准入)
`TAU_OVERLAP=0.7, TAU_DUAL=0.05, DUAL_PREFILTER=0.01`
定义于 `scripts/run_f6_evidence_package_20260613.py` L44-47;实现 `apply_gateA_add_only()` L242。
逻辑:保留全部 RGB boxes;逐个 dual box → conf<τ_dual 跳过 → 无同类 RGB admit → 同类 IoU max≥τ_overlap reject → 否则 admit;admitted∪RGB 过 class-aware 合并。

## LOCO split
- fold0:`splits/fold0_pilot_only_paired/`(yaml 定义 + image-id 列表;train∩val=0 已验证)
- fold1:`splits/loco_fold1/`(formal GateA 评估用,GT = fire_loco_fold1)
- fold2:`splits/loco_fold2_nirfree/`(formal GateA 评估用,GT = fire_loco_fold2_nirfree,NIR-free 设定;fold2 含 NIR 版 `loco_fold2/` 非 formal GateA 用,未收)

## bootstrap CI 复现说明
论文 bootstrap CI(n_boot=200)由 `scripts/run_f6_evidence_package_20260613.py` 从 `predictions/`(per-image JSONL) + `gt_labels/`(per-image GT) 计算。f6 原始设计未持久化 per-image matching/AP 贡献中间表(npz/pkl),只存汇总 `runs/f6_evidence_package_20260613/bootstrap_ci.csv`;提供 (JSONL 预测 + GT labels) 组合即可脱离图片/数据集/GPU,用 f6 脚本独立重算 bootstrap CI。
