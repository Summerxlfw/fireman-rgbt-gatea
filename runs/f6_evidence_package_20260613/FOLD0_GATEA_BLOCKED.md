# FOLD0 GATEA RETENTION: BLOCKED

## Status: BLOCKED (permanent)

Fold0 GateA retention **无法在 formal quality 下完成**。

## 原因

1. **无 formal fold0 dual checkpoint**：服务器上不存在任何 `phase_f2_dual_fold0_*` 目录。
   - `find /mnt/topic2_workspace/runs -name "best.pt" -path "*fold0*dual*"` 返回零结果。
   - `find /mnt/topic2_workspace/runs -name "best.pt" -path "*dual*fold0*"` 返回零结果。

2. **Phase F-1 pilot dual 不可替代**：唯一在 JAG 上评估过的 dual 模型是 `phase_f1_50epoch_seed42`，但其训练配置为 `fire_loco_pilot_only_paired`（已从其 `args.yaml` 确认），这是 pilot split，不是 formal fold0 LOCO split。训练数据分布、epoch 数、协议均不同，不可用于 formal GateA 评估。

3. **Fold0 RGB-only checkpoints 存在**（3 seeds: 42, 1337, 2024，cm10 模式），但缺少对应的 fold0 dual predictions，GateA 的 add-only 机制无法运行。

## 影响范围

- `statistical_significance_audit.md` 中 Fold0 GateA retention 行将标记为 `BLOCKED_no_dual_checkpoint`。
- 论文 claim boundary 不受影响：当前 claim 已限定为 "safe-admission protocol pass on fold1/fold2"，不依赖 fold0。
- Bootstrap CI 和 qualitative cases 仅覆盖 fold1/fold2，不受此 blocker 影响。

## 替代方案（未采纳）

曾考虑用 F-1 pilot dual 做 single-anchor diagnostic，但拒绝：
- 训练 split 不匹配 → GateA 的 acceptance/rejection 统计不可解释
- 单 seed，无统计意义
- 可能误导读者认为 fold0 有 dual 证据

## 日期

2026-06-13
