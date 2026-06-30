# AC-MPC Reproduction Workspace

本仓库是当前 AC-MPC 仿真复现工程的一体化源码快照，用于跨设备部署和继续开发。

```text
acmpc_reproduction_repo/
├── acmpc_public/   # AC-MPC、Gym、PPO、MPVE、训练与评估脚本
└── flightmare/     # 修改版 Flightmare RacingEnv 和 headless 构建代码
```

`acmpc_public/mpc.pytorch`、`acmpc_public/stable-baselines3` 和 `flightmare` 均作为包含本地修改的普通源码目录纳入本仓库，不需要再次拉取上游子模块。

完整安装、编译、赛道安装、模型迁移和验收流程见：

```text
acmpc_public/docs/acmpc_cross_device_migration_guide_zh.md
```

训练输出 `acmpc_public/runs/`、临时文件、Flightmare 构建缓存和平台相关 `.pyd` 不进入 Git。已训练模型需要单独归档；`flightgym` 应在目标设备按迁移指南重新编译。
