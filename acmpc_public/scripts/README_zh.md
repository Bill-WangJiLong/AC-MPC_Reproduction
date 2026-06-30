# AC-MPC `scripts` 目录说明

## 1. 文档范围

本文说明 `scripts/` 中当前全部 27 个 `.py`、`.ps1` 和 `.sh` 文件的用途、调用关系、主要输入和输出。

脚本分为两层：

- `.py`：实际实现训练、评估、绘图、验证或配置转换；
- `.ps1/.sh`：解析环境变量和命令行参数，定位 Conda/Python，再调用对应 Python 脚本。

除训练入口外，Python 文件均不会因为被 import 就自动启动长时间任务；主逻辑位于 `if __name__ == "__main__"` 入口。

## 2. 总体调用关系

```text
核心验证
  smoke_test_acmpc_forward.py
  validate_acmpc_core.py

Gym
  run_train_acmpc_gym.ps1 -> train_acmpc_gym.py
  run_eval_acmpc_gym.ps1  -> eval_acmpc_gym.py

MPVE
  run_mpve_horizontal_pipeline.ps1
    -> train_acmpc_gym_mpve.py
    -> eval_acmpc_gym.py
    -> plot_training_metrics.py
    -> plot_trajectories.py

Flightmare
  build_flightmare_racing_env.ps1
  run_smoke_test_flightmare_racing_env.ps1
    -> smoke_test_flightmare_racing_env.py
  run_train_acmpc_flightmare_track.ps1
    -> install_gym_track_into_flightmare.py
    -> verify_flightmare_track_runtime.py
    -> run_train_acmpc_flightmare.ps1
    -> train_acmpc_flightmare.py
  run_eval_acmpc_flightmare.ps1
    -> eval_acmpc_flightmare.py

绘图
  run_plot*_metrics.ps1/.sh -> plot_training_metrics.py
  run_plot*_trajectories.ps1 -> plot_trajectories.py
  plot_gym_tracks.py -> track_visualizations/*.png
```

## 3. AC-MPC 核心验证脚本

### `smoke_test_acmpc_forward.py`

最小 AC-MPC forward 验证，不启动 PPO 训练。

主要工作：

- 验证 `import mpc` 和 `import drone`；
- 调用 `DroneDx.forward()` 并检查 finite；
- 构造简单 Q/p，运行一次 `IL_Env.mpc()`；
- 检查 MPC state/action 轨迹 shape 和 NaN/Inf；
- 设置 `ACMPC_T` 并初始化 `MlpMpcPolicy`。

用于环境安装后的第一层验收。

### `validate_acmpc_core.py`

比 smoke test 更完整的核心数值验证。

检查：

- DroneDx forward；
- 四元数 norm 行为；
- 解析 Jacobian 维度和有限性；
- MPC action 是否满足 thrust/body-rate 边界；
- `MlpMpcPolicy.forward_actor()` 的 batch size 1、8、64；
- `self.predictions` 是否为预期 `[batch,T,14]`。

不创建竞速环境，不进行 PPO 更新。

## 4. Python Gym 环境脚本

### `smoke_test_racing_gym.py`

Gym 环境、SB3 wrapper 和 state plumbing 的回归测试集合。

覆盖：

- gate 平面相交、穿门和门框碰撞；
- observation/action/state spaces 为 36/4/13；
- 随机动作完整 rollout；
- 越界、地面碰撞、timeout 和终点；
- 同一步通过最后门并进入终点球，奖励应为 gate+finish；
- `StateDummyVecEnv.get_state()`；
- observation normalization，state 保持原始物理量；
- 从真实 env observation/state 调用 AC-MPC policy；
- 修改版 SB3 能把 state 写入 rollout buffer 并完成短 update。

该脚本用于 Gym 相关代码修改后的首要回归验证。

### `train_acmpc_gym.py`

标准 AC-MPC PPO Gym 训练主体。

负责：

- 解析 track、`ACMPC_T`、PPO、归一化、checkpoint、resume 和 CUDA memory 参数；
- 创建 `RacingEnvConfig` 和 `StateDummyVecEnv/VecNormalize`；
- 可把指定赛道临时裁成 single-gate 课程；
- 设置随机 seed 和线性学习率；
- 创建 `PPO(MlpMpcPolicy, env)`；
- 保存 `config.json`、checkpoint、TensorBoard/SB3 CSV；
- 用 `EpisodeCsvCallback` 记录 episode 结果；
- 可用 `CudaMemoryCsvCallback` 记录 allocated/reserved/peak CUDA memory；
- 在正常结束或异常进入 `finally` 时保存 `final_model.zip` 和 `vecnormalize.pkl`。

默认输出：

```text
runs/acmpc_gym/<timestamp>_<track>_T<horizon>/
```

该脚本会真正启动训练。

### `run_train_acmpc_gym.ps1`

Gym 训练的通用 PowerShell 入口。

它会：

- 根据 `-TrackName` 检查 JSON 是否存在；
- 验证 JSON name、gate 数量和 finish；
- 解析 Conda 环境或显式 `CondaEnvPath`；
- 把参数转换成 `train_acmpc_gym.py` 命令；
- 支持 single-gate、normalization 和 `ValidateOnly`。

`ValidateOnly` 只验证赛道，不启动训练；否则会启动 PPO。

### `eval_acmpc_gym.py`

Gym 模型的确定性/随机评估主体。

负责：

- 选择显式 run、latest run、final model 或指定 checkpoint；
- 配套加载 `vecnormalize.pkl`；
- 从 run `config.json` 恢复 track、动力学、reward、episode 和 `ACMPC_T`；
- 支持覆盖 track、seed、random reset 和最大步数；
- 默认使用 deterministic action，也可启用 stochastic；
- 逐 episode 记录 position、velocity、omega、action、物理推力/body-rate、reward 和 gate；
- 计算 success rate、collision rate、timeout rate、平均速度、最大速度、飞行时间等；
- 写出 `track.json`、`metadata.json`、`summary.csv/json` 和逐 episode trajectory CSV。

默认评估目录：

```text
<run>/eval/<timestamp>/
```

该脚本只推理，不更新网络。

### `run_eval_acmpc_gym.ps1`

Gym 评估的简化 PowerShell 入口。

主要通过环境变量接收 `RUN_DIR`、`MODEL_PATH`、`VECNORMALIZE_PATH`、`TRACK_NAME`、`EPISODES`、`SEED` 等；未给 `RUN_DIR` 时使用 latest Gym run，然后调用 `eval_acmpc_gym.py`。

## 5. MPVE 脚本

### `train_acmpc_gym_mpve.py`

Gym 下 AC-MPC+MPVE 的训练实现。

新增：

- `MPVERolloutBufferSamples`：普通 PPO sample 加 prediction 数据；
- `MPVERolloutBuffer`：保存 MPC 预测 observation/reward/valid/terminal；
- `compute_mpve_targets()`：按 gamma 累加预测 reward，处理 terminal 和 value bootstrap；
- `MPVEPPO.collect_rollouts()`：从 `MlpMpcPolicy.predictions` 取得 `[x10,u4]` 并调用环境预测评估；
- `MPVEPPO.train()`：在标准 PPO value loss 上增加 `mpve_coef * mpve_value_loss`；
- MPVE loss、valid fraction 等日志。

当前没有独立 MPVE lambda；`gae_lambda` 仍属于真实 rollout 的 GAE。MPVE 当前只用于 Gym，不用于 Flightmare。

默认输出：

```text
runs/acmpc_gym_mpve/<run_name>/
```

该脚本会真正启动训练。

### `run_mpve_horizontal_pipeline.ps1`

水平赛道 MPVE 的训练、评估和绘图流水线。

默认顺序：

1. 调用 `train_acmpc_gym_mpve.py`；
2. 调用 `plot_training_metrics.py`；
3. 调用 `eval_acmpc_gym.py --model-class mpve`；
4. 调用 `plot_trajectories.py`。

支持 `SkipTraining`、`SkipEvaluation`、`SkipPlots`，也支持只使用已有 run 重做评估。默认会启动训练。

## 6. Flightmare 编译和验证脚本

### `build_flightmare_racing_env.ps1`

修改版 Flightmare `flightgym` 的 Windows headless 构建入口。

负责：

- 解析 `FlightmarePath` 和 Conda Python；
- 优先通过 `vswhere` 选择 Visual Studio Build Tools 自带 CMake；
- 避免系统 CMake 4.x 与旧 Flightmare 配置冲突；
- 设置 `FLIGHTMARE_PATH`；
- 执行 `python -m pip install -e .\flightlib --no-deps -v`；
- 生成 `flightgym.cp310-win_amd64.pyd`。

`--no-deps` 用于防止 Flightmare 原始旧依赖污染 AC-MPC 环境。该脚本会编译 C++，但不启动 PPO。

### `smoke_test_flightmare_racing_env.py`

对编译后的 `flightgym.RacingEnv_v1` 做最小运行验证：

- import `flightgym`；
- 创建 C++ 向量环境；
- 检查 env 数量及 36/4/13 维度；
- 调用 reset、getState、step；
- 检查 observation/state/reward finite；
- 打印 extra-info names、最后 reward 和 done。

### `run_smoke_test_flightmare_racing_env.ps1`

上述 Flightmare smoke test 的 PowerShell 包装器，负责设置路径、选择 Conda Python和传入 step 数。

## 7. Flightmare 赛道安装脚本

### `install_gym_track_into_flightmare.py`

把 Gym JSON 作为规范源安装到 Flightmare。

负责：

- 验证 name、start、finish、bounds 和每个 gate 字段；
- 验证 normal/up/width/height/frame thickness；
- 生成 `flightlib/configs/tracks/<track>.yaml`；
- 更新 `flightlib/configs/racing_env.yaml` 的 `track_path`；
- 首次安装时创建 `.pre_track_install.bak`，之后不覆盖原始备份；
- 支持 `--restore` 恢复配置。

只修改配置，不编译、不训练。

### `verify_flightmare_track_runtime.py`

验证“文件已写入”和“C++ 运行时实际加载”是否一致。

它会创建真实 `FlightmareRacingVecEnv/RacingEnv_v1`，检查：

- 当前 track name；
- gate count；
- obs/action/state dimensions；
- reset observation/state；
- runtime config 和 Python metadata 是否一致。

不训练策略。

### `run_train_acmpc_flightmare_track.ps1`

Flightmare 的推荐通用赛道入口。

执行顺序：

1. 检查 `acmpc_racing_gym/tracks/assets/<TrackName>.json`；
2. 调用 `install_gym_track_into_flightmare.py`；
3. 调用 `verify_flightmare_track_runtime.py`；
4. 若不是 `InstallOnly`，调用 `run_train_acmpc_flightmare.ps1`。

支持 `InstallOnly`、`SkipInstall` 和 `SkipVerify`。默认会在安装和验证后启动训练。

## 8. Flightmare 训练和评估脚本

### `train_acmpc_flightmare.py`

修改版 Flightmare 环境上的标准 AC-MPC PPO 训练主体。

负责：

- 写入本次 run 专用 `flightmare_vec_env.yaml`；
- 创建 C++ `FlightmareRacingVecEnv`；
- 添加/恢复 `VecNormalize`；
- 检查实际 env 数量并计算 rollout/batch size；
- 创建 `PPO(MlpMpcPolicy, env)`；
- 保存 config、checkpoint、episode CSV、SB3/TensorBoard 和 CUDA memory log；
- 保存 `final_model.zip`、`vecnormalize.pkl`。

默认输出：

```text
runs/acmpc_flightmare/<run_name>/
```

该脚本会真正启动训练。当前不包含 MPVE。

### `run_train_acmpc_flightmare.ps1`

Flightmare Python 训练主体的参数包装器。解析 Conda 环境、Flightmare path、T、并行环境、线程、PPO 参数、normalization、run name 和 CUDA memory log，再调用 `train_acmpc_flightmare.py`。

它不安装赛道；通常应优先使用 `run_train_acmpc_flightmare_track.ps1`。

### `eval_acmpc_flightmare.py`

Flightmare 模型评估主体。

负责：

- 加载 run config、模型和 VecNormalize；
- 创建评估用 C++ VecEnv；
- 默认 deterministic inference；
- 记录 trajectory、normalized action、反归一化推力/body-rate、state 和 extra info；
- 计算 success/collision/timeout、速度、飞行时间等指标；
- 保存 `track.json`、`metadata.json`、`summary.csv/json` 和 trajectory CSV；
- 默认直接调用 `plot_trajectories.py` 生成速度热力图和平均控制输入，可用 `--no-plots` 禁用。

不更新网络。

### `run_eval_acmpc_flightmare.ps1`

Flightmare 评估的 PowerShell 入口。支持 run/model/normalizer/config/output、episode、seed、T、最大步数、速度色条范围、stochastic 和 `NoPlots`，未指定 run 时选择 latest Flightmare run。

## 9. 训练指标绘图脚本

### `plot_training_metrics.py`

Gym、MPVE 和 Flightmare 共用的训练日志绘图实现。

读取：

```text
<run>/csv/episodes.csv
<run>/sb3/progress.csv
```

生成：

```text
episode_metrics.png
ppo_rollout_metrics.png
ppo_loss_metrics.png
mpve_metrics.png（有 MPVE 字段时）
```

支持横轴：

- `paper-step`：按论文 25000 samples/step 换算；
- `global-timesteps`；
- `updates`。

同时在终端输出近期 return、success、collision、gate index、value loss、KL 和 policy std。

### `run_plot_training_metrics.ps1`

通用 Windows PowerShell 绘图入口。未指定 `RunDir` 时使用 latest Gym run；支持 window、DPI、横轴模式和输出目录。

### `run_plot_training_metrics.sh`

Git Bash/Linux Bash 版训练指标入口。尝试执行 Conda bash hook 和 activate，再调用 `plot_training_metrics.py`。

该脚本不适用于没有 `/bin/bash` 的 WSL 配置；Windows 上优先使用 `.ps1`。

### `run_plot_flightmare_training_metrics.ps1`

Flightmare 专用包装器。未指定 run 时从 `runs/acmpc_flightmare` 选择最新目录，再调用通用 `plot_training_metrics.py`。

## 10. 轨迹与赛道绘图脚本

### `plot_gym_tracks.py`

读取 Gym track JSON，绘制：

- XY、XZ、YZ 投影；
- 3D gate frame；
- gate normal；
- start、finish 和 world bounds；
- 单赛道图片和总览图。

默认输出 `track_visualizations/`，并在终端打印 gate 参数表。它不需要模型，不进行仿真。

### `plot_trajectories.py`

读取评估目录中的 `track.json`、`metadata.json`、`summary.csv` 和 trajectory CSV。

生成：

- `trajectories_speed_top_side.png`：XY/XZ 轨迹速度热力图；
- `mean_control_inputs.png`：平均 normalized/physical 控制输入及标准差。

支持成功、碰撞、timeout 端点标记、速度上下限和最大 episode 数。Gym 和 Flightmare 共用。

### `run_plot_trajectories.ps1`

Gym 轨迹绘图入口。按优先级使用：

1. `EVAL_DIR`；
2. `RUN_DIR/eval` 下最新评估；
3. latest Gym run 的最新评估。

然后调用 `plot_trajectories.py`。

### `run_plot_flightmare_trajectories.ps1`

Flightmare 轨迹绘图入口。要求显式提供 `EvalDir/EVAL_DIR`，支持最大 episode、速度颜色范围、输出目录和 `Show`。

## 11. 典型输出目录

训练 run：

```text
<run>/
├── config.json
├── final_model.zip
├── vecnormalize.pkl
├── checkpoints/
├── csv/
│   ├── episodes.csv
│   └── cuda_memory.csv（启用时）
├── sb3/
│   └── progress.csv
└── plots/
```

Flightmare run 额外包含 `flightmare_vec_env.yaml`。

评估目录：

```text
<run>/eval/<timestamp>/
├── track.json
├── metadata.json
├── summary.csv
├── summary.json
├── trajectories/
└── plots/
```

## 12. 推荐执行顺序

新设备或大改后：

```text
smoke_test_acmpc_forward.py
-> validate_acmpc_core.py
-> smoke_test_racing_gym.py
-> Gym 短训练和评估
-> build_flightmare_racing_env.ps1
-> run_smoke_test_flightmare_racing_env.ps1
-> run_train_acmpc_flightmare_track.ps1 -InstallOnly
-> Flightmare 短训练和评估
-> 正式长训练
```

不要用“能够 import”作为完整验收；至少应验证一次环境 step、一次 AC-MPC forward、一次 PPO update、一次 checkpoint 保存加载和一次确定性评估。
