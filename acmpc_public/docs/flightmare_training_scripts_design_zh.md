# Flightmare 训练、推理、评估与可视化脚本设计说明

本文档说明当前仓库中为 modified Flightmare 竞速环境新增的 Python 训练、评估、推理和可视化脚本设计。目标是让 AC-MPC 方法可以在已经编译好的 Flightmare `RacingEnv_v1` 上运行，并尽量复用前期 Python Gym 阶段已经验证过的 PPO、AC-MPC policy、日志、评估和绘图流程。

## 1. 设计目标

当前阶段的目标不是重新实现 Flightmare，而是把已经编译得到的 `flightgym.RacingEnv_v1` 接入现有 AC-MPC 训练流程。

具体目标如下：

1. 使用 modified Flightmare 的 C++ 环境负责真实环境交互：
   - `reset`
   - `step`
   - `getState`
   - gate passing
   - collision
   - reward
   - done

2. Python 侧只做训练工程适配：
   - 将 `RacingEnv_v1` 封装成 SB3 可用的 `VecEnv`
   - 对接 AC-MPC policy 所需的 `obs` 和 `state`
   - 保存 checkpoint、VecNormalize、CSV、TensorBoard 日志
   - 保存评估轨迹
   - 复用现有轨迹绘图脚本

3. 保持接口与 Python Gym 原型一致：
   - observation dimension: `36`
   - state dimension: `13`
   - action dimension: `4`
   - action range: `[-1, 1]`

4. 不接入 Unity gate 可视化。

## 2. 新增代码结构

新增 Python 包：

```text
acmpc_flightmare/
  __init__.py
  track.py
  vec_env.py
```

新增脚本：

```text
scripts/
  train_acmpc_flightmare.py
  eval_acmpc_flightmare.py
  run_train_acmpc_flightmare.ps1
  run_eval_acmpc_flightmare.ps1
  run_plot_flightmare_trajectories.ps1
  run_plot_flightmare_training_metrics.ps1
```

另外对已有文件做了一个兼容性补充：

```text
scripts/eval_acmpc_gym.py
```

该文件中的 checkpoint 自动搜索逻辑现在也能识别 Flightmare 训练生成的 checkpoint 前缀。

## 3. Flightmare VecEnv 封装

核心文件：

```text
acmpc_flightmare/vec_env.py
```

主要类：

```python
FlightmareRacingVecEnv
```

该类继承 SB3 的 `VecEnv`，负责把 `flightgym.RacingEnv_v1` 包装成 PPO 可以直接使用的向量化环境。

### 3.1 输入输出维度

封装后暴露给 PPO 的空间为：

```text
observation_space: Box(-inf, inf, shape=(36,), dtype=float32)
action_space:      Box(-1, 1, shape=(4,), dtype=float32)
state_space:       Box(-inf, inf, shape=(13,), dtype=float32)
```

其中：

```text
obs   -> 给神经网络 cost map 和 critic 使用
state -> 给 AC-MPC 内部可微 MPC 作为当前动力学状态使用
```

这保持了前期 Python Gym 环境中的设计：

```text
PPO policy input = obs(36) + state(13)
```

### 3.2 reset

调用方式：

```python
obs = env.reset()
```

内部逻辑：

1. 调用 C++ 环境 `RacingEnv_v1.reset(obs_buffer)`
2. 调用 `getState(state_buffer)`
3. 返回 shape 为 `(n_envs, 36)` 的 observation

### 3.3 step

调用方式：

```python
obs, reward, done, infos = env.step(action)
```

内部逻辑：

1. 将 action clip 到 `[-1, 1]`
2. 调用 C++ 环境：

```python
flightgym_env.step(action, obs, reward, done, extra)
```

3. 调用 `getState`
4. 将 Flightmare extra info 转成 Python 字典

输出：

```text
obs:    (n_envs, 36)
reward: (n_envs,)
done:   (n_envs,)
infos:  List[Dict]
```

### 3.4 info 字段

从 Flightmare extra info 和 state 中整理出以下常用字段：

```text
gate_index
gate_passed
finished
collision
collision_code
collision_type
out_of_bounds
timeout
speed
position
quaternion
velocity
omega
terminal_observation
```

其中 `position/velocity/omega` 来自 `getState()` 的 13 维状态。

### 3.5 get_state

AC-MPC policy 训练时需要额外输入 13 维状态，所以 wrapper 提供：

```python
state = env.get_state()
```

返回：

```text
shape = (n_envs, 13)
```

SB3 fork 的 rollout 逻辑会调用：

```python
state_np = env.get_state()
actions, values, log_probs = policy(obs_tensor, state_tensor)
```

因此 Flightmare wrapper 可以直接接入当前 AC-MPC 训练逻辑。

## 4. Flightmare 轨道解析

核心文件：

```text
acmpc_flightmare/track.py
```

作用：

1. 读取 Flightmare 的 `racing_env.yaml`
2. 解析 track name、start pose、world bounds、gate list
3. 根据 gate 的 `center/normal/up/width/height` 计算四个角点
4. 输出与 Python Gym 评估绘图一致的 `track.json` 格式

这样 `plot_trajectories.py` 可以同时绘制 Python Gym 和 Flightmare 的评估结果。

### 4.1 轨道来源

默认读取：

```text
D:\MyProjects\flightmare\flightlib\configs\racing_env.yaml
```

如果需要使用其他配置，可以在训练或评估时传：

```powershell
$env:RACING_CONFIG_PATH="D:\path\to\racing_env.yaml"
```

或在 Python 脚本里使用：

```text
--racing-config-path D:\path\to\racing_env.yaml
```

### 4.2 YAML 解析策略

代码会优先尝试使用 `PyYAML`。

如果环境里没有 `PyYAML`，则使用一个轻量级 fallback parser，只解析当前 `racing_env.yaml` 所需的简单字段。

## 5. 训练脚本

核心文件：

```text
scripts/train_acmpc_flightmare.py
```

PowerShell 启动文件：

```text
scripts/run_train_acmpc_flightmare.ps1
```

### 5.1 训练流程

训练脚本执行流程：

1. 设置环境变量：

```text
ACMPC_T
FLIGHTMARE_PATH
```

2. 创建 run 目录：

```text
runs/acmpc_flightmare/<timestamp>_flightmare_T<ACMPC_T>/
```

3. 在 run 目录下生成 Flightmare vector env 配置：

```text
flightmare_vec_env.yaml
```

内容类似：

```yaml
env:
  seed: 0
  scene_id: 0
  num_envs: 8
  num_threads: 8
  render: no
```

4. 构造 `FlightmareRacingVecEnv`
5. 可选包裹 `VecNormalize`
6. 创建 `PPO(MlpMpcPolicy, env, ...)`
7. 启动 PPO 训练
8. 保存：

```text
final_model.zip
vecnormalize.pkl
config.json
csv/episodes.csv
csv/cuda_memory.csv
sb3/progress.csv
sb3/tensorboard logs
checkpoints/
```

### 5.2 默认训练参数

默认参数与 Python Gym 阶段保持一致：

```text
ACMPC_T = 2
total_timesteps = 200000
n_envs = 8
num_threads = 8
n_steps = 250
n_epochs = 10
gamma = 0.98
gae_lambda = 0.95
clip_range = 0.2
learning_rate_start = 3e-4
learning_rate_end = 1e-5
ent_coef = 0.001
vf_coef = 0.5
max_grad_norm = 0.5
log_std_init = -1.2
```

### 5.3 启动训练

基础命令：

```powershell
cd D:\MyProjects\acmpc_public
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_acmpc_flightmare.ps1
```

指定 Flightmare 路径：

```powershell
$env:FLIGHTMARE_PATH="D:\MyProjects\flightmare"
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_acmpc_flightmare.ps1
```

指定训练步数和并行环境数：

```powershell
$env:TOTAL_TIMESTEPS="500000"
$env:N_ENVS="8"
$env:NUM_THREADS="8"
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_acmpc_flightmare.ps1
```

指定设备：

```powershell
$env:DEVICE="cuda"
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_acmpc_flightmare.ps1
```

## 6. 推理与评估脚本

核心文件：

```text
scripts/eval_acmpc_flightmare.py
```

PowerShell 启动文件：

```text
scripts/run_eval_acmpc_flightmare.ps1
```

### 6.1 评估流程

评估脚本执行流程：

1. 读取训练 run 的 `config.json`
2. 自动选择模型：
   - 优先 `final_model.zip`
   - 否则选择最新 checkpoint
3. 自动选择 `vecnormalize.pkl`
4. 创建单环境 Flightmare eval vec env
5. 加载 PPO 模型
6. 逐 episode 执行 deterministic 或 stochastic policy
7. 保存：

```text
summary.csv
summary.json
metadata.json
track.json
trajectories/trajectory_episode_0000.csv
trajectories/trajectory_episode_0001.csv
...
```

### 6.2 评估指标

评估输出中包含：

```text
return_mean
return_std
success_rate
crash_rate
timeout_rate
average_length
average_velocity
average_lap_time_successes
final_gate_index_mean
```

单条轨迹 CSV 中包含：

```text
episode
step
time_s
reward
done
gate_index
gate_passed
collision
collision_type
finished
timeout
x, y, z
vx, vy, vz
speed
wx, wy, wz
action_thrust
action_wx
action_wy
action_wz
```

### 6.3 启动评估

评估最新 Flightmare run：

```powershell
cd D:\MyProjects\acmpc_public
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_acmpc_flightmare.ps1
```

指定 run：

```powershell
$env:RUN_DIR="D:\MyProjects\acmpc_public\runs\acmpc_flightmare\<run_name>"
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_acmpc_flightmare.ps1
```

指定评估 episode 数：

```powershell
$env:EPISODES="64"
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_acmpc_flightmare.ps1
```

使用随机策略采样，而非确定性推理：

```powershell
$env:STOCHASTIC="1"
powershell -ExecutionPolicy Bypass -File .\scripts\run_eval_acmpc_flightmare.ps1
```

## 7. 训练过程可视化

PowerShell 启动文件：

```text
scripts/run_plot_flightmare_training_metrics.ps1
```

该脚本复用已有：

```text
scripts/plot_training_metrics.py
```

因为 Flightmare 训练脚本保存的日志格式与 Python Gym 训练一致：

```text
csv/episodes.csv
sb3/progress.csv
```

### 7.1 启动训练曲线绘制

默认绘制最新 Flightmare run：

```powershell
cd D:\MyProjects\acmpc_public
powershell -ExecutionPolicy Bypass -File .\scripts\run_plot_flightmare_training_metrics.ps1
```

指定 run：

```powershell
$env:RUN_DIR="D:\MyProjects\acmpc_public\runs\acmpc_flightmare\<run_name>"
powershell -ExecutionPolicy Bypass -File .\scripts\run_plot_flightmare_training_metrics.ps1
```

输出默认保存到：

```text
<run_dir>/plots/
```

主要图片：

```text
episode_metrics.png
ppo_rollout_metrics.png
ppo_loss_metrics.png
mpve_metrics.png
```

## 8. 轨迹可视化

PowerShell 启动文件：

```text
scripts/run_plot_flightmare_trajectories.ps1
```

该脚本复用已有：

```text
scripts/plot_trajectories.py
```

前提是先运行评估脚本，生成：

```text
track.json
summary.csv
trajectories/*.csv
```

### 8.1 启动轨迹绘制

```powershell
cd D:\MyProjects\acmpc_public
$env:EVAL_DIR="D:\MyProjects\acmpc_public\runs\acmpc_flightmare\<run_name>\eval\<timestamp>"
powershell -ExecutionPolicy Bypass -File .\scripts\run_plot_flightmare_trajectories.ps1
```

输出默认保存到：

```text
<eval_dir>/plots/
```

主要图片：

```text
trajectories_speed_top_side.png
trajectories_speed_top_view.png
```

轨迹颜色含义：

```text
蓝色 -> 速度慢
红色 -> 速度快
```

## 9. 与 Python Gym 阶段的关系

Flightmare 脚本不是重新写一套算法，而是把环境后端从 Python Gym 替换为 Flightmare。

保持不变的部分：

```text
MlpMpcPolicy
PPO
SB3 state plumbing
VecNormalize
checkpoint 保存方式
episode CSV logging
training metrics plotting
trajectory plotting
```

替换的部分：

```text
Python Gym RacingEnv -> Flightmare RacingEnv_v1
Python 动力学 -> Flightmare C++ 动力学
Python reward/done -> Flightmare C++ reward/done
Python gate passing -> Flightmare C++ gate passing
```

## 10. 数据流

训练时的数据流如下：

```text
Flightmare RacingEnv_v1
  -> obs(36)
  -> state(13)
  -> MlpMpcPolicy
  -> neural cost map
  -> differentiable MPC
  -> normalized action(4)
  -> Flightmare step
  -> reward/done/info
  -> PPO rollout buffer
  -> PPO update
```

其中：

```text
obs(36)   用于神经网络 cost map 和 critic
state(13) 用于 MPC 内部动力学初始状态
action(4) 是 thrust/body-rate normalized command
```

## 11. 当前验证情况

已完成以下非训练验证：

1. 新增 Python 文件通过 `py_compile`
2. `train_acmpc_flightmare.py --help` 正常
3. `eval_acmpc_flightmare.py --help` 正常
4. `FlightmareRacingVecEnv` smoke test 正常

烟测结果：

```text
n_envs = 2
obs_dim = 36
act_dim = 4
state_dim = 13
obs_shape = (2, 36)
state_shape = (2, 13)
step reward shape = (2,)
done shape = (2,)
```

这说明 Flightmare 环境接口已经能被当前 AC-MPC PPO 流程调用。

## 12. 注意事项

### 12.1 修改轨道后是否需要重新编译

如果只是修改：

```text
D:\MyProjects\flightmare\flightlib\configs\racing_env.yaml
```

通常不需要重新编译。

如果修改 C++ 文件，例如：

```text
flightlib/src/envs/racing_env/racing_env.cpp
flightlib/include/flightlib/envs/racing_env/racing_env.hpp
flightlib/wrapper/pybind11/flightgym.cpp
```

则需要重新编译 Flightmare。

### 12.2 VecNormalize 必须保存和加载

训练时如果启用了 observation normalization，评估时应使用同一个：

```text
vecnormalize.pkl
```

否则 policy 输入分布会变，评估结果不可信。

### 12.3 Flightmare pyd 与设备绑定

当前使用的是：

```text
flightgym.cp310-win_amd64.pyd
```

它依赖：

```text
Python 3.10
Windows x64
MSVC runtime
当前 Flightmare 编译产物
```

如果换设备，Python 版本、系统、编译器运行库、依赖路径不同，可能需要重新编译。

### 12.4 Unity 可视化未接入

当前脚本是 headless 训练和评估，不依赖 Unity。

如果后续要做 Unity 可视化，需要额外处理：

```text
connectUnity
disconnectUnity
render config
Unity scene
gate mesh visualization
```

这不是当前脚本的目标。

## 13. 后续建议

建议后续按以下顺序推进：

1. 先用当前 horizontal track 跑短训练，确认 Flightmare 训练不会崩溃。
2. 用同一套绘图脚本查看：
   - reward evolution
   - success rate
   - collision rate
   - gate progress
   - PPO loss
3. 将 `racing_env.yaml` 改成手工设计的 Split-S track。
4. 先评估 gate passing 和 reward 是否符合预期。
5. 再启动较长时间训练。
6. 若训练卡在某个门，优先检查：
   - gate normal 是否正确
   - gate 顺序是否合理
   - world bounds 是否过窄
   - reset 初始点是否太远或太偏
   - reward 是否能给到连续 progress 信号
7. 最后再考虑接 Unity 可视化或更真实的 domain randomization 接口。

