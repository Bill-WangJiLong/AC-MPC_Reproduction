# AC-MPC 仿真复现计划

## 1. 目的

本文档定义 Actor-Critic Model Predictive Control (AC-MPC) 方法在仿真中的技术复现计划。

本计划基于四类资料：

- AC-MPC 论文：`Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning for Agile Flight`。
- AC-MPC 代码仓库：`D:/MyProjects/acmpc_public`。
- Flightmare 论文：`Flightmare: A Flexible Quadrotor Simulator`。
- Flightmare 基础代码仓库：`D:/MyProjects/flightmare`。

目标是技术复现，不是数值复现论文中的图表或指标。

## 2. 目标范围

### 最终目标

实现并运行 AC-MPC 在四旋翼竞速任务中的仿真训练和推理。

最终系统必须做到：

- 使用论文中的 AC-MPC actor 结构：神经代价映射加可微 MPC。
- 使用 PPO 训练策略。
- 从训练好的 checkpoint 运行确定性推理。
- 最终阶段使用 Flightmare 作为外部仿真器。
- 使用受论文启发的穿门竞速任务：Horizontal、Vertical 和 SplitS-like 赛道。
- 保持观测、动作、奖励和 MPC 接口与 AC-MPC 论文一致。
- 为后续 sim-to-real 工作保留明确接口。

最终系统不需要做到：

- 复现 AC-MLP、tracking MPC 或 L1-MPC baseline。
- 复现论文精确指标、精确图像或精确 gate 坐标。
- 使用 BEM 或 NeuroBEM。
- 部署到真实硬件。
- 实现 Agilicious 集成。

### 中间目标

在修改 Flightmare 之前，先实现一个 Python Gym 竞速环境，用于展示完整的 AC-MPC 训练和推理闭环。

中间系统必须做到：

- 使用 Python Gym 风格的 `reset()` 和 `step()`。
- 使用非平凡的四旋翼竞速环境，包含 gate、碰撞检查、控制约束、执行器滞后，以及可选阻力。
- 使用本仓库现有的 AC-MPC policy 代码。
- 使用 PPO 训练。
- 保存和加载 checkpoint。
- 运行确定性评估。
- 记录轨迹和基础指标。

这个阶段用于在引入 Flightmare C++/pybind 复杂度之前，验证 AC-MPC 方法链路。

## 3. 当前仓库事实

### AC-MPC 仓库

相关文件：

- `diff_mpc_drones/drone.py`
  - 定义 `DroneDx`，即 MPC 使用的可微内部四旋翼模型。
  - 状态：
    - `x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz]`
    - 维度：10
  - 控制：
    - `u = [collective_thrust, omega_x, omega_y, omega_z]`
    - 维度：4
  - 提供前向动力学和解析雅可比。

- `diff_mpc_drones/il_env.py`
  - 封装 `mpc.pytorch`。
  - 构建控制下界和上界。
  - 调用 `mpc.MPC(...)(xinit, QuadCost(Q, p), dx)`。
  - 返回预测状态序列和控制序列。

- `training_modules/mlp_mpc_policy.py`
  - 定义 `MlpMpcPolicy`。
  - actor 网络输出 MPC 代价参数。
  - MPC horizon 从环境变量读取：
    - `ACMPC_T`
  - actor 返回 MPC 第一拍动作，作为归一化策略动作。
  - 将 MPC 预测保存到：
    - `self.predictions`

- `training_modules/mlp_only_policy.py`
  - 定义 AC-MLP baseline。
  - 当前复现目标不需要这个文件。

- `mpc.pytorch`
  - 可微 MPC 依赖。
  - 当前需要该子模块才能运行 policy。

- `stable-baselines3`
  - 作者用于 AC-MPC 集成的 fork。
  - 应用于 PPO 训练，而不是使用 Flightmare 旧的 TensorFlow PPO 栈。

### Flightmare 仓库

相关模块：

- `flightlib`
  - C++ 仿真核心。
  - 包含四旋翼动力学、四旋翼对象、向量化环境、Unity bridge 和 pybind wrapper。

- `flightrl`
  - TensorFlow `stable_baselines==2.10.1` PPO2 示例。
  - 可作为参考，但不应作为 AC-MPC 的主训练栈。

- `flightros`
  - ROS 示例。
  - 包含 gate 可视化和轨迹示例。
  - 不是强化学习竞速环境。

- `flightrender`
  - Unity 渲染端。
  - Unity binary 是外部文件。

重要 Flightmare 文件：

- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp`
  - 默认环境。
  - 这是稳定/悬停环境，不是竞速环境。
  - 观测：`[p, euler, v, omega]`，维度 12。
  - 动作：归一化单电机推力，维度 4。

- `flightlib/src/objects/quadrotor.cpp`
  - 同时支持单电机推力命令和 rate-thrust 命令。
  - rate-thrust 命令形式：
    - `Command(t, collective_thrust, omega)`
  - 这与 AC-MPC 论文中的动作类型一致。

- `flightlib/include/flightlib/objects/static_gate.hpp`
  - 为 Unity 渲染提供静态 gate 对象。

- `flightlib/src/wrapper/pybind_wrapper.cpp`
  - 暴露 `QuadrotorEnv_v1`。
  - 后续必须暴露新的 `RacingEnv_v1`。

## 4. 方法映射

### AC-MPC 论文组件映射

| 论文组件 | 仓库组件 | 需要做的工作 |
|---|---|---|
| 神经代价映射 | `training_modules/mlp_mpc_policy.py` | 复用，并修复路径和依赖 |
| 可微 MPC | `mpc.pytorch` 和 `diff_mpc_drones/il_env.py` | 初始化子模块并验证 |
| 内部四旋翼模型 | `diff_mpc_drones/drone.py` | 复用 |
| Actor 输出动作 | `MlpMpcPolicy.forward_actor()` | 复用 |
| PPO 训练 | `stable-baselines3` AC-MPC fork | 复用并接入环境 |
| 外部仿真器 | 先 Python Gym，后 Flightmare | 实现 |
| 竞速 gates | 当前不存在 | 实现 |
| 竞速奖励 | 当前不存在 | 实现 |
| Flightmare modified racing env | 当前不存在 | 实现 |
| MPVE | 未完整实现 | 若要求覆盖完整扩展方法，在基础 AC-MPC 稳定后添加 |

### 观测契约

AC-MPC 论文定义：

```text
o_quad = [v_t, R_t] in R12
```

其中：

- `v_t`：四旋翼线速度，维度 3。
- `R_t`：旋转矩阵，维度 9。

赛道观测使用未来 gates：

```text
o_track = future G=2 gate geometry
```

实现时使用：

```text
从四旋翼中心到未来两个 gate 的四个角点的相对位置
```

每个 gate 贡献：

```text
4 个角点 * 3 个坐标 = 12 个值
```

两个未来 gate 贡献：

```text
24 个值
```

推荐的 policy feature vector：

```text
features = [v, R, future_gate_corner_features]
```

推荐的 MPC state vector：

```text
mpc_state = [p, q, v]
```

policy 集成必须同时向 `MlpMpcPolicy` 提供两者，因为 MPC block 需要当前 `[p, q, v]`。

### 动作契约

AC-MPC 论文使用：

```text
a = [c, omega_x, omega_y, omega_z]
```

其中：

- `c`：质量归一化 collective thrust。
- `omega`：body rates。

policy 输出归一化动作。环境必须把归一化动作映射成物理命令。

推荐的归一化动作映射：

```text
thrust_normalized in [-1, 1]
omega_normalized in [-1, 1]^3

c = force_mean + force_std * thrust_normalized
omega = omega_max * omega_normalized
```

这应与 `MlpMpcPolicy` 中的归一化逻辑保持镜像一致。

### 奖励契约

使用受论文启发的竞速奖励：

```text
if collision:
    terminate episode
if gate passed:
    reward += 10
if race finished:
    reward += 10
otherwise:
    reward += progress_reward - body_rate_coeff * ||omega||
```

progress reward：

```text
progress_reward = distance(previous_position, target_gate_center)
                  - distance(current_position, target_gate_center)
```

默认系数：

```text
body_rate_coeff = 0.01
```

不要把环境做得过于理想化：

- 添加 gate 几何和 gate 框架检查。
- 添加碰撞或越界终止。
- 添加推力和 body-rate 饱和。
- 添加执行器或 body-rate 滞后。
- 在外部环境中添加可选 drag。

## 5. 建议的中间 Python 代码结构

在 `D:/MyProjects/acmpc_public` 下添加：

```text
acmpc_gym/
  __init__.py
  envs/
    __init__.py
    racing_env.py
    dynamics.py
    gates.py
    tracks.py
    reward.py
    sim_params.py
  wrappers/
    __init__.py
    acmpc_obs_wrapper.py
    vec_env_adapter.py
configs/
  train/
    acmpc_n2.yaml
    acmpc_n5.yaml
  tracks/
    horizontal.yaml
    vertical.yaml
    splits_like.yaml
  env/
    python_nominal.yaml
scripts/
  train_acmpc_gym.py
  eval_acmpc_gym.py
  plot_trajectories.py
  smoke_test_acmpc_forward.py
tests/
  test_drone_dx.py
  test_mpc_solver.py
  test_mlp_mpc_policy_forward.py
  test_gate_geometry.py
  test_racing_env_step.py
docs/
  acmpc_reproduction_plan.md
  training_and_eval_protocol.md
  known_gaps_vs_paper.md
```

### Python 环境职责

`racing_env.py`：

- 实现 Gym `reset()` 和 `step()`。
- 维护仿真器状态。
- 调用外部动力学积分。
- 跟踪当前 gate。
- 计算观测、奖励和 done。

`dynamics.py`：

- 实现外部仿真动力学。
- 不应以完全相同的形式简单调用 MPC 内部模型。
- 应包含可选：
  - drag
  - 执行器滞后
  - 命令饱和
  - 外力

`gates.py`：

- 定义 gate 几何。
- 计算角点位置。
- 检查 gate crossing。
- 检查门框碰撞。

`tracks.py`：

- 加载 track YAML。
- 为观测提供未来 gates。

`reward.py`：

- 实现奖励项并记录奖励分解。

`sim_params.py`：

- 定义类型化仿真参数。
- 后续应直接映射到 Flightmare 和 sim-to-real 参数块。

## 6. 建议的 Flightmare 代码结构

不要为了竞速修改原始 `QuadrotorEnv`。新增一个环境。

在 `D:/MyProjects/flightmare` 下添加：

```text
flightlib/include/flightlib/envs/racing_env/racing_env.hpp
flightlib/src/envs/racing_env/racing_env.cpp
flightlib/configs/racing_env.yaml
```

修改：

```text
flightlib/src/wrapper/pybind_wrapper.cpp
```

添加绑定：

```cpp
py::class_<VecEnv<RacingEnv>>(m, "RacingEnv_v1")
```

新的 `RacingEnv` 应该：

- 复用 `Quadrotor`。
- 使用 `Command(t, collective_thrust, omega)`，而不是单电机推力动作。
- 从 YAML 加载 track 和 reward 参数。
- 返回 AC-MPC 兼容观测。
- 通过 `VecEnv` 支持向量化执行。
- 可选地添加 `StaticGate` 对象到 Unity 进行可视化。
- 暴露 extra info：
  - gate index
  - success
  - collision
  - progress reward
  - body-rate penalty
  - lap time
  - average velocity

最终 Flightmare Python 流程：

```text
flightgym.RacingEnv_v1
-> Python VecEnv wrapper
-> stable-baselines3-acmpc PPO
-> MlpMpcPolicy
-> mpc.pytorch
```

不要使用 Flightmare 的 `flightrl` TensorFlow PPO2 作为 AC-MPC 的主训练栈。

## 7. 可配置量

所有重要量都必须定义在 YAML 或 dataclass 风格 config 中。除稳定数学常量外，避免硬编码。

### AC-MPC 和 MPC 设置

| 名称 | 含义 | 初始值 |
|---|---|---|
| `ACMPC_T` | `MlpMpcPolicy` 使用的 MPC horizon | `2` |
| `mpc_lqr_iter` | MPC iLQR 迭代次数 | policy forward 中先用 `1`，测试中可更高 |
| `mpc_eps` | MPC 收敛容差 | 来自 `DroneDx` |
| `linesearch_decay` | MPC line search 衰减 | 来自 `DroneDx` |
| `max_linesearch_iter` | MPC line search 迭代次数 | 来自 `DroneDx` |
| `cost_q_min` | diagonal Q 下界 | `0.1` |
| `cost_q_max` | diagonal Q 上界范围 | `100000` |
| `cost_p_range` | p 项范围 | `100000` |

### 四旋翼动力学设置

| 名称 | 含义 | 初始来源 |
|---|---|---|
| `dt` | 控制步长 | `0.02` |
| `mass` | 四旋翼质量 | 先与 `DroneDx` 对齐，后续可配置 |
| `inertia` | 惯量对角项 | 先与 `DroneDx` 对齐 |
| `thrust_min` | 最小推力 | 来自 `DroneDx` 或 Flightmare |
| `thrust_max` | 最大推力 | 来自 `DroneDx` 或 Flightmare |
| `omega_max` | body-rate 限制 | 初始来自 `DroneDx` |
| `actuator_tau` | 命令滞后 | 在外部环境中启用 |
| `drag_coeff` | 线性或二次 drag | 可选但推荐 |
| `external_force` | 风或扰动 | 默认关闭 |

### 竞速环境设置

| 名称 | 含义 | 初始值 |
|---|---|---|
| `track_name` | 选择的赛道 | `horizontal` |
| `gate_centers` | gate 位置 | 自定义、受论文启发的值 |
| `gate_orientations` | gate 四元数或 yaw-pitch-roll | 自定义 |
| `gate_size` | gate 宽度和高度 | 可配置 |
| `gate_frame_radius` | 碰撞余量 | 可配置 |
| `future_gate_count` | 观测中的未来 gate 数量 | `2` |
| `world_box` | 有效飞行区域 | 可配置 |
| `max_episode_time` | 超时 | 可配置 |
| `init_position_cube` | reset 位置随机化 | 训练时 `1.0 m` |
| `eval_position_cube` | 评估初始位置随机化 | 根据测试使用 `0.5 m` 或 `3.0 m` |

### 奖励设置

| 名称 | 含义 | 初始值 |
|---|---|---|
| `gate_pass_reward` | 通过 gate 的奖励 | `10.0` |
| `race_finish_reward` | 完成赛道的奖励 | `10.0` |
| `collision_reward` | 碰撞终止奖励 | `0.0` 或可配置负值 |
| `progress_scale` | progress reward 缩放 | `1.0` |
| `body_rate_coeff` | body-rate 惩罚 | `0.01` |
| `action_smooth_coeff` | 动作平滑惩罚 | 初始关闭，预留 |

### PPO 设置

| 名称 | 论文值 |
|---|---|
| `gamma` | `0.98` |
| `gae_lambda` | `0.95` |
| `n_steps` | `250` |
| `batch_size` | 目标 `25000`，根据可用 `num_envs` 调整 |
| `n_epochs` | `10` |
| `clip_range` | `0.2` |
| `learning_rate` | 从 `3e-4` 到 `1e-5` 的 schedule |
| `ent_coef` | `0.001` |
| `vf_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `init_log_std` | AC-MPC: `-2.2` |

### 日志和评估设置

| 名称 | 含义 |
|---|---|
| `seed` | 可复现随机种子 |
| `num_envs` | 并行训练环境数量 |
| `eval_episodes` | 评估 rollout 数量 |
| `deterministic_eval` | 评估时不采样动作 |
| `trajectory_log_rate` | 轨迹记录间隔 |
| `checkpoint_interval` | checkpoint 保存间隔 |
| `plot_sample_count` | 绘图轨迹数量 |

## 8. Sim-to-Real 接口预留

Sim-to-real 不在当前目标中，但设计不能阻塞它。

从一开始预留这些接口。

### 执行器模型接口

创建 `ActuatorModel` 抽象：

```text
normalized_action -> commanded_physical_action -> applied_physical_action
```

它应支持：

- thrust map
- motor lag
- command delay
- command saturation
- action smoothing penalty
- 单电机推力限制

初始实现：

- 简单推力和 body-rate 缩放
- 可选一阶滞后

未来实现：

- 实测 thrust map
- latency queue
- 电机响应模型

### 气动模型接口

创建 `AeroModel` 抽象：

```text
state, action -> extra_force, extra_torque
```

初始实现：

- 无 aero
- 可选简单线性 drag

未来实现：

- rotor drag
- 学习型 drag
- BEM 或 NeuroBEM adapter

### 传感器和状态估计接口

创建 `SensorModel` 或 `ObservationModel` 抽象：

```text
true_state, track_state -> policy_observation, mpc_state
```

它应支持：

- 观测归一化
- 测量噪声
- 状态延迟
- 部分观测
- 不同真实世界状态估计器输出

初始实现：

- 精确状态
- running mean 和 std 归一化

### 动力学参数提供器

创建 `DynamicsParams` config 对象：

```text
mass
inertia
thrust limits
body-rate limits
drag
latency
motor_tau
```

它应支持：

- nominal values
- fixed perturbations
- domain randomization ranges
- 用于鲁棒性测试的运行时参数更新

初始实现：

- 仅 nominal values

未来实现：

- 每个 episode 随机化
- 每个环境随机化

### Track 和 Gate Schema

Python Gym 和 Flightmare 使用相同的 track YAML schema：

```yaml
track:
  name: horizontal
  gates:
    - id: gate_0
      center: [0.0, 0.0, 2.0]
      rpy: [0.0, 0.0, 0.0]
      size: [2.0, 2.0]
    - id: gate_1
      center: [5.0, 0.0, 2.0]
      rpy: [0.0, 0.0, 0.0]
      size: [2.0, 2.0]
```

这允许同一套任务定义从 Python Gym 迁移到 Flightmare。

### 日志 Schema

始终记录足够的数据，以便后续 sim-to-real 分析：

```text
time
seed
env_id
track_name
state_true
state_observed
mpc_state
normalized_action
commanded_action
applied_action
dynamics_params
gate_id
reward_terms
done_reason
success
collision
```

## 9. 实现阶段和退出标准

### Phase 0: 依赖和子模块准备

任务：

- 初始化 `mpc.pytorch`。
- 初始化 `stable-baselines3` AC-MPC fork。
- 修复 `mlp_mpc_policy.py` 中的 `DRONE_PATH`。
- 记录 Python 和 PyTorch 版本。
- 创建 AC-MPC forward smoke script。

退出标准：

- `import mpc` 可用。
- `import drone` 可用。
- `MlpMpcPolicy` 可以初始化。
- 单次 MPC 求解返回有限状态序列和动作序列。

### Phase 1: AC-MPC 核心验证

任务：

- 测试 `DroneDx.forward()`。
- 测试四元数 norm 行为。
- 测试解析雅可比维度。
- 测试 `IL_Env.mpc()`。
- 测试 `MlpMpcPolicy.forward_actor()`。

退出标准：

- forward pass 中无 NaN。
- MPC action 满足边界。
- batch size `1`、`8` 和 `64` 至少都可 forward。
- `self.predictions` 具有期望 shape。

### Phase 2: Python Gym 竞速环境

任务：

- 实现 gate 几何。
- 实现 track 加载。
- 实现外部四旋翼动力学。
- 实现 AC-MPC observation builder。
- 实现 reward 和 done 逻辑。
- 添加 SB3 wrappers。

退出标准：

- 随机动作可以运行完整 episode。
- 单元测试能正确检测 gate passing。
- 碰撞和越界终止可用。
- 观测空间和动作空间符合 policy 期望。

### Phase 3: Python Gym 训练

任务：

- 实现 `train_acmpc_gym.py`。以PPO方式进行训练。
- 实现 checkpoint 保存。
- 实现 TensorBoard 或 CSV logging。
- 先训练 水平轨道
- 经我认可后再训练竖直和赛道轨道
- 参数选择先按论文和gym设置，其次看一下设备限制，有参数相对于论文更改时，写一个单独文件记录原本的数值和本地复现的数值

退出标准：

- 训练不崩溃。
- 生成 reward logs。
- checkpoint 可以保存和加载。
- 确定性评估可运行。
- 训练后可以在轨道成功穿门。

### Phase 4: Python Gym 评估和绘图

任务：

- 实现 `eval_acmpc_gym.py`。
- 实现轨迹记录。
- 实现 `plot_trajectories.py`。
- 计算 success rate、average velocity、lap time 和 crash rate。

退出标准：

- 评估输出指标和轨迹文件。
- 图中显示 gates 和轨迹。
- 固定 seed 下确定性 policy 可复现。

### Phase 5: MPVE 方法扩展

这是扩展版 AC-MPC 论文方法的一部分，对应论文 Algorithm 2：

```text
Actor-Critic Model Predictive Control with Model-Predictive Value Expansion
```

它不是新的 actor，也不是新的 MPC 求解器，而是在基础 AC-MPC 的 PPO 训练中增加一个 critic 训练项。其核心思想是：每次 AC-MPC actor forward 时，diffMPC 已经预测出一段短 horizon 的状态和动作序列，但基础 AC-MPC 只执行第一拍动作，后续预测被丢弃。MPVE 将这些预测状态和动作复用起来，计算预测奖励和预测 value target，从而额外训练 value function。

当前仓库状态：

- `training_modules/mlp_mpc_policy.py` 已经在 `MlpMpcPolicy.forward_actor()` 中计算 MPC 预测：
  - `nom_x`: MPC 预测状态序列。
  - `nom_u`: MPC 预测控制序列。
  - `self.predictions = cat(nom_x, nom_u)`。
- 当前 `self.predictions` 只是被保存，没有进入 rollout buffer。
- 当前 PPO 仍然只使用普通 TD(lambda) value loss：
  - `value_loss = mse(rollout_data.returns, values_pred)`。
- 当前代码还不是完整 Algorithm 2。

因此，本阶段目标是把当前“已经暴露的 MPC predictions”接入 PPO 的 critic 训练，使 value loss 变为：

```text
value_loss = TD(lambda) value loss + mpve_coef * MPVE value loss
```

基础 AC-MPC 未稳定前，不应启用本阶段。MPVE 必须作为可关闭的训练增强项实现，默认关闭。

#### 论文对应关系

论文 Algorithm 2 的关键新增步骤是：

```text
Collect set of trajectories and predictions with
x_{k:k+H}, u_{k:k+H} ~ N{diffMPC(x_k, Q(s_k), p(s_k)), Sigma}

Compute reward-to-go for predictions R_hat_{k:k+H}
and value targets using TD k-trick

Fit value function by regression on mean-squared error
using a sum of the TD(lambda) value loss and Eq. (9)
```

在代码中应对应为：

```text
MlpMpcPolicy.predictions
-> collect_rollouts() 取出预测序列
-> RolloutBuffer 保存预测状态/动作/奖励/观测
-> PPO.train() 中计算 Eq. (9) MPVE value loss
-> 将 MPVE value loss 加到普通 value loss
```

#### 数据约定

`MlpMpcPolicy.predictions` 当前应视为一个训练中间量，而不是环境状态。

建议约定：

```text
predictions.shape = [batch, H, n_state_mpc + n_ctrl]
                  = [batch, H, 10 + 4]
                  = [batch, H, 14]
```

其中：

```text
prediction[..., 0:10]  = x_mpc = [p, q, v]
prediction[..., 10:14] = u_mpc = [collective_thrust, wx, wy, wz]
```

这里的 `H` 建议先取：

```text
H = ACMPC_T
```

也就是 MPVE horizon 等于 diffMPC horizon。后续如需单独调参，可增加 `mpve_horizon`，但第一版不建议和 `ACMPC_T` 分离。

预测序列必须 `detach()` 后进入 MPVE critic loss：

```text
MPVE 训练 critic，不应通过 MPVE value loss 额外更新 actor/cost map。
```

actor/cost map 仍然只通过 PPO policy loss 和 diffMPC backward 更新。

#### 需要新增或修改的模块

1. `training_modules/mlp_mpc_policy.py`

需要整理 `self.predictions` 的语义：

- 保证每次 `forward_actor()` 后 `self.predictions` shape 稳定。
- 使用 batch-first 格式：
  - `[batch, H, 14]`。
- 保证保存的是 detached tensor：
  - 不让 MPVE critic loss 通过 predictions 回传到 actor。
- 增加可读注释，说明：
  - predictions 仅用于 logging、visualization 和 MPVE critic target。
  - 真正执行的 action 仍然是第一拍 `nom_u[:, 0, :]`。

2. `stable-baselines3/stable_baselines3/common/buffers.py`

需要扩展 `RolloutBuffer`，保存预测序列。

新增字段建议：

```text
self.prediction_states
self.prediction_actions
self.prediction_obs
self.prediction_rewards
self.prediction_valid
```

第一版可以先只保存：

```text
prediction_states:  [n_steps, n_envs, H, 10]
prediction_actions: [n_steps, n_envs, H, 4]
prediction_rewards: [n_steps, n_envs, H]
prediction_obs:     [n_steps, n_envs, H, obs_dim]
prediction_valid:   [n_steps, n_envs, H]
```

其中 `prediction_valid` 用于处理：

- horizon 内发生碰撞。
- horizon 内越界。
- horizon 内穿过 gate 后 target gate 变化。
- episode 已经结束。

`RolloutBufferSamples` 也需要补充对应字段，否则 `PPO.train()` 无法从 mini-batch 中读取 MPVE 数据。

3. `stable-baselines3/stable_baselines3/common/on_policy_algorithm.py`

需要在 rollout 采样阶段，从 policy 中取出 MPC predictions。

当前基础逻辑是：

```text
obs, state
-> policy(obs, state)
-> action, value, log_prob
-> env.step(action)
-> rollout_buffer.add(...)
```

MPVE 版需要变为：

```text
obs, state
-> policy(obs, state)
-> action, value, log_prob
-> policy.mlp_extractor.predictions
-> env.compute_prediction_rollout(...)
-> prediction_obs, prediction_rewards, prediction_valid
-> env.step(action)
-> rollout_buffer.add(..., prediction_*)
```

注意：

- predictions 必须与当前 `obs/state/action` 对齐。
- predictions 必须在 `env.step(action)` 前读取，因为它对应当前时刻的 MPC 解。
- 如果 policy 被包装在 SB3 policy 内，访问路径可能是：
  - `self.policy.mlp_extractor.predictions`。
- 若后续使用多 GPU 或并行 env，应避免用全局状态覆盖 predictions。

4. Python Gym 环境和后续 Flightmare 环境

需要提供一个纯函数式预测评估接口，不改变真实环境状态。

建议接口：

```python
compute_prediction_data(
    prediction_states,
    prediction_actions,
    current_track_state,
) -> dict
```

返回：

```text
prediction_obs
prediction_rewards
prediction_valid
```

该函数必须满足：

- 不推进真实环境。
- 不修改真实 gate index。
- 不修改真实 collision state。
- 使用和 `step()` 完全一致的 reward、collision、gate passing 逻辑。
- 可以根据预测状态模拟 horizon 内 gate target 的变化。

第一版可在 Python Gym 中实现；迁移 Flightmare 时在 C++/pybind 中实现同名接口。

5. Observation builder

MPVE 需要把预测状态 `x_mpc` 转成 critic 可输入的 observation。

因此 observation builder 必须支持：

```text
build_observation_from_state(state, track_context)
```

而不是只能从环境当前内部状态读取。

对 racing 任务，预测 observation 必须包含：

- 预测状态对应的速度 `v`。
- 预测状态对应的旋转矩阵 `R`。
- 当前预测 target gate 的 corner observation。
- 下一个 gate 的 corner observation。
- 当前配置的 gate observation mode：
  - `vehicle_relative`
  - `chained_gate_relative`

如果后续确认论文使用的是另一种 gate corner 差分方式，只应改 observation builder，不应改 MPVE 主流程。

6. `stable-baselines3/stable_baselines3/ppo/ppo.py`

需要增加 MPVE value loss。

基础 PPO value loss：

```python
td_value_loss = F.mse_loss(rollout_data.returns, values_pred)
```

MPVE 需要额外计算：

```text
mpve_value_loss =
mean_t mean_h valid_mask *
(
    V(prediction_obs_{t,h})
    - prediction_target_{t,h}
)^2
```

其中 prediction target 使用论文 Eq. (9) 的 TD k-trick：

```text
target_h =
sum_{k=h}^{H-1} gamma^(k-h) * r_hat_k
+ gamma^(H-h) * V(prediction_obs_H)
```

第一版实现可简化为：

```text
对每个 h in [0, H-1]：
target_h = discounted predicted rewards from h to H-1
           + bootstrap from final predicted observation
```

bootstrap value 必须 `detach()`，避免 target 自身参与梯度：

```python
target_h = target_h.detach()
```

最终：

```python
value_loss = td_value_loss + mpve_coef * mpve_value_loss
loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
```

建议新增日志：

```text
train/td_value_loss
train/mpve_value_loss
train/mpve_valid_fraction
train/value_loss
```

#### 配置项

建议新增训练配置：

```text
use_mpve: false
mpve_coef: 1.0
mpve_horizon: null
mpve_bootstrap: true
mpve_detach_predictions: true
mpve_valid_mask: true
```

含义：

- `use_mpve`
  - 是否启用 Algorithm 2。
  - 默认 `false`。
- `mpve_coef`
  - MPVE value loss 权重。
  - 第一版建议 `1.0`，后续再调。
- `mpve_horizon`
  - 默认 `None`，表示使用 `ACMPC_T`。
- `mpve_bootstrap`
  - 是否使用 horizon 末端 value bootstrap。
  - 对应公式中的 `gamma^H V(s_H)`。
- `mpve_detach_predictions`
  - 是否对 predictions detach。
  - 第一版必须为 `true`。
- `mpve_valid_mask`
  - 是否对无效预测步 mask。
  - racing 任务建议为 `true`。

#### 推荐实现顺序

1. 只在 Python Gym 中实现预测 reward/obs 计算接口。
2. 扩展 `RolloutBuffer`，但先不改 PPO loss。
3. 在 smoke test 中验证 predictions 被正确保存到 buffer：
   - shape 正确。
   - 数值有限。
   - 与当前 rollout timestep 对齐。
4. 增加 `compute_mpve_targets()` 独立函数。
5. 为 `compute_mpve_targets()` 写单元测试：
   - `H=1` 时退化为 one-step bootstrap。
   - `gamma=1` 时 target 等于 rewards 累加加 bootstrap。
   - `valid_mask=0` 的项不进入 loss。
6. 在 `PPO.train()` 中接入 `mpve_value_loss`。
7. 在 hover/stabilization task 上验证无 NaN。
8. 再在 single-gate racing task 上启用。
9. 最后再迁移到 Flightmare。

#### 不建议第一版做的事情

- 不要让 MPVE value loss 反向更新 actor/cost map。
- 不要在基础 AC-MPC 还未稳定时启用 MPVE。
- 不要一开始就在 SplitS-like racing 上调 MPVE。
- 不要把 MPVE 和 sim-to-real domain randomization 同时引入。
- 不要在没有 valid mask 的情况下直接使用所有预测点。
- 不要让预测接口修改真实环境状态。

任务：

- 使用 `MlpMpcPolicy.predictions` 作为 MPC 预测序列来源。
- 扩展 rollout buffer，保存 MPC 预测状态、动作、观测、奖励和 valid mask。
- 在 Python Gym 环境中实现 prediction evaluation 接口。
- 对 MPC 预测计算 predicted rewards 和 predicted observations。
- 基于论文 H-step value expansion 和 TD k-trick 添加 MPVE value loss。
- 在 PPO 日志中记录 TD value loss、MPVE value loss 和 valid fraction。
- 在进入 racing 前，先用 hover/stabilization task 验证。

退出标准：

- MPVE 关闭时，基础 AC-MPC 训练行为不变。
- MPVE 可通过 config 启用。
- rollout buffer 中 prediction tensors shape 正确。
- predicted value loss 是有限值。
- `compute_mpve_targets()` 单元测试通过。
- hover task 训练无 NaN。
- single-gate racing task 启用 MPVE 后训练不崩溃。

### Phase 6: Flightmare 竞速环境

任务：

- 添加 `RacingEnv` C++ class。
- 添加 `racing_env.yaml`。
- 添加 pybind class `RacingEnv_v1`。
- 复用相同 track YAML schema。
- 使用 rate-thrust command mode。
- 可选添加 Unity gate 可视化。

退出标准：

- `RacingEnv_v1.reset()` 可从 Python 调用。
- `RacingEnv_v1.step()` 可用于向量化环境。
- 观测/动作维度与 Python Gym 版本一致。
- gate passing 和 collision 逻辑与 Python 测试一致。

### Phase 7: Flightmare 训练和推理

任务：

- 实现 `train_acmpc_flightmare.py`。
- 实现 `eval_acmpc_flightmare.py`。
- 保持 PPO 使用 SB3 AC-MPC fork，而不是 Flightmare TensorFlow PPO2。
- 在同一 track 上比较 Flightmare rollout 行为和 Python Gym rollout 行为。

退出标准：

- AC-MPC 可使用 `RacingEnv_v1` 训练。
- checkpoint 可加载并运行确定性推理。
- 指标和轨迹被记录。
- 可选 Unity 渲染显示四旋翼和 gates。

## 10. 验证计划

### 单元测试

必需测试：

- Dynamics：
  - hover stability sanity check
  - step 后状态有限
  - 四元数归一化
  - action saturation

- Gate geometry：
  - 通过 gate 中心穿越
  - 在 gate 边界外穿越
  - 与 gate 平面平行移动时不触发穿越
  - 当前 gate 正确递增

- Reward：
  - 正 progress
  - 负或零 progress
  - gate pass bonus
  - race finish bonus
  - body-rate penalty

- AC-MPC：
  - `IL_Env.mpc()` shape 和边界
  - `MlpMpcPolicy` 输出 shape
  - prediction tensor shape
  - 随机有效 batch 下无 NaN

### 集成测试

必需测试：

- 随机 rollout 一个完整 episode。
- AC-MPC forward 在环境 loop 中运行 100 steps。
- PPO 训练极少量 steps。
- 保存和加载 checkpoint。
- 加载后确定性评估。

### 性能检查

跟踪：

- `policy_forward_ms`
- `mpc_solve_ms`
- `env_step_ms`
- `training_update_ms`
- 使用 CUDA 时的 GPU memory
- CPU memory

初始目标：

- Python Gym 训练阶段可以慢于实时。
- 对于 `ACMPC_T=2`，应测量推理时间并与论文 20 ms 控制预算对比，但在 Flightmare 集成稳定前这不是硬性要求。

### Flightmare 专项检查

- C++ `RacingEnv` 可编译。
- pybind 模块可 import。
- `getObsDim()` 和 `getActDim()` 返回期望值。
- `VecEnv` 并行 step 可用。
- Unity bridge 保持可选。
- 无 Unity 的 headless training 可用。

## 11. 与论文相比的已知差异

这些差异是有意的，必须在报告中记录：

- AC-MPC 论文没有提供精确的 Horizontal、Vertical、Circle 和 SplitS gate 坐标。
- 本复现将使用受论文启发的 tracks，而不是作者完全一致的 tracks。
- Baselines 不在范围内。
- BEM 和 NeuroBEM 不在范围内。
- 真实世界部署不在范围内。
- Agilicious 集成不在范围内。
- 不声称复现精确论文奖励、图像、lap time 或 success rate。

有效声明是：

```text
The AC-MPC method was technically reproduced in simulation with a gate-racing task,
including neural cost-map training, differentiable MPC inference, PPO training,
checkpointed inference, and later Flightmare simulator integration.
```

中文含义：

```text
在 gate-racing 任务中完成了 AC-MPC 方法的仿真技术复现，
包括神经代价映射训练、可微 MPC 推理、PPO 训练、checkpoint 推理，
以及后续 Flightmare 仿真器集成。
```

## 12. 开发建议

- 从一开始保持 Python Gym 和 Flightmare 的任务 schema 对齐。
- 避免在代码中硬编码 tracks。使用 YAML。
- 将动作转换保留在单独模块中。
- 将观测构造保留在单独模块中。
- 分别记录 reward terms。
- 明确区分内部 MPC state 和外部仿真器 state。
- 将 `DroneDx` 视为控制器内部预测模型，而不一定是完整外部世界。
- 从 `ACMPC_T=2` 开始；只有完整 pipeline 稳定后，再移动到 `ACMPC_T=5`。
- 先实现 single-gate task，再实现 multi-gate tracks。
- 只有 Horizontal 和 Vertical 正常后，再添加 SplitS-like。
- Python Gym 训练和推理验证前，不迁移到 Flightmare。
- 基础 AC-MPC 稳定前，不添加 MPVE。
- nominal simulation 稳定前，不添加 sim-to-real randomization。

## 13. 交接检查清单

新开发者应能从这个清单开始：

1. 阅读本文档。
2. 阅读 `README.md`。
3. 阅读：
   - `diff_mpc_drones/drone.py`
   - `diff_mpc_drones/il_env.py`
   - `training_modules/mlp_mpc_policy.py`
4. 初始化子模块。
5. 运行 AC-MPC smoke tests。
6. 运行 Python Gym random rollout。
7. 运行短 AC-MPC training。
8. 运行确定性 evaluation。
9. 之后再开发 Flightmare `RacingEnv`。

每个阶段都应留下：

- config file
- test
- script
- short documentation note
- example output path

这样项目容易继续开发，也容易调试。
