# AC-MPC Python Gym 竞速环境设计与实现方案

本文档说明如何在 `D:/MyProjects/acmpc_public` 中实现一个用于 AC-MPC 主方法初步验证的 Python Gym 竞速环境。

该环境的定位是：

```text
AC-MPC RacingEnv 的 Python 原型实现
```

它不是 Flightmare 的复刻，也不是 BEM / NeuroBEM 高保真仿真器。它的目标是在接入 Flightmare 之前，先用纯 Python 环境验证 AC-MPC 的核心训练和推理闭环：

```text
obs(36) + state(13)
    -> MlpMpcPolicy
    -> MLP 输出 MPC cost map
    -> IL_Env.mpc() 求解控制动作
    -> Gym env.step(action)
    -> 外部四旋翼动力学、gate、reward、done
    -> PPO 更新神经网络
```

设计标准：

1. 第一标准：对外接口、观测、动作、reward 和训练流程优先贴合 AC-MPC 论文和当前代码仓库。
2. 第二标准：环境结构、动力学模块和接口命名尽量贴近 Flightmare，方便后续迁移到 C++ / pybind / Flightmare VecEnv。

## 1. 为什么需要这个 Python Gym 环境

AC-MPC 的核心仓库已经提供：

```text
diff_mpc_drones/drone.py
    MPC 内部可微四旋翼动力学 DroneDx

diff_mpc_drones/il_env.py
    调用 mpc.pytorch 的 MPC 求解封装

training_modules/mlp_mpc_policy.py
    神经网络 + 可微 MPC 的 AC-MPC policy

stable-baselines3/
    作者 fork 的 SB3，用于 PPO 训练
```

但当前仓库没有完整的 racing Gym 环境。要训练 AC-MPC，需要一个外部环境负责：

```text
reset 初始状态
step 执行动作
构造论文 36 维 observation
提供 MPC 所需 13 维 state
推进外部四旋翼动力学
检测 gate passing
检测 collision / out of bounds
计算 reward
给 PPO 返回 done 和 info
```

因此需要新增一个环境包，而不是只写临时脚本。

## 2. 总体数据流

完整数据流如下：

```text
AcMpcRacingEnv.reset()
    -> obs(36)
    -> state(13)

PPO collect_rollouts
    -> policy(obs, state)

MlpMpcPolicy
    -> MLP(obs) 输出 MPC cost map Q, p
    -> 从 state[:10] 取 [p, q, v] 作为 MPC 初始状态
    -> IL_Env.mpc(dx, xinit, Q, p)
    -> 输出归一化 action(4)

AcMpcRacingEnv.step(action)
    -> 反归一化 action
    -> 得到 collective thrust + body-rate command
    -> Flightmare-like 外部四旋翼动力学推进状态
    -> gate / collision / bounds 检测
    -> reward / done / info
    -> next obs(36), next state(13)
```

需要严格区分：

```text
obs(36):
    给神经网络 MLP 看，用于输出 cost map。

state(13):
    给 MPC 和外部动力学用，是物理状态。
```

不要把二者混成一个向量。这样后续做观测噪声、状态估计、sim-to-real 或 Flightmare 迁移时接口更清晰。

## 3. 推荐代码结构

新增包：

```text
acmpc_racing_gym/
  __init__.py
  config.py

  dynamics/
    __init__.py
    params.py
    state.py
    integrator.py
    flightmare_like_dynamics.py

  tracks/
    __init__.py
    gate.py
    track.py
    loader.py
    assets/
      split_s_like.json
      vertical.json
      horizontal.json

  observations/
    __init__.py
    acmpc_observation.py

  rewards/
    __init__.py
    racing_reward.py

  envs/
    __init__.py
    racing_env.py

  wrappers/
    __init__.py
    state_vec_env.py
    sb3_make_env.py

tests/
  test_gate_geometry.py
  test_observation_shape.py
  test_dynamics_step.py
  test_racing_env_random_rollout.py
  test_sb3_state_plumbing.py

scripts/
  random_rollout_racing_env.py
  train_acmpc_racing_smoke.py
```

核心逻辑放在 `acmpc_racing_gym/` 中。`scripts/` 只作为运行入口，不承载环境实现。

## 4. Observation 设计：严格 36 维

AC-MPC 论文的 racing observation 分成两部分：

```text
o = [o_quad, o_track]
```

其中：

```text
o_quad = [v, R]
v: 世界系线速度，3 维
R: 姿态旋转矩阵，9 维
o_quad 总计 12 维
```

track observation 使用未来两个 gate：

```text
G = 2
每个 gate 用 4 个角点表示
每个角点是 3 维
每个 gate = 4 * 3 = 12 维
两个 gate = 24 维
```

所以：

```text
obs_dim = 12 + 24 = 36
```

### 4.1 基础排列

建议统一排列为：

```text
obs[0:3]    = linear velocity v
obs[3:12]   = rotation matrix R.flatten(row-major)
obs[12:24]  = gate_1 observation
obs[24:36]  = gate_2 observation
```

输出类型：

```text
np.ndarray
shape = (36,)
dtype = np.float32
```

Gym space：

```python
spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(36,),
    dtype=np.float32,
)
```

### 4.2 两种 track observation 模式

论文对第二个 gate 的表达有两种可能解释。因此环境需要同时支持两种模式，通过配置标志切换。

配置项：

```python
track_obs_mode: str = "vehicle_relative"
```

支持值：

```text
vehicle_relative
chained_gate_relative
```

#### 模式 1：vehicle_relative

两个未来 gate 的四个角点都相对无人机当前位置：

```text
obs[12:24] = gate_1_corners - drone_position
obs[24:36] = gate_2_corners - drone_position
```

这个模式最直观，debug 简单，建议作为第一版默认值。

#### 模式 2：chained_gate_relative

第一个 gate 相对无人机，第二个 gate 相对第一个 gate：

```text
obs[12:24] = gate_1_corners - drone_position
obs[24:36] = gate_2_corners - gate_1_corners
```

该模式用于兼容论文里可能存在的“consecutive gates corner difference”解释。

### 4.3 Observation builder 接口

实现位置：

```text
acmpc_racing_gym/observations/acmpc_observation.py
```

建议接口：

```python
class TrackObservationMode:
    VEHICLE_RELATIVE = "vehicle_relative"
    CHAINED_GATE_RELATIVE = "chained_gate_relative"


@dataclass
class ObservationConfig:
    future_gate_count: int = 2
    track_obs_mode: str = TrackObservationMode.VEHICLE_RELATIVE
    normalize: bool = False


class AcMpcObservationBuilder:
    @property
    def obs_dim(self) -> int:
        return 36

    def build(self, drone_state, track_state) -> np.ndarray:
        ...
```

要求：

```text
1. 两种 track_obs_mode 都必须返回 shape=(36,)。
2. observation builder 不修改环境状态。
3. normalization 第一版可以关闭，但接口保留。
4. 后续如果确认论文只用一种模式，只改配置或 builder，不改环境主体。
```

## 5. State 设计：单独 13 维

当前 `MlpMpcPolicy.forward_actor()` 不是只接收 observation，它还需要 `states`：

```python
forward_actor(features, states)
```

内部会取：

```python
states = states[:, 0:10]
```

这 10 维是：

```text
[p, q, v]
3 + 4 + 3 = 10
```

但外部环境应维护完整 13 维：

```text
state = [p, q, v, omega]

p:     位置，3 维
q:     四元数，4 维，顺序 [qw, qx, qy, qz]
v:     世界系线速度，3 维
omega: 机体系角速度，3 维

total = 13
```

Gym env 需要提供：

```python
def get_state(self) -> np.ndarray:
    return state13
```

state space：

```python
spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(13,),
    dtype=np.float32,
)
```

为什么保留 `omega`：

```text
1. 外部动力学需要当前角速度。
2. Flightmare 底层状态包含 omega。
3. 当前 stable-baselines3 fork 的 RolloutBuffer 已经按 13 维 state 设计。
4. 未来迁移到 Flightmare 或真实系统时，[p, q, v, omega] 是自然状态接口。
```

## 6. Action 设计

Gym action space：

```python
spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(4,),
    dtype=np.float32,
)
```

AC-MPC 输出归一化 action：

```text
action = [
    thrust_normalized,
    wx_normalized,
    wy_normalized,
    wz_normalized,
]
```

其物理含义：

```text
thrust_normalized:
    归一化后的 mass-normalized collective thrust

wx, wy, wz:
    归一化后的 body-rate command
```

反归一化必须镜像 `training_modules/mlp_mpc_policy.py`：

```python
normalization_max = 8.5
force_mean = (normalization_max * 4 / mass) / 2.0
force_std = (normalization_max * 4 / mass) / 2.0

c = action[0] * force_std + force_mean
collective_thrust_N = mass * c

omega_cmd = action[1:4] * np.array([10.0, 10.0, 4.0])
```

其中：

```text
c:
    mass-normalized thrust，单位 m/s^2

collective_thrust_N:
    总推力，单位 N

omega_cmd:
    期望机体角速度，单位 rad/s
```

为了贴近 Flightmare 的 `Command(t, collective_thrust, omega)`，内部可以同时保存：

```text
mass_normalized_collective_thrust = c
collective_thrust_N = mass * c
body_rate_cmd = omega_cmd
```

## 7. 外部四旋翼动力学设计

外部动力学不能只是一个过于简单的 `[p, q, v]` Euler 更新。应参考 Flightmare 代码实现较真实的刚体四旋翼模型。

参考代码：

```text
D:/MyProjects/flightmare/flightlib/src/objects/quadrotor.cpp
D:/MyProjects/flightmare/flightlib/src/dynamics/quadrotor_dynamics.cpp
D:/MyProjects/flightmare/flightlib/include/flightlib/common/quad_state.hpp
D:/MyProjects/flightmare/flightlib/configs/quadrotor_env.yaml
```

### 7.1 动力学状态

外部动力学内部状态：

```text
p:           位置
q:           姿态四元数
v:           世界系线速度
omega:       机体系角速度
motor_omega: 四个电机转速
```

对外 `get_state()` 返回：

```text
[p, q, v, omega]
```

`motor_omega` 是动力学内部变量，不作为 policy state 暴露。

### 7.2 参数

实现位置：

```text
acmpc_racing_gym/dynamics/params.py
```

建议参数：

```python
@dataclass
class QuadrotorParams:
    mass: float = 0.752
    gravity: float = 9.8066
    dt: float = 0.02
    substep_dt: float = 0.0025

    arm_l: float = 0.17
    inertia_diag: tuple[float, float, float] = (0.0025, 0.0021, 0.0043)
    kappa: float = 0.016

    thrust_min_per_motor: float = 0.0
    thrust_max_per_motor: float = 8.5

    omega_cmd_max: tuple[float, float, float] = (10.0, 10.0, 4.0)
    rate_gain: tuple[float, float, float] = (16.6, 16.6, 5.0)

    motor_tau: float = 0.02
    motor_omega_min: float = 150.0
    motor_omega_max: float = 3000.0
    thrust_map: tuple[float, float, float] = (
        1.3298253500372892e-06,
        0.0038360810526746033,
        -1.7689986848125325,
    )

    linear_drag: tuple[float, float, float] = (0.05, 0.05, 0.08)
```

说明：

```text
mass、thrust_max_per_motor、omega_cmd_max 优先贴合 AC-MPC DroneDx。
arm_l、rate_gain、motor model、RK4 子步长参考 Flightmare。
linear_drag 默认较小，但保留，避免环境过于理想化。
```

### 7.3 控制输入流程

每一步 `step(action)`：

```text
normalized action
    -> mass-normalized collective thrust c
    -> body-rate command omega_cmd
    -> rate controller
    -> desired body torque tau_des
    -> allocation inverse
    -> desired motor thrusts
    -> motor first-order lag
    -> actual motor thrusts
    -> actual total thrust and body torque
    -> rigid-body dynamics
    -> RK4 integration
```

### 7.4 Body-rate controller

Flightmare 的 `runFlightCtl()` 思路是：

```text
force = mass * collective_thrust
omega_error = omega_cmd - omega
tau_des = J * K_rate * omega_error + omega x (J * omega)
```

这里：

```text
J:
    惯性矩阵

K_rate:
    body-rate controller gain

omega x (J * omega):
    刚体旋转陀螺项补偿
```

得到：

```text
wrench_des = [force, tau_x, tau_y, tau_z]
```

### 7.5 Allocation matrix

将四个电机推力映射到总推力和力矩：

```text
[F, tau_x, tau_y, tau_z] = B @ [f1, f2, f3, f4]
```

`B` 参考 Flightmare：

```text
第一行：四个电机推力求和
第二、三行：由 arm length 产生 roll / pitch 力矩
第四行：由 kappa 产生 yaw 力矩
```

控制器中需要：

```text
motor_thrust_des = B_inv @ wrench_des
```

并进行 per-motor thrust clamp：

```text
f_i in [thrust_min_per_motor, thrust_max_per_motor]
```

### 7.6 电机模型

参考 Flightmare 的电机一阶响应：

```text
motor_omega_des = thrust_to_motor_omega(motor_thrust_des)
motor_omega = c * motor_omega + (1 - c) * motor_omega_des
c = exp(-dt / motor_tau)
motor_thrust_actual = motor_omega_to_thrust(motor_omega)
```

推力映射：

```text
thrust = a * motor_omega^2 + b * motor_omega + c
```

第一版要注意：

```text
1. thrust_map 可能产生负推力，需要 clamp。
2. motor_omega 和 thrust 都必须 finite。
3. motor_tau 不要设得过小，否则近似无电机动态。
```

### 7.7 刚体动力学

动力学方程：

```text
p_dot = v
q_dot = 0.5 * q ⊗ [0, omega]
v_dot = R(q) @ [0, 0, F_actual] / mass + gravity + drag
omega_dot = J^-1 * (tau_actual - omega × J omega)
```

drag：

```text
drag_acc = -linear_drag * v / mass
```

积分：

```text
控制步长 dt = 0.02
内部积分 substep_dt = 0.0025
每个控制步使用 RK4 多子步积分
每个子步后归一化 quaternion
```

RK4 接口：

```text
acmpc_racing_gym/dynamics/integrator.py

rk4_step(f, state, dt)
```

### 7.8 与 DroneDx 的关系

`DroneDx` 是 AC-MPC 内部模型：

```text
state = [p, q, v]
control = [collective thrust, body rates]
必须可微，必须快，服务 MPC 求解
```

Python Gym 外部动力学是环境模型：

```text
state = [p, q, v, omega] + motor_omega
包含 rate controller、电机响应、drag、RK4
不需要可微，服务训练反馈
```

两者不必完全一致。这个差异正是训练时有价值的地方：policy 内部用简化 MPC 预测，外部环境给出更真实执行结果。

## 8. Gate 几何设计

实现位置：

```text
acmpc_racing_gym/tracks/gate.py
```

`Gate` 数据结构：

```python
@dataclass
class Gate:
    center: np.ndarray
    normal: np.ndarray
    up: np.ndarray
    width: float
    height: float
    frame_thickness: float
```

派生量：

```text
right = normalize(cross(up, normal))
up = normalize(up)
normal = normalize(normal)
```

需要提供：

```python
corners_world() -> np.ndarray shape=(4, 3)
world_to_gate(point) -> np.ndarray shape=(3,)
segment_plane_intersection(p_prev, p_curr)
check_pass(p_prev, p_curr, direction_required=True)
check_frame_collision(p_prev, p_curr, drone_radius)
```

### 8.1 过门判定

基础逻辑：

```text
1. 取上一帧位置 p_prev 和当前帧位置 p_curr。
2. 判断线段是否穿过 gate 平面。
3. 求交点。
4. 把交点转到 gate 局部坐标。
5. 如果交点在 opening 内，并且方向正确，则 gate passed。
```

opening 内部条件：

```text
abs(local_x) <= width / 2
abs(local_y) <= height / 2
```

方向条件建议默认开启：

```text
dot(p_prev - center, normal) < 0
dot(p_curr - center, normal) >= 0
```

### 8.2 Gate frame collision

碰撞判定不需要一开始做到精确 mesh collision，但不能没有。

第一版建议：

```text
如果线段穿过 gate 平面：
    交点不在 opening 内
    但落在 gate 外框包围盒附近
    则 collision
```

带 frame thickness：

```text
outer_width = width + 2 * frame_thickness
outer_height = height + 2 * frame_thickness

inside_outer = abs(local_x) <= outer_width / 2 and abs(local_y) <= outer_height / 2
inside_inner = abs(local_x) <= width / 2 and abs(local_y) <= height / 2

frame_collision = inside_outer and not inside_inner
```

同时考虑无人机半径 `drone_radius`：

```text
opening 可按 drone_radius 收缩
outer frame 可按 drone_radius 扩张
```

## 9. Track 设计

实现位置：

```text
acmpc_racing_gym/tracks/track.py
acmpc_racing_gym/tracks/loader.py
```

`Track` 职责：

```python
current_gate()
future_gates(count=2)
advance_gate()
is_finished()
reset()
```

未来 gate 不足两个时：

```text
如果 race 未完成，可以重复最后一个 gate 或返回 finish ghost gate。
第一版建议重复最后一个 gate，保持 observation shape 固定。
```

track 文件建议 JSON：

```json
{
  "name": "split_s_like",
  "gates": [
    {
      "center": [0.0, 0.0, 2.0],
      "normal": [1.0, 0.0, 0.0],
      "up": [0.0, 0.0, 1.0],
      "width": 1.5,
      "height": 1.5,
      "frame_thickness": 0.12
    }
  ],
  "start": {
    "position": [-2.0, 0.0, 2.0],
    "yaw": 0.0
  }
}
```

第一阶段至少提供：

```text
horizontal.json
vertical.json
split_s_like.json
```

注意：`split_s_like` 是受论文 SplitS 启发的技术验证赛道，不承诺完全等同论文作者赛道参数。

## 10. Reward 设计

实现位置：

```text
acmpc_racing_gym/rewards/racing_reward.py
```

默认 reward 按 AC-MPC 论文：

```text
if collision:
    reward = -10.0
elif gate_passed:
    reward = +10.0
elif race_finished:
    reward = +10.0
else:
    reward = distance(prev_position, target_gate_center)
             - distance(curr_position, target_gate_center)
             - 0.01 * norm(body_rate)
```

说明：

```text
progress_reward 为正，表示更接近当前目标 gate。
body_rate penalty 抑制过激旋转。
gate_passed 后 target gate 切换到下一个。
race_finished 是所有 gates 通过后的终止奖励。
```

可以预留但默认关闭：

```text
action_smooth_penalty
action_magnitude_penalty
control_jitter_penalty
energy_penalty
```

原因：论文 sim-to-real 部分提到需要平滑控制项，但当前阶段不是 sim-to-real。为了保持主实验逻辑清晰，第一版不要默认加入额外 reward。

## 11. Done 逻辑

`done=True` 条件：

```text
1. race_finished
2. gate frame collision
3. ground collision
4. out of bounds
5. state contains NaN / Inf
6. max episode steps reached
```

建议 `info` 中写明终止原因：

```python
info = {
    "gate_index": int,
    "gate_passed": bool,
    "collision": bool,
    "collision_type": str | None,
    "finished": bool,
    "out_of_bounds": bool,
    "timeout": bool,
    "position": np.ndarray,
    "velocity": np.ndarray,
    "omega": np.ndarray,
    "physical_command": {
        "mass_normalized_thrust": float,
        "collective_thrust_N": float,
        "body_rate_cmd": np.ndarray,
    },
}
```

## 12. RacingEnv 主接口

实现位置：

```text
acmpc_racing_gym/envs/racing_env.py
```

接口：

```python
class AcMpcRacingEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, config: RacingEnvConfig):
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Box(...)
        self.state_space = spaces.Box(...)

    def reset(self):
        ...
        return obs

    def step(self, action):
        ...
        return obs, reward, done, info

    def get_state(self):
        return self.dynamics.get_state13()

    def render(self, mode="human"):
        ...
```

`step()` 顺序：

```text
1. 保存 prev_position。
2. clip action 到 [-1, 1]。
3. 反归一化 action。
4. 调用 dynamics.step(command)。
5. 得到 curr_position。
6. 检查 current gate passing / collision。
7. 如果 gate passed，track.advance_gate()。
8. 检查 ground / bounds / finite / timeout。
9. 计算 reward。
10. 构造 next obs。
11. 返回 obs, reward, done, info。
```

## 13. SB3 state plumbing 必须补齐

这是训练前必须解决的问题。

当前 fork 里存在半完成状态：

```text
MlpMpcPolicy.forward(obs, states) 需要 states。
RolloutBuffer 有 states 字段。
BasePolicy.predict() 已经增加 drone_state 参数。
```

但 PPO 主流程还没完全接上：

```text
on_policy_algorithm.py collect_rollouts 仍只调用 policy(obs_tensor)
ppo.py train 中 evaluate_actions 仍没有传 states
RolloutBuffer 初始化没有传 state_space
BaseBuffer 里 state_dim 目前硬编码为 13
```

需要修改：

### 13.1 VecEnv wrapper 暴露 state

实现位置：

```text
acmpc_racing_gym/wrappers/state_vec_env.py
```

要求：

```python
def get_state(self) -> np.ndarray:
    """
    Return shape = (n_envs, 13)
    """
```

对于 `DummyVecEnv`，可以通过：

```text
env.env_method("get_state")
```

取每个子环境 state，再 stack。

### 13.2 OnPolicyAlgorithm._setup_model

创建 RolloutBuffer 时传入：

```python
state_space=self.env.state_space
```

如果某些普通环境没有 `state_space`，可以 fallback：

```python
spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)
```

但 AC-MPC 环境必须显式提供 `state_space`。

### 13.3 collect_rollouts

旧逻辑：

```python
obs_tensor = obs_as_tensor(self._last_obs, self.device)
actions, values, log_probs = self.policy(obs_tensor)
```

新逻辑：

```python
state_np = env.get_state()
state_tensor = obs_as_tensor(state_np, self.device)
obs_tensor = obs_as_tensor(self._last_obs, self.device)
actions, values, log_probs = self.policy(obs_tensor, state_tensor)
```

buffer add：

```python
rollout_buffer.add(
    self._last_obs,
    actions,
    state_np,
    rewards,
    self._last_episode_starts,
    values,
    log_probs,
)
```

### 13.4 PPO.train

旧逻辑：

```python
values, log_prob, entropy = self.policy.evaluate_actions(
    rollout_data.observations,
    actions,
)
```

新逻辑：

```python
values, log_prob, entropy = self.policy.evaluate_actions(
    rollout_data.observations,
    actions,
    rollout_data.states,
)
```

### 13.5 predict_values

当前 `predict_values(obs)` 不接收 state，而 value network 当前只依赖 observation features，不依赖 MPC state。第一版可以保持不变。

如果后续 critic 也需要 state，再扩展：

```python
predict_values(obs, states=None)
```

### 13.6 RolloutBuffer state_dim

当前 `BaseBuffer` 里 `state_dim = 13` 是硬编码。建议改为从 `state_space` 推导：

```python
self.state_shape = get_obs_shape(state_space)
self.state_dim = int(np.prod(self.state_shape))
```

对于当前任务仍是 13，但代码不再写死。

## 14. 与 Flightmare 后续迁移的接口对应

Python Gym 设计应当为 Flightmare 迁移保留一一对应关系：

| Python Gym | Flightmare 迁移目标 |
|---|---|
| `AcMpcRacingEnv.reset()` | C++ `RacingEnv::reset()` |
| `AcMpcRacingEnv.step(action)` | C++ `RacingEnv::step()` |
| `get_obs()` | C++ `getObs()` |
| `get_state()` | C++ `getState()` 或 pybind 暴露 |
| `Gate` | `StaticGate` + racing gate geometry |
| `Track` | C++ track manager |
| `FlightmareLikeDynamics` | Flightmare `Quadrotor` / `QuadrotorDynamics` |
| `StateVecEnv` | Flightmare `VecEnv<RacingEnv>` |
| `track_obs_mode` | YAML config 参数 |

迁移原则：

```text
1. 先保证 Python 版接口稳定。
2. C++ 版 Flightmare RacingEnv 不重新发明 observation/reward/done 语义。
3. 保持 obs(36)、state(13)、action(4) 完全一致。
4. Python 版测试用例可以作为 C++ 迁移验收标准。
```

## 15. 验证计划

按以下顺序验证，不能直接跳到长训练。

### 15.1 Gate geometry 单元测试

测试：

```text
1. 正向穿过 gate 中心，passed=True。
2. 反向穿过 gate，direction_required=True 时 passed=False。
3. 穿过 gate 平面但落在 opening 外，collision=True。
4. 没有穿过 gate 平面，passed=False 且 collision=False。
5. drone_radius 收缩 opening 后，边界穿越会被判为 collision。
```

### 15.2 Observation shape 测试

测试：

```text
1. vehicle_relative 模式返回 shape=(36,)。
2. chained_gate_relative 模式返回 shape=(36,)。
3. obs 全部 finite。
4. v 和 R 部分排列正确。
5. gate corner 部分和手算结果一致。
```

### 15.3 Dynamics step 测试

测试：

```text
1. hover action 附近状态不发散。
2. quaternion norm 保持接近 1。
3. action clip 后物理命令满足边界。
4. motor thrust 满足 per-motor bounds。
5. state 全部 finite。
```

### 15.4 Random rollout 测试

测试：

```text
1. env.reset() 返回 obs shape=(36,)。
2. env.get_state() 返回 state shape=(13,)。
3. 随机 action 可以运行直到 done 或 max_episode_steps。
4. 每一步 obs/reward/state 都 finite。
5. info 包含 gate_index、collision、finished 等字段。
```

### 15.5 AC-MPC forward 集成测试

测试：

```text
1. 用 env.reset() 的 obs 和 get_state() 的 state 调用 MlpMpcPolicy。
2. policy 输出 action shape=(4,)。
3. action 在 [-1, 1] 内。
4. env.step(action) 可执行。
5. policy.mlp_extractor.predictions shape=(batch, T, 14)。
```

### 15.6 PPO state plumbing 测试

测试：

```text
1. collect_rollouts 可以填满 rollout buffer。
2. rollout_buffer.states shape 正确。
3. PPO.train 调用 evaluate_actions 时传入 states。
4. 一次短训练不出现 NaN。
5. TensorBoard 或日志中 reward 有合理数值。
```

### 15.7 短训练验收

第一阶段只要求：

```text
1. total_timesteps 较小的 smoke training 能跑完。
2. reward 不为 NaN。
3. episode 能结束。
4. 部分 rollout 出现 gate_passed。
```

不要求达到论文图表数值。

## 16. 推荐实现顺序

```text
Step 1:
    实现 config、state、quaternion/rotation 工具。

Step 2:
    实现 Gate、Track、track loader 和两种 track_obs_mode。

Step 3:
    实现 ObservationBuilder，完成 36 维观测单测。

Step 4:
    实现 FlightmareLikeDynamics，包括 rate controller、allocation、电机模型、RK4。

Step 5:
    实现 RacingReward 和 done 逻辑。

Step 6:
    实现 AcMpcRacingEnv，并完成 random rollout。

Step 7:
    修复 SB3 state plumbing。

Step 8:
    跑 MlpMpcPolicy + env 单步 forward。

Step 9:
    跑 PPO rollout buffer smoke test。

Step 10:
    跑短训练，观察 reward、gate_passed、collision。
```

## 17. 第一阶段不做的内容

明确暂不实现：

```text
BEM / NeuroBEM
真实 Unity 渲染
Agilicious 部署
真实 sim-to-real domain randomization
论文完整图表复现
baseline AC-MLP / tracking MPC / L1-MPC
视觉输入
多机并行 C++ Flightmare VecEnv
```

但接口要预留：

```text
action latency
drag
mass randomization
thrust map randomization
initial state randomization
observation noise
state estimator
Flightmare backend adapter
```

## 18. 最终退出标准

Python Gym 第一阶段完成标准：

```text
1. 随机动作可以运行完整 episode。
2. 单元测试能正确检测 gate passing。
3. 碰撞和越界终止可用。
4. observation_space 为 Box shape=(36,)。
5. action_space 为 Box shape=(4,), range=[-1, 1]。
6. get_state() 返回 shape=(13,)。
7. MlpMpcPolicy 能用 env obs 和 state 完成 forward。
8. MPC action 可被 env.step() 执行。
9. PPO collect_rollouts 能保存 states。
10. 短训练能跑完，reward 和 loss 不出现 NaN。
```

完成这些后，该 Python Gym 环境就可以可靠承担“替代 Flightmare 做 AC-MPC 主方法初步算法验证”的角色。


