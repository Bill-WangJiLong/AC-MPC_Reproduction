# AC-MPC 仿真复现完整设计、实现与变更记录

## 1. 文档范围

本文忠实记录截至 2026-06-30，本项目在作者原始代码基础上完成的修改和新增实现，覆盖：

1. `diff_mpc_drones`；
2. `mpc.pytorch`；
3. 作者 fork 的 `stable-baselines3`；
4. `training_modules`；
5. 新增 Python Gym 竞速环境及其训练、MPVE、评估和绘图工具；
6. 修改版 Flightmare C++ `RacingEnv`、pybind、Python SB3 adapter、训练和评估工具；
7. 当前实现与论文、内部 MPC 模型和外部仿真器之间已知的不一致与限制。

用户所写的 `diff_mpc_drone` 实际目录名为 `diff_mpc_drones`。本文只把 Git diff 能确认的内容称为“对作者代码的修改”；原仓库已有但未被我们修改的内容会明确标注。

## 2. 审计基线

| 代码区域 | 对比基线提交 | 当前差异 |
|---|---|---|
| AC-MPC 主仓库 | `c59e53aec11c1fffa8b69d99b0ee7879ba7ccb28` | 修改 2 个原文件，新增 Gym、Flightmare adapter、脚本和文档 |
| `mpc.pytorch` | `63732fa85ab2a151045493c4e67653210ca3d7ff` | 修改 `mpc/mpc.py` 1 个文件 |
| 作者 SB3 fork | `152c353863d3b05fb5feed4deb37b952bb4beb7b` | 修改 4 个文件 |
| Flightmare | `d4218aedac18cbe9364a0a0df10ab992c4b65e4f` | 修改 14 个原文件，新增 RacingEnv、配置和赛道 |

Flightmare 的 `.pyd` 和 `racing_env.yaml.pre_track_install.bak` 是生成产物，不属于源码并已从远程仓库排除。

## 3. 当前总体架构

```text
36维原始观测 -> VecNormalize -> neural cost map -> Q,p --+
                                                          v
13维外部状态 p,q,v,omega -> 取前10维 p,q,v ----------> diffMPC
                                                          |
                                               第一拍4维控制均值
                                                          |
                                      PPO Gaussian exploration（训练）
                                                          |
                                       [-1,1] normalized CTBR action
                                                          |
                                      Gym 或 Flightmare 外部动力学
```

| 数据 | 维度 | 顺序 | 默认归一化 |
|---|---:|---|---|
| observation | 36 | `v(3), R(9), gate1 corners(12), gate2 corners(12)` | 是，仅 VecNormalize |
| state | 13 | `p(3), q(wxyz)(4), v(3), omega(3)` | 否 |
| action | 4 | `normalized collective thrust, normalized wx, wy, wz` | 本身定义在归一化空间 |

内部 MPC 使用 10 维状态 `p,q,v` 和 4 维控制 `collective thrust, body rates`。外部环境需要保存当前角速度，因此额外 state 是 13 维；MPC 将 body rate 当作控制输入，不把它放入内部状态。

## 4. `diff_mpc_drones` 的修改

### 4.1 `drone.py`：作者原有、未修改

该文件实现可微 CTBR 模型、离散 forward 和解析 Jacobian。当前关键参数为质量 `0.752 kg`、`dt=0.02 s`、惯量 `[0.0025,0.0021,0.0043]`、body-rate 限制 `[10,10,4]`、单电机推力 `[0,8.5] N`。当前相对主仓库基线没有代码修改；`.T` warning 来自作者原实现。

### 4.2 `il_env.py`

把：

```text
mpc(..., lqr_iter_override=None)
```

扩展为：

```text
mpc(..., lqr_iter_override=None, backprop=True)
```

并把 `backprop` 传给 `mpc.MPC`。PPO rollout/inference 可关闭可微反向图，PPO update 仍保持 diffMPC backward，目的在于减少无梯度阶段的 CUDA graph 和显存占用。

当前必须注意：传入的 `u_init` 会在 `IL_Env.mpc()` 内被重新赋值为悬停序列。因此 policy 虽传入 `u_prev_chunk`，当前没有真正使用跨调用 warm start。

## 5. `mpc.pytorch` 的修改

仅修改 `mpc.pytorch/mpc/mpc.py`。作者 `MPC.__init__()` 原本已有 `backprop` 参数，但原始 forward 无论该值为何都会继续做动力学线性化、代价近似和用于隐式反向的 LQR 包装。

本项目在 iLQR forward 得到最佳 `x/u/costs` 后增加：

```python
if not self.backprop:
    return detach(x), detach(u), detach(costs)
```

效果：

- `backprop=False`：返回 detach 的 forward 最优轨迹，不构造反向路径；
- `backprop=True`：继续作者原 differentiable MPC 路径；
- AC-MPC 的可微性没有被全局关闭，仅关闭无梯度阶段的冗余图。

## 6. 作者 SB3 fork 的修改

### 6.1 作者 fork 原有但未接通的 state 结构

基线 fork 已有 `ActorCriticPolicy.forward(obs,states)`、`evaluate_actions(...,states)`、rollout buffer 的 state 数组和 sample 字段。但训练主流程仍调用 `policy(obs)`、buffer add 不传 state、PPO update 不传 state，且 reset 强制假定返回 `(obs,state)`，因此标准 VecEnv 无法训练。

### 6.2 `common/base_class.py`

`_setup_learn()` 现在同时兼容：

- 旧入口 `reset() -> (obs,state)`；
- 标准入口 `reset() -> obs`，随后调用 `env.get_state()`；
- 没有直接方法时退回 `env.env_method("get_state")`。

### 6.3 `common/buffers.py`

把硬编码 `state_dim=13` 改为从 `state_space` shape 计算维数，并拒绝 Dict state。当前结果仍是 13，但合同不再写死在 buffer 内。

### 6.4 `common/on_policy_algorithm.py`

- 从环境读取 `state_space`，缺失时退回 13 维 Box；
- 构造 rollout buffer 时传入 state space；
- 每个 rollout step 在执行 action 前取得 state；
- 调用 `policy(obs_tensor,state_tensor)`；
- 把对齐的 state 写入 buffer；
- 新增 `_get_env_state()` 统一直接方法和 `env_method()`。

### 6.5 `ppo/ppo.py`

PPO minibatch 的 `evaluate_actions(observations,actions)` 改为 `evaluate_actions(observations,actions,states)`，使重新计算 log probability 时 MPC 使用与采样时一致的物理初态。

### 6.6 完整 state 数据流

```text
env.get_state()
-> policy(obs,state)
-> rollout_buffer.add(obs,action,state,reward,...)
-> rollout_data.states
-> evaluate_actions(obs,action,state)
-> PPO clipped loss
-> diffMPC backward
-> cost-map network
```

Critic 仍只接收 observation；13 维 side-channel 专供 MPC actor 初态。

## 7. `training_modules` 的修改

### 7.1 `mlp_mpc_policy.py`

实际修改：

1. 把 `<YOUR_ACMPC_FOLDER>` 占位路径改为基于 `__file__` 的 `diff_mpc_drones` 相对路径；
2. 调用 MPC 时传 `backprop=torch.is_grad_enabled()`；
3. 保存 `u_prev` 时增加 `.detach()`，避免成员变量保留旧 autograd graph；
4. 增加 36 维 observation、28T cost-map、actor/critic 和 13->10 state 切片的注释。

`u_prev` detach 能消除图引用，但由于 `IL_Env` 覆盖 `u_init`，当前尚未形成真正 warm start。

### 7.2 当前 policy 实际结构

以下主要是作者原代码行为：

- 输入当前为 36；
- cost-map 是三层 512 GELU 隐层和 `28*T` Sigmoid；
- critic feature 网络是两层 512 GELU，SB3 再接标量 head；
- 每步输出 Q 的 14 个对角量和 p 的 14 个线性量；
- Q 映射到约 `[0.1,100000.1]`，p 大部分映射到约 `[-50000,50000]`；
- MPC batch 按 1024 分块，每次 `lqr_iter_override=1`；
- 13 维 state 只取前 10 维；
- `self.predictions` 保存 detach 后的 `[batch,T,14]=[x10,u4]`；
- MPC 第一拍经推力/body-rate 归一化后成为 Gaussian mean；
- `distr_identity=True`，MPC 后没有附加 action MLP；PPO `log_std` 控制探索。

论文文字描述两层 512 ReLU cost-map，而公开代码实际为三层 512 GELU。本复现保留公开代码结构，没有擅自改写。

### 7.3 `mlp_only_policy.py`

作者 AC-MLP baseline，当前相对基线无修改，本复现主实验不使用。

## 8. Python Gym 环境总体设计

`acmpc_racing_gym` 是本项目新增的算法原型环境，用于先验证 AC-MPC/PPO/MPVE 全链路，再迁移 Flightmare。它不是 Flightmare 的逐行 Python 重写，也不包含 NeuroBEM。

设计优先级：论文 observation/action/reward；作者 policy 的 36/4/13 接口；接近 Flightmare 的刚体、CTBR、motor lag 和赛道几何；Gym 0.21/SB3 兼容；与 Flightmare 复用赛道和 info 合同。

### 8.1 Observation

```text
linear velocity                       3
rotation matrix row-major             9
next gate four corner differences    12
second gate four corner differences  12
total                                36
```

第二门支持：

- `vehicle_relative`：两个门角点都减无人机位置，当前默认；
- `chained_gate_relative`：第一门减无人机位置，第二门角点减第一门对应角点。

环境始终输出原始物理量；`VecNormalize` 在线维护统计并向 policy 提供归一化 observation。13 维 state 不归一化。

### 8.2 Action

动作空间是 `Box(-1,1,(4,))`。第一维反归一化为 mass-normalized collective thrust，再乘质量得到 N；后三维乘 `[10,10,4]` 得 body-rate command。外部动力学经 rate controller、allocation、motor lag 和刚体积分推进。

### 8.3 Reward

```text
collision                         -10
gate passed                       +10
race finished                     +10
same step last gate + finish      +20
otherwise                         distance progress - 0.01*||omega||
```

碰撞优先级最高。连续项实际计算两个欧氏距离之差，不直接用速度近似。

### 8.4 Done

当前目标门框碰撞、触地、越界、非有限 state、通过全部门后线段与终点球相交、达到最大步数都会结束 episode。终点使用线段-球相交，避免高速一步穿过漏判；已经修复末门与终点同一步触发，并累计 `+20`。

## 9. `acmpc_racing_gym` 每个文件的职责

### 9.1 包入口

| 文件 | 职责 |
|---|---|
| `__init__.py` | 导出 `AcMpcRacingEnv`、`RacingEnvConfig` |
| `dynamics/__init__.py` | 导出动力学、参数和状态类 |
| `envs/__init__.py` | 导出 Gym 环境 |
| `observations/__init__.py` | 导出 builder、配置和模式 |
| `rewards/__init__.py` | 导出 reward 和配置 |
| `tracks/__init__.py` | 导出 Gate、Track、loader |
| `wrappers/__init__.py` | 导出 StateDummyVecEnv 和 factory |

### 9.2 `config.py`

- `InitialStateConfig`：初始位置/速度/yaw 及均匀噪声；
- `RacingEnvConfig`：赛道、随机 reset、最大步数、无人机半径、边界、动力学、观测、奖励和 seed；
- `world_bounds_array()`：生成 `(3,2)` NumPy bounds。

配置默认赛道是 `split_s`，训练脚本默认参数是 `horizontal`。

### 9.3 Dynamics

#### `dynamics/params.py`

定义质量、重力、20 ms 控制步、2.5 ms 子步、臂长、惯量、kappa、推力范围、rate gain、电机时间常数、转速、thrust map 和线性阻力。

#### `dynamics/state.py`

- `normalize_quat()`：单位化，非法值退回 identity；
- `quat_multiply()`：Hamilton 乘法；
- `quat_to_rotmat()`：四元数转旋转矩阵；
- `yaw_to_quat()`：yaw 转四元数；
- `QuadrotorState`：13 维状态、vector 转换和深复制。

#### `dynamics/integrator.py`

实现固定步长 RK4 的 `k1..k4` 积分。

#### `dynamics/flightmare_like_dynamics.py`

- `PhysicalCommand` 保存反归一化 CTBR；
- constructor 构造惯量、rate gain、allocation 和 motor state；
- `reset()` 重置状态并把电机设为悬停；
- `get_state13()` 提供 PPO side-channel；
- `action_to_command()` 完成动作反归一化和裁剪；
- `step()` 把 20 ms 拆成最多 8 个 2.5 ms 子步；
- `_substep()` 执行 rate controller、motor allocation、motor lag、thrust map 和 RK4；
- `_desired_motor_thrusts()` 计算 `J*K*rate_error + omega x J omega`；
- `_derivative()` 计算 p/q/v/omega 导数，包含重力和线性 drag；
- 其余函数实现 X 构型 allocation、推力/转速映射和裁剪。

称为 Flightmare-like 是因为结构接近，但参数和方程并非逐项等同 Flightmare。

### 9.4 Observation 和 Reward

#### `observations/acmpc_observation.py`

定义两种 track mode、固定 future gates=2 的配置，并生成/校验 36 维 finite float32 observation。`normalize=True` 只控制 factory 是否添加 VecNormalize，builder 不修改物理量。

#### `rewards/racing_reward.py`

定义 `-10/+10/+10/b=0.01`，按 collision、finish、gate、continuous progress 的优先级计算，并支持末门+终点同一步加和。

### 9.5 Track 和 Gate

#### `tracks/gate.py`

- 正交化 normal/up 并计算 right；
- 计算四个世界角点；
- 世界坐标转 gate 局部坐标；
- 连续线段与门平面求交；
- 按法向方向和扣除 drone radius 后的开口判穿门；
- 按 inner/outer rectangle 环带判门框碰撞。

#### `tracks/track.py`

- `TrackStart` 保存起点/yaw；
- `FinishRegion` 验证终点球并实现点包含/线段相交；
- `Track` 维护 current index、future gates、finish phase 和 target；
- 门不足两扇时用终点构造虚拟 observation gate 补齐。

#### `tracks/loader.py`

从 UTF-8 JSON 构造 Gate、TrackStart、FinishRegion 和 world bounds，要求 finish 存在并验证 bounds shape。

#### `tracks/assets/*.json`

| 文件 | 当前内容 |
|---|---|
| `horizontal.json` | 3 门水平直线；起点 `[-2,0,2]`，终点 `[6,0,2]`，半径 0.5 |
| `vertical.json` | z=10 向下穿 5/3/1 m 三门；终点 `[0,0,0.5]`，半径 0.5 |
| `split_s.json` | 7 门 Split-S-inspired 三维赛道 |
| `race_loop_like.json` | 7 门 loop-like 赛道，边界扩展到约 `+-15` |

赛道为论文启发的技术复现，不宣称逐厘米还原作者赛道。

### 9.6 `envs/racing_env.py`

- `__init__()` 装配 track/dynamics/observation/reward 并声明 36/4/13 spaces；
- `seed()` 更新 RNG；
- `reset()` 重载赛道、清 gate index、采样初态、重置动力学；
- `step()` 执行动作，连续检测 gate/frame/finish，处理 collision/timeout/reward/info；
- `get_state()` 暴露原始 13 维 state；
- `compute_prediction_rollout()` 在不修改真实环境状态的情况下评估 `[H,14]=[x10,u4]`，供 MPVE；
- `render("human")` 只打印位置/门序号；
- `render("rgb_array")` 返回黑图占位，不是真实渲染；
- 私有函数负责初态、边界覆盖和 collision type。

内部 MPC 没有 omega state，因此预测 rollout 把预测 body-rate control 作为预测状态的 omega。函数退出时恢复真实 gate index。

### 9.7 Wrappers

#### `wrappers/state_vec_env.py`

继承 `DummyVecEnv`，在同一 Python 进程顺序执行多个 env；暴露 `state_space`，并把各子环境 state 组成 `[n_envs,13]`。

#### `wrappers/sb3_make_env.py`

- 返回深拷贝 config 的零参数 factory；
- 创建 `StateDummyVecEnv`；
- 可选添加 `VecNormalize`；
- 默认 observation normalization 开、reward normalization 关；
- 把 state space 显式挂到外层，state 始终不归一化。

## 10. Gym 训练、MPVE、评估和绘图文件

| 文件 | 职责 |
|---|---|
| `scripts/train_acmpc_gym.py` | 标准 PPO；env、学习率、checkpoint、CSV/TensorBoard、normalizer 保存恢复 |
| `scripts/train_acmpc_gym_mpve.py` | PPO+MPVE critic 扩展 |
| `scripts/eval_acmpc_gym.py` | 模型/normalizer 选择、固定 seed 评估、trajectory/summary/指标 |
| `scripts/plot_gym_tracks.py` | 2D/3D 绘制门、法向、起终点和 bounds |
| `scripts/plot_trajectories.py` | 速度热力图、状态终点、平均控制输入和标准差 |
| `scripts/plot_training_metrics.py` | return/success/collision/gate progress/PPO loss 曲线 |
| `scripts/smoke_test_racing_gym.py` | gate、rollout、finish、normalization、state plumbing、policy 回归测试 |
| `scripts/run_train_acmpc_gym.ps1` | Gym 通用训练入口和 track/finish 校验 |
| `scripts/run_eval_acmpc_gym.ps1` | Gym 评估入口 |
| `scripts/run_mpve_horizontal_pipeline.ps1` | 水平赛道 MPVE 训练/评估/绘图流水线 |
| `scripts/run_plot_training_metrics.ps1` | Windows PowerShell 指标绘图入口 |
| `scripts/run_plot_training_metrics.sh` | Bash/Git Bash 指标绘图入口 |
| `scripts/run_plot_trajectories.ps1` | 轨迹绘图入口 |
| `scripts/smoke_test_acmpc_forward.py` | import、DroneDx、MPC、policy init smoke |
| `scripts/validate_acmpc_core.py` | forward、四元数、Jacobian、MPC、batch 1/8/64 验证 |

标准训练默认 `track=horizontal, T=2, n_envs=8, n_steps=250, rollout=2000, batch=2000, epochs=10, gamma=0.98, GAE lambda=0.95, clip=0.2, lr=3e-4->1e-5`。observation normalization 开，reward normalization 关。

`EpisodeCsvCallback` 记录 episode return/length/gate/finished/collision/timeout；`CudaMemoryCsvCallback` 在 rollout 边界记录 allocated/reserved/peak memory。

`track_visualizations/` 中的 PNG 是 `plot_gym_tracks.py` 生成的当前赛道静态可视化，包括各单赛道和总览；它们是展示资产，不参与环境动力学或训练。

## 11. 当前 MPVE 实现

`MPVERolloutBuffer` 增加：

```text
prediction_observations [batch,H,36]
prediction_rewards      [batch,H]
prediction_valid        [batch,H]
prediction_terminal     [batch,H]
```

采样时从 `MlpMpcPolicy.predictions` 取 `[x10,u4]`，调用 Gym `compute_prediction_rollout()` 生成预测 observation/reward/mask，再执行真实环境一步。

训练时：

```text
value_loss = TD(GAE) MSE + mpve_coef * MPVE prediction MSE
total_loss = PPO policy loss + entropy term + vf_coef*value_loss
```

`compute_mpve_targets()` 从每个预测起点累加折扣 reward，terminal 后停止，并可用最后预测 observation 的 critic value bootstrap。

必须明确：

- MPVE 目前只用于 Python Gym，Flightmare 脚本仍是标准 PPO；
- 没有独立 MPVE lambda，`gae_lambda` 只作用于真实 rollout GAE；
- MPVE 参数是 coef、horizon、bootstrap 和 valid mask；
- actor 仍通过 PPO+diffMPC 更新，MPVE 额外项主要训练 critic；
- prediction 已 detach，MPVE critic loss 不通过预测状态反向进入 MPC。

## 12. Flightmare 改造目标

Flightmare 阶段把 Gym 合同迁移到 C++ 外部仿真器：

```text
RacingEnv C++ -> Flightmare VecEnv/OpenMP -> pybind RacingEnv_v1
-> FlightmareRacingVecEnv -> VecNormalize -> 同一个 MlpMpcPolicy/PPO
```

目标是 36/4/13 接口、论文 reward、与 Gym 对齐的 gate/finish/collision、复用 Gym track schema、Windows headless 编译。明确不加入 Unity gate 可视化、BEM/NeuroBEM、sim-to-real 和 domain randomization。

## 13. Flightmare 原文件的逐文件修改

### 13.1 `flightlib/CMakeLists.txt`

- 设置 `CMP0091` 并统一 MSVC 动态 CRT，解决 `/MD`/`/MT` 冲突；
- 默认关闭 tests 和 Unity bridge tests；
- 新增 `BUILD_UNITY_BRIDGE`，默认 OFF；
- 只有 Unity build 才查找 OpenCV、编译 bridge/camera、链接 ZMQ/OpenCV；
- headless 只编译 dynamics、基础 object、IMU、env、common；
- MSVC 使用 `/O2 /MP`，不传 GCC 的 `-fPIC/-march/-Ofast`；
- 非 MSVC 才链接 `stdc++fs`；
- 可用时显式链接 OpenMP target。

### 13.2 `cmake/pybind11_download.cmake`

pybind11 从 `master` 固定到 `v2.10.4`，并使用 shallow clone，提高可复现性。

### 13.3 `bridges/unity_message_types.hpp`

移除顶部 OpenCV include，减少 headless 依赖。Unity 相关源码只在 bridge 开启时构建。

### 13.4 `env_base.hpp/.cpp`

- 移除 Windows 不可用的 `<unistd.h>`；
- 新增虚方法 `getState()`，默认 false；
- 新增 `state_dim_` 和 getter；
- constructor 初始化 state dimension。

### 13.5 `vec_env.hpp/.cpp`

- UnityBridge 改 forward declaration 和编译宏；
- 引入 `RacingEnv`；
- 从子环境读取 state dimension；
- 新增批量 `getState(Matrix)` 和 shape 验证；
- headless 下禁用 render/connect/disconnect 并 warning；
- `RenderMessage_t` 只在 Unity build 存在；
- SceneID 默认使用数值以去除 Unity enum 依赖；
- 显式实例化 `VecEnv<RacingEnv>`；
- 保留 OpenMP 并行 step。

### 13.6 `quadrotor_env.hpp/.cpp`

UnityBridge 改 forward declaration；只在 Unity build include；headless 下 `addObjectsToUnity()` 为空操作。

### 13.7 `objects/quadrotor.hpp/.cpp`

- RGBCamera 改 forward declaration，避免 headless 引入 OpenCV camera；
- 补充标准库 include；
- 修正 collective thrust clamp：使用四电机总推力除质量得到 mass-normalized 上下限，而不是把 collective scalar 当单电机 thrust clamp。

### 13.8 `dynamics/quadrotor_dynamics.cpp`

YAML 更新 motor omega 和 thrust map 后，重新计算单电机 `thrust_min/max`；最小推力限制不小于 0。避免动态参数更新后仍使用 constructor 旧限制。

### 13.9 `flightlib/setup.py`

- editable build 显式关闭 tests/Unity；
- 支持 `FLIGHTMARE_CMAKE_ARGS` 追加 CMake 参数；
- Windows x64 使用 Release 和并行 MSBuild。

### 13.10 `wrapper/pybind_wrapper.cpp`

新增 `flightgym.RacingEnv_v1`，暴露 constructors、reset、step、testStep、seed、close、terminal、curriculum、Unity stubs、env 数量、obs/action/state dimensions、getState 和 extra-info names。原 `QuadrotorEnv_v1` 保留。

## 14. Flightmare 新增 `RacingEnv`

### 14.1 `racing_env.hpp`

固定 `obs=36, action=4, state=13`。`RacingGate` 保存 frame 和尺寸并提供角点、坐标转换、线段求交、穿门、门框碰撞。`RacingEnv` 继承 EnvBase，保存 Flightmare Quadrotor、track、初态噪声、动作范围、reward、episode 状态和 extra info。

### 14.2 `racing_env.cpp` 方法

| 方法 | 当前实现 |
|---|---|
| `normalizeFrame()` | 归一化 normal，正交化 up，计算 right |
| `cornersWorld()` | 生成四个 observation 角点 |
| `segmentPlaneIntersection()` | 连续线段穿平面，可要求正向穿越 |
| `checkPass()` | 使用扣除 drone radius 后的有效开口 |
| `checkFrameCollision()` | 检查 inner/outer rectangle 环带 |
| constructor | 加载 YAML、更新 Flightmare dynamics、设置 36/4/13、加载 track |
| `loadParam()` | 读取 env、initial state、action、reward、observation、track |
| `loadTrack()` | 支持内嵌 track 或外部 `track_path` |
| `loadTrackFromNode()` | 构造 gates/start/bounds/finish/虚拟 finish gate |
| `reset()` | gate index 和 episode 标志清零并 reset quadrotor |
| `resetQuadState()` | 均匀 position/velocity/yaw noise |
| `actionToCommand()` | `[-1,1]^4` 转 mass-normalized thrust/body rates |
| `step()` | Flightmare 推进、gate/frame/finish/bounds/ground/finite/timeout/reward |
| `getObs()` | `v3+R9+corner12+corner12`，支持两种 mode |
| `getState()` | `p3+q4+v3+omega3` |
| `computeReward()` | 论文分支，支持末门+终点同一步 `+20` |
| `isTerminalState()` | collision、finished、timeout |
| `updateExtraInfo()` | gate、finish、collision code、speed、xyz |
| `finishReached()` | finish phase 下线段-球最近点检查 |
| `collisionCode()` | 0无、1越界、2地面、3门框、4非法 state |

step 先记录本步目标，推进 Flightmare，再做连续几何检查；gate advance 后立即检查新的 finish phase，因此已修复同一步末门/终点漏判。

### 14.3 当前碰撞范围

检查当前目标门框、地面、world bounds 和非法 state。没有检查非当前门、场景障碍物、完整无人机网格或 Unity 物体。`drone_radius=0.18` 是球形近似。

## 15. Flightmare 配置和赛道

### 15.1 `configs/racing_env.yaml`

新增 20 ms 仿真步、500 step、随机 reset、半径/边界、quadrotor dynamics、初态噪声、action limits、observation mode、reward 和当前 track path。

当前 `track_path` 是本机绝对路径。跨设备 clone 后必须运行 track install 脚本写入新设备实际路径。

### 15.2 `configs/tracks/*.yaml`

新增 `horizontal`、`vertical`、`split_s`、`race_loop_like`，由 Gym JSON 转换，字段包括 name/start/finish/bounds/gates/frame/label。只改 JSON/YAML 不需要重编 `.pyd`。

## 16. Flightmare Python adapter

### 16.1 `acmpc_flightmare/__init__.py`

导出 VecEnv、factory、track loader 和 metadata loader。

### 16.2 `acmpc_flightmare/track.py`

- 优先 PyYAML，缺失时使用当前 schema 的最小 parser；
- 解析 nested track path；
- 标准化 gate frame 并计算角点；
- 生成评估/绘图统一 track dict；
- 提取 dt、steps、mass、action limits、track name。

fallback parser 不是通用 YAML 实现。

### 16.3 `acmpc_flightmare/vec_env.py`

- 设置 `FLIGHTMARE_PATH` 和 import path；
- 创建已向量化的 C++ `RacingEnv_v1`；
- 读取 dimensions 和 extra-info names；
- 分配 obs/state/reward/done/extra/action buffers；
- `step_async()` 保存并裁剪动作；
- `step_wait()` 调 C++ batch step、刷新 state、构造 infos；
- `get_state()` 提供 `[n_envs,13]`；
- 实现 SB3 get/set attr、env_method、seed、close；
- collision code 转字符串；
- factory 可选 VecNormalize，state 不归一化。

当前 C++ VecEnv 在 done 时自动 reset 后再返回 observation。extra info 是 reset 前复制，因此 terminal xyz/speed/collision 可用；但 Python 随后取得的 13 维 state 已是 reset 后状态，wrapper 设置的 `terminal_observation` 也实际是 reset observation。该限制会影响 timeout value bootstrap 的严格语义，尚未修复。

当前 binary 的 RacingEnv 默认 constructor 只读 `<FLIGHTMARE_PATH>/flightlib/configs/racing_env.yaml`，adapter 因此拒绝用其他 config 只改变 Python metadata；必须先安装目标赛道到 runtime config。

## 17. Flightmare 工具和入口

| 文件 | 职责 |
|---|---|
| `build_flightmare_racing_env.ps1` | 解析 Conda Python、优先 VS CMake、`pip install -e flightlib --no-deps -v` |
| `smoke_test_flightmare_racing_env.py` | import、36/4/13、reset/getState/step/finite 验证 |
| `run_smoke_test_flightmare_racing_env.ps1` | smoke PowerShell 入口 |
| `install_gym_track_into_flightmare.py` | 验证 Gym JSON、生成 YAML、更新 runtime track path、备份恢复 |
| `verify_flightmare_track_runtime.py` | 创建真实 C++ env，核对 track/gates/接口 |
| `train_acmpc_flightmare.py` | PPO、VecNormalize、checkpoint、CSV/TensorBoard、CUDA memory log |
| `run_train_acmpc_flightmare.ps1` | 训练参数封装 |
| `run_train_acmpc_flightmare_track.ps1` | 通用赛道安装、验证和训练入口 |
| `eval_acmpc_flightmare.py` | 模型/normalizer、确定性评估、指标和轨迹 |
| `run_eval_acmpc_flightmare.ps1` | 评估入口 |
| `run_plot_flightmare_training_metrics.ps1` | Flightmare run 训练曲线入口 |
| `run_plot_flightmare_trajectories.ps1` | 轨迹/速度/控制输入绘图入口 |

Flightmare C++ VecEnv 可通过 OpenMP 并行各环境；Gym DummyVecEnv 在一个 Python 进程顺序 step。

## 18. 参数一致性和已知差异

| 参数 | 内部 MPC | Python Gym 外部 | Flightmare 外部 |
|---|---:|---:|---:|
| mass | 0.752 | 0.752 | 0.752 |
| dt | 0.02 | 0.02 | 0.02 |
| inertia diag | 0.0025,0.0021,0.0043 | 同 MPC | 约 0.00815,0.00815,0.01268，由 mass/arm 公式生成 |
| arm/allocation | lx=.075, ly=.10 | arm=.17 X | arm=.17 X |
| kappa | .022 | .016 | .016 |
| motor tau | 参数 .033，但 CTBR 10维模型不显式积分 motor | .02 | .0001 |
| motor omega max | CTBR 不使用 | 3000 | 1700 |
| per-motor thrust cap | 8.5 | 8.5 | action 8.5，map limit 约 8.596 |
| body-rate max | 10,10,4 | 10,10,4 | 10,10,4 |
| rate gain | rate 直接作控制 | 16.6,16.6,5 | 16.6,16.6,5 |
| linear drag | 无 | .05,.05,.08 | 无该项 |

三个模型目前不严格一致，尤其 Flightmare 惯量和 motor tau 与 MPC/Gym 不同。这构成名义模型与外部模型失配，但不能描述为参数完全对齐。

### 18.1 已实现

- neural cost map + diffMPC + PPO；
- 36 维 observation 和 CTBR action；
- 论文 gate-progress reward 和 observation normalization；
- Gym/Flightmare 训练评估；
- Gym MPVE；
- 多赛道、门框、终点、轨迹、速度和平均控制可视化。

### 18.2 未实现或不完整

- 作者私有 modified Flightmare 的逐行还原；
- BEM/NeuroBEM、sim-to-real、latency、domain randomization；
- Unity gate 场景；
- 精确论文赛道和论文数值；
- Flightmare MPVE；
- 完整障碍/所有门碰撞；
- 真正生效的 MPC warm start；
- 正确保留 Flightmare terminal observation/state。

## 19. 验证记录

已实际验证：import mpc/drone；DroneDx forward；四元数 norm；解析 Jacobian；IL_Env MPC action bounds；policy batch 1/8/64；prediction shape；Gym gate/frame/bounds/rollout/normalization/state plumbing；同一步末门+终点；Flightmare Windows headless 编译；RacingEnv reset/getState/step 及 36/4/13；Gym/Flightmare PPO 保存 model/checkpoint/VecNormalize；评估指标、CSV、速度热力图和平均控制输入。

PyTorch 2.6.0 下仍有作者依赖的 `.T`、`torch.lu/lu_solve` 和 uint8 indexing deprecation warning。当前不阻止运行，但升级 PyTorch 前需要处理。

## 20. 维护和部署规则

远程仓库：

```text
https://github.com/Bill-WangJiLong/AC-MPC_Reproduction.git
```

远程中的 `mpc.pytorch`、`stable-baselines3`、`flightmare` 是包含本地修改的普通源码目录，不要用上游版本覆盖。

- Gym JSON 是赛道规范源；
- 同步 Flightmare 需运行 track installer；
- 只改赛道不重编；
- 修改 C++/CMake/setup.py/pybind 必须重编 `.pyd`；
- 修改 observation/action/state 顺序必须同步 Gym、C++、adapter、SB3、policy 和测试；
- 模型必须与 `vecnormalize.pkl`、`ACMPC_T`、赛道配置配套保存。

本文是当前源码事实记录。后续算法、动力学、reward、赛道合同或 Flightmare 行为变化，应同步更新本文并增加回归测试。
