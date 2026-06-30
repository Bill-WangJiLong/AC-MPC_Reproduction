# Flightmare RacingEnv 设计与改动说明

本文档记录当前为了把 AC-MPC 复现从 Python Gym 原型迁移到 Flightmare 而做的设计和代码改动。目标是构建一个 Flightmare 侧的竞速环境，使其尽量贴近论文方法，同时尊重 Flightmare 原有环境结构，并方便后续接入当前仓库里的 `MlpMpcPolicy` 和 PPO 训练流程。

## 设计优先级

本阶段按以下顺序决策：

1. **优先参考 AC-MPC 论文**  
   保持论文中的 quadrotor racing observation、rate-thrust action、gate reward 和 AC-MPC 所需的物理 state 接口。

2. **其次尊重 Flightmare 原设计**  
   不重写 Flightmare 动力学，不绕过 `Quadrotor` 对象，不改变 `VecEnv` 的基本并行环境调用方式。

3. **最后参考当前 Python Gym 原型和训练代码**  
   保持 Gym 原型已验证过的 observation/action/state/reward/done 语义，方便后续从 Gym 训练脚本迁移到 Flightmare。

## 总体架构

新增环境为 Flightmare 中的一个独立 C++ 环境：

```text
Python PPO / AC-MPC policy
        |
        | obs: 36
        | state: 13
        | action: 4 normalized
        v
flightgym.RacingEnv_v1
        |
        v
VecEnv<RacingEnv>
        |
        v
RacingEnv
        |
        v
Flightmare Quadrotor + QuadrotorDynamics
```

`RacingEnv` 只负责竞速任务逻辑：track、gate、observation、reward、done、collision、state 输出。真实推进仍然使用 Flightmare 的 `Quadrotor::run()` 和 `QuadrotorDynamics`。

## Flightmare 侧新增文件

### `D:\MyProjects\flightmare\flightlib\include\flightlib\envs\racing_env\racing_env.hpp`

新增 `RacingEnv` 和 `RacingGate` 的声明。

核心定义：

```text
obs_dim   = 36
act_dim   = 4
state_dim = 13
```

主要职责：

```text
RacingGate:
  gate 坐标系构建
  四角点计算
  穿门检测
  门框碰撞检测

RacingEnv:
  reset
  step
  getObs
  getState
  isTerminalState
  updateExtraInfo
  loadParam / loadTrack
```

### `D:\MyProjects\flightmare\flightlib\src\envs\racing_env\racing_env.cpp`

新增 `RacingEnv` 的完整实现。

已实现内容：

```text
1. 加载 racing_env.yaml
2. 加载 track schema
3. 使用 Flightmare Quadrotor 动力学推进状态
4. 将归一化动作反归一化为 rate-thrust command
5. 构造论文风格 36 维 observation
6. 输出 AC-MPC 需要的 13 维 state
7. 实现 gate passing
8. 实现 gate frame collision
9. 实现 out_of_bounds / ground / non_finite_state 终止
10. 实现论文风格 reward
11. 写入 extra_info
```

## Flightmare 侧新增配置

### `D:\MyProjects\flightmare\flightlib\configs\racing_env.yaml`

新增默认竞速环境配置。

当前默认配置是一个水平三门轨道，用于 smoke test 和接口验证：

```yaml
track:
  name: horizontal
  start:
    position: [-2.0, 0.0, 2.0]
    yaw: 0.0
  finish:
    position: [6.0, 0.0, 2.0]
    radius: 0.5
  gates:
    - center: [0.0, 0.0, 2.0]
    - center: [2.0, 0.0, 2.0]
    - center: [4.0, 0.0, 2.0]
```

配置包含：

```text
racing_env:
  sim_dt
  max_t
  max_episode_steps
  random_reset
  drone_radius
  world_box

quadrotor_dynamics:
  mass
  arm_l
  motor limits
  thrust_map
  kappa
  omega_max

initial_state:
  position
  velocity
  yaw
  reset noise

action:
  thrust_max_per_motor
  omega_max

observation:
  future_gate_count
  track_obs_mode

reward:
  collision_reward
  gate_pass_reward
  finish_reward
  body_rate_coeff

track:
  name
  start
  finish
  world_bounds
  gates
```

当前 schema 与 Python Gym 原型一致，可直接安装 `horizontal`、`vertical`、`split_s`、`race_loop_like` 或按同一格式新增的赛道。

## 修改 Flightmare 原有文件

### `D:\MyProjects\flightmare\flightlib\include\flightlib\envs\env_base.hpp`

新增通用 state 接口：

```cpp
virtual bool getState(Ref<Vector<>> state);
inline int getStateDim() { return state_dim_; };
```

新增成员：

```cpp
int state_dim_;
```

原因：AC-MPC policy 不是只用 observation。MLP cost map 输入是 36 维 observation，但 MPC 初始状态需要 13 维物理 state：

```text
[p(3), q(4), v(3), omega(3)]
```

Flightmare 原始 `VecEnv` 没有这个接口，因此必须补。

### `D:\MyProjects\flightmare\flightlib\src\envs\env_base.cpp`

初始化 `state_dim_`，并给 `getState()` 提供默认实现：

```cpp
bool EnvBase::getState(Ref<Vector<>> state) {
  (void)state;
  return false;
}
```

这样不会强制破坏已有环境。只有 `RacingEnv` 实现有效 state 输出。

### `D:\MyProjects\flightmare\flightlib\include\flightlib\envs\vec_env.hpp`

新增：

```cpp
bool getState(Ref<MatrixRowMajor<>> state);
inline int getStateDim(void) { return state_dim_; };
```

并包含：

```cpp
#include "flightlib/envs/racing_env/racing_env.hpp"
```

### `D:\MyProjects\flightmare\flightlib\src\envs\vec_env.cpp`

新增批量 state 输出：

```cpp
bool VecEnv<EnvBase>::getState(Ref<MatrixRowMajor<>> state)
```

并显式实例化：

```cpp
template class VecEnv<RacingEnv>;
```

这样 Python 侧可以调用：

```python
env.getState(state_array)
```

其中 `state_array.shape == [num_envs, 13]`。

### `D:\MyProjects\flightmare\flightlib\src\wrapper\pybind_wrapper.cpp`

新增 pybind class：

```cpp
py::class_<VecEnv<RacingEnv>>(m, "RacingEnv_v1")
```

暴露接口：

```text
reset
step
testStep
setSeed
close
isTerminalState
curriculumUpdate
connectUnity
disconnectUnity
getNumOfEnvs
getObsDim
getActDim
getStateDim
getState
getExtraInfoNames
```

其中 `connectUnity/disconnectUnity` 保留 Flightmare 原接口，但本阶段不添加 gate Unity 可视化。

### `D:\MyProjects\flightmare\flightlib\src\objects\quadrotor.cpp`

修正 rate-thrust command 的推力裁剪逻辑。

Flightmare 的 `Command.collective_thrust` 注释定义为：

```text
Collective mass-normalized thrust in [m/s^2]
```

原代码在 `setCommand()` 中用单电机 thrust 范围裁剪 `collective_thrust`，这对 rate-thrust 模式不合适。

现改为按总推力除以质量后的范围裁剪：

```cpp
collective_thrust_min = dynamics_.collective_thrust_min() / dynamics_.getMass();
collective_thrust_max = dynamics_.collective_thrust_max() / dynamics_.getMass();
```

这样 `RacingEnv` 的动作空间才能和论文保持一致：

```text
action = [mass-normalized thrust, wx, wy, wz]
```

### `D:\MyProjects\flightmare\flightlib\src\dynamics\quadrotor_dynamics.cpp`

修正 `updateParams()` 后推力上下界没有跟随 YAML 参数更新的问题。

新增在读取 `motor_omega_min/max` 和 `thrust_map` 后重新计算：

```cpp
thrust_min_
thrust_max_
```

原因：`racing_env.yaml` 使用接近 AC-MPC 代码的参数：

```text
mass = 0.752
thrust_max_per_motor = 8.5
omega_max = [10, 10, 4]
```

如果 `thrust_max_` 不随 YAML 更新，rate-thrust action 的物理边界会不一致。

## Observation 设计

论文描述：

```text
o_quad = [v_t, R_t] in R12
o_track = future gates, G=2, each gate four corners
total obs = 12 + 12 + 12 = 36
```

当前 `RacingEnv` 实现：

```text
obs[0:3]    = linear velocity v
obs[3:12]   = rotation matrix R, row-major flatten
obs[12:24]  = next gate 4 corners relative to vehicle position
obs[24:36]  = second future gate 4 corners relative to vehicle position
```

默认：

```yaml
observation:
  future_gate_count: 2
  track_obs_mode: vehicle_relative
```

也保留了：

```yaml
track_obs_mode: chained_gate_relative
```

该模式下第二个 gate 的 corner observation 使用：

```text
gate2_corner - gate1_corner
```

这对应前面讨论过的另一种可能解释。

注意：Flightmare C++ 环境输出的是原始物理量，不在环境内部归一化。归一化应继续放在 PPO wrapper / VecNormalize 层。

## Action 设计

论文动作空间是：

```text
a = [c, wx, wy, wz]
```

其中：

```text
c      = mass-normalized collective thrust
omega  = body rates
```

当前 `RacingEnv` 对外仍接收归一化动作：

```text
action range = [-1, 1]^4
```

内部反归一化：

```text
mass_normalized_thrust = action[0] * force_std + force_mean
body_rate_cmd          = action[1:4] * omega_max
```

其中：

```text
force_mean = (thrust_max_per_motor * 4 / mass) / 2
force_std  = force_mean
omega_max  = [10, 10, 4]
```

之后构造 Flightmare 原生 `Command`：

```cpp
Command(t, mass_normalized_thrust, omega_cmd)
```

再由 `Quadrotor::runFlightCtl()` 转换为单电机推力。

## Dynamics 设计

不在 `RacingEnv` 内部重写动力学。

推进链路为：

```text
RacingEnv::step()
  -> actionToCommand()
  -> Quadrotor::run(cmd, sim_dt)
  -> Quadrotor::runFlightCtl()
  -> QuadrotorDynamics
  -> RK4 integrator
```

这样做的原因：

```text
1. 尊重 Flightmare 原设计
2. 避免 Python Gym 里的动力学重复移植错误
3. 后续接 BEM/NeuroBEM 或更真实动力学时接口更自然
```

本阶段不加入 BEM/NeuroBEM，不加入 sim-to-real domain randomization，但接口上保留了配置入口。

## Gate 与 Track 设计

`RacingGate` 包含：

```text
center
normal
up
right
width
height
frame_thickness
label
```

gate 坐标系：

```text
normal = gate forward direction
up     = gate vertical direction, projected to be orthogonal to normal
right  = up cross normal
```

穿门检测：

```text
1. 上一步位置 p_prev 和当前位置 p_curr 形成线段
2. 检查线段是否从 gate normal 的负侧穿到正侧
3. 计算线段和平面的交点
4. 判断交点是否在 gate 内框内
```

门框碰撞：

```text
1. 线段与 gate 平面相交
2. 交点在外框内
3. 交点不在内框内
```

该逻辑与 Python Gym 原型保持一致。

终点采用显式球体，不复用最后一个门：

```text
finish.position = 球心世界坐标
finish.radius   = 球体半径
```

穿过最后一个门后，`current_gate_idx` 进入终点阶段，但 episode 保持运行。只有后续位置线段与终点球相交时才设置 `race_finished=true`。终点判定使用线段到球心的最近距离，避免较大仿真步长或高速运动造成穿透漏检。

## Reward 与 Done 设计

reward 使用论文风格：

```text
collision:     -10
gate passed:   +10
race finished: +10
otherwise:
  distance_progress - 0.01 * ||omega||
```

最后一个门与终点是两个独立事件：最后一门产生一次 `gate passed: +10`；至少在后续一个仿真步命中终点球后，再产生一次 `race finished: +10`。因此完整通过 N 个门并到达终点的离散事件奖励为 `10*N + 10`，其间仍可能包含连续进度奖励。

对应配置：

```yaml
reward:
  collision_reward: -10.0
  gate_pass_reward: 10.0
  finish_reward: 10.0
  body_rate_coeff: 0.01
```

done 条件：

```text
collision
race_finished
timeout
```

collision 类型：

```text
out_of_bounds
ground
gate_frame
non_finite_state
```

## Extra Info 设计

`RacingEnv::updateExtraInfo()` 输出：

```text
gate_index
gate_passed
collision
collision_code
finished
finish_phase
finish_distance
out_of_bounds
timeout
speed
x
y
z
```

这些字段用于后续训练日志、评估和轨迹绘图。

## AC-MPC 训练接口关系

AC-MPC policy 需要两个输入：

```text
obs   -> 36 维，输入 neural cost map / critic
state -> 13 维，输入 MPC 初始状态
```

当前 Flightmare 侧提供：

```python
obs = np.zeros((num_envs, 36), dtype=np.float32)
state = np.zeros((num_envs, 13), dtype=np.float32)

env.reset(obs)
env.getState(state)
env.step(action, obs, reward, done, extra_info)
```

后续需要写 Flightmare 对接当前 SB3 fork 的 wrapper，使其提供：

```text
reset()
step()
get_state()
observation_space
action_space
state_space
```

这样才能直接复用当前已经补好的 PPO state plumbing。

## 不做的内容

本阶段明确不做：

```text
1. 不添加 Unity gate 可视化
2. 不加入 BEM / NeuroBEM
3. 不加入 sim-to-real domain randomization
4. 不复现 baseline
5. 不试图精确还原论文 SplitS 数值 track
```

但设计上没有阻断这些扩展：

```text
1. track 已配置化
2. dynamics 参数已配置化
3. action/state/obs 接口和论文一致
4. extra_info 可继续扩展
5. rate-thrust command mode 已接入 Flightmare 原动力学
```

## 当前验证状态

已完成：

```text
1. Flightmare 代码已落盘
2. Python smoke test 脚本已写
3. PowerShell 构建脚本已写
4. PowerShell smoke test 脚本已写
5. Python 脚本语法检查通过
6. acmpc 环境版本已恢复确认：
   numpy 1.26.4
   gym 0.21.0
   torch 2.6.0+cu118
```

未完成：

```text
1. C++ 编译未完成
2. flightgym.RacingEnv_v1 尚未 import 验证
3. reset/step/getState 尚未在 Python 中真实调用验证
```

阻断原因：

```text
当前 acmpc 环境 PATH 中没有 cmake。
```

构建失败信息：

```text
RuntimeError: CMake must be installed to build the following extensions: flightlib
```

## 构建方式

不要直接运行：

```powershell
pip install -e .\flightlib
```

原因：Flightmare 原始 `setup.py` 会尝试安装旧依赖，例如：

```text
gym==0.11
stable_baselines==2.10.1
PyOpenGL
```

这会污染当前 AC-MPC 的 PyTorch/SB3 环境。

应使用：

```powershell
cd D:\MyProjects\acmpc_public
powershell -ExecutionPolicy Bypass -File .\scripts\build_flightmare_racing_env.ps1
```

该脚本固定使用：

```text
pip install -e .\flightlib --no-deps -v
```

## Smoke Test

构建成功后运行：

```powershell
cd D:\MyProjects\acmpc_public
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_test_flightmare_racing_env.ps1
```

期望输出：

```text
RacingEnv_v1 smoke test passed
n_envs: ...
obs_dim: 36
act_dim: 4
state_dim: 13
```

测试内容：

```text
1. import flightgym.RacingEnv_v1
2. env.getObsDim() == 36
3. env.getActDim() == 4
4. env.getStateDim() == 13
5. reset(obs) 可调用
6. getState(state) 可调用
7. step(action, obs, reward, done, extra) 可调用
8. obs/state/reward 均为 finite
```

## 后续开发建议

1. **先安装 CMake 并完成 C++ 编译**

   只有编译通过后，才能确认模板实例化、pybind Eigen 绑定、Windows 编译环境是否完全可用。

2. **编译通过后立刻跑 smoke test**

   不要直接进入训练。先确认 `RacingEnv_v1` 的 reset/step/getState 形状和数值正确。

3. **实现 Flightmare SB3 wrapper**

   需要把 `RacingEnv_v1` 包装成当前 SB3 fork 可用的 vec env，并提供：

   ```text
   get_state()
   state_space = Box(shape=(13,))
   observation_space = Box(shape=(36,))
   action_space = Box(shape=(4,), low=-1, high=1)
   ```

4. **把 Python Gym 的 gate tests 迁移一份到 Flightmare smoke**

   尤其验证：

   ```text
   正向穿门成功
   反向穿门不成功
   门框碰撞
   越界终止
   timeout
   ```

5. **再迁移训练脚本**

   建议新建：

   ```text
   scripts/train_acmpc_flightmare.py
   scripts/eval_acmpc_flightmare.py
   scripts/plot_flightmare_trajectories.py
   ```

   不要直接改现有 Gym 训练脚本，以便对比 Gym 和 Flightmare。

6. **最后再替换 SplitS-like track**

   先用水平三门 track 跑通训练闭环，再使用论文启发式 SplitS track。

## 当前风险

1. **C++ 尚未编译验证**

   目前不能保证没有 Eigen Ref 或 pybind 编译细节问题。

2. **Windows Flightmare 构建依赖可能不完整**

   除 CMake 外，后续可能还需要 Visual Studio C++ build tools、OpenCV、OpenMP、ZeroMQ、yaml-cpp 等依赖。

3. **Flightmare 原 setup.py 依赖过旧**

   必须避免让它修改当前 PyTorch/SB3 环境。

4. **`VecEnv` 增加 state 接口是框架级改动**

   设计上兼容旧环境，但仍需编译验证。

5. **当前 track 是 smoke test track**

   它不是论文 SplitS 的精确数值复现，只用于接口和训练闭环验证。
