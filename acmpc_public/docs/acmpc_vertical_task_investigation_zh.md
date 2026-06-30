# AC-MPC 垂直任务专项排查与实验计划

## 1. 目标

本专项用于判断 AC-MPC 是否能够在垂直向下竞速任务中跳出“零推力自由落体”局部最优，并学习论文描述的高速策略：

1. 尽快滚转或俯仰，使机体正 Z 轴指向下方。
2. 翻转后施加正推力，与重力共同向下加速。
3. 依次穿过所有门并到达终点。

本阶段保持论文的 gate-progress reward，不通过额外的翻转奖励、姿态奖励或推力奖励直接指定动作。

## 2. 当前已确认事实

当前模型：

```text
runs/acmpc_flightmare/20260628_170625_flightmare_vertical_T2
```

确定性评估表明：

- 平均质量归一化推力约为 `1.52 m/s^2`。
- 悬停需要约 `9.81 m/s^2`。
- 最大推力约为 `6.80 m/s^2`，全程低于重力补偿。
- 前 `0.5 s` 的平均推力接近零。
- 实际角速度积分约为 `0.44 rad`，远低于完成翻转所需的约 `pi rad`。
- 策略主要依靠重力完成赛道，没有实现论文描述的翻转后向下加推力。

当前 run 还存在以下训练尺度差异：

| 项目 | 当前设置 | 论文设置或目标 |
|---|---:|---:|
| 实际训练样本 | 188k | 2M |
| 单次 rollout | 8 x 250 = 2,000 | 约 25,000 |
| PPO epochs | 10 | 10 |
| 初始 `log_std` | -1.2 | AC-MPC 参考值 -1.2 |
| 初始位置随机范围 | x/y 约 0.1 m 全宽 | 1 m 立方体 |

## 3. 核心问题

### 3.1 探索强度

训练动作服从：

```text
a ~ Normal(mu_AC-MPC, std^2)
```

需要测试以下初始探索标准差：

| std | `log_std_init = log(std)` |
|---:|---:|
| 0.22 | -1.514 |
| 0.30 | -1.204 |
| 0.44 | -0.821 |
| 0.60 | -0.511 |

每个 std 至少使用三个 seed：

```text
seed = 0, 1, 2
```

第一轮共 12 次训练。条件允许时扩展到五个 seed。

需要回答：

- 小 std 是否稳定收敛到自由落体。
- 中等 std 是否能够发现连续翻转动作。
- 大 std 是否使 MPC 均值被噪声淹没并导致碰撞。
- 哪个 std 在不同 seed 下最稳定，而不是只在单次训练中偶然成功。

### 3.2 训练样本和 PPO rollout

当前 `2,000 samples/update` 与论文的 `25,000` 存在明显差异。需要比较：

1. 当前配置：`8 envs x 250 steps`。
2. 中间配置：在显存允许范围内增加并行环境。
3. 论文尺度：累计约 `25,000` 个样本后再执行 PPO 更新。

如无法同时运行约 100 个环境，应实现 rollout 累计或梯度更新前的数据聚合，而不是改变每个环境的 250 步时间长度。

所有正式结论应至少基于 `2M` 环境样本，不能用 188k 的早期模型判断最终能力。

### 3.3 MPC horizon

分别训练：

```text
ACMPC_T = 2
ACMPC_T = 5
```

`T=2` 是论文中的有效设置，但在 `dt=0.02 s` 时只覆盖 `0.04 s`。翻转需要更长时间，因此需要确认 `T=5` 是否能提高发现和维持激进动作的能力。

注意：论文图中的“10 MPC predictions”不等价于 MPC horizon 必须为 10，不能据此直接设置 `T=10`。

### 3.4 MPC warm start

作者公开代码中，policy 会传入上一轮 MPC 控制量：

```python
u_init=u_prev_chunk
```

但 `IL_Env.mpc()` 会将其覆盖为悬停推力和零 body rate。因此当前实际行为是每次从固定点初始化。

需要进行严格消融：

1. `fixed_hover_init`：保留作者公开代码行为。
2. `previous_solution_init`：真正使用上一轮 MPC 解。
3. 可选 `shifted_sequence_init`：使用上一轮最优序列左移后的控制序列。

比较 MPC 收敛、控制连续性、显存占用和翻转成功率。修复 warm start 时必须切断跨时间步的 autograd graph。

### 3.5 名义动力学与 Flightmare 刚体动力学

当前两侧并非同一个有效模型：

| 部分 | MPC 内部 `DroneDx` | 外部 Flightmare |
|---|---|---|
| 状态 | `[p, q, v]`，10 维 | `[p, q, v, omega]`，13 维 |
| 输入 | collective thrust + body rates | collective thrust + body-rate command |
| body rate | 立即用于姿态积分 | 低层角速度控制器跟踪 |
| 电机分配与饱和 | 未显式推进 | 显式计算 |
| 惯量作用 | 当前 CTBR 推进不使用 | 影响力矩和饱和 |

声明的惯量也不同：

```text
DroneDx:    [0.0025, 0.0021, 0.0043] kg*m^2
Flightmare: [0.00815, 0.00815, 0.01268] kg*m^2
```

仅修改 Flightmare 惯量数值不能消除模型差异，因为 `DroneDx.forward_CTBR()` 当前不使用惯量。

建议保留两个环境模式：

1. `nominal_ctbr`：body rate 直接推进姿态，与 `DroneDx` 的有效模型一致，用于论文方法的名义训练验证。
2. `rigid_body`：使用 Flightmare 低层控制器、惯量、电机分配和饱和，用于模型失配与鲁棒性评估。

在没有 BEM/NeuroBEM 的条件下，`rigid_body` 只能称为更完整的刚体环境，不能称为论文的真实 BEM 模拟器。

### 3.6 初始状态分布

当前位置噪声：

```yaml
position_noise: [0.05, 0.05, 0.02]
```

这使自由落体很容易保持在门中心。需要增加论文尺度配置：

```yaml
position_noise: [0.5, 0.5, 0.5]
```

若 `uniform_dist` 取值为 `[-1, 1]`，该配置对应边长 1 m 的立方体。

训练与评估至少区分：

- nominal start：无随机化。
- training distribution：1 m 立方体。
- robustness distribution：3 m 立方体，仅用于论文风格泛化评估。

## 4. 不应优先修改的内容

以下修改会改变论文任务，应只作为后续诊断消融，不作为首选修复：

- 增加“必须翻转”的姿态奖励。
- 增加正推力奖励。
- 对自由落体直接施加惩罚。
- 强制终点速度或终点姿态。
- 将垂直门改成大幅横向错位来迫使策略转向。

当前 gate-progress、gate passed、collision、race finished 和 body-rate penalty 应先保持论文定义。

## 5. 必须补充的记录量

每个评估 episode 应记录：

```text
position
velocity
quaternion / rotation matrix
body rates
normalized action
mass-normalized thrust command
actual collective thrust
single-rotor thrusts
body-rate command
gate index
reward components
MPC predicted states and controls
```

必须计算以下指标：

### 5.1 任务指标

- success rate
- collision rate
- episode return
- lap time
- average velocity
- gate passing time

### 5.2 翻转指标

- 机体正 Z 轴在世界坐标系中的 z 分量。
- 首次满足 `body_z_world_z < -0.8` 的时间。
- 最大姿态翻转角。
- 翻转后推力是否显著大于重力补偿。
- 推力饱和持续时间。

建议定义论文目标策略判据：

```text
success
AND body_z_world_z < -0.8 at least once
AND post_flip_thrust > 9.81 m/s^2 for a sustained interval
```

仅有 success 不能说明学到了论文的高速翻转策略。

### 5.3 学习指标

- 首次达到滚动 90% success 的样本数。
- 首次达到翻转策略判据的样本数。
- 不同 seed 的最终均值和标准差。
- PPO KL、clip fraction、value loss、explained variance 和 policy std。
- CUDA allocated/reserved memory 随 samples 的变化。

## 6. 实验矩阵

### 阶段 A：最小诊断

固定：

```text
T=2
2M samples
论文 reward
nominal CTBR
1 m 初始位置随机化
```

扫描：

```text
std in [0.22, 0.30, 0.44, 0.60]
seed in [0, 1, 2]
```

### 阶段 B：horizon 对比

使用阶段 A 最稳定的 std：

```text
T in [2, 5]
seed in [0, 1, 2]
```

### 阶段 C：warm-start 消融

```text
fixed_hover_init
previous_solution_init
shifted_sequence_init
```

### 阶段 D：动力学迁移

将阶段 A-C 中的最佳策略分别部署到：

```text
nominal_ctbr
rigid_body
```

比较成功率、速度、翻转时序和控制饱和。

## 7. 实施顺序

1. 修复并确认 CUDA 显存增长问题，保证 2M 样本可完成。
2. 补全姿态、实际推力和单电机推力记录。
3. 实现翻转策略自动判据。
4. 实现论文尺度初始状态随机化。
5. 实现约 25,000 samples/update 的 rollout 聚合。
6. 完成阶段 A 的 std 多 seed 实验。
7. 完成 T=2/T=5 对比。
8. 完成 warm-start 消融。
9. 增加 nominal CTBR 与 rigid-body 双环境模式。
10. 汇总均值、标准差、收敛速度、控制输入和姿态图。

## 8. 退出标准

只有同时满足以下条件，才能认为复现了论文垂直任务的关键效果：

- 至少三个 seed 中多数能够稳定完成赛道。
- 策略明确将机体正 Z 轴转向下方。
- 翻转后持续施加正推力向下加速。
- 平均速度显著高于自由落体局部最优。
- 结论在独立确定性评估中成立，而不是依赖训练噪声。
- 名义环境与刚体环境的结果分别报告，不混合解释。

若策略只能零推力下落，即使 success rate 为 100%，也只能判定为任务跑通，不能判定为复现论文所强调的 AC-MPC 高速翻转行为。
