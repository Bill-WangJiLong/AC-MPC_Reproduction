# Gym 与 Flightmare 通用赛道工作流

## 1. 设计原则

项目使用 Gym 赛道 JSON 作为唯一赛道数据源：

```text
acmpc_racing_gym/tracks/assets/<track_name>.json
```

同一份 JSON 同时服务于两类环境：

- Python Gym 直接通过赛道名称加载 JSON。
- Flightmare 通过通用安装器将 JSON 转换为 YAML，并让 `RacingEnv_v1` 加载该 YAML。

以后新增赛道只需要增加一个 JSON 文件，不需要新增 Python 或 PowerShell 脚本，也不需要重新编译 Flightmare。

通用 Flightmare 入口为：

```text
scripts/run_train_acmpc_flightmare_track.ps1
```

## 2. 新增赛道 JSON

在以下目录创建赛道文件：

```text
D:\MyProjects\acmpc_public\acmpc_racing_gym\tracks\assets\my_track.json
```

示例：

```json
{
  "name": "my_track",
  "start": {
    "position": [0.0, 0.0, 2.0],
    "yaw": 0.0
  },
  "finish": {
    "position": [4.0, 0.0, 2.0],
    "radius": 0.5
  },
  "world_bounds": [
    [-10.0, 10.0],
    [-10.0, 10.0],
    [0.0, 12.0]
  ],
  "gates": [
    {
      "center": [2.0, 0.0, 2.0],
      "normal": [1.0, 0.0, 0.0],
      "up": [0.0, 0.0, 1.0],
      "width": 1.5,
      "height": 1.5,
      "frame_thickness": 0.12
    }
  ]
}
```

字段约束：

- 文件名必须与 `name` 一致，例如 `my_track.json` 对应 `"name": "my_track"`。
- `start.position` 是无人机初始世界坐标 `[x, y, z]`。
- `start.yaw` 是初始偏航角，单位为弧度。
- `finish.position` 是终点球心的世界坐标 `[x, y, z]`，必须位于最后一个门之后。
- `finish.radius` 是终点球半径，单位为米，必须大于 0。
- `world_bounds` 依次表示 x、y、z 三轴允许范围；起点和所有门必须位于边界内。
- `gates` 数组顺序就是必须完成的穿门顺序。
- `center` 是门中心的世界坐标。
- `normal` 是允许穿门的正方向。穿越检测要求轨迹从法向量反侧运动到正侧。
- `up` 用于确定门平面内的朝向，不能与 `normal` 平行。
- `width`、`height` 和 `frame_thickness` 的单位均为米。
- `label` 是可选字段，不参与控制、奖励或碰撞计算。

例如，从上向下穿越水平门时可使用：

```json
{
  "normal": [0.0, 0.0, -1.0],
  "up": [0.0, 1.0, 0.0]
}
```

## 3. 在 Gym 中验证和可视化

把 JSON 放入 `tracks/assets` 后，Gym 已经可以按文件名自动加载，无需额外安装。

首先生成赛道图：

```powershell
cd D:\MyProjects\acmpc_public

C:\Users\王纪龙\.conda\envs\acmpc\python.exe `
  .\scripts\plot_gym_tracks.py `
  --track-name my_track
```

输出目录：

```text
runs/track_visualizations/
```

检查内容：

- 起点是否位于边界内。
- 门的排列顺序是否正确。
- 法向箭头是否指向预期穿越方向。
- `up` 是否产生了正确的门姿态。
- 门之间是否留有可飞行空间。
- 终点球是否位于最后一个门的正向一侧，并且完全处于可飞行边界内。

当前结束逻辑为：

1. 穿过最后一个门时只获得 `gate_pass_reward=+10`，并进入终点阶段，episode 不立即结束。
2. 进入终点阶段后，控制目标和 36 维观测中的未来门几何切换为终点球位置的代理门。
3. 无人机位置进入终点球，或一个仿真步的运动线段与终点球相交时，获得 `finish_reward=+10` 并结束 episode。
4. 终点检测使用线段与球体相交判定，避免高速飞行时一步跨过球体而漏检。

## 4. 启动 Gym 训练

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_gym.ps1 `
  -TrackName my_track `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -AcmPcT 2 `
  -TotalTimesteps 2000000
```

`train_acmpc_gym.py` 会动态枚举 `tracks/assets/*.json`，因此新增赛道后不需要修改赛道白名单。

只检查 JSON 文件名、内部名称和门数量，不启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_gym.ps1 `
  -TrackName my_track `
  -ValidateOnly
```

## 5. 安装到 Flightmare，但不启动训练

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName my_track `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -FlightmarePath "D:\MyProjects\flightmare" `
  -InstallOnly
```

该命令会：

1. 读取 Gym JSON，并检查文件名、`name`、门数量和终点球参数。
2. 调用 `install_gym_track_into_flightmare.py` 验证赛道结构。
3. 生成 `D:\MyProjects\flightmare\flightlib\configs\tracks\my_track.yaml`。
4. 更新 Flightmare 的 `flightlib/configs/racing_env.yaml`，令其指向新 YAML。
5. 创建原始配置备份；已有备份不会被覆盖。
6. 使用编译后的 `RacingEnv_v1` 验证 `reset()`、`getState()`、观测门角点和 `step()`。
7. 检查接口维度为 observation 36、action 4、state 13。

赛道 YAML 是运行时配置，所以新增或修改赛道后不需要重新编译 `flightgym`。

## 6. 启动 Flightmare 训练

安装、验证并使用指定赛道启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName my_track `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -FlightmarePath "D:\MyProjects\flightmare"
```

指定常用训练参数：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName my_track `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -FlightmarePath "D:\MyProjects\flightmare" `
  -AcmPcT 2 `
  -TotalTimesteps 2000000 `
  -NEnvs 8 `
  -NumThreads 8 `
  -NSteps 250 `
  -BatchSize 2000 `
  -NEpochs 10 `
  -CudaMemoryLog
```

如果赛道已经安装并且活动 `racing_env.yaml` 未被切换，可以跳过转换：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName my_track `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -SkipInstall
```

此时运行时验证仍会执行，若当前 Flightmare 活动赛道不是 `my_track`，脚本会停止而不是在错误赛道上训练。只有在明确不需要检查时才使用 `-SkipVerify`。

## 7. 切换已有赛道

无需修改配置文件，直接替换 `TrackName`：

```powershell
# 垂直赛道
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc"

# Split-S 赛道
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName split_s `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc"
```

Flightmare 当前只有一个活动 `racing_env.yaml`。不要在训练进程运行期间安装或切换赛道，否则活动配置与该次运行记录可能不一致。

## 8. 文件职责

```text
acmpc_racing_gym/tracks/assets/<name>.json
  唯一赛道源；供 Gym 直接加载。

scripts/run_train_acmpc_flightmare_track.ps1
  面向用户的统一入口；安装、验证并按需启动训练。

scripts/run_train_acmpc_gym.ps1
  Gym 通用入口；验证赛道并按名称启动 Gym PPO 训练。

scripts/install_gym_track_into_flightmare.py
  JSON 到 Flightmare YAML 的通用转换器。

scripts/verify_flightmare_track_runtime.py
  验证 C++ RacingEnv_v1 实际加载的赛道与接口。

scripts/run_train_acmpc_flightmare.ps1
  不关心赛道的底层 Flightmare PPO 训练入口。

D:/MyProjects/flightmare/flightlib/configs/tracks/<name>.yaml
  自动生成的 Flightmare 运行时赛道文件，不应作为主数据源手动维护。
```

## 9. 当前垂直赛道示例

只安装并验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc" `
  -InstallOnly
```

启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath "C:\Users\王纪龙\.conda\envs\acmpc"
```
