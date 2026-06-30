# Flightmare RacingEnv 编译记录

本文档记录本次在 Windows 上编译修改版 Flightmare `RacingEnv_v1` 的实际操作、遇到的问题、处理方式和验证结果。目标是把 `flightgym` Python 扩展编译出来，并验证 `RacingEnv_v1.reset()`、`step()`、`getState()` 可以被 Python 调用。

## 基本环境

- AC-MPC 仓库：`D:\MyProjects\acmpc_public`
- Flightmare 仓库：`D:\MyProjects\flightmare`
- Python 环境：`C:\Users\王纪龙\.conda\envs\acmpc`
- Python 版本：3.10.20
- 编译器：Visual Studio Build Tools 2022 / MSVC 19.44
- CMake：优先使用 VS Build Tools 自带 CMake 3.31.6
- 构建脚本：`D:\MyProjects\acmpc_public\scripts\build_flightmare_racing_env.ps1`
- Smoke test 脚本：`D:\MyProjects\acmpc_public\scripts\run_smoke_test_flightmare_racing_env.ps1`

## 安全约束

本次没有执行以下危险操作：

- 没有执行 `git reset`、`git checkout --`、`git clean`。
- 没有删除 Flightmare 源码目录。
- 没有安装 conda OpenCV 包，因为 dry-run 显示会大范围替换 Python、OpenSSL、libffi 等核心包。
- 没有修改系统级 PATH、注册表或 Visual Studio 安装。

需要注意：Flightmare 原始 `flightlib/setup.py` 在构建时会清理 `flightlib/externals` 和 `flightlib/build` 下的构建缓存，这是原项目已有行为。本次未额外清理源码文件。

## 实际命令

编译命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_flightmare_racing_env.ps1 `
  -CondaEnvPath "$env:USERPROFILE\.conda\envs\acmpc" `
  -FlightmarePath D:\MyProjects\flightmare
```

Smoke test 命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_smoke_test_flightmare_racing_env.ps1 `
  -CondaEnvPath "$env:USERPROFILE\.conda\envs\acmpc" `
  -FlightmarePath D:\MyProjects\flightmare `
  -Steps 5
```

## 构建过程与处理

### 1. CMake 版本问题

之前系统 CMake 4.x 会拒绝 Flightmare 的旧 `cmake_minimum_required`。本次使用构建脚本优先选择 VS Build Tools 自带 CMake：

```text
CMake: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\...\cmake.exe
cmake version 3.31.6-msvc6
```

处理方式：继续使用 `build_flightmare_racing_env.ps1` 中的 VS CMake 优先逻辑。

### 2. pybind11 下载问题

Flightmare 原始 `pybind11_download.cmake` 使用：

```cmake
GIT_TAG master
```

这会拉取当前 pybind11 master，既不稳定也不可复现。处理方式：

```cmake
GIT_TAG           v2.10.4
GIT_SHALLOW       TRUE
```

影响文件：

- `D:\MyProjects\flightmare\flightlib\cmake\pybind11_download.cmake`

### 3. OpenCV 缺失问题

首次构建进入 CMake 配置后失败：

```text
Could not find a package configuration file provided by "OpenCV"
```

我检查了当前 conda 环境和磁盘，没有找到 `OpenCVConfig.cmake`。随后做了：

```powershell
conda install -p "$env:USERPROFILE\.conda\envs\acmpc" -c conda-forge libopencv --dry-run
```

dry-run 显示会下载约 385 MB 依赖，并替换 Python、OpenSSL、libffi 等核心包。因此没有执行安装。

处理方式：改为 headless 构建路线，不构建 Unity/OpenCV bridge。原因是当前目标只要求 Python `RacingEnv_v1` 训练接口，不需要 Unity gate 可视化。

### 4. Headless Flightmare 构建

修改 `D:\MyProjects\flightmare\flightlib\CMakeLists.txt`：

- 默认关闭 `BUILD_TESTS`。
- 新增并默认关闭 `BUILD_UNITY_BRIDGE`。
- 默认关闭 `BUILD_UNITY_BRIDGE_TESTS`。
- 只有启用 `BUILD_UNITY_BRIDGE` 时才查找 OpenCV。
- headless 模式下不编译 `src/bridges/*.cpp`、`rgb_camera.cpp`、`unity_camera.cpp`。
- MSVC 下不使用 GCC/Clang 专用优化参数。
- MSVC 下不链接 `stdc++fs`。
- 启用 `CMP0091` 并统一 MSVC 运行库为动态 CRT。

同时调整：

- `quadrotor.hpp`：用前向声明替代强制包含 `rgb_camera.hpp`。
- `vec_env.hpp/.cpp`：headless 下隔离 UnityBridge 调用。
- `quadrotor_env.hpp/.cpp`：headless 下隔离 UnityBridge 调用。
- `racing_env.hpp/.cpp`：headless 下隔离 UnityBridge 调用。
- `unity_message_types.hpp`：删除无实际使用的 OpenCV include。

### 5. gtest 缓存问题

即使 CMake 默认关闭 tests，旧 `CMakeCache.txt` 里仍有：

```text
BUILD_TESTS=ON
BUILD_UNITY_BRIDGE_TESTS=ON
```

处理方式：修改 `flightlib/setup.py`，让 Python editable install 显式传入：

```text
-DBUILD_TESTS=OFF
-DBUILD_UNITY_BRIDGE=OFF
-DBUILD_UNITY_BRIDGE_TESTS=OFF
```

并保留扩展接口：

```text
FLIGHTMARE_CMAKE_ARGS
```

### 6. MSVC 运行库不一致

链接阶段曾出现：

```text
RuntimeLibrary mismatch: MD_DynamicRelease vs MT_StaticRelease
```

原因：`yaml-cpp` 使用 `/MD`，而 pybind 模块被生成为 `/MT`。处理方式：在顶层 CMake 设置：

```cmake
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
```

之后链接通过。

## 最终结果

最终构建成功：

```text
Successfully installed flightgym
flightgym.vcxproj -> D:\MyProjects\flightmare\flightlib\flightgym.cp310-win_amd64.pyd
```

生成产物：

```text
D:\MyProjects\flightmare\flightlib\flightgym.cp310-win_amd64.pyd
```

产物大小约 416 KB。

## Smoke Test 结果

Smoke test 通过：

```text
RacingEnv_v1 smoke test passed
n_envs: 100
obs_dim: 36
act_dim: 4
state_dim: 13
extra_info: ['collision_code', 'gate_index', 'out_of_bounds', 'finished', 'gate_passed', 'collision', 'speed', 'timeout', 'x', 'y', 'z']
last_done: [False, ...]
```

验证内容：

- `from flightgym import RacingEnv_v1` 成功。
- `RacingEnv_v1()` 创建成功。
- `reset(obs)` 成功。
- `step(action, obs, reward, done, extra)` 成功。
- `getState(state)` 成功。
- 观测维度为 36。
- 动作为 4 维。
- 状态为 13 维。
- forward 过程中没有 NaN。

## 构建日志

本次产生的日志位于：

```text
D:\MyProjects\acmpc_public\runs\flightmare_build_logs
```

关键日志：

- `build_20260625_235310.log`：OpenCV 缺失。
- `build_20260626_000612.log`：headless 后进入 gtest 缓存问题。
- `build_20260626_001045.log`：进入 C++ 编译，暴露 Unity/OpenCV 残余 include。
- `build_20260626_001456.log`：暴露 MSVC 运行库不一致。
- `build_20260626_001815.log`：最终编译成功。
- `smoke_20260626_002101.log`：Smoke test 通过。

## 当前 Flightmare 改动范围

主要改动文件：

- `flightlib/CMakeLists.txt`
- `flightlib/setup.py`
- `flightlib/cmake/pybind11_download.cmake`
- `flightlib/include/flightlib/bridges/unity_message_types.hpp`
- `flightlib/include/flightlib/envs/env_base.hpp`
- `flightlib/src/envs/env_base.cpp`
- `flightlib/include/flightlib/envs/vec_env.hpp`
- `flightlib/src/envs/vec_env.cpp`
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp`
- `flightlib/src/envs/quadrotor_env/quadrotor_env.cpp`
- `flightlib/include/flightlib/objects/quadrotor.hpp`
- `flightlib/src/objects/quadrotor.cpp`
- `flightlib/src/dynamics/quadrotor_dynamics.cpp`
- `flightlib/src/wrapper/pybind_wrapper.cpp`
- `flightlib/configs/racing_env.yaml`
- `flightlib/include/flightlib/envs/racing_env/racing_env.hpp`
- `flightlib/src/envs/racing_env/racing_env.cpp`

新增/生成文件：

- `flightlib/flightgym.cp310-win_amd64.pyd`

## 当前限制

- 当前构建是 headless 版本，不包含 Unity bridge、OpenCV 相机、Unity gate 可视化。
- `connectUnity()` 在 headless 构建下会返回 false。
- 当前验证只覆盖 Python API smoke test，尚未接入 PPO 训练循环。
- 构建日志里仍有 MSVC warning，例如数值截断、`-fPIC` 被 MSVC 忽略。这些没有阻止构建，也不影响当前 smoke test。

## 后续建议

1. 下一步先写 Flightmare 版 Python wrapper，使其接口对齐当前 Python Gym wrapper：
   - `reset`
   - `step`
   - `get_state`
   - `obs_dim=36`
   - `action_dim=4`
   - `state_dim=13`

2. 再写一个短 rollout 脚本，随机动作跑 1 到 2 个 episode，记录：
   - reward
   - done
   - gate_index
   - collision
   - position
   - speed

3. 最后再把 PPO 训练脚本从 Python Gym env 切换到 Flightmare env。

4. 如果以后需要 Unity 可视化，再单独恢复 `BUILD_UNITY_BRIDGE=ON`，并安装 C++ OpenCV、ZMQ、ZMQPP 依赖。这个不应和当前 headless 训练接口混在一起做。
