# Flightmare 编译前准备说明

本文档记录在当前 AC-MPC 复现工程中编译 `D:\MyProjects\flightmare` 前需要准备的工具、依赖、检查命令和注意事项。目标是先把 Flightmare 的 C++/pybind 扩展编译到当前 `acmpc` Python 环境中，使后续可以从 Python 调用 `flightgym.RacingEnv_v1`。

## 当前目标

当前 Flightmare 侧已经完成 `RacingEnv` 代码改造，但尚未完成 C++ 编译和 Python import 验证。下一步目标是：

```text
编译 flightlib Python 扩展
-> import flightgym
-> import flightgym.RacingEnv_v1
-> 通过 reset / step / getState smoke test
```

在 smoke test 通过前，不建议直接进入 Flightmare 训练。

## 必需工具

### 1. CMake

作用：

```text
读取 Flightmare 的 CMakeLists.txt
生成 Windows/MSVC 可编译的工程
驱动 C++ 扩展构建流程
```

建议版本：

```text
CMake >= 3.16
推荐 CMake 3.20+
```

安装时需要勾选：

```text
Add CMake to system PATH
```

检查命令：

```powershell
cmake --version
```

如果 PowerShell 中找不到 `cmake`，重新打开 PowerShell；仍然找不到则需要检查系统 PATH。

### 2. Visual Studio C++ Build Tools

作用：

```text
提供 Windows C++ 编译器 cl.exe
提供链接器和 MSBuild
提供 Windows SDK
真正执行 C++ 编译和链接
```

建议安装：

```text
Visual Studio Build Tools 2022
```

建议勾选组件：

```text
Desktop development with C++
MSVC v143 C++ build tools
Windows 10/11 SDK
C++ CMake tools for Windows
```

检查命令：

```powershell
cl
```

普通 PowerShell 中找不到 `cl` 不一定代表没装，也可能是 MSVC 环境变量未加载。必要时可使用：

```text
x64 Native Tools Command Prompt for VS 2022
```

或者让 CMake 自动寻找 Visual Studio generator。

### 3. Git

作用：

```text
管理 Flightmare 源码和子模块
部分构建流程可能读取 Git 信息
```

检查命令：

```powershell
git --version
```

### 4. Conda / Python 环境

作用：

```text
Flightmare 会编译 Python 扩展模块 flightgym
该扩展必须和当前 Python 版本、ABI、NumPy 环境匹配
```

当前建议使用已有环境：

```text
conda env: acmpc
Python: 3.10
PyTorch: 2.6.0+cu118
CUDA runtime: 11.8
gym: 0.21.0
numpy: 1.26.4
```

检查命令：

```powershell
conda activate acmpc
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

注意：不要让 Flightmare 的 `setup.py` 自动安装旧依赖。

## Python 依赖注意事项

### NumPy

作用：

```text
Python/C++ 数组交互
pybind 扩展编译和运行时依赖
```

建议版本：

```text
numpy 1.26.4
```

不建议当前阶段使用 NumPy 2.x，因为老版 `gym`、本地 SB3 fork 和 Flightmare 老依赖更容易出现兼容问题。

检查命令：

```powershell
python -c "import numpy; print(numpy.__version__)"
```

### PyTorch / CUDA

作用：

```text
AC-MPC 训练和可微 MPC 使用 PyTorch
CUDA 用于训练加速
```

Flightmare C++ 扩展本身通常不依赖 CUDA。当前 CUDA 11.8 与 `torch 2.6.0+cu118` 匹配，不需要为了编译 Flightmare 单独升级 CUDA。

检查命令：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## C++ 库依赖

以下依赖不建议一开始盲目安装。先运行构建，缺什么再根据 CMake 报错处理。

### Eigen

作用：

```text
Flightmare 动力学、矩阵、向量运算
RacingEnv observation/state 也依赖 Eigen 类型
```

Flightmare 通常通过源码或子模块提供 Eigen。如果 CMake 报找不到 Eigen，再单独配置。

### OpenCV

作用：

```text
Flightmare 原工程的相机、图像、渲染相关模块可能依赖 OpenCV
```

虽然当前 `RacingEnv` 不做图像观测，也不添加 Unity gate 可视化，但原始 CMake 可能仍会检查 OpenCV。

常见错误：

```text
Could not find OpenCV
```

出现后再安装或配置 `OpenCV_DIR`。

### ZeroMQ / ZMQ

作用：

```text
Flightmare 与 Unity 通信常用 ZeroMQ
```

当前阶段不需要 Unity gate 可视化，但原工程可能仍编译相关通信模块。

常见错误：

```text
Could not find ZeroMQ
Could not find zmq
```

出现后再安装或配置对应路径。

### yaml-cpp

作用：

```text
Flightmare C++ 环境读取 YAML 配置
例如 racing_env.yaml
```

常见错误：

```text
Could not find yaml-cpp
```

出现后再安装或配置 `yaml-cpp`。

## 环境变量

建议运行时设置：

```powershell
$env:FLIGHTMARE_PATH = "D:\MyProjects\flightmare"
```

作用：

```text
让 Flightmare 运行时找到 configs、resources、Unity 相关路径
```

当前构建脚本会自动设置：

```powershell
$env:FLIGHTMARE_PATH = $FlightmarePath
```

因此通常不需要手动设置。

## 构建命令

不要直接运行：

```powershell
pip install -e .\flightlib
```

原因：

```text
Flightmare 原 setup.py 可能尝试安装旧依赖
例如旧版 gym、stable_baselines、PyOpenGL 等
这可能污染当前已经调通的 acmpc 环境
```

应使用当前仓库提供的构建脚本：

```powershell
cd D:\MyProjects\acmpc_public

powershell -ExecutionPolicy Bypass -File .\scripts\build_flightmare_racing_env.ps1 `
  -FlightmarePath D:\MyProjects\flightmare
```

该脚本内部使用：

```powershell
pip install -e .\flightlib --no-deps -v
```

含义：

```text
editable 安装 flightlib
跳过依赖自动安装
输出详细构建日志
```

## 编译后 smoke test

构建成功后，先运行 smoke test：

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
import flightgym.RacingEnv_v1
构造 RacingEnv_v1
检查 obs_dim == 36
检查 act_dim == 4
检查 state_dim == 13
调用 reset(obs)
调用 getState(state)
调用 step(action, obs, reward, done, extra)
检查 obs/state/reward 均为有限数
```

## 推荐检查顺序

在编译前按顺序执行：

```powershell
cmake --version
git --version
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

然后执行：

```powershell
cd D:\MyProjects\acmpc_public

powershell -ExecutionPolicy Bypass -File .\scripts\build_flightmare_racing_env.ps1 `
  -FlightmarePath D:\MyProjects\flightmare
```

最后执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke_test_flightmare_racing_env.ps1
```

## 当前已知卡点

当前已经遇到过的直接卡点：

```text
CMake is not available on PATH
```

对应处理：

```text
安装 CMake
勾选 Add CMake to system PATH
重新打开 PowerShell
确认 cmake --version 可用
```

如果后续出现 OpenCV、yaml-cpp、ZeroMQ、MSVC 等错误，不要一次性乱改环境。应保留完整报错，再按缺失依赖逐个处理。

## 后续步骤

完成编译和 smoke test 后，下一步才是：

```text
实现 Flightmare Python SB3 wrapper
暴露 reset / step / get_state
复用当前 AC-MPC PPO state plumbing
新增 train_acmpc_flightmare.py
新增 eval_acmpc_flightmare.py
先用简单 horizontal track 验证
再迁移到 split_s / SplitS-like track
```

不要在 `RacingEnv_v1` smoke test 通过前直接进入训练阶段。
