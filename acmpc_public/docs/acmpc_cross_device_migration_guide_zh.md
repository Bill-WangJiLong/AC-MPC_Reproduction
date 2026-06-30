# AC-MPC 仿真工程跨设备迁移指南

## 1. 文档目的与适用范围

本文说明如何把当前已经调通的 AC-MPC 工程从一台 Windows 设备迁移到另一台 Windows x64 设备，并恢复以下能力：

- Python Gym 竞速环境训练、评估和绘图；
- AC-MPC 可微 MPC 与 PPO 训练；
- MPVE 扩展训练；
- 修改版 Flightmare `RacingEnv_v1` 的编译与调用；
- Flightmare 赛道安装、训练、评估和轨迹可视化；
- 已训练 checkpoint、`VecNormalize` 统计量和训练日志的继续使用。

本文以当前已验证环境为基准：

| 项目 | 当前基准 |
|---|---|
| 操作系统 | Windows x64 |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu118 |
| CUDA runtime | 11.8，由 PyTorch wheel 提供 |
| NumPy | 1.26.4 |
| Gym | 0.21.0 |
| Stable-Baselines3 | 作者 fork 1.7.0a1，加本项目 state plumbing 修改 |
| mpc.pytorch | 作者仓库子模块，加本项目兼容性修改 |
| 编译器 | Visual Studio Build Tools 2022 / MSVC v143 |
| Flightmare 构建模式 | Windows headless，不包含 Unity/OpenCV bridge |

Linux、WSL、macOS 以及 Unity 图形版 Flightmare 不在本迁移流程的验证范围内。

## 2. 最重要的迁移原则

当前工程不是两个上游仓库的原始版本，不能只在新设备上重新执行 `git clone`。

必须迁移两套带有本地修改的源码：

```text
acmpc_public
flightmare
```

其中：

- `acmpc_public` 包含 Gym 环境、训练脚本、评估脚本、修改后的 SB3 和 mpc.pytorch；
- `flightmare` 包含新增的 C++ `RacingEnv`、pybind 接口、赛道配置、headless 构建修改及最新终点判定逻辑；
- 当前两个目录都有未提交或未跟踪的源码，直接克隆上游仓库会丢失这些内容；
- Conda 环境目录不要直接复制，新设备应重新创建环境；
- `flightgym.cp310-win_amd64.pyd` 不应作为跨设备正式安装产物，新设备应重新编译。

推荐的新设备目录结构如下，盘符可以不同：

```text
<PROJECT_ROOT>\
├── acmpc_public\
└── flightmare\
```

后续命令假定在 PowerShell 中先设置：

```powershell
$AcmPcRoot = "D:\MyProjects\acmpc_public"
$FlightmareRoot = "D:\MyProjects\flightmare"
$CondaEnvPath = "$env:USERPROFILE\.conda\envs\acmpc"
```

如果新设备目录不同，只修改这三个变量，不要批量修改源码中的默认路径。

### 2.1 一体化 Git 仓库

为避免两个原始仓库、两个 AC-MPC 子模块以及未提交修改在迁移时丢失，当前工程提供一个一体化源码仓库：

```text
acmpc_reproduction_repo/
├── acmpc_public/
└── flightmare/
```

该仓库中的 `mpc.pytorch`、`stable-baselines3` 和 `flightmare` 都是包含当前修改的普通源码目录，不依赖目标设备再次拉取上游子模块。不要在 clone 后用 `git submodule update --init --recursive` 覆盖这些目录。

仓库不包含：

```text
acmpc_public/runs/
临时文件和绘图缓存
Python __pycache__
Flightmare build/externals 构建缓存
旧设备生成的 flightgym *.pyd
```

训练结果需要按第 3.3 节单独迁移，Flightmare `.pyd` 需要按第 7 节在目标设备重新编译。

### 2.2 从远程 Git 仓库 clone

将本地一体化仓库推送到自己的 GitHub、GitLab 或其他私有 Git 服务后，在目标设备执行：

```powershell
Set-Location D:\MyProjects
git clone <你的远程仓库URL> acmpc_reproduction_repo
Set-Location .\acmpc_reproduction_repo

$AcmPcRoot = (Resolve-Path .\acmpc_public).Path
$FlightmareRoot = (Resolve-Path .\flightmare).Path
$CondaEnvPath = "$env:USERPROFILE\.conda\envs\acmpc"
```

检查两个源码目录：

```powershell
Test-Path "$AcmPcRoot\training_modules\mlp_mpc_policy.py"
Test-Path "$AcmPcRoot\stable-baselines3\stable_baselines3"
Test-Path "$AcmPcRoot\mpc.pytorch\mpc"
Test-Path "$FlightmareRoot\flightlib\src\envs\racing_env\racing_env.cpp"
```

四项都应返回 `True`。之后从第 4 节开始安装目标设备依赖。

### 2.3 没有远程服务器时使用 Git bundle

源设备可以把完整 Git 历史导出为单文件：

```powershell
git -C D:\MyProjects\acmpc_reproduction_repo bundle create `
  D:\MyProjects\acmpc_reproduction_repo.bundle --all
```

把 `acmpc_reproduction_repo.bundle` 复制到移动硬盘或目标设备，然后使用标准 `git clone`：

```powershell
Set-Location D:\MyProjects
git clone E:\Transfer\acmpc_reproduction_repo.bundle acmpc_reproduction_repo
```

bundle 包含已经提交的源码和 Git 历史，但同样不包含被 `.gitignore` 排除的训练结果。

### 2.4 clone 后更新源码

如果配置了远程仓库，后续更新使用：

```powershell
Set-Location D:\MyProjects\acmpc_reproduction_repo
git pull --ff-only
```

拉取后如果 Flightmare C++、pybind、CMake 或 `setup.py` 有变化，应重新执行第 7 节编译。仅赛道 JSON/YAML 变化时，不需要重新编译，重新安装目标赛道即可。

## 3. 迁移前需要备份的内容

### 3.1 必须迁移的 AC-MPC 内容

最稳妥的做法是复制整个 `acmpc_public` 目录。至少必须包含：

```text
acmpc_flightmare/
acmpc_racing_gym/
diff_mpc_drones/
training_modules/
mpc.pytorch/
stable-baselines3/
scripts/
docs/
.gitmodules
README.md
```

`mpc.pytorch` 和 `stable-baselines3` 不能用上游版本替换，因为当前训练流程依赖本地修改。

### 3.2 必须迁移的 Flightmare 内容

复制整个修改后的 `flightmare` 源码目录。尤其要确认以下内容存在：

```text
flightlib/include/flightlib/envs/racing_env/
flightlib/src/envs/racing_env/
flightlib/configs/racing_env.yaml
flightlib/configs/tracks/
flightlib/src/wrapper/pybind_wrapper.cpp
flightlib/CMakeLists.txt
flightlib/setup.py
flightlib/cmake/pybind11_download.cmake
```

此外还包含 headless 构建、动力学和通用 VecEnv 接口的修改，因此不能只复制 `racing_env` 文件夹。

### 3.3 已训练模型和日志

如果需要保留训练结果，复制所需的完整 run 目录：

```text
runs/acmpc_gym/<run_name>/
runs/acmpc_flightmare/<run_name>/
```

每个可复用 run 至少保留：

```text
final_model.zip                 # 最终策略和价值网络
vecnormalize.pkl                # 观测归一化统计量，通常必须与模型配套
config.json                     # 训练参数和赛道快照
flightmare_vec_env.yaml         # Flightmare 向量环境参数，仅 Flightmare run
checkpoints/                    # 中间 checkpoint，可选但建议保留
csv/                            # episode 日志
sb3/                            # TensorBoard / SB3 日志
plots/                          # 已生成图片，可重新生成
```

只迁移 `final_model.zip` 而漏掉 `vecnormalize.pkl`，会改变模型收到的 36 维观测，评估结果可能明显错误。

### 3.4 可以不迁移的生成文件

以下内容不是恢复工程所必需，可不复制以节省空间：

```text
**/__pycache__/
tmp/
tmp_*/
tmp_pdf_pages/
output/                         # 仅在不需要其中导出结果时
flightmare/flightlib/build/
flightmare/flightlib/externals/
flightmare/flightlib/*.pyd
```

`flightlib/build`、`externals` 和 `.pyd` 会在目标设备重新生成。

### 3.5 建议记录模型校验值

在源设备上可为关键模型记录 SHA-256：

```powershell
$RunDir = "D:\MyProjects\acmpc_public\runs\acmpc_flightmare\<run_name>"
Get-FileHash "$RunDir\final_model.zip", "$RunDir\vecnormalize.pkl" -Algorithm SHA256
```

在新设备执行相同命令，校验哈希一致后再评估。

## 4. 目标设备前置软件

### 4.1 必需软件

目标设备安装：

1. Git for Windows；
2. Miniconda 或 Anaconda；
3. Visual Studio Build Tools 2022；
4. NVIDIA 驱动，训练时需要；
5. PowerShell 5.1 或更高版本。

Visual Studio Build Tools 2022 安装器中勾选：

```text
Desktop development with C++
MSVC v143 C++ x64/x86 build tools
Windows 10 SDK 或 Windows 11 SDK
C++ CMake tools for Windows
```

使用 2022 是因为当前工程已经用 MSVC v143 验证，且构建脚本会通过 `vswhere.exe` 优先寻找其自带 CMake。无需因为系统上存在更高年份的 Visual Studio 就更换已验证工具链。

### 4.2 工具检查

```powershell
git --version
conda --version
cmake --version
```

普通 PowerShell 中无法直接识别 `cl` 不等价于 Build Tools 未安装。可用以下命令检查 v143 组件：

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
  -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
  -property installationPath
```

只要能返回 Build Tools 安装目录，项目构建脚本就可以继续寻找 MSVC 和 VS 自带 CMake。

### 4.3 CUDA 说明

- Flightmare 的当前 headless C++ 扩展本身不依赖 CUDA；
- AC-MPC、PPO 和可微 MPC 使用 PyTorch CUDA；
- 使用 `torch==2.6.0+cu118` 时通常不需要单独安装完整 CUDA Toolkit；
- 必须安装足够新的 NVIDIA 驱动，使其能够运行 CUDA 11.8 runtime；
- 目标 GPU 显存不同不会影响模型文件加载，但可能需要调整 `n_envs`、batch size 或关闭 CUDA memory log。

## 5. 在目标设备重建 Conda 环境

### 5.1 创建 Python 3.10 环境

不要使用 Python 3.14。Flightmare 扩展文件名中的 `cp310` 表明当前接口按 CPython 3.10 构建。

```powershell
conda create -n acmpc python=3.10.20 -y
```

如果 PowerShell 的 `conda activate` 受执行策略影响，可以全程使用 `conda run`，不必先解决激活问题。

### 5.2 固定构建工具版本

Gym 0.21 与较新的 setuptools 构建规则不兼容，先固定当前已验证版本：

```powershell
conda run -n acmpc python -m pip install `
  pip==23.2.1 setuptools==65.5.0 wheel==0.38.4
```

### 5.3 安装 PyTorch CUDA 11.8

```powershell
conda run -n acmpc python -m pip install `
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 `
  --index-url https://download.pytorch.org/whl/cu118
```

如果目标设备没有 NVIDIA GPU，也可以安装 CPU 版 PyTorch完成接口测试，但训练性能会很低；不要把 CPU 环境与当前 GPU 实验结果直接比较。

### 5.4 安装基础 Python 依赖

```powershell
conda run -n acmpc python -m pip install numpy==1.26.4
conda run -n acmpc python -m pip install gym==0.21.0 --no-build-isolation
conda run -n acmpc python -m pip install `
  cloudpickle==3.1.2 pandas==2.3.3 matplotlib==3.10.9 scipy==1.15.3 `
  setproctitle==1.3.7 tensorboard==2.20.0 ruamel.yaml==0.19.1 `
  pyglet==2.1.14 PyOpenGL==3.1.10
```

Gym 启动时会显示项目停止维护的提示，这是旧版 Gym 的已知提示，不代表当前 AC-MPC 运行失败。当前阶段不要擅自升级到 Gym 0.26 或 Gymnasium，因为作者 SB3 fork 使用 Gym 0.21 API。

### 5.5 安装本地修改版 mpc.pytorch 和 SB3

```powershell
Set-Location $AcmPcRoot

conda run -n acmpc python -m pip install -e .\mpc.pytorch --no-deps
conda run -n acmpc python -m pip install -e .\stable-baselines3 --no-deps
```

必须使用迁移过来的本地目录。不要执行从 PyPI 或原始 GitHub 仓库安装同名包，否则会覆盖 state plumbing、MPVE 和兼容性修改。

### 5.6 Python 环境检查

```powershell
conda run -n acmpc python --version
conda run -n acmpc python -c "import gym, numpy, torch; print(gym.__version__, numpy.__version__, torch.__version__, torch.version.cuda, torch.cuda.is_available())"
conda run -n acmpc python -c "import mpc, stable_baselines3; print(mpc.__file__); print(stable_baselines3.__file__)"
```

期望关键输出：

```text
Python 3.10.x
gym 0.21.0
numpy 1.26.4
torch 2.6.0+cu118
mpc 指向 <AcmPcRoot>\mpc.pytorch
stable_baselines3 指向 <AcmPcRoot>\stable-baselines3
```

## 6. 验证 AC-MPC 和 Python Gym

进入主仓库：

```powershell
Set-Location $AcmPcRoot
```

依次执行：

```powershell
conda run -n acmpc python .\scripts\smoke_test_acmpc_forward.py
conda run -n acmpc python .\scripts\validate_acmpc_core.py
conda run -n acmpc python .\scripts\smoke_test_racing_gym.py
```

验收条件：

- `import mpc`、`import drone` 成功；
- `MlpMpcPolicy` 可以初始化；
- MPC 输出无 NaN/Inf；
- batch size 1、8、64 测试通过；
- Gym 随机 rollout、穿门、碰撞、终点和同一步末门/终点回归测试通过；
- 观测维度为 36，动作维度为 4，额外 MPC state 为 13。

`drone.py` 中 `.T`、`torch.lu`、`lu_solve` 和 `uint8` 索引可能产生 deprecation warning。在固定 PyTorch 2.6.0 时这些 warning 不代表测试失败；应以进程退出码、NaN 检查和 smoke test 结果为准。

## 7. 在目标设备重新编译 Flightmare

### 7.1 为什么必须重新编译

`flightgym.cp310-win_amd64.pyd` 与以下因素相关：

- 操作系统和 CPU 架构；
- CPython ABI，本项目为 CPython 3.10；
- MSVC runtime；
- 当前 Flightmare C++ 源码和 pybind 接口。

即使两台设备都是 Windows x64，也建议从迁移后的源码重新构建，避免旧 `.pyd` 与目标 Python 或 MSVC runtime 不一致。

### 7.2 执行构建

```powershell
Set-Location $AcmPcRoot

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_flightmare_racing_env.ps1 `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot
```

该脚本会：

- 设置 `FLIGHTMARE_PATH`；
- 通过 `vswhere` 优先选择 VS Build Tools 自带 CMake；
- 执行 `pip install -e .\flightlib --no-deps -v`；
- 关闭 tests、Unity bridge 和 Unity bridge tests；
- 重新编译 C++/pybind 扩展；
- 生成新的 `flightlib\flightgym.cp310-win_amd64.pyd`。

不要直接执行不带 `--no-deps` 的 Flightmare `pip install -e`，其原始依赖声明包含旧 Gym、旧 stable_baselines 和图形依赖，可能污染已经调通的 `acmpc` 环境。

### 7.3 Flightmare smoke test

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_smoke_test_flightmare_racing_env.ps1 `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot `
  -Steps 5
```

期望输出包含：

```text
RacingEnv_v1 smoke test passed
obs_dim: 36
act_dim: 4
state_dim: 13
```

当前已验证接口还会返回 `gate_index`、`gate_passed`、`finish_phase`、`finish_distance`、`finished`、`collision`、`speed` 和位置等 extra info。

## 8. 迁移和选择赛道

Gym 侧 JSON 是赛道的规范源：

```text
acmpc_racing_gym/tracks/assets/horizontal.json
acmpc_racing_gym/tracks/assets/vertical.json
acmpc_racing_gym/tracks/assets/split_s.json
acmpc_racing_gym/tracks/assets/race_loop_like.json
```

把指定赛道安装到 Flightmare 并验证，但不启动训练：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot `
  -InstallOnly
```

该过程会把 Gym JSON 转成 Flightmare YAML，并更新当前 `racing_env.yaml` 所使用的赛道。只改变 JSON/YAML 赛道参数时不需要重新编译 `.pyd`；修改 C++ 环境、动力学或 pybind 接口后才需要重新编译。

训练和评估前都应确认当前安装的是目标赛道。不要只依赖旧 run 的 `config.json`，因为其中可能保留源设备绝对路径或旧版赛道参数。

## 9. 训练启动验证

### 9.1 Gym 训练

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_gym.ps1 `
  -TrackName horizontal `
  -CondaEnvPath $CondaEnvPath `
  -TotalTimesteps 200000 `
  -NEnvs 8
```

### 9.2 Flightmare 训练

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot `
  -TotalTimesteps 2000000 `
  -NEnvs 8 `
  -NumThreads 8
```

首次迁移验收建议先把 `TotalTimesteps` 临时设为 2000，只验证一次 PPO rollout/update、checkpoint 和日志写入；通过后再启动正式训练。短测试只能证明流程可运行，不能用于判断策略收敛。

## 10. 已训练模型的评估与绘图

### 10.1 Flightmare 模型评估

先安装该模型对应的赛道，再评估：

```powershell
$RunDir = "$AcmPcRoot\runs\acmpc_flightmare\<run_name>"

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_train_acmpc_flightmare_track.ps1 `
  -TrackName vertical `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot `
  -InstallOnly

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_eval_acmpc_flightmare.ps1 `
  -RunDir $RunDir `
  -CondaEnvPath $CondaEnvPath `
  -FlightmarePath $FlightmareRoot `
  -Episodes 32
```

评估脚本会读取 `final_model.zip` 和 `vecnormalize.pkl`，输出 success rate、collision rate、速度、飞行时间和轨迹文件，并绘制带速度色图的轨迹。

### 10.2 训练曲线

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_plot_flightmare_training_metrics.ps1 `
  -RunDir $RunDir `
  -CondaEnvPath $CondaEnvPath
```

### 10.3 历史模型与新版环境的区别

本项目后续修复过以下环境行为：

- 同一步通过最后一个门并到达终点时的漏判；
- gate reward 与 finish reward 的同一步累计；
- 垂直赛道终点位置/半径设置。

因此，旧模型在迁移后的最新环境中可以评估，但这属于“旧策略在新版环境上的评估”，不等于精确恢复其训练时的历史环境。严谨对比时应记录源码版本、赛道 JSON/YAML、`config.json` 和评估环境版本。

## 11. 路径迁移注意事项

当前脚本的默认 Flightmare 路径是：

```text
D:\MyProjects\flightmare
```

但主要 PowerShell 入口都支持 `-FlightmarePath`，因此新设备无需保持相同盘符。推荐每次显式传入：

```powershell
-FlightmarePath $FlightmareRoot
```

也可以在当前 PowerShell 会话设置：

```powershell
$env:FLIGHTMARE_PATH = $FlightmareRoot
$env:CONDA_ENV_PATH = $CondaEnvPath
```

`training_modules/mlp_mpc_policy.py` 中的 `DRONE_PATH` 已按文件相对位置计算，不依赖旧设备的绝对工程路径。

旧 run 的 `config.json` 会保留源设备绝对路径，这是实验记录，不应直接全局替换。运行新评估命令时通过 `-FlightmarePath`、`-RunDir` 等参数覆盖实际路径。

## 12. 常见迁移故障

### 12.1 `conda activate acmpc` 失败

直接使用：

```powershell
conda run -n acmpc python --version
```

或给项目脚本传入：

```powershell
-CondaEnvPath $CondaEnvPath
```

这样不依赖 PowerShell profile 和 `conda init`。

### 12.2 实际 Python 是 3.14

说明命令使用了系统 Python，而不是 `acmpc` 环境。检查：

```powershell
& "$CondaEnvPath\python.exe" --version
```

Flightmare 必须由这个 Python 3.10 解释器构建。

### 12.3 Gym 0.21 安装失败

确认先固定：

```text
pip 23.2.1
setuptools 65.5.0
wheel 0.38.4
```

然后使用：

```powershell
conda run -n acmpc python -m pip install gym==0.21.0 --no-build-isolation
```

### 12.4 `import stable_baselines3` 不是本地 fork

检查模块路径：

```powershell
conda run -n acmpc python -c "import stable_baselines3; print(stable_baselines3.__file__)"
```

必须指向迁移后的 `acmpc_public\stable-baselines3`。否则重新执行本地 editable 安装。

### 12.5 Flightmare 编译使用了系统 CMake 4.x

旧 Flightmare CMake 配置可能与 CMake 4.x 不兼容。项目构建脚本会优先选择 VS Build Tools 自带 CMake；不要绕过脚本直接手工调用 `pip install`。

### 12.6 `import flightgym` 失败或 `.pyd` 无法加载

检查：

1. Python 是否为 3.10 x64；
2. 是否使用迁移后的 Flightmare 源码重新编译；
3. VS Build Tools 2022 和 Windows SDK 是否安装；
4. `flightgym.cp310-win_amd64.pyd` 的生成时间是否为目标机本次构建时间；
5. editable 安装是否由 `$CondaEnvPath\python.exe` 执行。

### 12.7 CUDA 不可用

```powershell
conda run -n acmpc python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

若 `torch.version.cuda` 为 `11.8` 但 `is_available()` 为 false，优先检查 NVIDIA 驱动、设备管理器、GPU 权限和是否误安装 CPU 版 torch。

### 12.8 模型能加载但评估表现异常

按顺序检查：

1. 是否加载了该 run 的 `vecnormalize.pkl`；
2. 当前 Flightmare 是否安装了正确赛道；
3. `ACMPC_T` 是否与训练时一致；
4. 观测模式和 36 维顺序是否一致；
5. Flightmare C++ 源码是否为当前修改版；
6. 是否在用确定性评估；
7. 旧模型是否正在新版终点逻辑下评估。

## 13. 最终验收清单

迁移完成后逐项确认：

- [ ] 新设备同时存在完整的 `acmpc_public` 和修改版 `flightmare`；
- [ ] Python 为 3.10.x，而不是系统 Python 3.14；
- [ ] `torch==2.6.0+cu118` 且 `torch.cuda.is_available()` 为 true；
- [ ] `gym==0.21.0`、`numpy==1.26.4`；
- [ ] `mpc` 和 `stable_baselines3` 指向迁移后的本地源码；
- [ ] AC-MPC core smoke test 通过；
- [ ] Gym smoke test 通过；
- [ ] Flightmare 在目标机重新生成 `.pyd`；
- [ ] `RacingEnv_v1` smoke test 通过，维度为 36/4/13；
- [ ] 目标赛道已由 Gym JSON 安装并验证到 Flightmare；
- [ ] 一次短 PPO rollout/update 可以完成并写出日志；
- [ ] 迁移模型与对应 `vecnormalize.pkl` 同时存在；
- [ ] 固定 seed 的确定性评估能够输出指标和轨迹图；
- [ ] 关键模型文件 SHA-256 与源设备一致。

通过以上检查后，再启动长时间正式训练。不要把“可以 import”作为迁移完成标准，最终标准应包括一次实际环境 step、一次 AC-MPC forward、一次 PPO update 和一次模型确定性评估。
