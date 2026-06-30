param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$FlightmarePath = $(if ($env:FLIGHTMARE_PATH) { $env:FLIGHTMARE_PATH } else { "D:\MyProjects\flightmare" }),
    [string]$RacingConfigPath = $(if ($env:RACING_CONFIG_PATH) { $env:RACING_CONFIG_PATH } else { "" }),
    [int]$AcmPcT = $(if ($env:ACMPC_T) { [int]$env:ACMPC_T } else { 2 }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 200000 }),
    [int]$NEnvs = $(if ($env:N_ENVS) { [int]$env:N_ENVS } else { 8 }),
    [int]$NumThreads = $(if ($env:NUM_THREADS) { [int]$env:NUM_THREADS } else { 8 }),
    [int]$NSteps = $(if ($env:N_STEPS) { [int]$env:N_STEPS } else { 250 }),
    [int]$NEpochs = $(if ($env:N_EPOCHS) { [int]$env:N_EPOCHS } else { 10 }),
    [double]$Gamma = $(if ($env:GAMMA) { [double]$env:GAMMA } else { 0.98 }),
    [double]$GaeLambda = $(if ($env:GAE_LAMBDA) { [double]$env:GAE_LAMBDA } else { 0.95 }),
    [double]$ClipRange = $(if ($env:CLIP_RANGE) { [double]$env:CLIP_RANGE } else { 0.2 }),
    [double]$LearningRateStart = $(if ($env:LR_START) { [double]$env:LR_START } else { 0.0003 }),
    [double]$LearningRateEnd = $(if ($env:LR_END) { [double]$env:LR_END } else { 0.00001 }),
    [double]$EntCoef = $(if ($env:ENT_COEF) { [double]$env:ENT_COEF } else { 0.001 }),
    [double]$VfCoef = $(if ($env:VF_COEF) { [double]$env:VF_COEF } else { 0.5 }),
    [double]$MaxGradNorm = $(if ($env:MAX_GRAD_NORM) { [double]$env:MAX_GRAD_NORM } else { 0.5 }),
    [double]$LogStdInit = $(if ($env:LOG_STD_INIT) { [double]$env:LOG_STD_INIT } else { -1.2 }),
    [int]$CheckpointFreq = $(if ($env:CHECKPOINT_FREQ) { [int]$env:CHECKPOINT_FREQ } else { 25000 }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "auto" }),
    [string]$RunName = $(if ($env:RUN_NAME) { $env:RUN_NAME } else { "" }),
    [int]$BatchSize = $(if ($env:BATCH_SIZE) { [int]$env:BATCH_SIZE } else { 0 }),
    [switch]$NoNormalizeObs = $($env:NO_NORMALIZE_OBS -eq "1"),
    [switch]$NormalizeReward = $($env:NORMALIZE_REWARD -eq "1"),
    [switch]$CudaMemoryLog = $($env:CUDA_MEMORY_LOG -eq "1")
)

$ErrorActionPreference = "Stop"

function Resolve-CondaPython {
    param([string]$EnvName, [string]$EnvPath)

    if ($EnvPath -ne "") {
        $python = Join-Path $EnvPath "python.exe"
        if (-not (Test-Path $python)) {
            throw "Could not find python.exe in CondaEnvPath: $EnvPath"
        }
        return $python
    }
    if ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -eq $EnvName)) {
        $python = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $python) {
            return $python
        }
    }
    $candidateEnv = Join-Path $HOME ".conda\envs\$EnvName"
    $candidatePython = Join-Path $candidateEnv "python.exe"
    if (Test-Path $candidatePython) {
        return $candidatePython
    }
    return ""
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot
$env:FLIGHTMARE_PATH = $FlightmarePath

$pythonArgs = @(
    "scripts\train_acmpc_flightmare.py",
    "--flightmare-path", $FlightmarePath,
    "--acmpc-t", "$AcmPcT",
    "--total-timesteps", "$TotalTimesteps",
    "--n-envs", "$NEnvs",
    "--num-threads", "$NumThreads",
    "--n-steps", "$NSteps",
    "--n-epochs", "$NEpochs",
    "--gamma", "$Gamma",
    "--gae-lambda", "$GaeLambda",
    "--clip-range", "$ClipRange",
    "--learning-rate-start", "$LearningRateStart",
    "--learning-rate-end", "$LearningRateEnd",
    "--ent-coef", "$EntCoef",
    "--vf-coef", "$VfCoef",
    "--max-grad-norm", "$MaxGradNorm",
    "--log-std-init", "$LogStdInit",
    "--checkpoint-freq", "$CheckpointFreq",
    "--device", $Device
)

if ($RacingConfigPath -ne "") { $pythonArgs += @("--racing-config-path", $RacingConfigPath) }
if ($BatchSize -gt 0) { $pythonArgs += @("--batch-size", "$BatchSize") }
if ($RunName -ne "") { $pythonArgs += @("--run-name", $RunName) }
if ($NoNormalizeObs) { $pythonArgs += "--no-normalize-obs" }
if ($NormalizeReward) { $pythonArgs += "--normalize-reward" }
if ($CudaMemoryLog) { $pythonArgs += "--cuda-memory-log" }

Write-Host "Running from: $RepoRoot"
Write-Host "Flightmare path: $FlightmarePath"

$PythonExe = Resolve-CondaPython $CondaEnvName $CondaEnvPath
if ($PythonExe -ne "") {
    Write-Host "Python: $PythonExe"
    Write-Host "Command: $PythonExe $($pythonArgs -join ' ')"
    & $PythonExe @pythonArgs
} else {
    $CondaExe = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
    if (-not $CondaExe) { $CondaExe = "conda" }
    $condaArgs = @("run", "-n", $CondaEnvName, "python") + $pythonArgs
    Write-Host "Conda env: $CondaEnvName"
    Write-Host "Command: conda $($condaArgs -join ' ')"
    & $CondaExe @condaArgs
}
