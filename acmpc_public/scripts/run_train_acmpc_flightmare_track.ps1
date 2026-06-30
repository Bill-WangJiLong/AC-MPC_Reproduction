param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$TrackName,
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$FlightmarePath = $(if ($env:FLIGHTMARE_PATH) { $env:FLIGHTMARE_PATH } else { "D:\MyProjects\flightmare" }),
    [int]$AcmPcT = $(if ($env:ACMPC_T) { [int]$env:ACMPC_T } else { 2 }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 2000000 }),
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
    [switch]$CudaMemoryLog = $($env:CUDA_MEMORY_LOG -eq "1"),
    [switch]$InstallOnly,
    [switch]$SkipInstall,
    [switch]$SkipVerify
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
    $candidatePython = Join-Path (Join-Path $HOME ".conda\envs\$EnvName") "python.exe"
    if (Test-Path $candidatePython) {
        return $candidatePython
    }
    return ""
}

function Invoke-Python {
    param(
        [string]$PythonExe,
        [string]$EnvName,
        [string[]]$Arguments
    )

    if ($PythonExe -ne "") {
        Write-Host "Command: $PythonExe $($Arguments -join ' ')"
        & $PythonExe @Arguments
    } else {
        $condaExe = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
        if (-not $condaExe) { $condaExe = "conda" }
        $condaArgs = @("run", "-n", $EnvName, "python") + $Arguments
        Write-Host "Command: conda $($condaArgs -join ' ')"
        & $condaExe @condaArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
Set-Location $RepoRoot

$TrackJson = Join-Path $RepoRoot "acmpc_racing_gym\tracks\assets\${TrackName}.json"
if (-not (Test-Path $TrackJson)) {
    throw "Gym track asset does not exist: $TrackJson"
}
if (-not (Test-Path $FlightmarePath)) {
    throw "Flightmare path does not exist: $FlightmarePath"
}

try {
    $TrackData = Get-Content -Raw -Encoding UTF8 $TrackJson | ConvertFrom-Json
} catch {
    throw "Could not parse Gym track JSON ${TrackJson}: $($_.Exception.Message)"
}
$DeclaredTrackName = [string]$TrackData.name
if ($DeclaredTrackName -ne $TrackName) {
    throw "Track filename/name mismatch: requested '$TrackName', JSON declares '$DeclaredTrackName'"
}
$GateCount = @($TrackData.gates).Count
if ($GateCount -lt 1) {
    throw "Track '$TrackName' must contain at least one gate"
}
if ($null -eq $TrackData.finish -or @($TrackData.finish.position).Count -ne 3 -or [double]$TrackData.finish.radius -le 0.0) {
    throw "Track '$TrackName' must define finish.position[3] and a positive finish.radius"
}

$env:FLIGHTMARE_PATH = $FlightmarePath
$PythonExe = Resolve-CondaPython $CondaEnvName $CondaEnvPath

Write-Host "Repository: $RepoRoot"
Write-Host "Canonical Gym track: $TrackJson"
Write-Host "Track name: $TrackName"
Write-Host "Gate count: $GateCount"
Write-Host "Finish: position=$($TrackData.finish.position -join ',') radius=$($TrackData.finish.radius)"
Write-Host "Flightmare: $FlightmarePath"

if (-not $SkipInstall) {
    $installArgs = @(
        "scripts\install_gym_track_into_flightmare.py",
        "--flightmare-path", $FlightmarePath,
        "--track-json", $TrackJson
    )
    Invoke-Python $PythonExe $CondaEnvName $installArgs
}

if (-not $SkipVerify) {
    $verifyArgs = @(
        "scripts\verify_flightmare_track_runtime.py",
        "--flightmare-path", $FlightmarePath,
        "--expected-track-name", $TrackName,
        "--expected-gates", "$GateCount"
    )
    Invoke-Python $PythonExe $CondaEnvName $verifyArgs
}

if ($InstallOnly) {
    Write-Host "Track installation and verification completed; training was not started."
    return
}

if ($RunName -eq "") {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "${timestamp}_flightmare_${TrackName}_T${AcmPcT}"
}

$trainParams = @{
    CondaEnvName = $CondaEnvName
    CondaEnvPath = $CondaEnvPath
    FlightmarePath = $FlightmarePath
    AcmPcT = $AcmPcT
    TotalTimesteps = $TotalTimesteps
    NEnvs = $NEnvs
    NumThreads = $NumThreads
    NSteps = $NSteps
    NEpochs = $NEpochs
    Gamma = $Gamma
    GaeLambda = $GaeLambda
    ClipRange = $ClipRange
    LearningRateStart = $LearningRateStart
    LearningRateEnd = $LearningRateEnd
    EntCoef = $EntCoef
    VfCoef = $VfCoef
    MaxGradNorm = $MaxGradNorm
    LogStdInit = $LogStdInit
    CheckpointFreq = $CheckpointFreq
    Device = $Device
    RunName = $RunName
    BatchSize = $BatchSize
    NoNormalizeObs = $NoNormalizeObs
    NormalizeReward = $NormalizeReward
    CudaMemoryLog = $CudaMemoryLog
}

Write-Host "Starting Flightmare training"
Write-Host "Run name: $RunName"
& (Join-Path $ScriptDir "run_train_acmpc_flightmare.ps1") @trainParams
if ($LASTEXITCODE -ne 0) {
    throw "Flightmare training failed with exit code $LASTEXITCODE"
}
