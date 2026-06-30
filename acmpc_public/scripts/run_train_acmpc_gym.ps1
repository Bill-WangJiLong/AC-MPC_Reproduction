param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$TrackName = $(if ($env:TRACK_NAME) { $env:TRACK_NAME } else { "horizontal" }),
    [int]$AcmPcT = $(if ($env:ACMPC_T) { [int]$env:ACMPC_T } else { 2 }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 200000 }),
    [int]$NEnvs = $(if ($env:N_ENVS) { [int]$env:N_ENVS } else { 8 }),
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
    [switch]$SingleGate = $($env:SINGLE_GATE -eq "1"),
    [switch]$NoNormalizeObs = $($env:NO_NORMALIZE_OBS -eq "1"),
    [switch]$NormalizeReward = $($env:NORMALIZE_REWARD -eq "1"),
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$TrackJson = Join-Path $RepoRoot "acmpc_racing_gym\tracks\assets\${TrackName}.json"
if (-not (Test-Path $TrackJson)) {
    throw "Gym track asset does not exist: $TrackJson"
}
try {
    $TrackData = Get-Content -Raw -Encoding UTF8 $TrackJson | ConvertFrom-Json
} catch {
    throw "Could not parse Gym track JSON ${TrackJson}: $($_.Exception.Message)"
}
if ([string]$TrackData.name -ne $TrackName) {
    throw "Track filename/name mismatch: requested '$TrackName', JSON declares '$($TrackData.name)'"
}
$GateCount = @($TrackData.gates).Count
if ($GateCount -lt 1) {
    throw "Track '$TrackName' must contain at least one gate"
}
if ($null -eq $TrackData.finish -or @($TrackData.finish.position).Count -ne 3 -or [double]$TrackData.finish.radius -le 0.0) {
    throw "Track '$TrackName' must define finish.position[3] and a positive finish.radius"
}

Write-Host "Running from: $RepoRoot"
Write-Host "Canonical Gym track: $TrackJson"
Write-Host "Track name: $TrackName"
Write-Host "Gate count: $GateCount"
Write-Host "Finish: position=$($TrackData.finish.position -join ',') radius=$($TrackData.finish.radius)"

if ($ValidateOnly) {
    Write-Host "Gym track validation completed; training was not started."
    return
}

$argsList = @(
    "scripts\train_acmpc_gym.py",
    "--track-name", $TrackName,
    "--acmpc-t", "$AcmPcT",
    "--total-timesteps", "$TotalTimesteps",
    "--n-envs", "$NEnvs",
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

if ($BatchSize -gt 0) {
    $argsList += @("--batch-size", "$BatchSize")
}

if ($RunName -ne "") {
    $argsList += @("--run-name", $RunName)
}

if ($SingleGate) {
    $argsList += "--single-gate"
}

if ($NoNormalizeObs) {
    $argsList += "--no-normalize-obs"
}

if ($NormalizeReward) {
    $argsList += "--normalize-reward"
}

if ($CondaEnvPath -ne "") {
    $PythonExe = Join-Path $CondaEnvPath "python.exe"
    if (-not (Test-Path $PythonExe)) {
        throw "Could not find python.exe in CondaEnvPath: $CondaEnvPath"
    }
    Write-Host "Python: $PythonExe"
    Write-Host "Command: $PythonExe $($argsList -join ' ')"
    & $PythonExe @argsList
} elseif ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -eq $CondaEnvName)) {
    $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    Write-Host "Python: $PythonExe"
    Write-Host "Command: $PythonExe $($argsList -join ' ')"
    & $PythonExe @argsList
} else {
    $candidatePython = Join-Path (Join-Path $HOME ".conda\envs\$CondaEnvName") "python.exe"
    if (Test-Path $candidatePython) {
        Write-Host "Python: $candidatePython"
        Write-Host "Command: $candidatePython $($argsList -join ' ')"
        & $candidatePython @argsList
    } else {
        $CondaExe = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
        if (-not $CondaExe) { $CondaExe = "conda" }
        $condaArgs = @("run", "-n", $CondaEnvName, "python") + $argsList
        Write-Host "Command: conda $($condaArgs -join ' ')"
        & $CondaExe @condaArgs
    }
}
if ($LASTEXITCODE -ne 0) {
    throw "Gym training failed with exit code $LASTEXITCODE"
}
