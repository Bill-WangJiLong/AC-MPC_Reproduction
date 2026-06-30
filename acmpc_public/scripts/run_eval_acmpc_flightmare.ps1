param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$FlightmarePath = $(if ($env:FLIGHTMARE_PATH) { $env:FLIGHTMARE_PATH } else { "D:\MyProjects\flightmare" }),
    [string]$RunDir = $(if ($env:RUN_DIR) { $env:RUN_DIR } else { "" }),
    [string]$ModelPath = $(if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "" }),
    [string]$VecNormalizePath = $(if ($env:VECNORMALIZE_PATH) { $env:VECNORMALIZE_PATH } else { "" }),
    [string]$RacingConfigPath = $(if ($env:RACING_CONFIG_PATH) { $env:RACING_CONFIG_PATH } else { "" }),
    [string]$OutputDir = $(if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "" }),
    [int]$Episodes = $(if ($env:EPISODES) { [int]$env:EPISODES } else { 32 }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 1000 }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "auto" }),
    [int]$AcmPcT = $(if ($env:ACMPC_T) { [int]$env:ACMPC_T } else { 0 }),
    [int]$MaxEpisodeSteps = $(if ($env:MAX_EPISODE_STEPS) { [int]$env:MAX_EPISODE_STEPS } else { 0 }),
    [int]$PlotMaxEpisodes = $(if ($env:PLOT_MAX_EPISODES) { [int]$env:PLOT_MAX_EPISODES } else { 64 }),
    [double]$SpeedVMin = $(if ($env:SPEED_VMIN) { [double]$env:SPEED_VMIN } else { 0.0 }),
    [double]$SpeedVMax = $(if ($env:SPEED_VMAX) { [double]$env:SPEED_VMAX } else { -1.0 }),
    [switch]$Stochastic = $($env:STOCHASTIC -eq "1"),
    [switch]$AllowMissingVecNormalize = $($env:ALLOW_MISSING_VECNORMALIZE -eq "1"),
    [switch]$NoPlots = $($env:NO_PLOTS -eq "1")
)

$ErrorActionPreference = "Stop"

function Resolve-CondaPython {
    param([string]$EnvName, [string]$EnvPath)
    if ($EnvPath -ne "") {
        $python = Join-Path $EnvPath "python.exe"
        if (-not (Test-Path $python)) { throw "Could not find python.exe in CondaEnvPath: $EnvPath" }
        return $python
    }
    if ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -eq $EnvName)) {
        $python = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $python) { return $python }
    }
    $candidatePython = Join-Path (Join-Path $HOME ".conda\envs\$EnvName") "python.exe"
    if (Test-Path $candidatePython) { return $candidatePython }
    return ""
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot
$env:FLIGHTMARE_PATH = $FlightmarePath

$pythonArgs = @(
    "scripts\eval_acmpc_flightmare.py",
    "--flightmare-path", $FlightmarePath,
    "--episodes", "$Episodes",
    "--seed", "$Seed",
    "--device", $Device
)

if ($RunDir -ne "") { $pythonArgs += @("--run-dir", $RunDir) } else { $pythonArgs += "--latest" }
if ($ModelPath -ne "") { $pythonArgs += @("--model-path", $ModelPath) }
if ($VecNormalizePath -ne "") { $pythonArgs += @("--vecnormalize-path", $VecNormalizePath) }
if ($RacingConfigPath -ne "") { $pythonArgs += @("--racing-config-path", $RacingConfigPath) }
if ($OutputDir -ne "") { $pythonArgs += @("--output-dir", $OutputDir) }
if ($AcmPcT -gt 0) { $pythonArgs += @("--acmpc-t", "$AcmPcT") }
if ($MaxEpisodeSteps -gt 0) { $pythonArgs += @("--max-episode-steps", "$MaxEpisodeSteps") }
$pythonArgs += @("--plot-max-episodes", "$PlotMaxEpisodes", "--speed-vmin", "$SpeedVMin")
if ($SpeedVMax -gt 0) { $pythonArgs += @("--speed-vmax", "$SpeedVMax") }
if ($Stochastic) { $pythonArgs += "--stochastic" }
if ($AllowMissingVecNormalize) { $pythonArgs += "--allow-missing-vecnormalize" }
if ($NoPlots) { $pythonArgs += "--no-plots" }

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
