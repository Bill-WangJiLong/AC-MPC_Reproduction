param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$EvalDir = $(if ($env:EVAL_DIR) { $env:EVAL_DIR } else { "" }),
    [string]$OutputDir = $(if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "" }),
    [int]$MaxEpisodes = $(if ($env:MAX_EPISODES) { [int]$env:MAX_EPISODES } else { 64 }),
    [double]$SpeedVMin = $(if ($env:SPEED_VMIN) { [double]$env:SPEED_VMIN } else { 0.0 }),
    [double]$SpeedVMax = $(if ($env:SPEED_VMAX) { [double]$env:SPEED_VMAX } else { -1.0 }),
    [switch]$Show = $($env:SHOW -eq "1")
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

if ($EvalDir -eq "") {
    throw "Set EVAL_DIR to an evaluation output folder, e.g. <run-dir>\eval\<timestamp>."
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

$pythonArgs = @(
    "scripts\plot_trajectories.py",
    "--eval-dir", $EvalDir,
    "--max-episodes", "$MaxEpisodes",
    "--speed-vmin", "$SpeedVMin"
)

if ($OutputDir -ne "") { $pythonArgs += @("--output-dir", $OutputDir) }
if ($SpeedVMax -gt 0) { $pythonArgs += @("--speed-vmax", "$SpeedVMax") }
if ($Show) { $pythonArgs += "--show" }

Write-Host "Running from: $RepoRoot"
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
