param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$RunDir = $(if ($env:RUN_DIR) { $env:RUN_DIR } else { "" }),
    [string]$OutputDir = $(if ($env:OUTPUT_DIR) { $env:OUTPUT_DIR } else { "" }),
    [int]$Window = $(if ($env:WINDOW) { [int]$env:WINDOW } else { 50 }),
    [int]$Dpi = $(if ($env:DPI) { [int]$env:DPI } else { 160 }),
    [ValidateSet("paper-step", "global-timesteps", "updates")]
    [string]$XAxis = $(if ($env:X_AXIS) { $env:X_AXIS } else { "paper-step" }),
    [double]$PaperStepSamples = $(if ($env:PAPER_STEP_SAMPLES) { [double]$env:PAPER_STEP_SAMPLES } else { 25000 }),
    [switch]$Show = $($env:SHOW -eq "1")
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

if ($CondaEnvPath -eq "") {
    if ($env:CONDA_PREFIX -and ((Split-Path -Leaf $env:CONDA_PREFIX) -eq $CondaEnvName)) {
        $candidatePython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $candidatePython) {
            $CondaEnvPath = $env:CONDA_PREFIX
        }
    }
}

if ($CondaEnvPath -eq "") {
    $candidateEnv = Join-Path $HOME ".conda\envs\$CondaEnvName"
    $candidatePython = Join-Path $candidateEnv "python.exe"
    if (Test-Path $candidatePython) {
        $CondaEnvPath = $candidateEnv
    }
}

$pythonArgs = @(
    "scripts\plot_training_metrics.py",
    "--window", "$Window",
    "--dpi", "$Dpi",
    "--x-axis", "$XAxis",
    "--paper-step-samples", "$PaperStepSamples"
)

if ($RunDir -ne "") {
    $pythonArgs += @("--run-dir", $RunDir)
} else {
    $pythonArgs += "--latest"
}

if ($OutputDir -ne "") {
    $pythonArgs += @("--output-dir", $OutputDir)
}

if ($Show) {
    $pythonArgs += "--show"
}

Write-Host "Running from: $RepoRoot"
if ($CondaEnvPath -ne "") {
    Write-Host "Conda env path: $CondaEnvPath"
} else {
    Write-Host "Conda env: $CondaEnvName"
}
Write-Host "X axis: $XAxis"
if ($XAxis -eq "paper-step") {
    Write-Host "Paper step samples: $PaperStepSamples"
}

if ($CondaEnvPath -ne "") {
    $PythonExe = Join-Path $CondaEnvPath "python.exe"
    if (-not (Test-Path $PythonExe)) {
        throw "Could not find python.exe in CondaEnvPath: $CondaEnvPath"
    }
    Write-Host "Command: $PythonExe $($pythonArgs -join ' ')"
    & $PythonExe @pythonArgs
} else {
    $CondaExe = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
    if (-not $CondaExe) {
        $CondaExe = "conda"
    }
    $condaArgs = @("run", "-n", $CondaEnvName, "python") + $pythonArgs
    Write-Host "Command: conda $($condaArgs -join ' ')"
    & $CondaExe @condaArgs
}
