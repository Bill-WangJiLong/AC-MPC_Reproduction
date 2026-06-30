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

if ($RunDir -eq "") {
    $runsRoot = Join-Path $RepoRoot "runs\acmpc_flightmare"
    if (-not (Test-Path $runsRoot)) {
        throw "No Flightmare run root found: $runsRoot"
    }
    $RunDir = (Get-ChildItem -Path $runsRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    if (-not $RunDir) {
        throw "No Flightmare runs found under: $runsRoot"
    }
}

$pythonArgs = @(
    "scripts\plot_training_metrics.py",
    "--run-dir", $RunDir,
    "--window", "$Window",
    "--dpi", "$Dpi",
    "--x-axis", "$XAxis",
    "--paper-step-samples", "$PaperStepSamples"
)

if ($OutputDir -ne "") { $pythonArgs += @("--output-dir", $OutputDir) }
if ($Show) { $pythonArgs += "--show" }

Write-Host "Running from: $RepoRoot"
Write-Host "Run dir: $RunDir"
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
