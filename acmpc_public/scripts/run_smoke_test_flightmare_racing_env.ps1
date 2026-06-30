param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$FlightmarePath = $(if ($env:FLIGHTMARE_PATH) { $env:FLIGHTMARE_PATH } else { "D:\MyProjects\flightmare" }),
    [int]$Steps = $(if ($env:STEPS) { [int]$env:STEPS } else { 5 })
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")
Set-Location $RepoRoot
$env:FLIGHTMARE_PATH = $FlightmarePath

$pythonArgs = @(
    "scripts\smoke_test_flightmare_racing_env.py",
    "--flightmare-path", $FlightmarePath,
    "--steps", "$Steps"
)

Write-Host "Repository: $RepoRoot"
Write-Host "Flightmare path: $FlightmarePath"

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

if ($CondaEnvPath -ne "") {
    $PythonExe = Join-Path $CondaEnvPath "python.exe"
    if (-not (Test-Path $PythonExe)) {
        throw "Could not find python.exe in CondaEnvPath: $CondaEnvPath"
    }
    Write-Host "Conda env path: $CondaEnvPath"
    Write-Host "Command: $PythonExe $($pythonArgs -join ' ')"
    & $PythonExe @pythonArgs
} else {
    $condaArgs = @("run", "-n", $CondaEnvName, "python") + $pythonArgs
    Write-Host "Conda env: $CondaEnvName"
    Write-Host "Command: conda $($condaArgs -join ' ')"
    & conda @condaArgs
}
