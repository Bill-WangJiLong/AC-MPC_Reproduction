param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [string]$CondaEnvPath = $(if ($env:CONDA_ENV_PATH) { $env:CONDA_ENV_PATH } else { "" }),
    [string]$FlightmarePath = $(if ($env:FLIGHTMARE_PATH) { $env:FLIGHTMARE_PATH } else { "D:\MyProjects\flightmare" }),
    [string]$CMakeExe = $(if ($env:CMAKE_EXE) { $env:CMAKE_EXE } else { "" })
)

$ErrorActionPreference = "Stop"

function Resolve-CMakeExe {
    param([string]$ExplicitCMakeExe)

    if ($ExplicitCMakeExe -ne "") {
        if (-not (Test-Path $ExplicitCMakeExe)) {
            throw "CMakeExe does not exist: $ExplicitCMakeExe"
        }
        return (Resolve-Path $ExplicitCMakeExe).Path
    }

    $programFilesX86 = ${env:ProgramFiles(x86)}
    if ($programFilesX86) {
        $vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vsInstallPath = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
            if ($vsInstallPath) {
                $vsCMake = Join-Path $vsInstallPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
                if (Test-Path $vsCMake) {
                    return $vsCMake
                }
            }
        }
    }

    $pathCMake = Get-Command cmake -ErrorAction SilentlyContinue
    if ($pathCMake) {
        return $pathCMake.Source
    }

    throw "CMake is not available. Install CMake or pass -CMakeExe."
}

$env:FLIGHTMARE_PATH = $FlightmarePath

$ResolvedCMake = Resolve-CMakeExe $CMakeExe
$ResolvedCMakeBin = Split-Path -Parent $ResolvedCMake
$env:Path = "$ResolvedCMakeBin;$env:Path"

Write-Host "CMake: $ResolvedCMake"
& $ResolvedCMake --version

Set-Location $FlightmarePath

$pipArgs = @(
    "-m", "pip", "install",
    "-e", ".\flightlib",
    "--no-deps",
    "-v"
)

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
    Write-Host "Command: $PythonExe $($pipArgs -join ' ')"
    & $PythonExe @pipArgs
} else {
    $condaArgs = @("run", "-n", $CondaEnvName, "python") + $pipArgs
    Write-Host "Conda env: $CondaEnvName"
    Write-Host "Command: conda $($condaArgs -join ' ')"
    & conda @condaArgs
}
