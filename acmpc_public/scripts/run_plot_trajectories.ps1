$ErrorActionPreference = "Stop"

$CondaEnv = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }
$MaxEpisodes = if ($env:MAX_EPISODES) { $env:MAX_EPISODES } else { "64" }

function Get-LatestDirectory($Path) {
    if (-not (Test-Path $Path)) {
        throw "Directory does not exist: $Path"
    }
    $Latest = Get-ChildItem -Path $Path -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $Latest) {
        throw "No child directories found under: $Path"
    }
    return $Latest.FullName
}

if ($env:EVAL_DIR) {
    $EvalDir = $env:EVAL_DIR
} elseif ($env:RUN_DIR) {
    $EvalDir = Get-LatestDirectory (Join-Path $env:RUN_DIR "eval")
} else {
    $LatestRun = Get-LatestDirectory "runs\acmpc_gym"
    $EvalDir = Get-LatestDirectory (Join-Path $LatestRun "eval")
}

$PlotArgs = @(
    "scripts\plot_trajectories.py",
    "--eval-dir", $EvalDir,
    "--max-episodes", $MaxEpisodes
)

if ($env:OUTPUT_DIR) {
    $PlotArgs += @("--output-dir", $env:OUTPUT_DIR)
}

Write-Host "Running: conda run -n $CondaEnv python $($PlotArgs -join ' ')"
conda run -n $CondaEnv python @PlotArgs
