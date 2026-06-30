$ErrorActionPreference = "Stop"

$CondaEnv = if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }
$Episodes = if ($env:EPISODES) { $env:EPISODES } else { "32" }
$Seed = if ($env:SEED) { $env:SEED } else { "1000" }
$Device = if ($env:DEVICE) { $env:DEVICE } else { "auto" }

$EvalArgs = @(
    "scripts\eval_acmpc_gym.py",
    "--episodes", $Episodes,
    "--seed", $Seed,
    "--device", $Device
)

if ($env:RUN_DIR) {
    $EvalArgs += @("--run-dir", $env:RUN_DIR)
} else {
    $EvalArgs += "--latest"
}

if ($env:MODEL_PATH) {
    $EvalArgs += @("--model-path", $env:MODEL_PATH)
}

if ($env:VECNORMALIZE_PATH) {
    $EvalArgs += @("--vecnormalize-path", $env:VECNORMALIZE_PATH)
}

if ($env:OUTPUT_DIR) {
    $EvalArgs += @("--output-dir", $env:OUTPUT_DIR)
}

if ($env:ACMPC_T) {
    $EvalArgs += @("--acmpc-t", $env:ACMPC_T)
}

if ($env:TRACK_NAME) {
    $EvalArgs += @("--track-name", $env:TRACK_NAME)
}

if ($env:TRACK_PATH) {
    $EvalArgs += @("--track-path", $env:TRACK_PATH)
}

if ($env:MAX_EPISODE_STEPS) {
    $EvalArgs += @("--max-episode-steps", $env:MAX_EPISODE_STEPS)
}

if ($env:RANDOM_RESET) {
    $EvalArgs += @("--random-reset", $env:RANDOM_RESET)
}

if ($env:STOCHASTIC -eq "1") {
    $EvalArgs += "--stochastic"
}

if ($env:ALLOW_MISSING_VECNORMALIZE -eq "1") {
    $EvalArgs += "--allow-missing-vecnormalize"
}

Write-Host "Running: conda run -n $CondaEnv python $($EvalArgs -join ' ')"
conda run -n $CondaEnv python @EvalArgs
