param(
    [string]$CondaEnvName = $(if ($env:CONDA_ENV_NAME) { $env:CONDA_ENV_NAME } else { "acmpc" }),
    [int]$AcmPcT = $(if ($env:ACMPC_T) { [int]$env:ACMPC_T } else { 2 }),
    [int]$MpveHorizon = $(if ($env:MPVE_HORIZON) { [int]$env:MPVE_HORIZON } else { 0 }),
    [double]$MpveCoef = $(if ($env:MPVE_COEF) { [double]$env:MPVE_COEF } else { 1.0 }),
    [int]$TotalTimesteps = $(if ($env:TOTAL_TIMESTEPS) { [int]$env:TOTAL_TIMESTEPS } else { 200000 }),
    [int]$NEnvs = $(if ($env:N_ENVS) { [int]$env:N_ENVS } else { 8 }),
    [int]$NSteps = $(if ($env:N_STEPS) { [int]$env:N_STEPS } else { 250 }),
    [int]$NEpochs = $(if ($env:N_EPOCHS) { [int]$env:N_EPOCHS } else { 10 }),
    [int]$BatchSize = $(if ($env:BATCH_SIZE) { [int]$env:BATCH_SIZE } else { 0 }),
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
    [int]$MaxEpisodeSteps = $(if ($env:MAX_EPISODE_STEPS) { [int]$env:MAX_EPISODE_STEPS } else { 500 }),
    [int]$Seed = $(if ($env:SEED) { [int]$env:SEED } else { 0 }),
    [string]$Device = $(if ($env:DEVICE) { $env:DEVICE } else { "auto" }),
    [string]$RunName = $(if ($env:RUN_NAME) { $env:RUN_NAME } else { "" }),
    [string]$OutputRoot = $(if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { "runs\acmpc_gym_mpve" }),
    [int]$PlotWindow = $(if ($env:WINDOW) { [int]$env:WINDOW } else { 50 }),
    [int]$PlotDpi = $(if ($env:DPI) { [int]$env:DPI } else { 160 }),
    [int]$EvalEpisodes = $(if ($env:EVAL_EPISODES) { [int]$env:EVAL_EPISODES } else { 32 }),
    [int]$EvalSeed = $(if ($env:EVAL_SEED) { [int]$env:EVAL_SEED } else { 1000 }),
    [ValidateSet("config", "true", "false")]
    [string]$EvalRandomReset = $(if ($env:EVAL_RANDOM_RESET) { $env:EVAL_RANDOM_RESET } else { "config" }),
    [int]$TrajectoryMaxEpisodes = $(if ($env:MAX_EPISODES) { [int]$env:MAX_EPISODES } else { 32 }),
    [switch]$SingleGate = $($env:SINGLE_GATE -eq "1"),
    [switch]$NoNormalizeObs = $($env:NO_NORMALIZE_OBS -eq "1"),
    [switch]$NormalizeReward = $($env:NORMALIZE_REWARD -eq "1"),
    [switch]$NoMpveBootstrap = $($env:NO_MPVE_BOOTSTRAP -eq "1"),
    [switch]$NoMpveValidMask = $($env:NO_MPVE_VALID_MASK -eq "1"),
    [switch]$CudaMemoryLog = $($env:CUDA_MEMORY_LOG -eq "1"),
    [switch]$SkipTraining = $($env:SKIP_TRAINING -eq "1"),
    [switch]$SkipEvaluation = $($env:SKIP_EVALUATION -eq "1"),
    [switch]$SkipPlots = $($env:SKIP_PLOTS -eq "1")
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

if ($MpveHorizon -le 0) {
    $MpveHorizon = $AcmPcT
}

if ($RunName -eq "") {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "${Timestamp}_horizontal_T${AcmPcT}_mpveH${MpveHorizon}"
    if ($SingleGate) {
        $RunName = "${RunName}_single_gate"
    }
}

$RunDir = Join-Path $OutputRoot $RunName
$EvalDir = Join-Path $RunDir ("eval\mpve_horizontal_seed{0}" -f $EvalSeed)

Write-Host "Repository: $RepoRoot"
Write-Host "Conda env: $CondaEnvName"
Write-Host "Run directory: $RunDir"
Write-Host "Track: horizontal"
Write-Host "ACMPC_T: $AcmPcT"
Write-Host "MPVE horizon: $MpveHorizon"
Write-Host "MPVE coef: $MpveCoef"

if (-not $SkipTraining) {
    $TrainArgs = @(
        "run", "-n", $CondaEnvName, "python", "scripts\train_acmpc_gym_mpve.py",
        "--track-name", "horizontal",
        "--run-name", $RunName,
        "--output-dir", $OutputRoot,
        "--acmpc-t", "$AcmPcT",
        "--mpve-horizon", "$MpveHorizon",
        "--mpve-coef", "$MpveCoef",
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
        "--max-episode-steps", "$MaxEpisodeSteps",
        "--seed", "$Seed",
        "--checkpoint-freq", "$CheckpointFreq",
        "--device", $Device
    )

    if ($BatchSize -gt 0) {
        $TrainArgs += @("--batch-size", "$BatchSize")
    }
    if ($SingleGate) {
        $TrainArgs += "--single-gate"
    }
    if ($NoNormalizeObs) {
        $TrainArgs += "--no-normalize-obs"
    }
    if ($NormalizeReward) {
        $TrainArgs += "--normalize-reward"
    }
    if ($NoMpveBootstrap) {
        $TrainArgs += "--no-mpve-bootstrap"
    }
    if ($NoMpveValidMask) {
        $TrainArgs += "--no-mpve-valid-mask"
    }
    if ($CudaMemoryLog) {
        $TrainArgs += "--cuda-memory-log"
    }

    Write-Host "Training command: conda $($TrainArgs -join ' ')"
    & conda @TrainArgs
} else {
    Write-Host "Skipping training because SkipTraining is set."
}

if (-not $SkipPlots) {
    $TrainingPlotArgs = @(
        "run", "-n", $CondaEnvName, "python", "scripts\plot_training_metrics.py",
        "--run-dir", $RunDir,
        "--window", "$PlotWindow",
        "--dpi", "$PlotDpi"
    )
    Write-Host "Training plot command: conda $($TrainingPlotArgs -join ' ')"
    & conda @TrainingPlotArgs
} else {
    Write-Host "Skipping training plots because SkipPlots is set."
}

if (-not $SkipEvaluation) {
    if (Test-Path $EvalDir) {
        $EvalDir = Join-Path $RunDir ("eval\mpve_horizontal_seed{0}_{1}" -f $EvalSeed, (Get-Date -Format "yyyyMMdd_HHmmss"))
    }

    $EvalArgs = @(
        "run", "-n", $CondaEnvName, "python", "scripts\eval_acmpc_gym.py",
        "--run-dir", $RunDir,
        "--output-dir", $EvalDir,
        "--model-class", "mpve",
        "--track-name", "horizontal",
        "--episodes", "$EvalEpisodes",
        "--seed", "$EvalSeed",
        "--device", $Device,
        "--acmpc-t", "$AcmPcT",
        "--random-reset", $EvalRandomReset
    )

    Write-Host "Evaluation command: conda $($EvalArgs -join ' ')"
    & conda @EvalArgs

    if (-not $SkipPlots) {
        $TrajectoryPlotArgs = @(
            "run", "-n", $CondaEnvName, "python", "scripts\plot_trajectories.py",
            "--eval-dir", $EvalDir,
            "--max-episodes", "$TrajectoryMaxEpisodes"
        )
        Write-Host "Trajectory plot command: conda $($TrajectoryPlotArgs -join ' ')"
        & conda @TrajectoryPlotArgs
    }
} else {
    Write-Host "Skipping evaluation because SkipEvaluation is set."
}

Write-Host "Done."
Write-Host "Training plots: $(Join-Path $RunDir 'plots')"
if (-not $SkipEvaluation) {
    Write-Host "Evaluation outputs: $EvalDir"
}
