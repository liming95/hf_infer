param(
    [string]$Model,
    [string]$RunName = "",
    [double]$Duration = 20,
    [int]$TargetCompletedRequests = 0,
    [double]$ArrivalRate = 10,
    [int]$BatchSize = 16,
    [int]$MaxConcurrent = 48,
    [int]$ChunkSize = 4,
    [double]$AcceptRate = 0.85,
    [int]$AvgPromptLen = 50,
    [int]$AvgMaxTokens = 200,
    [string]$WorkloadMode = "mixed",
    [int]$RolloutPullBatchSize = 8,
    [int]$RolloutPullTargetOutstanding = 16,
    [string]$Device = "cuda",
    [string]$DType = "float16"
)

$args = @(
    "sim\run_hf_real_pipeline.py",
    "--use-real-compute",
    "--model", $Model,
    "--duration", $Duration,
    "--arrival-rate", $ArrivalRate,
    "--batch-size", $BatchSize,
    "--max-concurrent", $MaxConcurrent,
    "--chunk-size", $ChunkSize,
    "--accept-rate", $AcceptRate,
    "--avg-prompt-len", $AvgPromptLen,
    "--avg-max-tokens", $AvgMaxTokens,
    "--workload-mode", $WorkloadMode,
    "--rollout-pull-batch-size", $RolloutPullBatchSize,
    "--rollout-pull-target-outstanding", $RolloutPullTargetOutstanding,
    "--device", $Device,
    "--dtype", $DType
)

if ($RunName -ne "") {
    $args += @("--run-name", $RunName)
}

if ($TargetCompletedRequests -gt 0) {
    $args += @("--target-completed-requests", $TargetCompletedRequests)
}

python @args
