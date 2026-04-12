param(
    [string]$Model,
    [string]$RunName = "",
    [double]$Duration = 20,
    [double]$ArrivalRate = 10,
    [int]$BatchSize = 16,
    [int]$MaxConcurrent = 48,
    [int]$ChunkSize = 4,
    [double]$AcceptRate = 0.85,
    [int]$AvgPromptLen = 50,
    [int]$AvgMaxTokens = 200,
    [string]$WorkloadMode = "mixed",
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
    "--device", $Device,
    "--dtype", $DType
)

if ($RunName -ne "") {
    $args += @("--run-name", $RunName)
}

python @args
