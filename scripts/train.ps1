param(
    [string]$Data = "data/tiny.txt",
    [int]$Steps = 500,
    [int]$ContextLength = 32,
    [int]$Seed = 42,
    [string]$Checkpoint = "checkpoints/feneco-token-0.1.pt"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    & $venvPython -m llm_from_scratch.train_token --data $Data --steps $Steps --context-length $ContextLength --seed $Seed --checkpoint $Checkpoint
    if ($LASTEXITCODE -ne 0) { throw "O treino terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
