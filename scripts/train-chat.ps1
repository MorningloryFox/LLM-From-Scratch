param(
    [string]$Data = "data/conversations/pilot/conversations",
    [string]$BaseCheckpoint = "checkpoints/feneco-token-livros-0.1.pt",
    [int]$Steps = 500,
    [int]$Seed = 42,
    [string]$Checkpoint = "checkpoints/feneco-chat-oasst1-token-0.1.pt"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    & $venvPython -m llm_from_scratch.train_chat_token --data-dir $Data --base-checkpoint $BaseCheckpoint --steps $Steps --seed $Seed --checkpoint $Checkpoint
    if ($LASTEXITCODE -ne 0) { throw "O ajuste de conversa terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
