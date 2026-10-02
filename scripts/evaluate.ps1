param(
    [string]$Data = "data/tiny.txt",
    [string]$Checkpoint = "checkpoints/feneco-char-0.1.pt",
    [int]$Batches = 100,
    [int]$Seed = 45
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    & $venvPython -m llm_from_scratch.evaluate --data $Data --checkpoint $Checkpoint --batches $Batches --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "A avaliação terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
