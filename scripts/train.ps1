param(
    [string]$Data = "data/tiny.txt",
    [int]$Steps = 500,
    [int]$ContextLength = 64,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    & $venvPython -m llm_from_scratch.train --data $Data --steps $Steps --context-length $ContextLength --seed $Seed
    if ($LASTEXITCODE -ne 0) { throw "O treino terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
