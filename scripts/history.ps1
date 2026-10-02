$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    & $venvPython -m llm_from_scratch.history
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível exibir o histórico ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
