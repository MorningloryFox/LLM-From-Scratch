param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Não foi possível criar o ambiente Python." }
    }
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível atualizar o pip." }
    & $venvPython -m pip install -e .
    if ($LASTEXITCODE -ne 0) { throw "Não foi possível instalar o projeto e suas dependências." }
    Write-Host "Ambiente local preparado. Próximo passo: .\scripts\train.ps1"
}
finally {
    Pop-Location
}
