param(
    [string]$Output = "data/conversations/pilot",
    [string[]]$Sources = @("oasst"),
    [int]$PtCorpusRows = 2000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$previousPythonUtf8 = $env:PYTHONUTF8
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    $env:PYTHONUTF8 = "1"
    $arguments = @("-m", "llm_from_scratch.prepare_chat_data", "--output", $Output, "--sources") + $Sources + @("--pt-corpus-rows", "$PtCorpusRows")
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "A preparação terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
    if ($null -eq $previousPythonUtf8) {
        Remove-Item Env:PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONUTF8 = $previousPythonUtf8
    }
}
