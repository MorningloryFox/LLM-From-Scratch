param(
    [string]$Path = "data",
    [switch]$Overwrite,
    [switch]$DeletePdf
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    $arguments = @("-m", "llm_from_scratch.extract_pdfs", "--path", $Path)
    if ($Overwrite) { $arguments += "--overwrite" }
    if ($DeletePdf) { $arguments += "--delete-pdf" }
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "A extração terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
