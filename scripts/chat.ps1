param(
    [string]$Checkpoint = "checkpoints/feneco-char-0.1.pt",
    [int]$Tokens = 100,
    [double]$Temperature = 0.8,
    [switch]$AssistantChat
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Ambiente local ausente. Execute .\scripts\setup.ps1 primeiro."
}

Push-Location $repoRoot
try {
    $arguments = @("-m", "llm_from_scratch.generate", "--checkpoint", $Checkpoint, "--tokens", $Tokens, "--temperature", $Temperature, "--chat")
    if ($AssistantChat) { $arguments += "--assistant-chat" }
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "A sessão terminou com erro ($LASTEXITCODE)." }
}
finally {
    Pop-Location
}
