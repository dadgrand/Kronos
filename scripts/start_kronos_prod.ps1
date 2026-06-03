param(
  [string]$Config = "prod\config.example.json",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Starting Kronos prod status preflight..."
& $Python -m prod.cli --config $Config preflight
Write-Host "Writing local dashboard..."
& $Python -m prod.cli --config $Config status

Write-Host "For a shadow tick run manually:"
Write-Host "$Python -m prod.cli --config $Config shadow-run"
