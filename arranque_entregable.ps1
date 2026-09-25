param(
    [switch]$SkipTests,
    [switch]$SkipProcessing
)

$ErrorActionPreference = 'Stop'

# Siempre ubica la ejecucion en la raiz del repo.
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Error "No se encontro .venv. Crea el entorno virtual e instala dependencias antes de ejecutar este script."
}

Write-Host "==> Usando Python: $pythonExe"

if (-not $SkipTests) {
    Write-Host "`n==> Ejecutando pruebas (pytest)..."
    & $pythonExe -m pytest -q
}

if (-not $SkipProcessing) {
    Write-Host "`n==> Procesando dataset TASS con nuevas caracteristicas..."
    & $pythonExe .\scripts\procesar_tweets_features.py
}

Write-Host "`nProceso finalizado correctamente."
Write-Host "Archivos generados:"
Write-Host " - data/tass/tass2018_es_train_features.csv"
Write-Host " - data/tass/tass2018_es_test_features.csv"
