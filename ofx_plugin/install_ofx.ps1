# Installe l'effet OFX « TAPNext Shapes » pour DaVinci Resolve (Windows).
# Lancé par INSTALLER_Windows.bat avec les droits administrateur (une seule
# fenêtre de confirmation Windows) : Resolve lit les plugins OFX dans
#   C:\Program Files\Common Files\OFX\Plugins
$ErrorActionPreference = 'Stop'
$src = Join-Path $PSScriptRoot 'dist\TAPNextShapes.ofx.bundle'
$dst = Join-Path ${env:CommonProgramFiles} 'OFX\Plugins'
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$target = Join-Path $dst 'TAPNextShapes.ofx.bundle'
if (Test-Path $target) { Remove-Item -Recurse -Force $target }
Copy-Item -Recurse -Force $src $dst
