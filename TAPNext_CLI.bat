@echo off
REM Ligne de commande : TAPNext_CLI.bat video.mov [options]   (voir --help)
REM Glisser une video sur ce fichier = traitement avec les reglages par defaut.
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if exist "%PY%" goto :run
echo L'outil n'est pas encore installe : lancez d'abord INSTALLER_Windows.bat
pause
exit /b 1
:run
if "%~1"=="" ("%PY%" "%~dp0tap_resolve_tool.py" --help) else ("%PY%" "%~dp0tap_resolve_tool.py" %*)
pause
