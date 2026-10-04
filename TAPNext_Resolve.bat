@echo off
REM Lance l'interface graphique. Glissez une video sur ce fichier pour la pre-remplir.
setlocal
cd /d "%~dp0"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" set "PYW=%~dp0.venv\Scripts\python.exe"
if exist "%PYW%" goto :run
echo L'outil n'est pas encore installe : lancez d'abord INSTALLER_Windows.bat
pause
exit /b 1
:run
start "" "%PYW%" "%~dp0tap_gui.py" %*
