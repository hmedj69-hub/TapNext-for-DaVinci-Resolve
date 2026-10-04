@echo off
REM ==========================================================================
REM  TAPNext++ pour DaVinci Resolve - installation automatique (Windows)
REM  Double-cliquez sur ce fichier. Aucun prerequis : Python, PyTorch (CUDA),
REM  TAPNext++, ffmpeg et le modele sont installes dans CE dossier.
REM  Relancer le script repare / met a jour l'installation.
REM ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"
title Installation TAPNext++ pour DaVinci Resolve

set "ROOT=%~dp0"
set "TOOLS=%ROOT%tools"
set "UV=%TOOLS%\uv.exe"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"
set "UV_PYTHON_INSTALL_DIR=%TOOLS%\python"
set "UV_LINK_MODE=copy"
set "UV_HTTP_TIMEOUT=300"

echo.
echo  ============================================================
echo    TAPNext++ pour DaVinci Resolve - Installation
echo  ============================================================
echo   Dossier : %ROOT%
echo   Telechargement total : ~5 Go (PyTorch CUDA + modele 2,5 Go)
echo.

REM ---------------------------------------------------------------- 1. uv
echo [1/9] Gestionnaire de paquets uv...
if exist "%UV%" goto :uv_ok
if not exist "%TOOLS%" mkdir "%TOOLS%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol='Tls12';" ^
  "$z=Join-Path $env:TEMP 'uv-win.zip';" ^
  "Invoke-WebRequest -UseBasicParsing 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $z;" ^
  "$d=Join-Path $env:TEMP 'uv-win'; if (Test-Path $d) { Remove-Item -Recurse -Force $d };" ^
  "Expand-Archive $z $d -Force;" ^
  "Copy-Item (Get-ChildItem $d -Recurse -Filter uv.exe | Select-Object -First 1).FullName '%UV%'"
if not exist "%UV%" goto :fail_uv
:uv_ok
echo       OK

REM ------------------------------------------------------------ 2. Python
echo [2/9] Python 3.11 (local, n'affecte pas le systeme)...
if exist "%PY%" goto :py_ok
"%UV%" venv "%VENV%" --python 3.11 --python-preference only-managed --seed
if errorlevel 1 goto :fail
:py_ok
echo       OK

REM ----------------------------------------------------------- 3. PyTorch
echo [3/9] PyTorch...
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :torch_cpu
set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
echo       GPU NVIDIA detecte : installation de la version CUDA 12.4
goto :torch_install
:torch_cpu
echo       ATTENTION : aucun GPU NVIDIA detecte (nvidia-smi absent).
echo       Installation de la version CPU (tres lente). Mettez a jour le
echo       pilote NVIDIA puis relancez ce script si vous avez une carte RTX.
:torch_install
"%UV%" pip install --python "%PY%" torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 goto :fail

REM ------------------------------------------------------- 4. Dependances
echo [4/9] OpenCV, ffmpeg, interface Qt et autres dependances...
"%UV%" pip install --python "%PY%" opencv-python numpy einops tqdm imageio-ffmpeg PySide6-Essentials scipy transformers
if errorlevel 1 goto :fail

REM ---------------------------------------------------------- 5. TAPNext++
echo [5/9] TAPNext++ (google-deepmind/tapnet)...
"%UV%" pip install --python "%PY%" --no-deps --reinstall-package tapnet "tapnet @ https://github.com/google-deepmind/tapnet/archive/refs/heads/main.zip"
if not errorlevel 1 goto :tapnet_ok
echo       Archive indisponible, essai via git...
"%UV%" pip install --python "%PY%" --no-deps "tapnet @ git+https://github.com/google-deepmind/tapnet.git"
if errorlevel 1 goto :fail
:tapnet_ok

REM ------------------------------------------------------------ 6. Modele
echo [6/9] Modele TAPNext++ 512 px (~2,5 Go, une seule fois)...
"%PY%" -c "import tap_resolve_tool as t; print('      ', t.ensure_checkpoint(None, 512))"
if errorlevel 1 goto :fail

REM ------------------------------------------------------- 7. Verification
echo [7/9] Verification...
"%PY%" -c "import PySide6, torch, cv2, imageio_ffmpeg, scipy, transformers; from tapnet.tapnextpp.votsp2026.model import TAPNextPP; c=torch.cuda.is_available(); print('       PyTorch', torch.__version__, '| CUDA :', c, '|', torch.cuda.get_device_name(0) if c else 'CPU'); print('       OpenCV', cv2.__version__, '| ffmpeg OK | Qt OK | 3D OK | TAPNext++ OK')"
if errorlevel 1 goto :fail

REM ------------------------------------------------ 8. Integration Resolve
echo [8/9] Integration dans DaVinci Resolve (Workspace ^> Scripts)...
"%PY%" "%ROOT%resolve_plugin\install_resolve_plugin.py"
if errorlevel 1 echo       ATTENTION : integration Resolve impossible, voir README.

REM ------------------------------------------------ 9. Effet OFX Resolve
echo [9/9] Effet OFX "TAPNext Shapes" pour DaVinci Resolve...
echo       Windows va demander une autorisation administrateur (copie dans
echo       C:\Program Files\Common Files\OFX\Plugins). Acceptez-la.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -Wait -WindowStyle Hidden -ArgumentList ('-NoProfile -ExecutionPolicy Bypass -File \"' + '%ROOT%ofx_plugin\install_ofx.ps1' + '\"')" >nul 2>&1
if exist "%CommonProgramFiles%\OFX\Plugins\TAPNextShapes.ofx.bundle\Contents\Win64\TAPNextShapes.ofx" (
  echo       OK - redemarrez DaVinci Resolve : Effets ^> OpenFX ^> TAPNext Shapes
) else (
  echo       ATTENTION : effet OFX non installe. Copiez le dossier
  echo       ofx_plugin\dist\TAPNextShapes.ofx.bundle dans
  echo       C:\Program Files\Common Files\OFX\Plugins  ^(droits administrateur^).
)

REM Raccourci sur le Bureau
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\TAPNext Studio.lnk');" ^
  "$s.TargetPath='%ROOT%TAPNext_Studio.bat'; $s.WorkingDirectory='%ROOT%'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,18'; $s.Save()" >nul 2>&1

echo.
echo  ============================================================
echo    Installation terminee !
echo    - Double-cliquez sur "TAPNext_Studio.bat" (ou le raccourci
echo      "TAPNext Studio" du Bureau) pour ouvrir l'application.
echo    - Vous pouvez aussi glisser une video sur TAPNext_Studio.bat.
echo    - Dans DaVinci Resolve (a redemarrer s'il etait ouvert) :
echo      Workspace ^> Scripts ^> TAPNext_Tracker
echo      et l'effet Effets ^> OpenFX ^> TAPNext Shapes
echo  ============================================================
echo.
pause
exit /b 0

:fail_uv
echo.
echo  ERREUR : impossible de telecharger uv. Verifiez la connexion Internet
echo  (ou un proxy/antivirus bloquant github.com), puis relancez.
pause
exit /b 1

:fail
echo.
echo  ERREUR pendant l'installation (voir les messages ci-dessus).
echo  Relancez ce script : il reprend la ou il s'est arrete.
pause
exit /b 1
