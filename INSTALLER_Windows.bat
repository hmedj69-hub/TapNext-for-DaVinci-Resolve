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
echo [1/7] Gestionnaire de paquets uv...
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
echo [2/7] Python 3.11 (local, n'affecte pas le systeme)...
if not exist "%PY%" goto :py_make
REM Environnement existant sans Tk (ancienne installation) : on le recree
"%PY%" -c "import tkinter" >nul 2>&1
if not errorlevel 1 goto :py_ok
rmdir /s /q "%VENV%"
:py_make
"%UV%" venv "%VENV%" --python 3.11 --python-preference only-managed --seed
if errorlevel 1 goto :fail
:py_ok
echo       OK

REM ----------------------------------------------------------- 3. PyTorch
echo [3/7] PyTorch...
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
echo [4/7] OpenCV, ffmpeg et autres dependances...
"%UV%" pip install --python "%PY%" opencv-python numpy einops tqdm imageio-ffmpeg
if errorlevel 1 goto :fail

REM ---------------------------------------------------------- 5. TAPNext++
echo [5/7] TAPNext++ (google-deepmind/tapnet)...
"%UV%" pip install --python "%PY%" --no-deps --reinstall-package tapnet "tapnet @ https://github.com/google-deepmind/tapnet/archive/refs/heads/main.zip"
if not errorlevel 1 goto :tapnet_ok
echo       Archive indisponible, essai via git...
"%UV%" pip install --python "%PY%" --no-deps "tapnet @ git+https://github.com/google-deepmind/tapnet.git"
if errorlevel 1 goto :fail
:tapnet_ok

REM ------------------------------------------------------------ 6. Modele
echo [6/7] Modele TAPNext++ 512 px (~2,5 Go, une seule fois)...
"%PY%" -c "import tap_resolve_tool as t; print('      ', t.ensure_checkpoint(None, 512))"
if errorlevel 1 goto :fail

REM ------------------------------------------------------- 7. Verification
echo [7/7] Verification...
"%PY%" -c "import tkinter, torch, cv2, imageio_ffmpeg; from tapnet.tapnextpp.votsp2026.model import TAPNextPP; c=torch.cuda.is_available(); print('       PyTorch', torch.__version__, '| CUDA :', c, '|', torch.cuda.get_device_name(0) if c else 'CPU'); print('       OpenCV', cv2.__version__, '| ffmpeg OK | Tk OK | TAPNext++ OK')"
if errorlevel 1 goto :fail

REM Raccourci sur le Bureau
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\TAPNext++ Resolve.lnk');" ^
  "$s.TargetPath='%ROOT%TAPNext_Resolve.bat'; $s.WorkingDirectory='%ROOT%'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,18'; $s.Save()" >nul 2>&1

echo.
echo  ============================================================
echo    Installation terminee !
echo    - Double-cliquez sur "TAPNext_Resolve.bat" (ou le raccourci
echo      "TAPNext++ Resolve" du Bureau) pour ouvrir l'interface.
echo    - Vous pouvez aussi glisser une video sur TAPNext_Resolve.bat.
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
