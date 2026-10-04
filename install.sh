#!/usr/bin/env bash
# TAPNext++ pour DaVinci Resolve — installation automatique (Linux / macOS).
# Usage : ./install.sh     (tout est installé dans ce dossier, rien dans le système)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"; TOOLS="$ROOT/tools"; VENV="$ROOT/.venv"; PY="$VENV/bin/python"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python" UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300
mkdir -p "$TOOLS"

echo "[1/8] Gestionnaire de paquets uv…"
if command -v uv >/dev/null 2>&1; then UV="$(command -v uv)"
else
  UV="$TOOLS/uv"
  if [ ! -x "$UV" ]; then
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64)  T=x86_64-unknown-linux-gnu ;;
      Linux-aarch64) T=aarch64-unknown-linux-gnu ;;
      Darwin-arm64)  T=aarch64-apple-darwin ;;
      Darwin-x86_64) T=x86_64-apple-darwin ;;
      *) echo "Plateforme non supportée : $(uname -sm)"; exit 1 ;;
    esac
    curl -LsSf "https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz" \
      | tar -xz -C "$TOOLS" --strip-components=1 "uv-$T/uv"
  fi
fi

echo "[2/8] Python 3.11 (local)…"
if [ -x "$PY" ] && ! "$PY" -c "import tkinter" 2>/dev/null; then rm -rf "$VENV"; fi
[ -x "$PY" ] || "$UV" venv "$VENV" --python 3.11 --python-preference only-managed --seed

echo "[3/8] PyTorch…"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "      GPU NVIDIA détecté → CUDA 12.4"
  "$UV" pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cu124
elif [ "$(uname -s)" = "Darwin" ]; then
  "$UV" pip install --python "$PY" torch torchvision
else
  echo "      Pas de GPU NVIDIA → version CPU (lente)"
  "$UV" pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "[4/8] OpenCV, ffmpeg et dépendances…"
"$UV" pip install --python "$PY" opencv-python numpy einops tqdm imageio-ffmpeg

echo "[5/8] TAPNext++ (google-deepmind/tapnet)…"
"$UV" pip install --python "$PY" --no-deps --reinstall-package tapnet \
    "tapnet @ https://github.com/google-deepmind/tapnet/archive/refs/heads/main.zip" \
  || "$UV" pip install --python "$PY" --no-deps "tapnet @ git+https://github.com/google-deepmind/tapnet.git"

echo "[6/8] Modèle TAPNext++ 512 px (~2,5 Go, une seule fois)…"
"$PY" -c "import tap_resolve_tool as t; print('      ', t.ensure_checkpoint(None, 512))"

echo "[7/8] Vérification…"
"$PY" -c "import tkinter, torch, cv2, imageio_ffmpeg; from tapnet.tapnextpp.votsp2026.model import TAPNextPP; c=torch.cuda.is_available(); print('       PyTorch', torch.__version__, '| CUDA :', c, '|', torch.cuda.get_device_name(0) if c else 'CPU'); print('       OpenCV', cv2.__version__, '| ffmpeg OK | Tk OK | TAPNext++ OK')"
echo "[8/8] Intégration dans DaVinci Resolve (Workspace > Scripts)…"
"$PY" resolve_plugin/install_resolve_plugin.py || echo "      ATTENTION : intégration Resolve impossible, voir README."
chmod +x TAPNext_Resolve.sh 2>/dev/null || true
echo
echo "Installation terminée. Interface : ./TAPNext_Resolve.sh"
echo "Dans DaVinci Resolve : Workspace > Scripts > TAPNext_Tracker"
