#!/usr/bin/env bash
# Lance TAPNext Studio (Linux / macOS). Argument optionnel : une vidéo.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then echo "Lancez d'abord ./install.sh"; exit 1; fi
exec .venv/bin/python tap_studio.py "$@"
