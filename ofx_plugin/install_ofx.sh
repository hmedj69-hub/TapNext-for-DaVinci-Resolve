#!/usr/bin/env bash
# Installe l'effet OFX (Linux) : sudo ./install_ofx.sh
set -e
cd "$(dirname "$0")"
mkdir -p /usr/OFX/Plugins
rm -rf /usr/OFX/Plugins/TAPNextShapes.ofx.bundle
cp -r dist/TAPNextShapes.ofx.bundle /usr/OFX/Plugins/
echo "Effet OFX installé dans /usr/OFX/Plugins (redémarrez Resolve)."
