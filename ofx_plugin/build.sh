#!/usr/bin/env bash
# Compile l'effet OFX « TAPNext Shapes ».
#   Windows (depuis Linux, MinGW-w64) et Linux : ./build.sh
#   macOS / MSVC : voir CMakeLists.txt
set -euo pipefail
cd "$(dirname "$0")"
B=dist/TAPNextShapes.ofx.bundle/Contents
INC="-Ithird_party/openfx -Ithird_party/stb"
mkdir -p "$B/Win64" "$B/Linux-x86-64"
if command -v x86_64-w64-mingw32-g++ >/dev/null; then
  x86_64-w64-mingw32-g++ -std=c++17 -O2 -shared -static -static-libgcc -static-libstdc++ \
    $INC src/TAPNextShapes.cpp -o "$B/Win64/TAPNextShapes.ofx"
  x86_64-w64-mingw32-strip "$B/Win64/TAPNextShapes.ofx"
  echo "Windows : $B/Win64/TAPNextShapes.ofx"
fi
g++ -std=c++17 -O2 -fPIC -shared -static-libstdc++ -static-libgcc $INC \
  src/TAPNextShapes.cpp -o "$B/Linux-x86-64/TAPNextShapes.ofx"
strip "$B/Linux-x86-64/TAPNextShapes.ofx"
echo "Linux   : $B/Linux-x86-64/TAPNextShapes.ofx"
# Test sans Resolve (mini hôte OFX) : test/mini_host.cpp
