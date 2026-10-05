#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Installe le script TAPNext_Tracker.lua dans DaVinci Resolve.

Le script apparaît ensuite dans Workspace > Scripts > TAPNext_Tracker
(toutes les pages). Appelé automatiquement par les installateurs ; peut être
relancé à la main si le dossier de l'outil a été déplacé :

    .venv/Scripts/python resolve_plugin/install_resolve_plugin.py      (Windows)
    .venv/bin/python     resolve_plugin/install_resolve_plugin.py      (Linux/macOS)

Option --uninstall pour le retirer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "TAPNext_Tracker.lua"
NAME = "TAPNext_Tracker.lua"


def resolve_scripts_dir() -> Path:
    """Dossier « Scripts/Utility » de Resolve (visible sur toutes les pages)."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        base = base / "Blackmagic Design" / "DaVinci Resolve" / "Support" / "Fusion"
    elif sys.platform == "darwin":
        base = (Path.home() / "Library" / "Application Support" / "Blackmagic Design"
                / "DaVinci Resolve" / "Fusion")
    else:
        base = Path.home() / ".local" / "share" / "DaVinciResolve" / "Fusion"
    return base / "Scripts" / "Utility"


def main() -> int:
    dest = resolve_scripts_dir() / NAME
    if "--uninstall" in sys.argv:
        if dest.exists():
            dest.unlink()
        print("Script Resolve retiré :", dest)
        return 0
    text = SRC.read_text(encoding="utf-8")
    root = str(ROOT)
    if "]]" in root:
        raise SystemExit("Chemin d'installation non supporté (contient « ]] ») : " + root)
    if sys.platform.startswith("win") and not root.isascii():
        print("ATTENTION : le dossier de l'outil contient des caractères accentués ("
              + root + "). Resolve ne pourra pas lancer TAPNext Studio depuis ce dossier : "
              "déplacez-le, par exemple dans C:\\TAPNext, puis relancez l'installateur.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text.replace("@@TAPNEXT_ROOT@@", root), encoding="utf-8")
    print("Script Resolve installé :", dest)
    print("Dans Resolve : Workspace > Scripts > TAPNext_Tracker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
