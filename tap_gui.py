#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tap_gui.py — Interface graphique (Tkinter) pour tap_resolve_tool.py.

Lancement : double-clic sur « TAPNext_Resolve.bat » (Windows) ou
« TAPNext_Resolve.command / .sh » (macOS / Linux). On peut aussi glisser une
vidéo sur le lanceur : elle est pré-remplie.

L'interface construit la ligne de commande de tap_resolve_tool.py, l'exécute
dans un processus séparé et affiche la progression et le journal en direct.
Les réglages sont mémorisés dans « tap_gui_settings.json ».
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "tap_resolve_tool.py")
SETTINGS = os.path.join(HERE, "tap_gui_settings.json")
IS_WIN = sys.platform.startswith("win")

# (clé, libellé, valeur par défaut) — les clés correspondent aux arguments CLI.
DEFAULTS = {
    "video": "", "output_dir": "",
    "points_mode": "grid", "grid": 10, "points": "", "roi": "",
    "start_frame": 0, "end_frame": -1, "backward": False,
    "radius": 40.0, "merge": 0.6, "threshold": 0.5, "edge_softness": 0.08,
    "motion_mode": "none", "motion_sensitivity": 0.05, "max_motion_scale": 3.0,
    "fade_in": 6, "fade_out": 8, "smooth_sigma": 1.0, "vis_threshold": 0.5,
    "codec": "prores", "invert": False, "preview": True,
    "input_res": "512", "fp16_weights": False, "use_certainty": False,
    "points_per_batch": 512,
    "fusion_mode": "both", "fusion_model": "similarity",
    "fusion_smooth": 0, "fusion_frame_offset": 0,
}


def python_console_exe() -> str:
    """python.exe (et non pythonw.exe) pour que le sous-processus ait une sortie."""
    exe = sys.executable
    if IS_WIN and exe.lower().endswith("pythonw.exe"):
        cand = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.isfile(cand):
            return cand
    return exe


class App(tk.Tk):
    def __init__(self, initial_video: str = ""):
        super().__init__()
        self.title("TAPNext++ pour DaVinci Resolve")
        self.minsize(900, 680)
        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.vars: dict[str, tk.Variable] = {}
        self._load_settings()
        if initial_video:
            self.vars["video"].set(initial_video)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    # ------------------------------------------------------------- réglages
    def _load_settings(self) -> None:
        data = dict(DEFAULTS)
        try:
            with open(SETTINGS, encoding="utf-8") as f:
                data.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
        except Exception:
            pass
        for k, v in data.items():
            if isinstance(DEFAULTS[k], bool):
                self.vars[k] = tk.BooleanVar(value=bool(v))
            else:
                self.vars[k] = tk.StringVar(value=str(v))

    def _save_settings(self) -> None:
        try:
            with open(SETTINGS, "w", encoding="utf-8") as f:
                json.dump({k: v.get() for k, v in self.vars.items()}, f, indent=1)
        except Exception:
            pass

    # ------------------------------------------------------------ interface
    def _build(self) -> None:
        pad = dict(padx=6, pady=3)
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        # --- Fichiers
        f = ttk.LabelFrame(root, text="Fichiers", padding=6)
        f.pack(fill="x", **pad)
        self._file_row(f, 0, "Vidéo source (MP4/MOV)", "video", self._browse_video)
        self._file_row(f, 1, "Dossier de sortie (vide = à côté de la vidéo)",
                       "output_dir", self._browse_outdir)
        f.columnconfigure(1, weight=1)

        cols = ttk.Frame(root)
        cols.pack(fill="x", **pad)
        left, mid, right = ttk.Frame(cols), ttk.Frame(cols), ttk.Frame(cols)
        for i, c in enumerate((left, mid, right)):
            c.grid(row=0, column=i, sticky="nsew", padx=4)
            cols.columnconfigure(i, weight=1)

        # --- Points
        p = ttk.LabelFrame(left, text="Points de tracking", padding=6)
        p.pack(fill="both", expand=True)
        ttk.Radiobutton(p, text="Grille automatique N×N", value="grid",
                        variable=self.vars["points_mode"]).grid(row=0, column=0, columnspan=2, sticky="w")
        self._field(p, 1, "N", "grid", 6)
        self._field(p, 2, "ROI x,y,l,h (option)", "roi", 16)
        ttk.Radiobutton(p, text="Cliquer les points sur l'image", value="pick",
                        variable=self.vars["points_mode"]).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(p, text="Liste « x,y;x,y » (pixels)", value="list",
                        variable=self.vars["points_mode"]).grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Entry(p, textvariable=self.vars["points"], width=26).grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Separator(p).grid(row=6, column=0, columnspan=2, sticky="ew", pady=6)
        self._field(p, 7, "Image de départ", "start_frame", 8)
        self._field(p, 8, "Image de fin (-1 = fin)", "end_frame", 8)
        ttk.Checkbutton(p, text="Suivre aussi en arrière", variable=self.vars["backward"]
                        ).grid(row=9, column=0, columnspan=2, sticky="w")

        # --- Matte
        m = ttk.LabelFrame(mid, text="Matte (page Color)", padding=6)
        m.pack(fill="both", expand=True)
        self._field(m, 0, "Rayon des blobs (px)", "radius", 8)
        self._field(m, 1, "Fusion (× rayon)", "merge", 8)
        self._field(m, 2, "Seuil metaball", "threshold", 8)
        self._field(m, 3, "Douceur du bord", "edge_softness", 8)
        self._combo(m, 4, "Réaction au mouvement", "motion_mode",
                    ["none", "scale", "stretch", "both"])
        self._field(m, 5, "Sensibilité mouvement", "motion_sensitivity", 8)
        self._field(m, 6, "Fondu apparition (img)", "fade_in", 8)
        self._field(m, 7, "Fondu disparition (img)", "fade_out", 8)
        self._field(m, 8, "Lissage trajectoires", "smooth_sigma", 8)
        self._combo(m, 9, "Format", "codec", ["prores", "dnxhr", "h264", "png"])
        ttk.Checkbutton(m, text="Inverser", variable=self.vars["invert"]
                        ).grid(row=10, column=0, sticky="w")
        ttk.Checkbutton(m, text="Vidéo de contrôle", variable=self.vars["preview"]
                        ).grid(row=10, column=1, sticky="w")

        # --- Modèle + Fusion
        r = ttk.LabelFrame(right, text="Modèle TAPNext++", padding=6)
        r.pack(fill="x")
        self._combo(r, 0, "Résolution interne", "input_res", ["512", "256"])
        self._field(r, 1, "Points par lot (VRAM)", "points_per_batch", 8)
        ttk.Checkbutton(r, text="Poids FP16 (moins de VRAM)", variable=self.vars["fp16_weights"]
                        ).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(r, text="Visibilité × certitude", variable=self.vars["use_certainty"]
                        ).grid(row=3, column=0, columnspan=2, sticky="w")
        fu = ttk.LabelFrame(right, text="Fusion", padding=6)
        fu.pack(fill="both", expand=True, pady=(6, 0))
        self._combo(fu, 0, "Nœuds générés", "fusion_mode",
                    ["both", "stabilize", "matchmove", "none"])
        self._combo(fu, 1, "Modèle de mouvement", "fusion_model", ["similarity", "translation"])
        self._field(fu, 2, "Lissage (0 = verrouillé)", "fusion_smooth", 8)
        self._field(fu, 3, "Décalage d'image", "fusion_frame_offset", 8)

        # --- Actions
        a = ttk.Frame(root)
        a.pack(fill="x", **pad)
        self.btn_run = ttk.Button(a, text="▶  Lancer", command=self._run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(a, text="■  Arrêter", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(a, text="Ouvrir le dossier de sortie", command=self._open_out).pack(side="left")
        ttk.Button(a, text="Réglages par défaut", command=self._reset).pack(side="right")
        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(root, textvariable=self.status).pack(fill="x", padx=6)
        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=6, pady=(0, 4))

        # --- Journal
        lf = ttk.Frame(root)
        lf.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(lf, height=12, wrap="word", state="disabled",
                           font=("Consolas" if IS_WIN else "Monospace", 9))
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _file_row(self, parent, row, label, key, cmd) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(row=row, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Parcourir…", command=cmd).grid(row=row, column=2, padx=4)

    def _field(self, parent, row, label, key, width) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        ttk.Entry(parent, textvariable=self.vars[key], width=width).grid(row=row, column=1, sticky="w", padx=4)

    def _combo(self, parent, row, label, key, values) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
        ttk.Combobox(parent, textvariable=self.vars[key], values=values, width=11,
                     state="readonly").grid(row=row, column=1, sticky="w", padx=4)

    def _browse_video(self) -> None:
        p = filedialog.askopenfilename(
            title="Choisir la vidéo",
            filetypes=[("Vidéos", "*.mp4 *.mov *.MP4 *.MOV *.mxf *.mkv"), ("Tous", "*.*")])
        if p:
            self.vars["video"].set(p)

    def _browse_outdir(self) -> None:
        p = filedialog.askdirectory(title="Dossier de sortie")
        if p:
            self.vars["output_dir"].set(p)

    def _reset(self) -> None:
        for k, v in DEFAULTS.items():
            if k not in ("video", "output_dir"):
                self.vars[k].set(v)

    # ------------------------------------------------------- ligne de commande
    def _build_cmd(self) -> list[str]:
        v = {k: var.get() for k, var in self.vars.items()}
        video = str(v["video"]).strip().strip('"')
        if not video or not os.path.isfile(video):
            raise ValueError("Choisissez une vidéo source valide.")
        cmd = [python_console_exe(), "-u", TOOL, video]
        if str(v["output_dir"]).strip():
            cmd += ["--output-dir", str(v["output_dir"]).strip()]
        mode = v["points_mode"]
        if mode == "grid":
            cmd += ["--grid", str(int(float(v["grid"])))]
            roi = [s for s in re.split(r"[,; ]+", str(v["roi"]).strip()) if s]
            if roi:
                if len(roi) != 4:
                    raise ValueError("Le ROI doit contenir 4 valeurs : x,y,largeur,hauteur.")
                cmd += ["--roi", *[str(float(s)) for s in roi]]
        elif mode == "pick":
            cmd.append("--pick")
        else:
            if not str(v["points"]).strip():
                raise ValueError("Saisissez au moins un point « x,y ».")
            cmd += ["--points", str(v["points"]).strip()]

        def num(key, cast=float):
            try:
                return str(cast(float(v[key])))
            except ValueError:
                raise ValueError(f"Valeur invalide pour « {key} » : {v[key]!r}")

        for key, cast in [("start_frame", int), ("end_frame", int), ("radius", float),
                          ("merge", float), ("threshold", float), ("edge_softness", float),
                          ("motion_sensitivity", float), ("max_motion_scale", float),
                          ("fade_in", int), ("fade_out", int), ("smooth_sigma", float),
                          ("vis_threshold", float), ("points_per_batch", int),
                          ("fusion_smooth", int), ("fusion_frame_offset", int)]:
            cmd += ["--" + key.replace("_", "-"), num(key, cast)]
        cmd += ["--motion-mode", v["motion_mode"], "--codec", v["codec"],
                "--input-res", str(v["input_res"]), "--fusion-mode", v["fusion_mode"],
                "--fusion-model", v["fusion_model"]]
        for flag in ("backward", "invert", "preview", "fp16_weights", "use_certainty"):
            if v[flag]:
                cmd.append("--" + flag.replace("_", "-"))
        return cmd

    # -------------------------------------------------------------- exécution
    def _run(self) -> None:
        if self.proc is not None:
            return
        try:
            cmd = self._build_cmd()
        except ValueError as e:
            messagebox.showerror("Paramètres", str(e))
            return
        self._save_settings()
        self._clear_log()
        self._append("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd[1:]) + "\n\n")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if IS_WIN else 0
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         cwd=HERE, env=env, creationflags=flags)
        except OSError as e:
            messagebox.showerror("Erreur", f"Impossible de lancer le traitement :\n{e}")
            return
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status.set("Traitement en cours…")
        self.progress["value"] = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        """Lit la sortie (tqdm utilise \\r) et la découpe en lignes."""
        assert self.proc is not None and self.proc.stdout is not None
        buf = b""
        while True:
            chunk = self.proc.stdout.read1(4096) if hasattr(self.proc.stdout, "read1") \
                else self.proc.stdout.read(1)
            if not chunk:
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for p in parts:
                if p.strip():
                    self.q.put(p.decode("utf-8", "replace"))
        if buf.strip():
            self.q.put(buf.decode("utf-8", "replace"))
        self.q.put(None)

    _PCT = re.compile(r"^(?P<desc>[^:]+):\s+(?P<pct>\d+)%\|")

    def _poll(self) -> None:
        try:
            while True:
                line = self.q.get_nowait()
                if line is None:
                    self._finished()
                    break
                m = self._PCT.match(line)
                if m:  # ligne de barre de progression tqdm → pas dans le journal
                    self.progress["value"] = int(m.group("pct"))
                    self.status.set(line.strip()[:160])
                else:
                    self._append(line + "\n")
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finished(self) -> None:
        code = self.proc.wait() if self.proc else -1
        self.proc = None
        self.btn_run.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if code == 0:
            self.progress["value"] = 100
            self.status.set("Terminé ✔  — fichiers dans le dossier de sortie.")
            self._append("\n✔ Terminé.\n")
        else:
            self.status.set(f"Échec (code {code}) — voir le journal.")
            self._append(f"\n✘ Le traitement s'est arrêté (code {code}).\n")

    def _stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self._append("\nArrêt demandé…\n")

    def _open_out(self) -> None:
        out = str(self.vars["output_dir"].get()).strip()
        if not out:
            video = str(self.vars["video"].get()).strip()
            out = os.path.dirname(os.path.abspath(video)) if video else HERE
        if not os.path.isdir(out):
            return
        if IS_WIN:
            os.startfile(out)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", out])

    def _append(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.proc is not None and not messagebox.askyesno(
                "Quitter", "Un traitement est en cours. L'arrêter et quitter ?"):
            return
        if self.proc is not None:
            self.proc.terminate()
        self._save_settings()
        self.destroy()


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else "").mainloop()
