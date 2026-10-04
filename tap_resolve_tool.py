#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tap_resolve_tool.py — Tracking TAPNext++ → Mattes alpha & données Fusion
=========================================================================

Outil autonome qui :
  1. suit des points sur une vidéo MP4/MOV (1080p / 4K, jusqu'à 1024+ images)
     avec TAPNext++ (Google DeepMind, dépôt ``google-deepmind/tapnet``) ;
  2. génère une vidéo **Alpha Matte N&B** à la résolution d'origine, prête à
     être utilisée comme « External Matte » dans la page Color de Resolve ;
  3. exporte les trajectoires (CSV + JSON) et un nœud **Transform Fusion**
     (.setting) pour la stabilisation ou le match-move.

Exemples
--------
  # Grille automatique 12×12, matte ProRes 422 HQ + CSV + nœud Fusion
  python tap_resolve_tool.py clip.mov --grid 12

  # Points précis (pixels de la vidéo source), blobs réactifs au mouvement
  python tap_resolve_tool.py clip.mp4 --points "960,540;1010,560;900,600" \\
         --radius 70 --motion-mode stretch --motion-sensitivity 0.08

  # Sélection interactive à la souris sur l'image 120, suivi avant + arrière
  python tap_resolve_tool.py clip.mov --pick --start-frame 120 --backward

Pipeline
--------
  lecture vidéo (OpenCV, en flux, aucune image 4K gardée en RAM)
    → redimensionnement INTER_AREA vers input_res×input_res (512 par défaut)
    → TAPNext++ image par image (état récurrent, fp16, points par lots)
    → coordonnées modèle [0..256] → pixels source
    → post-traitement (lissage temporel, vitesse, enveloppe de visibilité)
    → rendu des mattes (metaballs : cercles → flou gaussien → seuil)
    → encodage ffmpeg (ProRes / DNxHR / H.264 / PNG)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger("tap_resolve")

# =============================================================================
# 0. PARAMÈTRES PAR DÉFAUT (modifiables ici ou via la ligne de commande)
# =============================================================================

DEFAULTS = dict(
    # --- Modèle -------------------------------------------------------------
    checkpoint=None,             # None → checkpoints/<nom officiel selon input_res>
    input_res=512,               # résolution interne du modèle (256 ou 512)
    device="cuda",               # "cuda" ou "cpu"
    fp16_weights=False,          # poids en float16 (≈ -50 % VRAM pour les poids)
    points_per_batch=512,        # nb max de points traités ensemble (VRAM)
    use_certainty=False,         # visibilité × certitude de position (plus strict)
    # --- Points -------------------------------------------------------------
    grid=10,                     # grille N×N si aucun point n'est fourni
    grid_margin=0.05,            # marge de la grille (fraction de l'image)
    # --- Plage temporelle ---------------------------------------------------
    start_frame=0,               # image où les points sont définis
    end_frame=-1,                # dernière image suivie (-1 = fin du clip)
    # --- Post-traitement ----------------------------------------------------
    smooth_sigma=1.0,            # lissage gaussien des trajectoires (images)
    vis_threshold=0.5,           # seuil de visibilité (probabilité)
    fade_in=6,                   # durée du fondu d'apparition (images)
    fade_out=8,                  # durée du fondu de disparition (images)
    # --- Mattes -------------------------------------------------------------
    radius=40.0,                 # rayon de base des blobs (pixels source)
    merge=0.6,                   # flou de fusion = merge × rayon (metaballs)
    threshold=0.5,               # seuil de la fusion (plus bas = plus de fusion)
    edge_softness=0.08,          # douceur du bord (0 = bord dur)
    motion_mode="none",          # none | scale | stretch | both
    motion_sensitivity=0.05,     # gain appliqué à la vitesse (px/image @1080p)
    max_motion_scale=3.0,        # facteur d'agrandissement maximal
    render_max_side=1920,        # résolution de calcul du champ (upscale ensuite)
    codec="prores",              # prores | dnxhr | h264 | png
)

CHECKPOINT_URLS = {
    256: ("tapnextpp_ckpt.pt",
          "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"),
    512: ("tapnextpp_512.ckpt",
          "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt"),
}

# Espace de coordonnées des prédictions TAPNext++ (toujours 256×256, quelle que
# soit la résolution d'entrée : voir tapnet/tapnextpp/votsp2026/model.py).
MODEL_COORD_SIZE = 256


# =============================================================================
# 1. VIDÉO : PROBE, LECTURE, ÉCRITURE
# =============================================================================

@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int          # estimation OpenCV (corrigée après lecture)
    fps_rational: str = ""    # ex. "24000/1001" pour ffmpeg
    timecode: Optional[str] = None


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _probe_timecode(path: str) -> Optional[str]:
    """Lit le timecode de départ (ffprobe), utile pour la synchro dans Resolve."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "format_tags=timecode:stream_tags=timecode", "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(out or "{}")
        for s in data.get("streams", []):
            tc = s.get("tags", {}).get("timecode")
            if tc:
                return tc
        return data.get("format", {}).get("tags", {}).get("timecode")
    except Exception:
        return None


def probe_video(path: str) -> VideoInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("La vidéo ne contient aucune image lisible.")
    # La taille réelle des images décodées fait foi (rotation, anamorphose…).
    h, w = frame.shape[:2]
    frac = Fraction(fps).limit_denominator(1001)
    return VideoInfo(path, w, h, fps, n, f"{frac.numerator}/{frac.denominator}",
                     _probe_timecode(path))


def iter_frames(path: str, start: int = 0, end: int = -1) -> Iterator[Tuple[int, np.ndarray]]:
    """Itère (index, image BGR uint8) de start à end inclus, en flux.

    Les images précédant ``start`` sont sautées par ``grab()`` (lecture
    séquentielle exacte : le seek OpenCV n'est pas fiable sur H.264/H.265).
    """
    cap = cv2.VideoCapture(path)
    try:
        idx = 0
        while idx < start:
            if not cap.grab():
                return
            idx += 1
        while end < 0 or idx <= end:
            ok, frame = cap.read()
            if not ok:
                return
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class MatteWriter:
    """Écrit une séquence d'images N&B (uint8, H×W) dans un format lisible par
    DaVinci Resolve. Utilise ffmpeg en pipe (qualité maîtrisée) ou, à défaut,
    une séquence PNG."""

    CODECS = {
        # nom: (extension, arguments ffmpeg)
        "prores": (".mov", ["-c:v", "prores_ks", "-profile:v", "3",
                            "-vendor", "apl0", "-pix_fmt", "yuv422p10le"]),
        "dnxhr": (".mov", ["-c:v", "dnxhd", "-profile:v", "dnxhr_hq",
                           "-pix_fmt", "yuv422p"]),
        "h264": (".mp4", ["-c:v", "libx264", "-preset", "medium", "-crf", "8",
                          "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
    }

    def __init__(self, out_base: str, info: VideoInfo, codec: str = "prores",
                 suffix: str = "_matte", gray: bool = True):
        self.info = info
        self.gray = gray
        self.count = 0
        self.proc: Optional[subprocess.Popen] = None
        self.png_dir: Optional[str] = None
        ffmpeg = _find_ffmpeg()
        if codec != "png" and ffmpeg is None:
            log.warning("ffmpeg introuvable → export en séquence PNG.")
            codec = "png"
        self.codec = codec
        if codec == "png":
            self.png_dir = out_base + suffix + "_png"
            os.makedirs(self.png_dir, exist_ok=True)
            self.path = os.path.join(self.png_dir, "matte_%06d.png")
            return
        ext, cargs = self.CODECS[codec]
        self.path = out_base + suffix + ext
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24",
               "-s", f"{info.width}x{info.height}",
               "-r", info.fps_rational or str(info.fps), "-i", "-",
               *cargs]
        if info.timecode and ext == ".mov":
            cmd += ["-timecode", info.timecode]
        cmd.append(self.path)
        log.debug("ffmpeg: %s", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, img: np.ndarray) -> None:
        if self.png_dir is not None:
            cv2.imwrite(self.path % self.count, img)
        else:
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(np.ascontiguousarray(img).tobytes())
        self.count += 1

    def close(self) -> None:
        if self.proc is not None:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg a échoué lors de l'encodage de " + self.path)


# =============================================================================
# 2. INITIALISATION DES POINTS
# =============================================================================

def make_grid_points(width: int, height: int, n: int, margin: float = 0.05,
                     roi: Optional[Sequence[float]] = None) -> np.ndarray:
    """Grille N×N de points [x, y] (pixels) dans l'image ou dans un ROI."""
    if roi is not None:
        x0, y0, rw, rh = roi
    else:
        x0, y0 = width * margin, height * margin
        rw, rh = width * (1 - 2 * margin), height * (1 - 2 * margin)
    if n == 1:
        xs, ys = np.array([x0 + rw / 2]), np.array([y0 + rh / 2])
    else:
        xs = np.linspace(x0, x0 + rw, n)
        ys = np.linspace(y0, y0 + rh, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def parse_points(text: str) -> np.ndarray:
    """'x,y;x,y;…' → [N, 2]."""
    pts = []
    for chunk in text.replace("\n", ";").split(";"):
        chunk = chunk.strip().strip("[]()")
        if not chunk:
            continue
        x, y = (float(v) for v in chunk.replace(" ", ",").split(",") if v != "")
        pts.append((x, y))
    if not pts:
        raise ValueError("Aucun point valide dans --points")
    return np.asarray(pts, dtype=np.float32)


def load_points_file(path: str) -> np.ndarray:
    """JSON ([[x,y],…] ou {"points": [[x,y],…]}) ou CSV/TXT (x,y par ligne)."""
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("points") or data.get("query_points_xy")
        return np.asarray(data, dtype=np.float32).reshape(-1, 2)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals = [float(v) for v in line.replace(";", ",").split(",")[:2]]
            except ValueError:
                continue  # ligne d'en-tête
            rows.append(vals)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 2)


def pick_points_interactive(frame: np.ndarray) -> np.ndarray:
    """Fenêtre OpenCV : clic gauche = ajouter, clic droit = retirer le dernier,
    Entrée/Espace = valider, Échap = annuler."""
    h, w = frame.shape[:2]
    scale = min(1.0, 1600.0 / w, 900.0 / h)
    disp = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pts: List[Tuple[float, float]] = []
    win = "TAPNext++ - clic gauche: ajouter | droit: retirer | Entree: valider"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = disp.copy()
        for i, (x, y) in enumerate(pts):
            c = (int(x * scale), int(y * scale))
            cv2.circle(view, c, 5, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(view, str(i), (c[0] + 6, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 32):
            break
        if k == 27:
            pts.clear()
            break
    cv2.destroyWindow(win)
    if not pts:
        raise SystemExit("Aucun point sélectionné.")
    return np.asarray(pts, dtype=np.float32)


# =============================================================================
# 3. TRACKER TAPNext++
# =============================================================================

def ensure_checkpoint(path: Optional[str], input_res: int, download: bool = True) -> str:
    if input_res not in CHECKPOINT_URLS:
        raise ValueError("input_res doit être 256 ou 512 (checkpoints officiels).")
    name, url = CHECKPOINT_URLS[input_res]
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "checkpoints", name)
    if os.path.isfile(path):
        return path
    if not download:
        raise FileNotFoundError(f"Checkpoint absent : {path}\nTéléchargez-le : {url}")
    import torch

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    log.info("Téléchargement du checkpoint TAPNext++ (~2,5 Go) : %s", url)
    torch.hub.download_url_to_file(url, path, progress=True)
    return path


@dataclass
class TrackResult:
    """Trajectoires sur toute la durée du clip (T = nb d'images du clip).

    positions : [T, Q, 2] float32 (x, y) en pixels source, NaN hors plage suivie
    visibility: [T, Q]    float32 probabilité de visibilité (0 hors plage)
    tracked   : [T]       bool, image couverte par le tracking
    """
    positions: np.ndarray
    visibility: np.ndarray
    tracked: np.ndarray


class TAPNextPPTracker:
    """Inférence en ligne (image par image) de TAPNext++ avec état récurrent.

    * Les images sont redimensionnées en interne à input_res × input_res
      (INTER_AREA = anti-aliasing correct depuis la 4K) → VRAM maîtrisée.
    * Les points sont traités par lots (``points_per_batch``), chaque lot
      ayant son propre état récurrent : la VRAM ne dépend pas du nombre total
      de points.
    """

    def __init__(self, checkpoint: str, input_res: int = 512, device: str = "cuda",
                 fp16_weights: bool = False, points_per_batch: int = 512,
                 use_certainty: bool = False, certainty_radius: float = 8.0):
        import torch

        try:
            from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "Le paquet 'tapnet' est introuvable. Installez-le avec :\n"
                "  pip install git+https://github.com/google-deepmind/tapnet.git"
            ) from e

        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA indisponible → exécution sur CPU (très lent).")
            device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.input_res = int(input_res)
        self.points_per_batch = max(1, int(points_per_batch))
        self.use_certainty = use_certainty
        self.certainty_radius = certainty_radius

        log.info("Chargement de TAPNext++ (%s, entrée %d px) sur %s…",
                 os.path.basename(checkpoint), self.input_res, self.device)
        wrapper = TAPNextPP.from_checkpoint(
            checkpoint, device=self.device,
            half_precision=fp16_weights and self.is_cuda,
            input_resolution=self.input_res,
        )
        self.model = wrapper._model  # TAPNext (nn.Module) en mode eval
        if use_certainty:
            from tapnet.tapnext.tapnext_torch_utils import tracker_certainty
            self._certainty = tracker_certainty

    # ------------------------------------------------------------------ utils
    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR (H, W) → RGB uint8 (S, S). Gardé compact pour un éventuel cache."""
        small = cv2.resize(frame_bgr, (self.input_res, self.input_res),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, rgb_small: np.ndarray):
        t = self.torch.from_numpy(rgb_small).to(self.device, non_blocking=True)
        t = t.float().div_(127.5).sub_(1.0)          # [-1, 1]
        return t[None, None]                           # [1, 1, S, S, 3]

    def _autocast(self):
        if self.is_cuda:
            return self.torch.amp.autocast("cuda", dtype=self.torch.float16)
        return self.torch.amp.autocast("cpu", enabled=False)

    # --------------------------------------------------------------- tracking
    def track(self, frames: Iterable[np.ndarray], queries_xy: np.ndarray,
              width: int, height: int, n_frames_hint: int = 0,
              desc: str = "Tracking") -> Tuple[np.ndarray, np.ndarray]:
        """Suit ``queries_xy`` (pixels source, définis sur la 1re image fournie).

        Args:
            frames: itérable d'images déjà prétraitées (sortie de preprocess).
        Returns:
            positions [T, Q, 2] (pixels source) et visibility [T, Q] (probas).
        """
        torch = self.torch
        sx, sy = MODEL_COORD_SIZE / width, MODEL_COORD_SIZE / height
        q = queries_xy.shape[0]
        # Requêtes TAPNext : [t, y, x] dans l'espace modèle 256×256.
        qt = np.zeros((q, 3), np.float32)
        qt[:, 1] = queries_xy[:, 1] * sy
        qt[:, 2] = queries_xy[:, 0] * sx
        batches = [slice(i, min(i + self.points_per_batch, q))
                   for i in range(0, q, self.points_per_batch)]
        states: List[Optional[object]] = [None] * len(batches)
        pos_out: List[np.ndarray] = []
        vis_out: List[np.ndarray] = []

        bar = _progress(n_frames_hint, desc)
        with torch.inference_mode(), self._autocast():
            for rgb in frames:
                video = self._to_tensor(rgb)
                pos_f = np.empty((q, 2), np.float32)
                vis_f = np.empty((q,), np.float32)
                for bi, sl in enumerate(batches):
                    if states[bi] is None:
                        query = torch.from_numpy(qt[sl]).to(self.device)[None]
                        tracks, track_logits, vis_logits, states[bi] = self.model(
                            video=video, query_points=query)
                    else:
                        tracks, track_logits, vis_logits, states[bi] = self.model(
                            video=video, state=states[bi])
                    # tracks : [1, 1, Qb, 2] en (y, x) ; vis_logits : [1, 1, Qb, 1]
                    vis = torch.sigmoid(vis_logits[0, 0, :, 0].float())
                    if self.use_certainty:
                        cert = self._certainty(tracks.float(), track_logits.float(),
                                               self.certainty_radius)
                        vis = vis * cert[0, 0, :, 0]
                    yx = tracks[0, 0].float().cpu().numpy()
                    pos_f[sl, 0] = yx[:, 1] / sx
                    pos_f[sl, 1] = yx[:, 0] / sy
                    vis_f[sl] = vis.cpu().numpy()
                pos_out.append(pos_f)
                vis_out.append(vis_f)
                bar.update()
        bar.close()
        if self.is_cuda:
            log.info("VRAM max utilisée : %.2f Go",
                     torch.cuda.max_memory_allocated() / 1024 ** 3)
        if not pos_out:
            return np.zeros((0, q, 2), np.float32), np.zeros((0, q), np.float32)
        return np.stack(pos_out), np.stack(vis_out)


def run_tracking(tracker: TAPNextPPTracker, info: VideoInfo, queries_xy: np.ndarray,
                 start: int, end: int, backward: bool) -> TrackResult:
    """Suivi avant (start → end) et, optionnellement, arrière (start → 0)."""
    cache: List[np.ndarray] = []

    def forward_frames() -> Iterator[np.ndarray]:
        # On lit depuis 0 si le suivi arrière est demandé, pour mettre en cache
        # les images (déjà réduites à input_res, ~0,8 Mo/image en 512).
        for idx, frame in iter_frames(info.path, 0 if backward else start, end):
            small = tracker.preprocess(frame)
            if idx < start:
                cache.append(small)
                continue
            if backward and idx == start:
                cache.append(small)
            yield small

    n_hint = (end if end >= 0 else info.frame_count - 1) - start + 1
    pos_f, vis_f = tracker.track(forward_frames(), queries_xy, info.width, info.height,
                                 n_hint, "Tracking avant")
    n_total = max(info.frame_count, start + len(pos_f))
    q = queries_xy.shape[0]
    positions = np.full((n_total, q, 2), np.nan, np.float32)
    visibility = np.zeros((n_total, q), np.float32)
    tracked = np.zeros(n_total, bool)
    positions[start:start + len(pos_f)] = pos_f
    visibility[start:start + len(vis_f)] = vis_f
    tracked[start:start + len(pos_f)] = True

    if backward and start > 0 and cache:
        pos_b, vis_b = tracker.track(reversed(cache), queries_xy, info.width, info.height,
                                     len(cache), "Tracking arrière")
        # pos_b[0] = image start ; pos_b[k] = image start - k
        positions[:start] = pos_b[1:][::-1]
        visibility[:start] = vis_b[1:][::-1]
        tracked[:start] = True
    # On tronque si OpenCV a surestimé le nombre d'images.
    last = int(np.nonzero(tracked)[0].max()) + 1 if tracked.any() else 0
    n_real = max(last, min(n_total, info.frame_count))
    return TrackResult(positions[:n_real], visibility[:n_real], tracked[:n_real])


# =============================================================================
# 4. POST-TRAITEMENT : LISSAGE, VITESSE, ENVELOPPE DE VISIBILITÉ
# =============================================================================

def _gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def smooth_tracks(positions: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Lissage gaussien temporel pondéré par la visibilité (convolution
    normalisée) : les images occultées n'entraînent pas les voisines."""
    if sigma <= 0:
        return positions.copy()
    k = _gaussian_kernel(sigma)
    valid = np.isfinite(positions[..., 0])
    w = np.where(valid, np.maximum(weights, 1e-3), 0.0).astype(np.float32)
    p = np.nan_to_num(positions)
    out = positions.copy()
    conv = lambda a: np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    den = conv(w)
    for c in range(2):
        num = conv(p[..., c] * w)
        sm = num / np.maximum(den, 1e-6)
        out[..., c] = np.where(valid & (den > 1e-4), sm, positions[..., c])
    return out


def compute_velocity(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vitesse instantanée v = sqrt(dx² + dy²) (px/image, différence centrée).

    Returns: velocity [T, Q] et direction [T, Q, 2] (dx, dy).
    """
    p = positions
    d = np.zeros_like(p)
    if len(p) >= 3:
        d[1:-1] = (p[2:] - p[:-2]) * 0.5
    if len(p) >= 2:
        d[0] = p[1] - p[0]
        d[-1] = p[-1] - p[-2]
    d = np.nan_to_num(d)
    return np.hypot(d[..., 0], d[..., 1]).astype(np.float32), d.astype(np.float32)


def visibility_envelope(vis: np.ndarray, tracked: np.ndarray, threshold: float,
                        fade_in: int, fade_out: int) -> np.ndarray:
    """Opacité [T, Q] avec fondu d'apparition / disparition.

    La probabilité de visibilité est convertie en cible douce autour du seuil,
    puis l'opacité suit la cible avec une vitesse limitée (1/fade_in en montée,
    1/fade_out en descente) → pas de clignotement, fondus réguliers.
    """
    soft = np.clip((vis - (threshold - 0.1)) / 0.2, 0.0, 1.0)
    soft[~tracked] = 0.0
    up = 1.0 / max(1, fade_in)
    down = 1.0 / max(1, fade_out)
    alpha = np.zeros_like(soft)
    a = soft[0].copy() if len(soft) else None  # visible d'emblée à la 1re image
    for t in range(len(soft)):
        target = soft[t]
        a = np.where(target > a, np.minimum(target, a + up), np.maximum(target, a - down))
        alpha[t] = a
    return alpha


def hold_occluded_positions(positions: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Pendant une occultation, le blob reste à la dernière position fiable
    (évite qu'un masque en fondu ne « saute » sur une prédiction incertaine).
    Avant la première apparition d'un point, on utilise sa première position
    visible (utile pour le fondu d'apparition)."""
    out = positions.copy()
    ok = visible & np.isfinite(positions[..., 0])
    T, Q = ok.shape
    last = np.full((Q, 2), np.nan, np.float32)
    for t in range(T):
        last[ok[t]] = positions[t, ok[t]]
        hold = ~ok[t] & np.isfinite(last[:, 0])
        out[t, hold] = last[hold]
    seen = np.cumsum(ok, axis=0) > 0          # visible à t ou avant
    nxt = np.full((Q, 2), np.nan, np.float32)
    for t in range(T - 1, -1, -1):
        nxt[ok[t]] = positions[t, ok[t]]
        fill = ~seen[t] & np.isfinite(nxt[:, 0])
        out[t, fill] = nxt[fill]
    return out


# =============================================================================
# 5. RENDU DES MATTES (METABALLS + MOTION-REACTIVE)
# =============================================================================

@dataclass
class MatteParams:
    radius: float = 40.0
    merge: float = 0.6
    threshold: float = 0.5
    edge_softness: float = 0.08
    motion_mode: str = "none"
    motion_sensitivity: float = 0.05
    max_motion_scale: float = 3.0
    render_max_side: int = 1920
    invert: bool = False


class MatteRenderer:
    """Génère une matte N&B à la résolution source.

    Étapes par image :
      1. Calque « forme » : chaque point visible dessine une ellipse blanche
         (cercle, ou ellipse étirée selon la vitesse).
      2. Flou gaussien (σ = merge × rayon) → les blobs proches se rejoignent.
      3. Seuil doux (smoothstep autour de ``threshold``) → contour net et
         anti-aliasé, typique des metaballs.
      4. Opacité (fondus d'occultation) : convolution normalisée
         flou(Σ opacité) / flou(couverture) → moyenne locale des opacités,
         égale à 1 partout où les points sont pleinement visibles.
    Le champ est calculé à ``render_max_side`` puis interpolé à la résolution
    finale avant le seuil : bords nets en 4K pour un coût proche du 1080p.
    """

    SHIFT = 4  # précision sub-pixel OpenCV (1/16 px)

    def __init__(self, width: int, height: int, p: MatteParams):
        self.W, self.H, self.p = width, height, p
        self.rs = min(1.0, p.render_max_side / float(max(width, height)))
        self.w = max(1, int(round(width * self.rs)))
        self.h = max(1, int(round(height * self.rs)))
        # Normalisation de la vitesse : px/image ramenés à une hauteur 1080.
        self.vnorm = 1080.0 / height
        sigma = max(0.5, p.merge * p.radius * self.rs)
        # Le flou est calculé sur une version encore réduite si σ est grand.
        self.blur_scale = min(1.0, 4.0 / sigma)
        self.bw = max(1, int(round(self.w * self.blur_scale)))
        self.bh = max(1, int(round(self.h * self.blur_scale)))
        self.sigma_b = sigma * self.blur_scale

    def _axes(self, speed: float, d: np.ndarray) -> Tuple[float, float, float]:
        """Demi-axes (a, b) en pixels de travail + angle (degrés)."""
        r = self.p.radius * self.rs
        mode = self.p.motion_mode
        if mode == "none" or speed <= 0:
            return r, r, 0.0
        k = min(self.p.max_motion_scale, 1.0 + self.p.motion_sensitivity * speed * self.vnorm)
        ang = math.degrees(math.atan2(d[1], d[0]))
        if mode == "scale":
            return r * k, r * k, ang
        if mode == "stretch":   # étirement dans la direction du mouvement
            return r * k, r, ang
        # both : étirement + léger gonflement global
        g = math.sqrt(k)
        return r * k * g, r * g, ang

    def render(self, pos: np.ndarray, alpha: np.ndarray, speed: np.ndarray,
               direction: np.ndarray) -> np.ndarray:
        """pos [Q,2] pixels source, alpha [Q], speed [Q], direction [Q,2]."""
        shape = np.zeros((self.h, self.w), np.uint8)
        opac = np.zeros((self.h, self.w), np.uint8)
        cover = np.zeros((self.h, self.w), np.uint8)
        valid = (alpha > 1.0 / 255.0) & np.isfinite(pos[:, 0])
        idx = np.nonzero(valid)[0]
        if idx.size:
            # Ordre croissant d'opacité : le point le plus opaque l'emporte.
            idx = idx[np.argsort(alpha[idx], kind="stable")]
            f = float(1 << self.SHIFT)
            for i in idx:
                a, b, ang = self._axes(float(speed[i]), direction[i])
                c = (int(round(pos[i, 0] * self.rs * f)), int(round(pos[i, 1] * self.rs * f)))
                ax = (max(1, int(round(a * f))), max(1, int(round(b * f))))
                cv2.ellipse(shape, c, ax, ang, 0, 360, 255, -1, cv2.LINE_AA, self.SHIFT)
                # Opacité : empreinte un peu plus large que la forme.
                ax2 = (int(ax[0] * 1.5), int(ax[1] * 1.5))
                cv2.ellipse(opac, c, ax2, ang, 0, 360, int(round(alpha[i] * 255)), -1,
                            cv2.LINE_8, self.SHIFT)
                cv2.ellipse(cover, c, ax2, ang, 0, 360, 255, -1, cv2.LINE_8, self.SHIFT)
        field = self._blur(shape)
        opacity = self._blur(opac) / np.maximum(self._blur(cover), 1e-3)
        # Interpolation à la résolution finale puis seuil (bords nets en 4K).
        if (self.w, self.h) != (self.W, self.H):
            field = cv2.resize(field, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            opacity = cv2.resize(opacity, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        lo = self.p.threshold - self.p.edge_softness
        hi = self.p.threshold + self.p.edge_softness
        if hi - lo < 1e-4:
            m = (field >= self.p.threshold).astype(np.float32)
        else:
            m = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
            m = m * m * (3.0 - 2.0 * m)          # smoothstep
        m *= np.clip(opacity, 0.0, 1.0)
        if self.p.invert:
            m = 1.0 - m
        return (m * 255.0 + 0.5).astype(np.uint8)

    def _blur(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.float32) * (1.0 / 255.0)
        if self.blur_scale < 1.0:
            x = cv2.resize(x, (self.bw, self.bh), interpolation=cv2.INTER_AREA)
        x = cv2.GaussianBlur(x, (0, 0), self.sigma_b)
        if self.blur_scale < 1.0:
            x = cv2.resize(x, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return x


# =============================================================================
# 6. EXPORTS : CSV, JSON, FUSION
# =============================================================================

def export_csv(path: str, positions: np.ndarray, visibility: np.ndarray,
               velocity: np.ndarray, tracked: np.ndarray) -> int:
    """frame_index, point_id, x_pixels, y_pixels, visibility, velocity."""
    T, Q = visibility.shape
    frames = np.nonzero(tracked)[0]
    fr = np.repeat(frames, Q)
    pid = np.tile(np.arange(Q), len(frames))
    data = np.column_stack([
        fr, pid,
        positions[frames, :, 0].ravel(), positions[frames, :, 1].ravel(),
        visibility[frames].ravel(), velocity[frames].ravel(),
    ])
    ok = np.isfinite(data[:, 2]) & np.isfinite(data[:, 3])
    data = data[ok]
    np.savetxt(path, data, delimiter=",",
               fmt=["%d", "%d", "%.3f", "%.3f", "%.4f", "%.3f"],
               header="frame_index,point_id,x_pixels,y_pixels,visibility,velocity",
               comments="")
    return len(data)


def export_json(path: str, info: VideoInfo, queries: np.ndarray, args_dict: dict,
                frame_count: int, positions: Optional[np.ndarray] = None,
                visibility: Optional[np.ndarray] = None) -> None:
    payload = {
        "metadata": {
            "source": os.path.abspath(info.path),
            "width": info.width, "height": info.height,
            "fps": info.fps, "fps_rational": info.fps_rational,
            "frame_count": int(frame_count),
            "timecode": info.timecode,
            "query_frame": int(args_dict.get("start_frame", 0)),
            "coordinate_system": "pixels, origine haut-gauche, Y vers le bas",
            "model": "TAPNext++",
        },
        "parameters": args_dict,
        "query_points_xy": queries.round(3).tolist(),
    }
    if positions is not None and visibility is not None:
        payload["tracks"] = {
            "positions_xy": np.where(np.isfinite(positions), positions.round(3), None).tolist(),
            "visibility": visibility.round(4).tolist(),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)


# =============================================================================
# 7. ORCHESTRATION
# =============================================================================

def _progress(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(total=total or None, desc=desc, unit="img", dynamic_ncols=True)
    except ImportError:  # pragma: no cover
        class _Bar:
            n = 0

            def update(self, k=1):
                self.n += k
                if self.n % 50 == 0:
                    log.info("%s : %d images", desc, self.n)

            def close(self):
                pass
        return _Bar()


def render_outputs(args, info: VideoInfo, res: TrackResult, smoothed: np.ndarray,
                   alpha: np.ndarray, velocity: np.ndarray, direction: np.ndarray,
                   out_base: str) -> List[str]:
    params = MatteParams(
        radius=args.radius, merge=args.merge, threshold=args.threshold,
        edge_softness=args.edge_softness, motion_mode=args.motion_mode,
        motion_sensitivity=args.motion_sensitivity,
        max_motion_scale=args.max_motion_scale,
        render_max_side=args.render_max_side, invert=args.invert,
    )
    renderer = MatteRenderer(info.width, info.height, params)
    visible = res.visibility >= args.vis_threshold
    draw_pos = hold_occluded_positions(smoothed, visible)
    # Pendant un maintien (occultation) le blob ne doit pas s'étirer.
    speed = np.where(visible, velocity, 0.0)
    writer = MatteWriter(out_base, info, args.codec)
    outputs = [writer.path]
    preview = None
    src_iter = None
    if args.preview:
        preview = MatteWriter(out_base, info, "h264", suffix="_preview", gray=False)
        outputs.append(preview.path)
        src_iter = iter_frames(info.path)
    T = len(alpha)
    bar = _progress(T, "Rendu matte")
    colors = _point_colors(alpha.shape[1])
    try:
        for t in range(T):
            m = renderer.render(draw_pos[t], alpha[t], speed[t], direction[t])
            writer.write(m)
            if preview is not None:
                nxt = next(src_iter, None)
                frame = nxt[1] if nxt else np.zeros((info.height, info.width, 3), np.uint8)
                preview.write(_preview_frame(frame, m, smoothed[t], visible[t], colors))
            bar.update()
    finally:
        bar.close()
        writer.close()
        if preview is not None:
            preview.close()
    return outputs


def _point_colors(n: int) -> np.ndarray:
    hsv = np.zeros((max(n, 1), 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.arange(max(n, 1)) * 180 // max(n, 1)).astype(np.uint8)
    hsv[:, 0, 1:] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


def _preview_frame(frame: np.ndarray, matte: np.ndarray, pos: np.ndarray,
                   vis: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Contrôle visuel : matte en surimpression verte + points suivis."""
    out = frame.copy()
    tint = np.zeros_like(out)
    tint[..., 1] = 255
    a = (matte.astype(np.float32) / 255.0 * 0.45)[..., None]
    out = (out * (1 - a) + tint * a).astype(np.uint8)
    r = max(2, frame.shape[0] // 270)
    for i in range(len(pos)):
        if np.isfinite(pos[i, 0]):
            c = (int(pos[i, 0]), int(pos[i, 1]))
            col = tuple(int(v) for v in colors[i]) if vis[i] else (80, 80, 80)
            cv2.circle(out, c, r, col, -1, cv2.LINE_AA)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    d = DEFAULTS
    ap = argparse.ArgumentParser(
        description="TAPNext++ → Alpha Matte N&B + tracking Fusion pour DaVinci Resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video", help="Vidéo source (MP4/MOV, 1080p ou 4K)")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Dossier de sortie (défaut : à côté de la vidéo)")
    ap.add_argument("--name", default=None, help="Préfixe des fichiers de sortie")

    g = ap.add_argument_group("Modèle TAPNext++")
    g.add_argument("--checkpoint", default=d["checkpoint"],
                   help="Checkpoint .pt/.ckpt (téléchargé automatiquement si absent)")
    g.add_argument("--input-res", type=int, default=d["input_res"], choices=[256, 512],
                   help="Résolution interne d'inférence (VRAM ↔ précision)")
    g.add_argument("--device", default=d["device"])
    g.add_argument("--fp16-weights", action="store_true", default=d["fp16_weights"],
                   help="Charger les poids en float16 (économise de la VRAM)")
    g.add_argument("--points-per-batch", type=int, default=d["points_per_batch"])
    g.add_argument("--use-certainty", action="store_true", default=d["use_certainty"],
                   help="Visibilité × certitude de position (moins de faux positifs)")
    g.add_argument("--no-download", action="store_true",
                   help="Ne pas télécharger le checkpoint automatiquement")

    g = ap.add_argument_group("Points de tracking")
    g.add_argument("--grid", type=int, default=d["grid"], help="Grille N×N automatique")
    g.add_argument("--grid-margin", type=float, default=d["grid_margin"])
    g.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                   help="Restreint la grille à un rectangle (pixels source)")
    g.add_argument("--points", help='Liste "x,y;x,y;…" (pixels source)')
    g.add_argument("--points-file", help="Fichier JSON [[x,y],…] ou CSV x,y")
    g.add_argument("--normalized", action="store_true",
                   help="Les coordonnées fournies sont normalisées [0..1]")
    g.add_argument("--pick", action="store_true",
                   help="Sélection interactive des points à la souris")

    g = ap.add_argument_group("Plage temporelle")
    g.add_argument("--start-frame", type=int, default=d["start_frame"],
                   help="Image sur laquelle les points sont définis")
    g.add_argument("--end-frame", type=int, default=d["end_frame"])
    g.add_argument("--backward", action="store_true",
                   help="Suivre aussi en arrière de start-frame jusqu'à l'image 0")

    g = ap.add_argument_group("Post-traitement")
    g.add_argument("--smooth-sigma", type=float, default=d["smooth_sigma"],
                   help="Lissage temporel des trajectoires (images, 0 = brut)")
    g.add_argument("--vis-threshold", type=float, default=d["vis_threshold"])
    g.add_argument("--fade-in", type=int, default=d["fade_in"])
    g.add_argument("--fade-out", type=int, default=d["fade_out"])

    g = ap.add_argument_group("Matte (Color Page)")
    g.add_argument("--radius", type=float, default=d["radius"],
                   help="Rayon de base des blobs (pixels source)")
    g.add_argument("--merge", type=float, default=d["merge"],
                   help="Flou de fusion metaball, en multiple du rayon")
    g.add_argument("--threshold", type=float, default=d["threshold"],
                   help="Seuil metaball (↓ = blobs plus gros et plus fusionnés)")
    g.add_argument("--edge-softness", type=float, default=d["edge_softness"])
    g.add_argument("--motion-mode", choices=["none", "scale", "stretch", "both"],
                   default=d["motion_mode"], help="Réaction des blobs à la vitesse")
    g.add_argument("--motion-sensitivity", type=float, default=d["motion_sensitivity"],
                   help="Gain : facteur = 1 + gain × vitesse(px/image @1080p)")
    g.add_argument("--max-motion-scale", type=float, default=d["max_motion_scale"])
    g.add_argument("--render-max-side", type=int, default=d["render_max_side"],
                   help="Résolution de calcul du champ (perf. 4K)")
    g.add_argument("--invert", action="store_true", help="Inverser la matte")
    g.add_argument("--codec", choices=["prores", "dnxhr", "h264", "png"],
                   default=d["codec"])
    g.add_argument("--preview", action="store_true",
                   help="Exporter aussi une vidéo de contrôle (source + matte + points)")
    g.add_argument("--no-matte", action="store_true", help="Ne pas générer la matte")

    g = ap.add_argument_group("Fusion")
    g.add_argument("--fusion-mode", choices=["stabilize", "matchmove", "both", "none"],
                   default="both")
    g.add_argument("--fusion-model", choices=["similarity", "translation"],
                   default="similarity")
    g.add_argument("--fusion-smooth", type=int, default=0,
                   help="Stabilisation douce : rayon de lissage (0 = lock-off)")
    g.add_argument("--fusion-frame-offset", type=int, default=0,
                   help="Décalage des keyframes (ex. 1001 si la comp démarre à 1001)")
    g.add_argument("--fusion-point-paths", default="",
                   help="IDs de points exportés chacun dans un Transform (ex. 0,5)")
    g.add_argument("--json-full", action="store_true",
                   help="Inclure toutes les trajectoires dans le JSON")
    g.add_argument("-v", "--verbose", action="store_true")
    return ap


def resolve_queries(args, info: VideoInfo) -> np.ndarray:
    if args.points or args.points_file:
        pts = parse_points(args.points) if args.points else load_points_file(args.points_file)
        if args.normalized:
            pts = pts * np.array([info.width, info.height], np.float32)
    elif args.pick:
        frame = next(iter_frames(info.path, args.start_frame, args.start_frame), None)
        if frame is None:
            raise SystemExit("start-frame hors de la vidéo.")
        pts = pick_points_interactive(frame[1])
    else:
        roi = args.roi
        if roi is not None and args.normalized:
            roi = [roi[0] * info.width, roi[1] * info.height,
                   roi[2] * info.width, roi[3] * info.height]
        pts = make_grid_points(info.width, info.height, args.grid, args.grid_margin, roi)
    pts[:, 0] = np.clip(pts[:, 0], 0, info.width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, info.height - 1)
    return pts


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()

    info = probe_video(args.video)
    log.info("Vidéo : %dx%d @ %.3f i/s, ~%d images, TC=%s",
             info.width, info.height, info.fps, info.frame_count, info.timecode)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.video))
    os.makedirs(out_dir, exist_ok=True)
    stem = args.name or os.path.splitext(os.path.basename(args.video))[0]
    out_base = os.path.join(out_dir, stem)

    queries = resolve_queries(args, info)
    log.info("%d points de tracking (image %d)", len(queries), args.start_frame)

    # --- 1. Tracking ---------------------------------------------------------
    ckpt = ensure_checkpoint(args.checkpoint, args.input_res, not args.no_download)
    tracker = TAPNextPPTracker(ckpt, args.input_res, args.device, args.fp16_weights,
                               args.points_per_batch, args.use_certainty)
    res = run_tracking(tracker, info, queries, args.start_frame, args.end_frame,
                       args.backward)
    del tracker
    _free_gpu()
    log.info("Tracking terminé : %d images suivies sur %d", int(res.tracked.sum()),
             len(res.tracked))

    # --- 2. Post-traitement --------------------------------------------------
    smoothed = smooth_tracks(res.positions, res.visibility, args.smooth_sigma)
    velocity, direction = compute_velocity(smoothed)
    alpha = visibility_envelope(res.visibility, res.tracked, args.vis_threshold,
                                args.fade_in, args.fade_out)

    # --- 3. Exports données --------------------------------------------------
    csv_path = out_base + "_tracks.csv"
    n_rows = export_csv(csv_path, smoothed, res.visibility, velocity, res.tracked)
    json_path = out_base + "_tracks.json"
    params = {k: v for k, v in vars(args).items() if k != "video"}
    export_json(json_path, info, queries, params, len(res.tracked),
                smoothed if args.json_full else None,
                res.visibility if args.json_full else None)
    outputs = [csv_path, json_path]
    log.info("CSV : %s (%d lignes)", csv_path, n_rows)

    if args.fusion_mode != "none":
        import fusion_export as fx

        modes = ["stabilize", "matchmove"] if args.fusion_mode == "both" else [args.fusion_mode]
        pp = [int(v) for v in args.fusion_point_paths.replace(";", ",").split(",") if v.strip()]
        for mode in modes:
            path = f"{out_base}_fusion_{mode}.setting"
            fx.export_setting(
                csv_path, path, info.width, info.height, mode=mode,
                model=args.fusion_model, smooth_radius=args.fusion_smooth,
                frame_offset=args.fusion_frame_offset,
                min_visibility=args.vis_threshold, point_paths_ids=pp or None,
                outlier_px=max(2.0, 0.002 * info.width),
                ref_frame=args.start_frame,
            )
            outputs.append(path)

    # --- 4. Matte -------------------------------------------------------------
    if not args.no_matte:
        outputs += render_outputs(args, info, res, smoothed, alpha, velocity,
                                  direction, out_base)

    log.info("Terminé en %.1f s. Fichiers générés :", time.time() - t0)
    for p in outputs:
        log.info("  • %s", p)
    return 0


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
