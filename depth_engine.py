# -*- coding: utf-8 -*-
"""
depth_engine.py — Profondeur stable dans le temps, guidée par le suivi.

1. Profondeur par image avec Depth Anything V2 (Small, ~25 M paramètres) :
   carte de « disparité » relative (grand = proche), pour tous les pixels.
2. Problème connu de ces réseaux : chaque image est normalisée à sa façon →
   la profondeur « respire » et scintille. On la stabilise avec les points
   suivis par TAPNext++ : un même point physique doit garder (presque) la même
   disparité d'une image à l'autre. Pour chaque image on ajuste un gain et un
   décalage (moindres carrés robustes) sur l'image précédente, puis on lisse
   ces paramètres dans le temps.
3. Si la caméra 3D a été résolue (camera_solver), on cale les cartes sur la
   profondeur géométrique des points 3D : profondeur cohérente sur tout le
   plan, ce que l'IA seule ne garantit pas.

Convention de sortie : depth01 ∈ [0, 1], 0 = proche, 1 = loin (normalisé sur
le plan, en log-profondeur).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
_MODEL_CACHE: dict = {}


def _load_model(device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    key = device
    if key not in _MODEL_CACHE:
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
        if device.startswith("cuda"):
            model = model.half()
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = (proc, model, device)
    return _MODEL_CACHE[key]


@dataclass
class DepthClip:
    first: int                 # index de la 1re image
    disp: np.ndarray           # [n, h, w] float16 disparité alignée (grand = proche)
    depth01: np.ndarray        # [n, h, w] float16, 0 = proche, 1 = loin
    lo: float = 0.0            # bornes de normalisation (log-profondeur)
    hi: float = 1.0

    def frame01(self, t: int) -> Optional[np.ndarray]:
        i = t - self.first
        return self.depth01[i] if 0 <= i < len(self.depth01) else None


def estimate(get_frame: Callable[[int], Optional[np.ndarray]], t0: int, t1: int,
             device: str = "cuda", out_max_side: int = 480, batch: int = 4,
             progress=None, cancel=None) -> np.ndarray:
    """Disparités brutes [n, h, w] (float32) pour les images t0..t1."""
    import torch

    proc, model, dev = _load_model(device)
    out = []
    n = t1 - t0 + 1
    buf, idx = [], []
    size = None

    def flush():
        nonlocal size
        if not buf:
            return
        x = proc(images=[cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in buf], return_tensors="pt")
        pv = x["pixel_values"].to(dev)
        if dev.startswith("cuda"):
            pv = pv.half()
        with torch.inference_mode():
            pred = model(pixel_values=pv).predicted_depth.float().cpu().numpy()
        for p in pred:
            if size is None:
                h0, w0 = buf[0].shape[:2]
                s = out_max_side / float(max(h0, w0))
                size = (max(1, int(round(w0 * s))), max(1, int(round(h0 * s))))
            out.append(cv2.resize(p, size, interpolation=cv2.INTER_AREA))
        buf.clear()
        idx.clear()

    for k, t in enumerate(range(t0, t1 + 1)):
        if cancel is not None and cancel.is_set():
            from tap_resolve_tool import Cancelled
            raise Cancelled()
        img = get_frame(t)
        if img is None:
            img = np.zeros((270, 480, 3), np.uint8)
        buf.append(img)
        idx.append(t)
        if len(buf) >= batch:
            flush()
        if progress is not None and k % 4 == 0:
            progress("Profondeur (IA)", (k + 1) / n)
    flush()
    return np.stack(out).astype(np.float32)


def _sample(img: np.ndarray, xy: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Échantillonnage bilinéaire de img aux positions xy (pixels source)."""
    h, w = img.shape
    x = np.clip(xy[:, 0] * sx - 0.5, 0, w - 1.001)
    y = np.clip(xy[:, 1] * sy - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    ax, ay = x - x0, y - y0
    return ((img[y0, x0] * (1 - ax) + img[y0, x0 + 1] * ax) * (1 - ay)
            + (img[y0 + 1, x0] * (1 - ax) + img[y0 + 1, x0 + 1] * ax) * ay)


def _robust_affine(a: np.ndarray, b: np.ndarray, iters: int = 4):
    """b ≈ g·a + c, moindres carrés repondérés (Huber)."""
    if len(a) < 3:
        return 1.0, 0.0
    w = np.ones_like(a)
    g, c = 1.0, 0.0
    for _ in range(iters):
        A = np.stack([a * w, w], 1)
        sol, *_ = np.linalg.lstsq(A, b * w, rcond=None)
        g, c = float(sol[0]), float(sol[1])
        r = np.abs(g * a + c - b)
        s = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 1.5 * s, 1.0, 1.5 * s / np.maximum(r, 1e-9))
    return g, c


def stabilize(disp: np.ndarray, first: int, positions: np.ndarray, valid: np.ndarray,
              width: int, height: int, inv_depth: Optional[np.ndarray] = None,
              smooth: float = 3.0) -> DepthClip:
    """Aligne les disparités dans le temps avec les points suivis.

    positions/valid : [T, Q] (indices absolus). inv_depth : [T, Q] inverse de
    profondeur géométrique (solveur 3D) — si fourni, chaque image y est calée.
    """
    n, h, w = disp.shape
    sx, sy = w / float(width), h / float(height)
    gains = np.ones(n)
    offs = np.zeros(n)
    prev_vals = None
    prev_ok = None
    for i in range(n):
        t = first + i
        ok = valid[t] & np.isfinite(positions[t, :, 0])
        vals = np.full(positions.shape[1], np.nan)
        if ok.any():
            vals[ok] = _sample(disp[i], positions[t, ok], sx, sy)
        if inv_depth is not None and np.isfinite(inv_depth[t]).sum() >= 6:
            m = ok & np.isfinite(inv_depth[t])
            gains[i], offs[i] = _robust_affine(vals[m], inv_depth[t, m])
        elif prev_vals is not None and (ok & prev_ok & np.isfinite(prev_vals)).sum() >= 6:
            m = ok & prev_ok & np.isfinite(prev_vals)
            gains[i], offs[i] = _robust_affine(vals[m], prev_vals[m])
        elif i > 0:
            # Pas (assez) de points suivis : calage robuste sur la carte précédente
            # (grille de pixels ; les zones qui bougent sont rejetées par Huber).
            a = disp[i, ::6, ::6].ravel().astype(np.float64)
            b = (disp[i - 1, ::6, ::6].ravel() * gains[i - 1] + offs[i - 1]).astype(np.float64)
            gains[i], offs[i] = _robust_affine(a, b)
        aligned = np.where(ok, gains[i] * vals + offs[i], np.nan)
        prev_vals, prev_ok = aligned, ok
    if smooth > 0 and n > 2:
        r = int(np.ceil(3 * smooth))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
        k /= k.sum()
        pad = lambda v: np.pad(v, r, mode="edge")
        gains = np.convolve(pad(gains), k, mode="valid")
        offs = np.convolve(pad(offs), k, mode="valid")
    al = disp * gains[:, None, None] + offs[:, None, None]
    # Normalisation du plan entier en log-profondeur (0 = proche, 1 = loin).
    pos = al[al > 0]
    if pos.size == 0:
        al = al - al.min() + 1e-3
        pos = al.ravel()
    lo_d, hi_d = np.percentile(pos, [1, 99])
    al = np.clip(al, max(lo_d * 0.5, 1e-6), None)
    logz = -np.log(al)
    lo, hi = float(-np.log(hi_d)), float(-np.log(max(lo_d, 1e-6)))
    d01 = np.clip((logz - lo) / max(hi - lo, 1e-6), 0, 1)
    return DepthClip(first, al.astype(np.float16), d01.astype(np.float16), lo, hi)


def point_depth(clip: DepthClip, positions: np.ndarray, valid: np.ndarray,
                width: int, height: int, smooth: float = 2.0) -> np.ndarray:
    """Profondeur 0..1 de chaque point à chaque image ([T, Q], NaN si inconnue),
    lissée le long de la trajectoire."""
    T, Q = valid.shape
    out = np.full((T, Q), np.nan, np.float32)
    n, h, w = clip.depth01.shape
    sx, sy = w / float(width), h / float(height)
    for i in range(n):
        t = clip.first + i
        if t >= T:
            break
        ok = valid[t] & np.isfinite(positions[t, :, 0])
        if ok.any():
            out[t, ok] = _sample(clip.depth01[i].astype(np.float32), positions[t, ok], sx, sy)
    if smooth > 0:
        r = int(np.ceil(3 * smooth))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / smooth) ** 2)
        for q in range(Q):
            v = out[:, q]
            m = np.isfinite(v)
            if m.sum() < 3:
                continue
            num = np.convolve(np.where(m, v, 0), k, mode="full")[r:r + T]
            den = np.convolve(m.astype(float), k, mode="full")[r:r + T]
            out[m, q] = (num / np.maximum(den, 1e-9))[m]
    return out


def normalize_depth(z: np.ndarray) -> np.ndarray:
    """Profondeur géométrique z [T, Q] → 0..1 (log, percentiles 2–98), NaN conservés."""
    out = np.full(z.shape, np.nan, np.float32)
    ok = np.isfinite(z) & (z > 0)
    if ok.sum() < 2:
        return out
    lz = np.log(z[ok])
    lo, hi = np.percentile(lz, [2, 98])
    out[ok] = np.clip((lz - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def write_depth_video(out_base: str, info, codec: str, clip: DepthClip, n_total: int,
                      progress=None, cancel=None) -> str:
    """Vidéo N&B de profondeur à la résolution source (blanc = proche).
    Avant/après la plage calculée : première/dernière carte."""
    import tap_resolve_tool as eng

    writer = eng.MatteWriter(out_base, info, codec, suffix="_depth")
    n = len(clip.depth01)
    try:
        for t in range(n_total):
            if cancel is not None and cancel.is_set():
                raise eng.Cancelled()
            i = min(max(t - clip.first, 0), n - 1)
            d = 1.0 - clip.depth01[i].astype(np.float32)
            img = cv2.resize(d, (info.width, info.height), interpolation=cv2.INTER_CUBIC)
            writer.write((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
            if progress is not None and (t % 8 == 0 or t == n_total - 1):
                progress("Vidéo de profondeur", (t + 1) / max(1, n_total))
    finally:
        writer.close()
    return writer.path


def export_tapdepth(path: str, clip: DepthClip, src_w: int, src_h: int) -> str:
    """Fichier lu par l'effet OFX « TAPNext Profondeur & Temps » (voir
    TAPNextDepthTime.inc) : cartes 0..1 (0 = proche) en uint16."""
    import shape_engine as se

    n, h, w = clip.depth01.shape
    with open(path, "wb") as f:
        f.write(b"TAPDEP01")
        np.array([w, h, n, clip.first, src_w, src_h, 0, 0], "<i4").tofile(f)
        np.array([clip.lo, clip.hi], "<f4").tofile(f)
        for i in range(n):
            d = np.clip(clip.depth01[i].astype(np.float32), 0, 1)
            np.round(d * 65535).astype("<u2").tofile(f)
    se.remember_last_tapfx(path, "last_tapdepth.txt")
    return path
