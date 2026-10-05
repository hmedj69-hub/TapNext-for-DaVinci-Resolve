# -*- coding: utf-8 -*-
"""
stabilizer.py — Stabilisation « comme les outils de référence », en mieux.

Ce que font le stabilisateur de Resolve, Warp Stabilizer ou vid.stab :
  1. analyser le mouvement image par image sur TOUTE l'image (des centaines de
     points répartis partout), en rejetant ce qui bouge autrement (RANSAC) ;
  2. reconstruire la trajectoire de la caméra, la lisser (garder le mouvement
     voulu, retirer les tremblements) ;
  3. limiter le recadrage : là où la correction demanderait trop de zoom, on
     corrige un peu moins, au lieu de zoomer tout le plan ;
  4. rendre la vidéo stabilisée.

Ce que TAPNext ajoute :
  • les zones des sujets suivis (groupes de formes) sont exclues de l'analyse :
    un personnage qui traverse l'image ne fait plus « glisser » le décor ;
  • on peut aussi partir des trajectoires TAPNext d'un groupe posé sur le
    décor (calage direct sur l'image de référence : aucune dérive en mode
    verrouillé, même après des occultations).

Convention : les matrices sont des homographies 3×3 en pixels source, origine
en haut à gauche. path[t] envoie l'image de référence vers l'image t ;
la correction C[t] envoie l'image t vers l'image stabilisée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

ANALYSIS_MAX_SIDE = 960


# =============================================================================
# 1. Analyse du mouvement image par image
# =============================================================================

@dataclass
class Motion:
    """Mouvement image → image (D[t] : image t-1 → image t), pixels source."""
    D: np.ndarray            # [T, 3, 3] (identité si inconnu)
    ok: np.ndarray           # [T] estimation fiable
    inliers: np.ndarray      # [T]
    rms: np.ndarray          # [T] erreur résiduelle (px source)
    frames: Tuple[int, int]
    width: int
    height: int


def _gray(img: np.ndarray, k: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if abs(k - 1.0) > 1e-3:
        g = cv2.resize(g, (int(round(g.shape[1] * k)), int(round(g.shape[0] * k))),
                       interpolation=cv2.INTER_AREA)
    return g


def _features(g: np.ndarray, mask: Optional[np.ndarray], n: int = 700) -> np.ndarray:
    """Coins répartis sur toute l'image (grille 6×4) : un ciel uniforme ou un
    sujet très texturé ne monopolisent pas l'analyse."""
    h, w = g.shape
    gx, gy = 6, 4
    per = max(8, n // (gx * gy))
    md = max(5, int(min(w, h) / 70))
    p = cv2.goodFeaturesToTrack(g, n * 4, 0.002, md, mask=mask, blockSize=7)
    if p is None:
        return np.zeros((0, 2), np.float32)
    p = p.reshape(-1, 2)                         # triés par qualité décroissante
    cell = (np.minimum(p[:, 0] * gx // w, gx - 1) * gy + np.minimum(p[:, 1] * gy // h, gy - 1)).astype(int)
    keep = np.zeros(len(p), bool)
    count = np.zeros(gx * gy, int)
    for i, c in enumerate(cell):
        if count[c] < per:
            keep[i] = True
            count[c] += 1
    return p[keep].astype(np.float32)


def _sim(src, dst, thr):
    if len(src) < 6:
        return None, None
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=thr, maxIters=3000,
                                         confidence=0.999, refineIters=20)
    if M is None:
        return None, None
    inl = inl.ravel().astype(bool)
    if inl.sum() < 6:
        return None, None
    # Affinage moindres carrés sur les inliers (similitude exacte).
    M2, _ = cv2.estimateAffinePartial2D(src[inl], dst[inl], method=cv2.LMEDS)
    if M2 is not None:
        M = M2
    return np.vstack([M, [0, 0, 1]]), inl


def analyze(get_frame: Callable[[int], Optional[np.ndarray]], t0: int, t1: int,
            width: int, height: int,
            exclude: Optional[Callable[[int], Optional[np.ndarray]]] = None,
            progress=None, cancel=None) -> Motion:
    """Mouvement image par image sur [t0, t1].

    get_frame(t) : image BGR (n'importe quelle résolution, même rapport).
    exclude(t) : polygones [(N, 2) pixels source…] à ignorer (sujets).
    """
    T = t1 + 1
    D = np.tile(np.eye(3), (T, 1, 1))
    ok = np.zeros(T, bool)
    inl_n = np.zeros(T, int)
    rms = np.full(T, np.nan)
    prev_g = None
    n = t1 - t0 + 1
    for idx, t in enumerate(range(t0, t1 + 1)):
        if cancel is not None and cancel.is_set():
            from tap_resolve_tool import Cancelled
            raise Cancelled()
        img = get_frame(t)
        if img is None:
            prev_g = None
            continue
        k = min(1.0, ANALYSIS_MAX_SIDE / float(max(img.shape[:2])))
        g = _gray(img, k)
        ks = g.shape[1] / float(width)          # pixels d'analyse par pixel source
        if prev_g is not None and prev_g.shape == g.shape:
            mask = None
            polys = exclude(t - 1) if exclude is not None else None
            if polys:
                mask = np.full(g.shape, 255, np.uint8)
                for poly in polys:
                    cv2.fillPoly(mask, [np.round(np.asarray(poly) * ks).astype(np.int32)], 0)
                if (mask > 0).mean() < 0.25:     # sujets trop grands : on garde tout
                    mask = None
            p0 = _features(prev_g, mask)
            if len(p0) >= 8:
                lk = dict(winSize=(21, 21), maxLevel=4,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, p0, None, **lk)
                p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_g, p1, None, **lk)
                good = (st1.ravel() == 1) & (st2.ravel() == 1) & \
                    (np.linalg.norm(p0b - p0, axis=1) < 0.5)
                A, inl = _sim(p0[good], p1[good], 1.0)
                if A is not None:
                    S = np.diag([ks, ks, 1.0])
                    D[t] = np.linalg.inv(S) @ A @ S
                    ok[t] = True
                    inl_n[t] = int(inl.sum())
                    r = (np.c_[p0[good][inl], np.ones(inl.sum())] @ A.T)[:, :2] - p1[good][inl]
                    rms[t] = float(np.sqrt(np.mean(np.sum(r ** 2, 1)))) / ks
        prev_g = g
        if progress is not None and idx % 8 == 0:
            progress("Analyse du mouvement", (idx + 1) / n)
    return Motion(D, ok, inl_n, rms, (t0, t1), width, height)


def path_from_motion(m: Motion, ref: int) -> np.ndarray:
    """Trajectoire cumulée : path[t] = image ref → image t."""
    t0, t1 = m.frames
    T = len(m.D)
    path = np.full((T, 3, 3), np.nan)
    ref = int(min(max(ref, t0), t1))
    path[ref] = np.eye(3)
    for t in range(ref + 1, t1 + 1):
        path[t] = m.D[t] @ path[t - 1]
    for t in range(ref - 1, t0 - 1, -1):
        path[t] = np.linalg.inv(m.D[t + 1]) @ path[t + 1]
    return path


# =============================================================================
# 2. Lissage de trajectoire avec recadrage limité
# =============================================================================

@dataclass
class StabParams:
    mode: str = "smooth"           # "smooth" (garder le mouvement voulu) | "lock"
    smooth: float = 30.0           # force du lissage (images, ≈ σ gaussien)
    max_crop: float = 0.10         # recadrage maximal (0.10 = zoom ×1.11)
    position: bool = True
    rotation: bool = True
    scale: bool = True
    zoom: bool = True              # zoom pour cacher les bords


def _params(path: np.ndarray, P: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Similitude → (cx, cy, log échelle, angle) du centre image."""
    T = len(path)
    prm = np.full((T, 4), np.nan)
    for t in np.nonzero(ok)[0]:
        H = path[t]
        c = H @ np.array([P[0], P[1], 1.0])
        c = c[:2] / c[2]
        a, b = H[0, 0], H[1, 0]
        prm[t] = (c[0], c[1], math.log(max(1e-9, math.hypot(a, b))), math.atan2(b, a))
    idx = np.nonzero(ok)[0]
    if idx.size:
        prm[idx, 3] = np.unwrap(prm[idx, 3])
    return prm


def _matrix(p, P) -> np.ndarray:
    s = math.exp(p[2])
    R = np.array([[math.cos(p[3]), -math.sin(p[3])], [math.sin(p[3]), math.cos(p[3])]]) * s
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.asarray(p[:2]) - R @ P
    return M


def _gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    """Lissage gaussien avec bords réfléchis (pas de biais en début/fin)."""
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = min(r, len(x) - 1)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        v = x[:, j]
        # réflexion impaire : prolonge la tendance (pas d'aplatissement aux bords)
        left = 2 * v[0] - v[pad:0:-1]
        right = 2 * v[-1] - v[-2:-pad - 2:-1]
        vv = np.concatenate([np.repeat(left[:1], r - pad), left, v, right, np.repeat(right[-1:], r - pad)])
        out[:, j] = np.convolve(vv, k, mode="valid")
    return out


def required_zoom(C: np.ndarray, ok: np.ndarray, W: int, H: int) -> np.ndarray:
    """Zoom minimal (autour du centre) pour qu'aucun bord noir n'apparaisse,
    image par image (inf si impossible)."""
    P = np.array([W / 2.0, H / 2.0])
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], float) - P
    z = np.ones(len(C))
    for t in np.nonzero(ok)[0]:
        Ci = np.linalg.inv(C[t])
        q = Ci @ np.array([P[0], P[1], 1.0])
        q = q[:2] / q[2]
        B = Ci[:2, :2]
        u = 1.0                                   # u = 1/zoom, le plus grand possible
        if not (0 <= q[0] <= W and 0 <= q[1] <= H):
            z[t] = np.inf
            continue
        for v in corners:
            d = B @ v
            for comp, lim in ((0, W), (1, H)):
                if d[comp] > 1e-9:
                    u = min(u, (lim - q[comp]) / d[comp])
                elif d[comp] < -1e-9:
                    u = min(u, -q[comp] / d[comp])
        z[t] = 1.0 / max(u, 1e-6)
    return z


def _min_filter(a: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return a
    pad = np.pad(a, r, mode="edge")
    return np.min(np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1), axis=1)


@dataclass
class StabResult:
    C: np.ndarray            # [T, 3, 3] image t → image stabilisée (zoom compris)
    zoom: float
    strength: np.ndarray     # [T] part de la correction appliquée (1 = totale)
    shake_before: float      # tremblement (px/image) avant
    shake_after: float       # … après


def _shake(prm: np.ndarray, ok: np.ndarray) -> float:
    """Tremblement : écart moyen (px) entre la position et sa version lissée
    sur ±3 images (ce que l'œil perçoit comme vibration)."""
    idx = np.nonzero(ok)[0]
    if len(idx) < 8:
        return float("nan")
    x = prm[idx, :2]
    hf = x - _gauss(x, 3.0)
    return float(np.mean(np.linalg.norm(hf, axis=1)))


def stabilize(path: np.ndarray, W: int, H: int, p: StabParams,
              ref: Optional[int] = None) -> StabResult:
    """Corrections C[t] à partir de la trajectoire (référence → image t)."""
    T = len(path)
    ok = np.isfinite(path[:, 0, 0])
    P = np.array([W / 2.0, H / 2.0])
    prm = _params(path, P, ok)
    idx = np.nonzero(ok)[0]
    C = np.full((T, 3, 3), np.nan)
    if idx.size == 0:
        return StabResult(C, 1.0, np.zeros(T), float("nan"), float("nan"))
    raw = prm[idx]
    if p.mode == "lock":
        r = int(np.searchsorted(idx, ref)) if ref is not None else 0
        r = min(max(r, 0), len(idx) - 1)
        target = np.tile(raw[r], (len(idx), 1))
    else:
        target = _gauss(raw, max(0.5, p.smooth))
    if not p.position:
        target[:, :2] = raw[:, :2]
    if not p.scale:
        target[:, 2] = raw[:, 2]
    if not p.rotation:
        target[:, 3] = raw[:, 3]
    zmax = 1.0 / max(1e-3, 1.0 - p.max_crop) if p.max_crop > 0 else np.inf
    alpha = np.ones(len(idx))
    # Repli quand le recadrage est limité : une trajectoire légèrement lissée
    # (retire encore les vibrations, suit le mouvement voulu). On ne revient
    # jamais à la trajectoire brute → pas de tremblement réintroduit.
    fallback = _gauss(raw, 4.0 if p.mode == "lock" else min(4.0, max(0.5, p.smooth)))
    for j, on in enumerate((p.position, p.position, p.scale, p.rotation)):
        if not on:
            fallback[:, j] = raw[:, j]

    def build(alpha):
        tg = fallback + alpha[:, None] * (target - fallback)
        Cs = np.full((T, 3, 3), np.nan)
        for k, t in enumerate(idx):
            Cs[t] = _matrix(tg[k], P) @ np.linalg.inv(path[t])
        return Cs, tg

    C, tg = build(alpha)
    if p.zoom and np.isfinite(zmax):
        # Recadrage limité : on réduit localement la correction là où il faudrait
        # zoomer plus que la limite, avec des transitions douces.
        win = max(2, int(p.smooth / 3)) if p.mode != "lock" else 12
        for _ in range(40):
            z = required_zoom(C, ok, W, H)[idx]
            bad = z > zmax * 1.0005
            if not bad.any():
                break
            a = alpha.copy()
            a[bad] *= 0.8
            a = _min_filter(a, win)
            a = np.minimum(alpha, _gauss(a[:, None], win / 2.0)[:, 0])
            alpha = np.clip(a, 0.0, 1.0)
            C, tg = build(alpha)
    z = required_zoom(C, ok, W, H)[idx]
    zoom = float(min(np.max(z[np.isfinite(z)]) if np.isfinite(z).any() else 1.0,
                     zmax if np.isfinite(zmax) else 4.0)) if p.zoom else 1.0
    zoom = max(1.0, zoom)
    if zoom != 1.0:
        Z = np.array([[zoom, 0, P[0] * (1 - zoom)], [0, zoom, P[1] * (1 - zoom)], [0, 0, 1]])
        for t in idx:
            C[t] = Z @ C[t]
    strength = np.zeros(T)
    strength[idx] = alpha
    # tremblement résiduel : trajectoire vue dans l'image stabilisée
    stab = np.full((T, 4), np.nan)
    for k, t in enumerate(idx):
        stab[t] = _params(np.array([C[t] @ path[t]]), P, np.array([True]))[0]
    return StabResult(C, zoom, strength, _shake(prm, ok), _shake(stab, ok))


def fill_outside(C: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Images hors de la plage analysée : correction de l'image la plus proche."""
    C = C.copy()
    ok = np.isfinite(C[:, 0, 0])
    idx = np.nonzero(ok)[0]
    if not idx.size:
        return C
    for t in range(len(C)):
        if not ok[t]:
            C[t] = C[idx[np.argmin(np.abs(idx - t))]]
    return C


# =============================================================================
# 3. Rendu de la vidéo stabilisée
# =============================================================================

def render_video(out_base: str, info, codec: str, C: np.ndarray, progress=None, cancel=None) -> str:
    """Vidéo stabilisée pleine résolution (même durée et timecode que la source :
    elle remplace le clip image pour image dans Resolve)."""
    import tap_resolve_tool as eng

    C = fill_outside(C, 0, len(C) - 1)
    writer = eng.MatteWriter(out_base, info, codec, suffix="_stabilized", gray=False)
    W, H = info.width, info.height
    n = len(C)
    try:
        for t, frame in eng.iter_frames(info.path):
            if cancel is not None and cancel.is_set():
                raise eng.Cancelled()
            if t >= n or not np.isfinite(C[t, 0, 0]):
                out = frame
            else:
                M = C[t] / C[t][2, 2]
                if abs(M[2, 0]) < 1e-12 and abs(M[2, 1]) < 1e-12:
                    out = cv2.warpAffine(frame, M[:2], (W, H), flags=cv2.INTER_LANCZOS4,
                                         borderMode=cv2.BORDER_CONSTANT)
                else:
                    out = cv2.warpPerspective(frame, M, (W, H), flags=cv2.INTER_LANCZOS4,
                                              borderMode=cv2.BORDER_CONSTANT)
            writer.write(out)
            if progress is not None and t % 8 == 0:
                progress("Rendu de la vidéo stabilisée", (t + 1) / max(1, info.frame_count or n))
    finally:
        writer.close()
    return writer.path
