# -*- coding: utf-8 -*-
"""
motion_solver.py — Mouvement de caméra / d'objet à partir des trajectoires.

Inspiré des outils de référence (stabilisateur de Resolve, Warp Stabilizer,
Mocha, Fusion Tracker) :

  • 3 modèles : translation · similitude (position + rotation + échelle) ·
    perspective (homographie : surfaces planes, « planar tracking »).
  • Estimation robuste RANSAC (rejet des points qui bougent autrement :
    personnages devant le décor, reflets…).
  • Calage direct sur l'image de référence tant qu'assez de points y sont
    visibles (aucune dérive) ; sinon relais image par image (chaînage) — les
    trajectoires longues et la re-détection de TAPNext++ rendent le calage
    direct possible bien plus longtemps qu'avec un tracker classique.
  • Stabilisation : verrouillée ou lissée, composantes activables (position,
    rotation, échelle), zoom automatique pour masquer les bords.
  • Export Fusion : Transform (translation/similitude) ou CornerPositioner
    (perspective : stabilisation planaire ou insertion sur 4 coins).

Convention : H[t] envoie un point de l'image de référence vers l'image t
(pixels source, origine en haut à gauche).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import fusion_export as fx

MODELS = ("translation", "similarity", "perspective")
MIN_DIRECT = {"translation": 3, "similarity": 6, "perspective": 10}
MIN_INLIERS = {"translation": 2, "similarity": 3, "perspective": 6}


@dataclass
class Solve:
    H: np.ndarray                 # [T, 3, 3] référence → image t (NaN si inconnu)
    inliers: np.ndarray           # [T] nombre de points retenus
    rms: np.ndarray               # [T] erreur de reprojection (px)
    method: np.ndarray            # [T] 0 = aucun, 1 = direct, 2 = chaîné, 3 = maintenu
    ref: int
    model: str
    frames: Tuple[int, int]

    def ok(self, t: int) -> bool:
        return 0 <= t < len(self.H) and bool(np.isfinite(self.H[t, 0, 0]))


# =============================================================================
# Estimation
# =============================================================================

def _estimate(src: np.ndarray, dst: np.ndarray, model: str, thr: float):
    n = len(src)
    if model == "translation":
        if n < 1:
            return None, None
        d = dst - src
        med = np.median(d, axis=0)
        r = np.linalg.norm(d - med, axis=1)
        inl = r <= max(thr, 3.0 * float(np.median(r)) + 1e-6)
        t = d[inl].mean(axis=0)
        return np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], float), inl
    if model == "similarity":
        if n < 2:
            return None, None
        M, inl = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32),
                                             method=cv2.RANSAC, ransacReprojThreshold=thr,
                                             maxIters=2000, confidence=0.995, refineIters=10)
        if M is None:
            return None, None
        return np.vstack([M, [0, 0, 1]]).astype(float), inl.ravel().astype(bool)
    if n < 4:
        return None, None
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    Hm, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), method, thr,
                                 maxIters=3000, confidence=0.995)
    if Hm is None:
        return None, None
    return Hm.astype(float), inl.ravel().astype(bool)


def apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:3]


def solve_motion(pos: np.ndarray, valid: np.ndarray, ref: int, t0: int, t1: int,
                 model: str = "similarity", ransac_px: float = 2.0,
                 cols: Optional[np.ndarray] = None) -> Solve:
    """pos [T, Q, 2], valid [T, Q] ; cols = points utilisés (défaut : tous)."""
    T = pos.shape[0]
    if cols is not None:
        pos, valid = pos[:, cols], valid[:, cols]
    valid = valid & np.isfinite(pos[..., 0])
    H = np.full((T, 3, 3), np.nan)
    inl_n = np.zeros(T, int)
    rms = np.full(T, np.nan)
    method = np.zeros(T, np.int8)
    ref = int(min(max(ref, t0), t1))
    H[ref] = np.eye(3)
    inl_n[ref] = int(valid[ref].sum())
    rms[ref] = 0.0
    method[ref] = 1
    order = list(range(ref + 1, t1 + 1)) + list(range(ref - 1, t0 - 1, -1))
    for t in order:
        prev = t - 1 if t > ref else t + 1
        done = False
        c = valid[ref] & valid[t]
        if c.sum() >= MIN_DIRECT[model]:
            Ht, inl = _estimate(pos[ref, c], pos[t, c], model, ransac_px)
            if Ht is not None and inl.sum() >= MIN_INLIERS[model]:
                H[t] = Ht / Ht[2, 2]
                pr = apply_h(H[t], pos[ref, c][inl])
                rms[t] = float(np.sqrt(np.mean(np.sum((pr - pos[t, c][inl]) ** 2, 1))))
                inl_n[t], method[t], done = int(inl.sum()), 1, True
        if not done and np.isfinite(H[prev, 0, 0]):
            c = valid[prev] & valid[t]
            D, inl = _estimate(pos[prev, c], pos[t, c], model, ransac_px) if c.sum() else (None, None)
            if D is not None and inl.sum() >= MIN_INLIERS[model]:
                Ht = D @ H[prev]
                H[t] = Ht / Ht[2, 2]
                pr = apply_h(D, pos[prev, c][inl])
                rms[t] = float(np.sqrt(np.mean(np.sum((pr - pos[t, c][inl]) ** 2, 1))))
                inl_n[t], method[t] = int(inl.sum()), 2
            else:
                H[t], method[t] = H[prev], 3
    return Solve(H, inl_n, rms, method, ref, model, (t0, t1))


# =============================================================================
# Stabilisation
# =============================================================================

def _gauss_smooth(x: np.ndarray, sigma: float, ok: np.ndarray) -> np.ndarray:
    """Lissage gaussien le long de l'axe 0 (convolution normalisée)."""
    if sigma <= 0:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w = ok.astype(float)
    xf = np.where(ok.reshape(ok.shape + (1,) * (x.ndim - 1)), x, 0.0)
    flat = xf.reshape(len(x), -1)

    def conv(a):
        return np.convolve(a, k, mode="full")[r:r + len(a)]

    den = conv(w)
    out = np.stack([conv(flat[:, j] * w) for j in range(flat.shape[1])], 1) / np.maximum(den, 1e-9)[:, None]
    return np.where(ok.reshape(ok.shape + (1,) * (x.ndim - 1)), out.reshape(x.shape), x)


def _sim_params(H: np.ndarray, P: np.ndarray):
    """Similitude → (centre de P transformé, log échelle, angle)."""
    c = apply_h(H, P[None])[0]
    a, b = H[0, 0], H[1, 0]
    return c, math.log(max(1e-9, math.hypot(a, b))), math.atan2(b, a)


def _sim_matrix(c, ls, th, P) -> np.ndarray:
    s = math.exp(ls)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * s
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.asarray(c) - R @ P
    return M


def _point_in_quad(p, quad) -> bool:
    sgn = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) < 1e-9:
            continue
        s = 1 if cr > 0 else -1
        if sgn == 0:
            sgn = s
        elif s != sgn:
            return False
    return True


def auto_zoom(C: np.ndarray, W: int, Hh: int, ok: np.ndarray, max_zoom: float = 3.0) -> float:
    """Plus petit zoom (autour du centre) qui évite tout bord noir."""
    P = np.array([W / 2, Hh / 2])
    corners = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], float)
    worst = 1.0
    for t in np.nonzero(ok)[0]:
        quad = apply_h(C[t], corners)
        lo, hi = 1.0, max_zoom
        if all(_point_in_quad(c, quad) for c in corners):
            continue
        for _ in range(25):
            mid = (lo + hi) / 2
            rect = P + (corners - P) / mid
            if all(_point_in_quad(c, quad) for c in rect):
                hi = mid
            else:
                lo = mid
        worst = max(worst, hi)
    return float(min(worst, max_zoom))


@dataclass
class StabSettings:
    smooth: float = 0.0            # 0 = plan verrouillé ; sinon lissage (images)
    position: bool = True
    rotation: bool = True
    scale: bool = True
    zoom: bool = True              # zoom automatique (pas de bords noirs)


def stabilize(sol: Solve, W: int, Hh: int, st: StabSettings) -> Tuple[np.ndarray, float]:
    """Correction C[t] (image t → image stabilisée) et zoom appliqué."""
    T = len(sol.H)
    ok = np.isfinite(sol.H[:, 0, 0])
    C = np.full((T, 3, 3), np.nan)
    P = np.array([W / 2, Hh / 2])
    if sol.model == "perspective":
        corners = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], float)
        tr = np.full((T, 4, 2), np.nan)
        for t in np.nonzero(ok)[0]:
            tr[t] = apply_h(sol.H[t], corners)
        target = _gauss_smooth(tr, st.smooth, ok) if st.smooth > 0 else None
        for t in np.nonzero(ok)[0]:
            S = cv2.getPerspectiveTransform(corners.astype(np.float32),
                                            target[t].astype(np.float32)) if target is not None else np.eye(3)
            C[t] = S @ np.linalg.inv(sol.H[t])
    else:
        prm = np.full((T, 4), np.nan)
        for t in np.nonzero(ok)[0]:
            c, ls, th = _sim_params(sol.H[t], P)
            prm[t] = (c[0], c[1], ls, th)
        th = prm[:, 3].copy()                      # déroulement de l'angle
        idx = np.nonzero(ok)[0]
        if idx.size:
            th[idx] = np.unwrap(th[idx])
        prm[:, 3] = th
        tgt = _gauss_smooth(prm, st.smooth, ok) if st.smooth > 0 else \
            np.tile([P[0], P[1], 0.0, 0.0], (T, 1))
        if not st.position:
            tgt[:, :2] = prm[:, :2]
        if not st.scale or sol.model == "translation":
            tgt[:, 2] = prm[:, 2]
        if not st.rotation or sol.model == "translation":
            tgt[:, 3] = prm[:, 3]
        for t in idx:
            S = _sim_matrix(tgt[t, :2], tgt[t, 2], tgt[t, 3], P)
            C[t] = S @ np.linalg.inv(sol.H[t])
    z = auto_zoom(C, W, Hh, ok) if st.zoom else 1.0
    if z != 1.0:
        Z = np.array([[z, 0, P[0] * (1 - z)], [0, z, P[1] * (1 - z)], [0, 0, 1]])
        for t in np.nonzero(ok)[0]:
            C[t] = Z @ C[t]
    return C, z


# =============================================================================
# Export Fusion (.setting)
# =============================================================================

def _norm(xy, W, Hh):
    return (float(xy[0]) / W, 1.0 - float(xy[1]) / Hh)


def transform_keys(M: np.ndarray, W: int, Hh: int, frames: Sequence[int],
                   offset: int = 0) -> Dict[str, Dict[int, object]]:
    """Clés d'un nœud Transform réalisant la similitude M[t] (image → sortie).

    Fusion : sortie = Center + Size·R(Angle)·(entrée − Pivot), coordonnées
    normalisées Y vers le haut, angle positif anti-horaire.
    """
    P = np.array([W / 2, Hh / 2])
    keys: Dict[str, Dict[int, object]] = {"Center": {}, "Pivot": {}, "Angle": {}, "Size": {}}
    for t in frames:
        if not np.isfinite(M[t, 0, 0]):
            continue
        c = apply_h(M[t], P[None])[0]
        A = M[t, :2, :2]
        k = t + offset
        keys["Pivot"][k] = (0.5, 0.5)
        keys["Center"][k] = _norm(c, W, Hh)
        keys["Size"][k] = float(math.sqrt(abs(np.linalg.det(A))))
        keys["Angle"][k] = -math.degrees(math.atan2(A[1, 0], A[0, 0]))
    return keys


def _corner_tool(name: str, corners: Dict[str, Dict[int, Tuple[float, float]]],
                 pos: Tuple[int, int]) -> List[str]:
    inputs, extra = [], []
    for inp in ("TopLeft", "TopRight", "BottomLeft", "BottomRight"):
        prefix = name + inp
        inputs.append('\t\t\t\t%s = Input { SourceOp = "%sXYPath", Source = "Value", },' % (inp, prefix))
        extra += fx._xypath(prefix, corners[inp])
    tool = "\n".join(["\t\t%s = CornerPositioner {" % name, "\t\t\tInputs = {"] + inputs +
                     ["\t\t\t},", "\t\t\tViewInfo = OperatorInfo { Pos = { %d, %d } }," % pos, "\t\t},"])
    return [tool] + extra


def corner_keys(M: np.ndarray, quad: np.ndarray, W: int, Hh: int, frames: Sequence[int],
                offset: int = 0) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Coins (TL, TR, BR, BL de quad, pixels) transformés par M[t]."""
    names = ("TopLeft", "TopRight", "BottomRight", "BottomLeft")
    keys: Dict[str, Dict[int, Tuple[float, float]]] = {n: {} for n in names}
    for t in frames:
        if not np.isfinite(M[t, 0, 0]):
            continue
        q = apply_h(M[t], quad)
        for n, p in zip(names, q):
            keys[n][t + offset] = _norm(p, W, Hh)
    return keys


def write_setting(path: str, kind: str, M: np.ndarray, W: int, Hh: int,
                  frames: Sequence[int], offset: int = 0,
                  quad: Optional[np.ndarray] = None, name: Optional[str] = None) -> str:
    """kind : 'transform' (similitude) ou 'corner' (perspective)."""
    if kind == "corner":
        q = quad if quad is not None else np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], float)
        blocks = _corner_tool(name or "TAP_CornerPin", corner_keys(M, q, W, Hh, frames, offset), (0, 0))
        text = "{\n\tTools = ordered() {\n" + "\n".join(blocks) + \
            '\n\t},\n\tActiveTool = "%s"\n}\n' % (name or "TAP_CornerPin")
    else:
        text = fx.build_setting({name or "TAP_Transform": transform_keys(M, W, Hh, frames, offset)})
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
