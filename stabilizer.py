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
GX, GY = 16, 9            # maillage de la correction locale (cellules)


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
    local: Optional[np.ndarray] = None   # [T, GY+1, GX+1, 2] mouvement résiduel local (px source)


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


def _mesh_field(p0: np.ndarray, res: np.ndarray, w: int, h: int) -> np.ndarray:
    """Mouvement résiduel des points → sommets du maillage (MeshFlow) :
    médiane des points voisins de chaque sommet, puis médiane spatiale 3×3."""
    vx = np.linspace(0, w, GX + 1)
    vy = np.linspace(0, h, GY + 1)
    V = np.stack(np.meshgrid(vx, vy), -1).reshape(-1, 2)
    out = np.zeros((len(V), 2))
    if len(p0):
        R = 1.6 * max(w / GX, h / GY)
        d2 = ((V[:, None, :] - p0[None, :, :]) ** 2).sum(-1)
        near = d2 < R * R
        for i in range(len(V)):
            m = near[i]
            if m.sum() >= 3:
                out[i] = np.median(res[m], axis=0)
    F = out.reshape(GY + 1, GX + 1, 2)
    pad = np.pad(F, ((1, 1), (1, 1), (0, 0)), mode="edge")
    win = np.stack([pad[dy:dy + GY + 1, dx:dx + GX + 1] for dy in range(3) for dx in range(3)], 0)
    return np.median(win, axis=0)


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
    local = np.zeros((T, GY + 1, GX + 1, 2), np.float32)
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
                    # Mouvement local (parallaxe, rolling shutter) : résidus après
                    # le mouvement global, objets indépendants exclus (> 6 px).
                    q0, q1 = p0[good], p1[good]
                    res = q1 - (np.c_[q0, np.ones(len(q0))] @ A.T)[:, :2]
                    keep = np.linalg.norm(res, axis=1) < 6.0
                    local[t] = _mesh_field(q0[keep] / ks, res[keep] / ks, width, height)
        prev_g = g
        if progress is not None and idx % 8 == 0:
            progress("Analyse du mouvement", (idx + 1) / n)
    return Motion(D, ok, inl_n, rms, (t0, t1), width, height, local)


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
    style: str = "smooth"          # "smooth" (gaussien) | "cinema" (plans fixes / panos réguliers)
    horizon: bool = False          # horizon verrouillé (rotation constante)
    horizon_deg: float = 0.0       # inclinaison corrigée (°)
    local: bool = False            # correction locale (maillage : parallaxe, rolling shutter)
    fill: bool = False             # bords reconstruits avec les images voisines


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
    path: Optional[np.ndarray] = None    # trajectoire utilisée (bords reconstruits)
    L: Optional[np.ndarray] = None       # [T, GY+1, GX+1, 2] correction locale (px source)
    wobble_before: float = float("nan")  # gélatine / vibration locale (px)
    wobble_after: float = float("nan")


def _shake(prm: np.ndarray, ok: np.ndarray) -> float:
    """Tremblement : écart moyen (px) entre la position et sa version lissée
    sur ±3 images (ce que l'œil perçoit comme vibration)."""
    idx = np.nonzero(ok)[0]
    if len(idx) < 8:
        return float("nan")
    x = prm[idx, :2]
    hf = x - _gauss(x, 3.0)
    return float(np.mean(np.linalg.norm(hf, axis=1)))


def _l1_path(raw: np.ndarray, margin: float, w=(10.0, 1.0, 100.0)) -> Optional[np.ndarray]:
    """Trajectoire « cinéma » (Grundmann et al., stabilisateur de YouTube) :
    minimise |vitesse|, |accélération| et |à-coups| en norme L1 → segments
    immobiles, à vitesse constante ou à accélération douce, en restant à moins
    de « margin » de la trajectoire réelle."""
    try:
        from scipy.optimize import linprog
        from scipy.sparse import bmat, diags, identity
    except Exception:
        return None
    n = len(raw)
    if n < 5 or margin <= 0:
        return None
    sc = max(1e-9, float(np.std(raw)) + margin)          # conditionnement
    r = raw / sc
    m = margin / sc
    D1 = diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n), format="csr")
    D2 = (D1[:-1, :-1] @ D1).tocsr()
    D3 = (D1[:-2, :-2] @ D2).tocsr()
    Ds = (D1, D2, D3)
    sizes = [D.shape[0] for D in Ds]
    nv = n + sum(sizes)
    c = np.concatenate([np.zeros(n)] + [np.full(k, wk) for k, wk in zip(sizes, w)])
    rows = []
    for j, D in enumerate(Ds):
        I = identity(sizes[j], format="csr")
        for sgn in (1, -1):                 # ±D·q − e_j ≤ 0  ⇔  |D·q| ≤ e_j
            rows.append([sgn * D] + [(-I if jj == j else None) for jj in range(3)])
    A = bmat(rows, format="csr")
    b = np.zeros(A.shape[0])
    bounds = [(float(r[i] - m), float(r[i] + m)) for i in range(n)] + [(0, None)] * (nv - n)
    try:
        sol = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    except Exception:
        return None
    if not sol.success:
        return None
    return sol.x[:n] * sc


def fuse_paths(path_fast: np.ndarray, path_ref: np.ndarray, W: int, H: int,
               sigma: float = 20.0) -> np.ndarray:
    """Hybride : détails image par image du flux optique dense (path_fast,
    précis mais qui dérive) + basses fréquences des trajectoires TAPNext
    (path_ref, calage direct sans dérive)."""
    P = np.array([W / 2.0, H / 2.0])
    T = max(len(path_fast), len(path_ref))

    def pad(a):
        out = np.full((T, 3, 3), np.nan)
        out[:len(a)] = a
        return out
    path_fast, path_ref = pad(path_fast), pad(path_ref)
    okf = np.isfinite(path_fast[:, 0, 0])
    okr = np.isfinite(path_ref[:, 0, 0]) & okf
    pf = _params(path_fast, P, okf)
    pr = _params(path_ref, P, okr)
    diff = np.where(okr[:, None], pr - pf, 0.0)
    w = okr.astype(float)
    idx = np.nonzero(okf)[0]
    if okr.sum() < 3 or idx.size < 3:
        return path_fast
    num = _gauss(diff[idx] * w[idx, None], sigma)
    den = _gauss(w[idx, None], sigma)
    corr = num / np.maximum(den, 1e-6)
    out = path_fast.copy()
    for k, t in enumerate(idx):
        out[t] = _matrix(pf[t] + corr[k], P)
    return out


def wobble(local: Optional[np.ndarray], ok: np.ndarray) -> float:
    """Gélatine / vibration locale : partie rapide du mouvement local cumulé."""
    if local is None:
        return float("nan")
    idx = np.nonzero(ok)[0]
    if len(idx) < 8:
        return float("nan")
    R = np.cumsum(local[idx], axis=0).reshape(len(idx), -1)
    hf = R - _gauss(R, 3.0)
    return float(np.mean(np.abs(hf)))


def stabilize(path: np.ndarray, W: int, H: int, p: StabParams,
              ref: Optional[int] = None, motion: Optional[Motion] = None) -> StabResult:
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
        if p.style == "cinema":
            # Marges autorisées par le recadrage (80 % translation, 20 % rotation/échelle).
            c = max(p.max_crop, 0.02) * (2.0 if p.fill else 1.0)
            sx, sy = W * c / 2, H * c / 2
            half_diag = 0.5 * math.hypot(W, H)
            margins = (0.8 * sx, 0.8 * sy, math.log(1 + 0.2 * c), 0.2 * min(sx, sy) / half_diag)
            pre = _gauss(raw, 1.5)          # les micro-vibrations ne guident pas le tracé
            for j in range(4):
                q = _l1_path(pre[:, j], margins[j])
                if q is not None:
                    target[:, j] = q
    if p.horizon and p.rotation:
        r = int(np.searchsorted(idx, ref)) if ref is not None else 0
        r = min(max(r, 0), len(idx) - 1)
        target[:, 3] = raw[r, 3] + math.radians(p.horizon_deg)
    if not p.position:
        target[:, :2] = raw[:, :2]
    if not p.scale:
        target[:, 2] = raw[:, 2]
    if not p.rotation:
        target[:, 3] = raw[:, 3]
    zmax = 1.0 / max(1e-3, 1.0 - p.max_crop) if p.max_crop > 0 else np.inf
    # Bords reconstruits : la correction peut sortir du cadre de 15 % de plus,
    # les zones manquantes sont remplies avec les images voisines.
    zmax_corr = 1.0 / max(1e-3, 1.0 - min(0.6, p.max_crop + 0.15)) if p.fill else zmax
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
            bad = z > zmax_corr * 1.0005
            if not bad.any():
                break
            a = alpha.copy()
            a[bad] *= 0.8
            a = _min_filter(a, win)
            a = np.minimum(alpha, _gauss(a[:, None], win / 2.0)[:, 0])
            alpha = np.clip(a, 0.0, 1.0)
            C, tg = build(alpha)
    z = required_zoom(C, ok, W, H)[idx]
    zf = z[np.isfinite(z)]
    if p.fill:
        # Bords reconstruits : zoom suffisant pour 75 % des images, le reste est
        # rempli avec les images voisines.
        need = float(np.percentile(zf, 25)) if zf.size else 1.0
    else:
        need = float(np.max(zf)) if zf.size else 1.0
    zoom = float(min(need, zmax if np.isfinite(zmax) else 4.0)) if p.zoom else 1.0
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
    # Correction locale (maillage) : retire la partie rapide du mouvement local
    # (gélatine du rolling shutter, vibrations de parallaxe), garde la parallaxe
    # lente (le relief de la scène).
    L = None
    wb = wa = float("nan")
    if motion is not None and motion.local is not None:
        loc = motion.local[:T]
        wb = wobble(loc, ok)
        if p.local:
            R = np.zeros_like(loc, dtype=np.float64)
            R[idx] = np.cumsum(loc[idx], axis=0)
            flat = R[idx].reshape(len(idx), -1)
            sm = _gauss(flat, 6.0 if p.mode == "lock" else min(6.0, max(1.0, p.smooth)))
            L = np.zeros_like(R, dtype=np.float32)
            Lc = (sm - flat).reshape((len(idx),) + loc.shape[1:])
            lim = 0.03 * W
            nrm = np.linalg.norm(Lc, axis=-1, keepdims=True)
            L[idx] = (Lc * np.minimum(1.0, lim / np.maximum(nrm, 1e-9))).astype(np.float32)
            wa = wobble(np.diff(np.concatenate([np.zeros_like(L[:1]), L]), axis=0) + loc, ok)
        else:
            wa = wb
    return StabResult(C, zoom, strength, _shake(prm, ok), _shake(stab, ok), path, L, wb, wa)


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

def _sample_field(F: np.ndarray, qx: np.ndarray, qy: np.ndarray, W: int, H: int) -> np.ndarray:
    """Champ du maillage [GY+1, GX+1, 2] échantillonné aux points (qx, qy) (px source)."""
    mx = (qx * (GX / float(W))).astype(np.float32)
    my = (qy * (GY / float(H))).astype(np.float32)
    return cv2.remap(F.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _sample_points(img: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    """Échantillonnage bilinéaire d'une liste de points (cv2.remap limite chaque
    dimension à 32 767 : on range les points en tableau 2D)."""
    n = len(mx)
    cols = 4096
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    X = np.concatenate([mx, np.zeros(pad, np.float32)]).reshape(rows, cols)
    Y = np.concatenate([my, np.zeros(pad, np.float32)]).reshape(rows, cols)
    out = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * cols, -1)[:n]


def warp_frame(img: np.ndarray, t: int, C: np.ndarray, W: int, H: int,
               L: Optional[np.ndarray] = None, path: Optional[np.ndarray] = None,
               get_frame: Optional[Callable[[int], Optional[np.ndarray]]] = None,
               fill: bool = False, max_dist: int = 12, quality: bool = True) -> np.ndarray:
    """Image t stabilisée. img et les images voisines sont à la même échelle
    (n'importe laquelle) ; C, L et path sont en pixels source."""
    h, w = img.shape[:2]
    s = w / float(W)
    if t >= len(C) or not np.isfinite(C[t, 0, 0]):
        return img
    xs = (np.arange(w, dtype=np.float32) + 0.5) / s - 0.5
    ys = (np.arange(h, dtype=np.float32) + 0.5) / s - 0.5
    X, Y = np.meshgrid(xs, ys)
    Ci = np.linalg.inv(C[t])
    den = Ci[2, 0] * X + Ci[2, 1] * Y + Ci[2, 2]
    qx = (Ci[0, 0] * X + Ci[0, 1] * Y + Ci[0, 2]) / den
    qy = (Ci[1, 0] * X + Ci[1, 1] * Y + Ci[1, 2]) / den
    if L is not None and t < len(L) and np.any(L[t]):
        d = _sample_field(L[t], qx, qy, W, H)
        qx = qx - d[..., 0]
        qy = qy - d[..., 1]
    interp = cv2.INTER_LANCZOS4 if quality else cv2.INTER_LINEAR
    mapx = ((qx + 0.5) * s - 0.5).astype(np.float32)
    mapy = ((qy + 0.5) * s - 0.5).astype(np.float32)
    valid = (mapx >= -0.5) & (mapx <= w - 0.5) & (mapy >= -0.5) & (mapy <= h - 0.5)
    if valid.all():
        return cv2.remap(img, mapx, mapy, interp, borderMode=cv2.BORDER_REPLICATE)
    if not (fill and get_frame is not None and path is not None and np.isfinite(path[t, 0, 0])):
        return cv2.remap(img, mapx, mapy, interp, borderMode=cv2.BORDER_CONSTANT)
    cur = cv2.remap(img, mapx, mapy, interp, borderMode=cv2.BORDER_REPLICATE)
    # Bords reconstruits : les zones hors champ viennent des images voisines,
    # recalées par le mouvement de caméra (de la plus proche à la plus lointaine).
    filled = valid.copy()
    fillimg = cur.copy()
    Pinv = np.linalg.inv(path[t])
    for dist in range(1, max_dist + 1):
        if filled.all():
            break
        for tn in (t - dist, t + dist):
            if tn < 0 or tn >= len(path) or not np.isfinite(path[tn, 0, 0]):
                continue
            nb = get_frame(tn)
            if nb is None or nb.shape[:2] != (h, w):
                continue
            G = path[tn] @ Pinv
            hole = ~filled
            den2 = G[2, 0] * qx[hole] + G[2, 1] * qy[hole] + G[2, 2]
            nx = (G[0, 0] * qx[hole] + G[0, 1] * qy[hole] + G[0, 2]) / den2
            ny = (G[1, 0] * qx[hole] + G[1, 1] * qy[hole] + G[1, 2]) / den2
            mx = ((nx + 0.5) * s - 0.5).astype(np.float32)
            my = ((ny + 0.5) * s - 0.5).astype(np.float32)
            ok = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
            if not ok.any():
                continue
            vals = _sample_points(nb, mx, my)
            hy, hx = np.nonzero(hole)
            fillimg[hy[ok], hx[ok]] = vals[ok].reshape(fillimg[hy[ok], hx[ok]].shape)
            filled[hy[ok], hx[ok]] = True
    return fillimg


def render_video(out_base: str, info, codec: str, C: np.ndarray, L: Optional[np.ndarray] = None,
                 path: Optional[np.ndarray] = None, fill: bool = False,
                 progress=None, cancel=None) -> str:
    """Vidéo stabilisée pleine résolution (même durée et timecode que la source :
    elle remplace le clip image pour image dans Resolve)."""
    import tap_resolve_tool as eng

    C = fill_outside(C, 0, len(C) - 1)
    writer = eng.MatteWriter(out_base, info, codec, suffix="_stabilized", gray=False)
    W, H = info.width, info.height
    n = len(C)
    K = 12 if W * H <= 2_300_000 else 6
    buf: dict = {}
    it = eng.iter_frames(info.path)
    last = -1
    total = info.frame_count or n

    def pull(upto: int):
        nonlocal last
        while last < upto:
            nxt = next(it, None)
            if nxt is None:
                last = 10 ** 9
                return
            last = nxt[0]
            buf[last] = nxt[1]

    try:
        t = 0
        while True:
            if cancel is not None and cancel.is_set():
                raise eng.Cancelled()
            pull(t + (K if fill else 0))
            if t not in buf:
                break
            frame = buf[t]
            out = warp_frame(frame, t, C, W, H, L=L, path=path, get_frame=buf.get,
                             fill=fill, max_dist=K) if t < n else frame
            writer.write(out)
            for k in [k for k in buf if k < t - K]:
                del buf[k]
            if progress is not None and t % 8 == 0:
                progress("Rendu de la vidéo stabilisée", (t + 1) / max(1, total))
            t += 1
    finally:
        writer.close()
    return writer.path
