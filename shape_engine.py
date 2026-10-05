# -*- coding: utf-8 -*-
"""
shape_engine.py — Rendu des « groupes de formes » attachés aux points suivis.

Chaque groupe de points possède un style (ShapeStyle) : forme, mode
(révéler / découper), taille, rotation, réaction au mouvement, fondus,
traînée, fusion « metaball »… Le rendu combine tous les groupes en une
matte N&B, à la résolution voulue.

Combinaison des groupes :
    base   = max(groupes « Révéler »)   (blanc partout s'il n'y en a aucun)
    matte  = base × (1 − max(groupes « Découper »))
    option : inversion finale.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

import tap_resolve_tool as eng

SHAPES = ["circle", "square", "rounded", "diamond", "triangle", "hexagon",
          "star", "cross", "ring", "image"]
ECHO_MODES = ["none", "echo", "slit_h", "slit_v", "slit_radial", "slit_depth"]
ECHO_LABELS = {
    "none": "Aucun", "echo": "Écho (copies dans le temps)",
    "slit_h": "Slit-scan horizontal", "slit_v": "Slit-scan vertical",
    "slit_radial": "Slit-scan radial", "slit_depth": "Time-slice selon la profondeur (suivi)",
}

SHAPE_LABELS = {
    "circle": "Cercle", "square": "Carré", "rounded": "Carré arrondi",
    "diamond": "Losange", "triangle": "Triangle", "hexagon": "Hexagone",
    "star": "Étoile", "cross": "Croix", "ring": "Anneau", "image": "Image (PNG)…",
}


@dataclass
class ShapeStyle:
    # --- forme
    shape: str = "circle"
    image_path: str = ""
    mode: str = "add"                 # add = révèle (blanc) · subtract = découpe
    size: float = 4.0                 # rayon en pixels de la vidéo source
    width: float = 1.0                # largeur (× taille) : rectangles, ellipses…
    height: float = 1.0               # hauteur (× taille)
    rotation: float = 0.0             # degrés
    perspective: bool = False         # épouse la déformation locale de la surface (suivi TAPNext)
    perspective_amount: float = 1.0   # 0 = forme rigide, 1 = suit entièrement la surface
    follow_motion: bool = False       # orientée dans le sens du mouvement
    opacity: float = 1.0
    size_jitter: float = 0.0          # variation aléatoire de taille (0..1)
    # --- mouvement
    grow: float = 0.1                 # agrandissement avec la vitesse
    stretch: float = 0.195            # étirement dans la direction du mouvement
    max_scale: float = 3.0
    # --- apparition
    always_visible: bool = False      # ignore les occultations
    fade_in_on: bool = False
    fade_in: int = 6
    fade_out_on: bool = False
    fade_out: int = 8
    # --- fusion & bords
    merge: float = 0.10               # 0 = formes nettes et séparées
    threshold: float = 0.10
    softness: float = 0.0
    # --- effets
    trail: int = 0                    # traînée : nombre d'images
    smooth: float = 2.1               # lissage des trajectoires (images)
    # --- profondeur déduite du suivi (0 = proche, 1 = loin ; échelle locale de la surface)
    depth_scale: float = 0.0          # taille selon la profondeur (perspective)
    depth_near: float = 0.0           # plage de profondeur gardée
    depth_far: float = 1.0
    depth_feather: float = 0.05
    depth_fog: float = 0.0            # opacité réduite avec la distance
    # --- ombre portée
    shadow_on: bool = False
    shadow_angle: float = 135.0       # direction (degrés, 0 = droite, 90 = bas)
    shadow_dist: float = 12.0         # pixels source
    shadow_soft: float = 6.0          # flou (pixels source)
    shadow_opacity: float = 0.6
    shadow_depth: bool = False        # objets proches → ombre plus décalée
    # --- écho temporel / slit-scan
    echo_mode: str = "none"           # none · echo · slit_h · slit_v · slit_radial · slit_depth
    echo_count: int = 4
    echo_step: int = 3                # images entre deux échos
    echo_decay: float = 0.6           # opacité multipliée à chaque écho
    echo_scale: float = 1.0           # taille multipliée à chaque écho
    slit_span: int = 24               # décalage temporel maximal (images)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShapeStyle":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


PRESETS: Dict[str, dict] = {
    "Points nets": dict(shape="circle", size=6, merge=0.0, softness=0.0, grow=0.0, stretch=0.0,
                        fade_in_on=False, fade_out_on=False, trail=0),
    "Blobs fusionnés": dict(shape="circle", size=40, merge=0.6, threshold=0.5, softness=0.08,
                            grow=0.0, stretch=0.0, fade_in_on=True, fade_out_on=True),
    "Masque du sujet": dict(shape="circle", size=60, merge=0.9, threshold=0.35, softness=0.05,
                            grow=0.0, stretch=0.0, always_visible=True, smooth=2.0),
    "Traînées de mouvement": dict(shape="circle", size=8, merge=0.0, softness=0.1, grow=0.1,
                                  stretch=0.4, trail=10, follow_motion=True),
    "Particules": dict(shape="star", size=10, merge=0.0, size_jitter=0.6, rotation=0,
                       follow_motion=True, grow=0.2, stretch=0.0, fade_in_on=True, fade_out_on=True),
    "Découpe (trous)": dict(mode="subtract", shape="circle", size=30, merge=0.5, threshold=0.5,
                            softness=0.06, grow=0.0, stretch=0.0),
    "Échos fantômes": dict(shape="circle", size=24, merge=0.4, threshold=0.45, softness=0.08,
                           grow=0.0, stretch=0.0, echo_mode="echo", echo_count=6, echo_step=3,
                           echo_decay=0.65, echo_scale=0.92),
    "Slit-scan": dict(shape="circle", size=30, merge=0.6, threshold=0.5, softness=0.06,
                      grow=0.0, stretch=0.0, echo_mode="slit_h", slit_span=30, always_visible=True),
    "Perspective + ombre": dict(shape="square", size=16, merge=0.0, softness=0.0, grow=0.0,
                                stretch=0.0, perspective=True, shadow_on=True, shadow_depth=True,
                                shadow_dist=18, shadow_soft=8, shadow_opacity=0.5),
}


# =============================================================================
# Géométrie des formes (polygones unitaires, rayon 1)
# =============================================================================

def _regular(n: int, phase_deg: float = -90.0) -> np.ndarray:
    a = np.radians(phase_deg) + np.arange(n) * 2 * np.pi / n
    return np.stack([np.cos(a), np.sin(a)], -1)


def _unit_shapes() -> Dict[str, List[np.ndarray]]:
    star = _regular(10)
    star[1::2] *= 0.45
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    p = 4.0
    rounded = np.stack([np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2 / p),
                        np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2 / p)], -1)
    w = 0.34
    cross = np.array([[-w, -1], [w, -1], [w, -w], [1, -w], [1, w], [w, w], [w, 1], [-w, 1],
                      [-w, w], [-1, w], [-1, -w], [-w, -w]], float)
    return {
        "circle": [_regular(48, 0)],
        "square": [np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * 0.886],
        "rounded": [rounded * 0.93],
        "diamond": [np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], float) * 1.2],
        "triangle": [_regular(3) * 1.2],
        "hexagon": [_regular(6, 0)],
        "star": [star * 1.2],
        "cross": [cross],
        "ring": [_regular(48, 0)],
    }


UNIT = _unit_shapes()


def shape_icon(shape: str, size: int = 22) -> np.ndarray:
    """Petite vignette uint8 (pour les menus)."""
    img = np.zeros((size, size), np.uint8)
    r = size * 0.38
    c = np.array([size / 2, size / 2])
    if shape == "image":
        cv2.rectangle(img, (4, 4), (size - 5, size - 5), 255, 1)
        cv2.putText(img, "PNG", (3, size // 2 + 3), cv2.FONT_HERSHEY_PLAIN, 0.6, 255, 1)
        return img
    for poly in UNIT.get(shape, UNIT["circle"]):
        pts = np.round((poly * r + c) * 16).astype(np.int32)
        if shape == "ring":
            cv2.polylines(img, [pts], True, 255, 2, cv2.LINE_AA, 4)
        else:
            cv2.fillPoly(img, [pts], 255, cv2.LINE_AA, 4)
    return img


# =============================================================================
# Données par groupe (dépendent du suivi + lissage/fondus du style)
# =============================================================================

@dataclass
class GroupData:
    pos: np.ndarray        # [T, n, 2] positions de dessin (pixels source)
    alpha: np.ndarray      # [T, n]
    speed: np.ndarray      # [T, n] px/image
    direction: np.ndarray  # [T, n, 2]
    jitter: np.ndarray     # [n] facteur de taille
    ids: np.ndarray        # [n] indices des points (colonnes du suivi)
    depth: Optional[np.ndarray] = None   # [T, n] profondeur 0..1 (0 = proche), déduite du suivi
    affine: Optional[np.ndarray] = None  # [T, n, 2, 2] déformation locale de la surface


def local_affine(pos: np.ndarray, ok: np.ndarray, ref: np.ndarray, k: int = 8,
                 smooth: float = 2.0):
    """Perspective et profondeur DÉDUITES DU SUIVI TAPNext : pour chaque point,
    la déformation locale (matrice 2×2 : échelle, rotation, cisaillement de
    perspective) que subissent ses voisins entre l'image où il a été posé et
    chaque image. Une surface qui s'approche grandit, une surface qui tourne
    se raccourcit dans un sens : les formes posées dessus font pareil.

    pos [T, n, 2], ok [T, n], ref [n] image de pose → (A [T, n, 2, 2],
    profondeur relative [T, n] 0 = proche … 1 = loin)."""
    T, n = ok.shape
    A = np.tile(np.eye(2, dtype=np.float32), (T, n, 1, 1))
    if n < 4:
        return A, None
    P = np.nan_to_num(pos.astype(np.float64))
    for j in range(n):
        r = int(min(max(ref[j], 0), T - 1))
        cand = np.nonzero(ok[r])[0]
        cand = cand[cand != j]
        if len(cand) < 3 or not ok[r, j]:
            continue
        d = np.linalg.norm(P[r, cand] - P[r, j], axis=1)
        nb = cand[np.argsort(d)[:k]]
        X = P[r, nb] - P[r, j]                                  # [k, 2]
        Y = P[:, nb] - P[:, j:j + 1]                            # [T, k, 2]
        w = (ok[:, nb] & ok[:, j:j + 1]).astype(np.float64)     # [T, k]
        XX = np.einsum("tk,ki,kj->tij", w, X, X)
        YX = np.einsum("tk,tki,kj->tij", w, Y, X)
        det = XX[:, 0, 0] * XX[:, 1, 1] - XX[:, 0, 1] * XX[:, 1, 0]
        good = (w.sum(1) >= 3) & (np.abs(det) > 1e-6 * (np.abs(XX).max() + 1e-9) ** 2)
        inv = np.zeros_like(XX)
        inv[:, 0, 0], inv[:, 1, 1] = XX[:, 1, 1], XX[:, 0, 0]
        inv[:, 0, 1], inv[:, 1, 0] = -XX[:, 0, 1], -XX[:, 1, 0]
        inv /= np.where(np.abs(det) > 0, det, 1.0)[:, None, None]
        Aj = YX @ inv
        if not good.any():
            continue
        # images sans voisins : dernière déformation connue (puis première)
        idx = np.where(good, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        first = int(np.argmax(good))
        idx[:first] = first
        Aj = Aj[idx]
        if smooth > 0:
            rr = int(math.ceil(3 * smooth))
            ker = np.exp(-0.5 * (np.arange(-rr, rr + 1) / smooth) ** 2)
            ker /= ker.sum()
            flat = np.pad(Aj.reshape(T, 4), ((rr, rr), (0, 0)), mode="edge")
            Aj = np.stack([np.convolve(flat[:, c], ker, mode="valid") for c in range(4)], 1).reshape(T, 2, 2)
        # garde-fous : échelle 0,2–5
        sc = np.sqrt(np.abs(Aj[:, 0, 0] * Aj[:, 1, 1] - Aj[:, 0, 1] * Aj[:, 1, 0]))
        f = np.clip(sc, 0.2, 5.0) / np.maximum(sc, 1e-6)
        A[:, j] = (Aj * f[:, None, None]).astype(np.float32)
    sc = np.sqrt(np.abs(A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]))
    ls = np.log(np.maximum(sc, 1e-6))
    lo, hi = np.percentile(ls, [2, 98])
    if hi - lo < 0.05:
        depth = np.full((T, n), 0.5, np.float32)
    else:
        depth = np.clip((hi - ls) / (hi - lo), 0, 1).astype(np.float32)
    return A, depth


def valid_mask(res: "eng.TrackResult", cols: np.ndarray) -> np.ndarray:
    """Images où un point est suivi et n'a pas été coupé au décrochage."""
    T = len(res.tracked)
    ok = res.tracked[:, None] & np.isfinite(res.positions[:, cols, 0])
    if res.cut_frames is not None and res.query_frames is not None:
        tt = np.arange(T)[:, None]
        cut = res.cut_frames[cols][None]
        qf = res.query_frames[cols][None]
        fwd = (cut >= 0) & (cut > qf) & (tt >= cut)
        bwd = (cut >= 0) & (cut < qf) & (tt <= cut)
        ok &= ~(fwd | bwd)
    return ok


def prepare_group(res: "eng.TrackResult", cols: np.ndarray, st: ShapeStyle,
                  vis_threshold: float = 0.5, depth=None) -> GroupData:
    """depth : ignoré (ancienne profondeur IA) ; la profondeur vient du suivi."""
    cols = np.asarray(cols, int)
    sub = eng.TrackResult(res.positions[:, cols], res.visibility[:, cols], res.tracked)
    smoothed = eng.smooth_tracks(sub.positions, sub.visibility, st.smooth)
    velocity, direction = eng.compute_velocity(smoothed)
    if st.always_visible:
        ok = valid_mask(res, cols)
        alpha = ok.astype(np.float32)
        pos = smoothed.copy()
        speed = np.where(ok, velocity, 0.0).astype(np.float32)
    else:
        alpha = eng.visibility_envelope(sub.visibility, sub.tracked, vis_threshold,
                                        st.fade_in if st.fade_in_on else 1,
                                        st.fade_out if st.fade_out_on else 1)
        visible = sub.visibility >= vis_threshold
        pos = eng.hold_occluded_positions(smoothed, visible)
        speed = np.where(visible, velocity, 0.0).astype(np.float32)
    rng = np.random.default_rng(1234)
    jit = 1.0 + st.size_jitter * (rng.random(res.positions.shape[1]) * 2 - 1)
    ok = valid_mask(res, cols) & (sub.visibility >= vis_threshold)
    qf = res.query_frames[cols] if res.query_frames is not None else np.zeros(len(cols), int)
    A, d = local_affine(smoothed, ok, qf)
    return GroupData(pos, alpha.astype(np.float32), speed, direction,
                     jit[cols].astype(np.float32), cols, d, A)


# =============================================================================
# Rendu
# =============================================================================

class ShapeRenderer:
    """Rend une image de matte à partir des groupes (pixels source → sortie)."""

    SHIFT = 4

    def __init__(self, src_w: int, src_h: int, out_w: int, out_h: int,
                 work_max_side: int = 1920):
        self.src_w, self.src_h = src_w, src_h
        self.out_w, self.out_h = out_w, out_h
        self.rs = min(out_w / src_w, work_max_side / float(max(src_w, src_h)))
        self.w = max(1, int(round(src_w * self.rs)))
        self.h = max(1, int(round(src_h * self.rs)))
        self.vnorm = 1080.0 / src_h
        self._stamps: Dict[str, Optional[np.ndarray]] = {}

    # ------------------------------------------------------------- tampons
    def _stamp(self, path: str) -> Optional[np.ndarray]:
        if path not in self._stamps:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
            if img is not None:
                if img.ndim == 3 and img.shape[2] == 4:
                    g = img[..., 3]
                else:
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                if g.dtype != np.uint8:
                    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                s = 256.0 / max(g.shape)
                g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                               interpolation=cv2.INTER_AREA)
            else:
                g = None
            self._stamps[path] = g
        return self._stamps[path]

    def _draw(self, canvas: np.ndarray, st: ShapeStyle, cx: float, cy: float,
              Mx: np.ndarray, value: int) -> None:
        """Mx : matrice 2×2 forme unitaire → pixels de travail (taille, rotation,
        étirement, perspective)."""
        if st.shape == "image":
            stamp = self._stamp(st.image_path)
            if stamp is None:
                return self._draw_poly(canvas, "circle", cx, cy, Mx, value)
            h, w = stamp.shape
            k = 2.0 / max(h, w)
            A = Mx * k
            t = np.array([cx, cy]) - A @ np.array([w / 2, h / 2])
            M = np.hstack([A, t[:, None]]).astype(np.float32)
            r = int(math.ceil(np.linalg.norm(Mx, 2) * 1.5)) + 2
            x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
            x1, y1 = min(canvas.shape[1], int(cx) + r), min(canvas.shape[0], int(cy) + r)
            if x1 <= x0 or y1 <= y0:
                return
            M[:, 2] -= (x0, y0)
            warped = cv2.warpAffine(stamp, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            if value < 255:
                warped = (warped.astype(np.uint16) * value // 255).astype(np.uint8)
            roi = canvas[y0:y1, x0:x1]
            np.maximum(roi, warped, out=roi)
            return
        self._draw_poly(canvas, st.shape, cx, cy, Mx, value, ring=st.shape == "ring")

    def _draw_poly(self, canvas, shape, cx, cy, Mx, value, ring=False):
        f = float(1 << self.SHIFT)
        for poly in UNIT.get(shape, UNIT["circle"]):
            pts = poly @ Mx.T + (cx, cy)
            p = np.round(pts * f).astype(np.int32)
            if ring:
                th = max(1, int(round(0.3 * np.linalg.svd(Mx, compute_uv=False)[-1])))
                cv2.polylines(canvas, [p], True, value, th, cv2.LINE_AA, self.SHIFT)
            else:
                cv2.fillPoly(canvas, [p], value, cv2.LINE_AA, self.SHIFT)

    def _stamps_for(self, st: ShapeStyle, gd: GroupData, t: int, scale: float = 1.0,
                    shadow: bool = False):
        """Liste (cx, cy, matrice 2×2, alpha) en pixels de travail."""
        out = []
        T = len(gd.alpha)
        trail = max(0, int(st.trail))
        base = st.size * self.rs * scale
        n = len(gd.ids)
        # time-slice selon la profondeur : chaque point vit dans son propre passé
        tj0 = np.full(n, t, int)
        if st.echo_mode == "slit_depth" and gd.depth is not None and 0 <= t < T:
            dd = np.nan_to_num(gd.depth[t], nan=0.0)
            tj0 = t - np.round(dd * max(1, st.slit_span)).astype(int)
        ca, sa = math.cos(math.radians(st.shadow_angle)), math.sin(math.radians(st.shadow_angle))
        for k in range(0, trail + 1):
            fade = 1.0 - k / (trail + 1.0)
            for j in range(n):
                tk = int(tj0[j]) - k
                if tk < 0 or tk >= T:
                    continue
                al = float(gd.alpha[tk, j])
                x, y = gd.pos[tk, j]
                if al <= 1.0 / 255 or not np.isfinite(x):
                    continue
                spd = float(gd.speed[tk, j])
                v = spd * self.vnorm
                g = min(st.max_scale, 1.0 + st.grow * v)
                s = min(st.max_scale, 1.0 + st.stretch * v)
                r = base * float(gd.jitter[j]) * g * (fade if k else 1.0)
                d = float(gd.depth[tk, j]) if gd.depth is not None else float("nan")
                if np.isfinite(d):
                    if st.depth_scale > 0:
                        r *= (1 - st.depth_scale) + st.depth_scale * (1.6 - 1.2 * d)
                    if st.depth_near > 0 or st.depth_far < 1:
                        fe = max(1e-3, st.depth_feather)
                        a_in = np.clip((d - (st.depth_near - fe)) / fe, 0, 1)
                        a_out = np.clip(((st.depth_far + fe) - d) / fe, 0, 1)
                        al *= float(a_in * a_out)
                    if st.depth_fog > 0:
                        al *= 1.0 - st.depth_fog * d
                if al <= 1.0 / 255:
                    continue
                ang = st.rotation
                if (st.follow_motion or st.stretch > 0) and spd > 0.05:
                    dv = gd.direction[tk, j]
                    ang += math.degrees(math.atan2(dv[1], dv[0]))
                cx, cy = x * self.rs, y * self.rs
                sxx, syy = max(0.5, r * s * st.width), max(0.5, r * st.height)
                ra = math.radians(ang)
                Mx = np.array([[math.cos(ra) * sxx, -math.sin(ra) * syy],
                               [math.sin(ra) * sxx, math.cos(ra) * syy]])
                if st.perspective and gd.affine is not None:
                    Pm = np.eye(2) + st.perspective_amount * (gd.affine[tk, j].astype(np.float64) - np.eye(2))
                    Mx = Pm @ Mx
                if shadow:
                    dist = st.shadow_dist * self.rs
                    if st.shadow_depth and np.isfinite(d):
                        dist *= 0.25 + 1.5 * (1.0 - d)
                    cx, cy = cx + ca * dist, cy + sa * dist
                out.append((cx, cy, Mx, al * (fade if k else 1.0)))
        out.sort(key=lambda e: e[3])
        return out

    def _blur(self, img: np.ndarray, sigma: float) -> np.ndarray:
        x = img.astype(np.float32) * (1.0 / 255.0)
        if sigma <= 0.3:
            return x
        bs = min(1.0, 4.0 / sigma)
        if bs < 1.0:
            small = cv2.resize(x, (max(1, int(self.w * bs)), max(1, int(self.h * bs))),
                               interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigma * bs)
            return cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(x, (0, 0), sigma)

    def _blurf(self, x: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0.3:
            return x
        return self._blur((np.clip(x, 0, 1) * 255).astype(np.uint8), sigma)

    def _core(self, st: ShapeStyle, gd: GroupData, t: int, scale: float = 1.0,
              shadow: bool = False) -> np.ndarray:
        """Matte d'un groupe à l'instant t (sans écho, ombre ni opacité)."""
        stamps = self._stamps_for(st, gd, t, scale, shadow)
        if not stamps:
            return np.zeros((self.h, self.w), np.float32)
        base = max(0.5, st.size * self.rs * scale)
        if st.merge > 0:
            shape = np.zeros((self.h, self.w), np.uint8)
            amap = np.zeros_like(shape)
            cover = np.zeros_like(shape)
            for cx, cy, Mx, al in stamps:
                self._draw(shape, st, cx, cy, Mx, 255)
                self._draw(amap, st, cx, cy, Mx * 1.5, int(round(al * 255)))
                self._draw(cover, st, cx, cy, Mx * 1.5, 255)
            sigma = st.merge * base
            field_ = self._blur(shape, sigma)
            lo, hi = st.threshold - st.softness, st.threshold + st.softness
            if hi - lo < 1e-4:
                m = (field_ >= st.threshold).astype(np.float32)
            else:
                m = np.clip((field_ - lo) / (hi - lo), 0, 1)
                m = m * m * (3 - 2 * m)
            m *= np.clip(self._blur(amap, sigma) / np.maximum(self._blur(cover, sigma), 1e-3), 0, 1)
        else:
            canvas = np.zeros((self.h, self.w), np.uint8)
            for cx, cy, Mx, al in stamps:
                self._draw(canvas, st, cx, cy, Mx, int(round(al * 255)))
            m = self._blur(canvas, st.softness * base * 2.0)
        return m

    def _slit_map(self, mode: str) -> np.ndarray:
        key = (mode, self.w, self.h)
        if getattr(self, "_slit_key", None) != key:
            yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
            if mode == "slit_h":
                f = xx / max(1, self.w - 1)
            elif mode == "slit_v":
                f = yy / max(1, self.h - 1)
            else:
                cx, cy = (self.w - 1) / 2, (self.h - 1) / 2
                f = np.hypot(xx - cx, yy - cy) / math.hypot(cx, cy)
            self._slit_key, self._slit_f = key, f
        return self._slit_f

    def _timed(self, st: ShapeStyle, gd: GroupData, t: int) -> np.ndarray:
        """Écho temporel / slit-scan."""
        mode = st.echo_mode
        if mode == "echo":
            m = self._core(st, gd, t)
            for k in range(1, max(0, int(st.echo_count)) + 1):
                tk = t - k * max(1, int(st.echo_step))
                if tk < 0:
                    break
                mk = self._core(st, gd, tk, scale=st.echo_scale ** k) * (st.echo_decay ** k)
                m = np.maximum(m, mk)
            return m
        if mode in ("slit_h", "slit_v", "slit_radial"):
            span = max(1, int(st.slit_span))
            K = min(12, span + 1)
            times = [max(0, t - int(round(i * span / (K - 1)))) for i in range(K)]
            f = self._slit_map(mode) * (K - 1)
            out = np.zeros(f.shape, np.float32)
            for i, tk in enumerate(times):
                w = np.maximum(0.0, 1.0 - np.abs(f - i))
                if w.any():
                    out += self._core(st, gd, tk) * w
            return out
        return self._core(st, gd, t)

    def render_group(self, st: ShapeStyle, gd: GroupData, t: int) -> np.ndarray:
        """Matte float32 [h, w] (résolution de travail) d'un groupe."""
        m = self._timed(st, gd, t)
        if st.shadow_on and st.shadow_opacity > 0:
            if st.shadow_depth and gd.depth is not None:
                sh = self._core(st, gd, t, shadow=True)
            else:
                dist = st.shadow_dist * self.rs
                dx = math.cos(math.radians(st.shadow_angle)) * dist
                dy = math.sin(math.radians(st.shadow_angle)) * dist
                sh = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (self.w, self.h))
            sh = self._blurf(sh, st.shadow_soft * self.rs) * st.shadow_opacity
            m = m + sh * (1.0 - m)
        return m * float(st.opacity)

    def render(self, layers: Sequence[tuple], t: int, invert: bool = False,
               as_float: bool = False):
        """layers : séquence de (ShapeStyle, GroupData, actif). Sortie uint8
        [out_h, out_w] (ou float32 si as_float)."""
        add = None
        sub = None
        for st, gd, on in layers:
            if not on or gd is None or len(gd.ids) == 0:
                continue
            m = self.render_group(st, gd, t)
            if st.mode == "subtract":
                sub = m if sub is None else np.maximum(sub, m)
            else:
                add = m if add is None else np.maximum(add, m)
        if add is None:
            add = np.ones((self.h, self.w), np.float32) if sub is not None else \
                np.zeros((self.h, self.w), np.float32)
        out = add if sub is None else add * (1.0 - sub)
        if invert:
            out = 1.0 - out
        if (self.w, self.h) != (self.out_w, self.out_h):
            out = cv2.resize(out, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        if as_float:
            return out
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


    # --------------------------------------------------------- rush masqué
    def composite(self, layers: Sequence[tuple], t: int,
                  get_frame: Callable[[int], Optional[np.ndarray]],
                  background: str = "transparent"):
        """RUSH MASQUÉ + EFFETS : l'image d'origine visible dans les formes, et
        les effets appliqués à cette image (pas seulement au masque) :
        • écho : les images passées du sujet masqué, superposées et estompées ;
        • slit-scan : chaque zone montre le sujet masqué à un autre instant ;
        • ombre portée : l'ombre du sujet découpé sur le fond.
        background : transparent · black · original · dim · blur.
        → (BGR uint8 [out_h, out_w, 3], alpha uint8 [out_h, out_w])."""
        cache: Dict[int, Optional[np.ndarray]] = {}

        def img(tt: int) -> Optional[np.ndarray]:
            if tt not in cache:
                f = get_frame(tt) if tt >= 0 else None
                if f is not None and f.shape[:2] != (self.h, self.w):
                    f = cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
                cache[tt] = None if f is None else f.astype(np.float32) * (1.0 / 255.0)
            return cache[tt]

        cur = img(t)
        if cur is None:
            cur = np.zeros((self.h, self.w, 3), np.float32)
        if background == "original":
            col = cur.copy()
        elif background == "dim":
            col = cur * 0.3
        elif background == "blur":
            col = cv2.GaussianBlur(cur, (0, 0), max(2.0, self.w / 90.0))
        else:
            col = np.zeros_like(cur)
        bg = col.copy()
        alpha = np.zeros((self.h, self.w), np.float32)

        def over(c, a, src, m):
            m3 = m[..., None]
            return c * (1 - m3) + src * m3, np.maximum(a, m)

        act = [(st, gd) for st, gd, on in layers if on and gd is not None and len(gd.ids)]
        adds = [(st, gd) for st, gd in act if st.mode != "subtract"]
        subs = [(st, gd) for st, gd in act if st.mode == "subtract"]
        if not adds and subs:
            col, alpha = cur.copy(), np.ones_like(alpha)
        for st, gd in adds:
            base = lambda tt, sc=1.0: self._core(st, gd, tt, scale=sc) * float(st.opacity)
            m_now = base(t)
            mode = st.echo_mode
            fx_m = m_now
            layers_fx = []
            if mode == "echo":
                for k in range(max(0, int(st.echo_count)), 0, -1):    # du plus ancien au plus récent
                    tk = t - k * max(1, int(st.echo_step))
                    src = img(tk)
                    if tk < 0 or src is None:
                        continue
                    mk = base(tk, st.echo_scale ** k) * (st.echo_decay ** k)
                    layers_fx.append((src, mk))
                    fx_m = np.maximum(fx_m, mk)
            elif mode in ("slit_h", "slit_v", "slit_radial"):
                span = max(1, int(st.slit_span))
                K = min(12, span + 1)
                f = self._slit_map(mode) * (K - 1)
                m_sl = np.zeros_like(alpha)
                c_sl = np.zeros_like(cur)
                for i in range(K):
                    w = np.maximum(0.0, 1.0 - np.abs(f - i))
                    if not w.any():
                        continue
                    tk = max(0, t - int(round(i * span / (K - 1))))
                    src = img(tk)
                    mi = base(tk) * w
                    m_sl += mi
                    c_sl += (src if src is not None else cur) * mi[..., None]
                c_sl = c_sl / np.maximum(m_sl, 1e-6)[..., None]
                m_now, cur_g = m_sl, c_sl
                fx_m = m_sl
            elif mode == "slit_depth":
                m_now = self._timed(st, gd, t) * float(st.opacity)
                fx_m = m_now
            # ombre portée du sujet découpé (sous tout le reste)
            if st.shadow_on and st.shadow_opacity > 0:
                dist = st.shadow_dist * self.rs
                dx = math.cos(math.radians(st.shadow_angle)) * dist
                dy = math.sin(math.radians(st.shadow_angle)) * dist
                sh = cv2.warpAffine(fx_m, np.float32([[1, 0, dx], [0, 1, dy]]), (self.w, self.h))
                sh = self._blurf(sh, st.shadow_soft * self.rs) * st.shadow_opacity
                col = col * (1 - sh[..., None])
                alpha = np.maximum(alpha, sh)
            for src, mk in layers_fx:
                col, alpha = over(col, alpha, src, mk)
            col, alpha = over(col, alpha, cur_g if mode in ("slit_h", "slit_v", "slit_radial") else cur, m_now)
        if subs:
            hole = np.zeros_like(alpha)
            for st, gd in subs:
                hole = np.maximum(hole, self._core(st, gd, t) * float(st.opacity))
            col = col * (1 - hole[..., None]) + bg * hole[..., None]
            alpha = alpha * (1 - hole)
        if background != "transparent":
            alpha = np.ones_like(alpha)
        out = np.clip(col * 255 + 0.5, 0, 255).astype(np.uint8)
        a8 = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
        if (self.w, self.h) != (self.out_w, self.out_h):
            out = cv2.resize(out, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
            a8 = cv2.resize(a8, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
        return out, a8


def frames_back(layers: Sequence[tuple]) -> int:
    """Nombre d'images passées nécessaires au rendu masqué."""
    n = 0
    for st, gd, on in layers:
        if not on:
            continue
        if st.echo_mode == "echo":
            n = max(n, int(st.echo_count) * max(1, int(st.echo_step)))
        elif st.echo_mode.startswith("slit"):
            n = max(n, int(st.slit_span))
    return n


def write_composite_video(out_base: str, info: "eng.VideoInfo", codec: str, layers: Sequence[tuple],
                          background: str = "transparent", progress=None, cancel=None) -> str:
    """Vidéo « rush masqué + effets » pleine résolution. Fond transparent →
    ProRes 4444 avec alpha (ou PNG avec alpha) ; sinon le codec choisi."""
    T = len(layers[0][1].alpha) if layers else 0
    ren = ShapeRenderer(info.width, info.height, info.width, info.height,
                        work_max_side=max(info.width, info.height))
    alpha_out = background == "transparent"
    if alpha_out and codec not in ("png",):
        codec = "prores4444"
    writer = eng.MatteWriter(out_base, info, codec, suffix="_masque",
                             gray=False, channels=4 if alpha_out else 3)
    back = frames_back(layers)
    buf: Dict[int, np.ndarray] = {}
    try:
        for t, frame in eng.iter_frames(info.path):
            if cancel is not None and cancel.is_set():
                raise eng.Cancelled()
            buf[t] = frame
            for k in [k for k in buf if k < t - back]:
                del buf[k]
            if t < T:
                rgb, a = ren.composite(layers, t, buf.get, background)
            else:
                rgb, a = frame, np.full(frame.shape[:2], 255, np.uint8)
            writer.write(np.dstack([rgb, a]) if alpha_out else rgb)
            if progress is not None and (t % 4 == 0):
                progress("Rendu du rush masqué", (t + 1) / max(1, info.frame_count or T))
    finally:
        writer.close()
    return writer.path


# =============================================================================
# Fichier de suivi pour l'effet OFX « TAPNext Shapes » (.tapfx)
# =============================================================================

def export_tapfx(path: str, res: "eng.TrackResult", point_groups: np.ndarray,
                 width: int, height: int, fps: float, depth=None) -> str:
    """Écrit le fichier binaire lu par le plugin OFX (voir TAPNextShapes.cpp),
    format v3 : + profondeur et déformation locale (perspective) DÉDUITES DU
    SUIVI, calculées groupe par groupe comme dans Studio.

    point_groups : [Q] numéro de groupe (0, 1, 2… dans l'ordre de Studio).
    depth : ignoré (ancienne profondeur IA).
    """
    T, Q = res.visibility.shape
    valid = valid_mask(res, np.arange(Q))
    dep = np.full((T, Q), np.nan, np.float32)
    aff = np.tile(np.eye(2, dtype=np.float32), (T, Q, 1, 1))
    qf = res.query_frames if res.query_frames is not None else np.zeros(Q, int)
    for g in np.unique(point_groups):
        cols = np.nonzero(point_groups == g)[0]
        if len(cols) < 4:
            continue
        sm = eng.smooth_tracks(res.positions[:, cols], res.visibility[:, cols], 2.1)
        ok = valid[:, cols] & (res.visibility[:, cols] >= 0.5)
        A, d = local_affine(sm, ok, qf[cols])
        aff[:, cols] = A
        if d is not None:
            dep[:, cols] = d
    with open(path, "wb") as f:
        f.write(b"TAPFX03\0")
        np.array([width, height, T, Q, 0, 0, 0, 0], "<i4").tofile(f)
        np.array([fps], "<f4").tofile(f)
        np.ascontiguousarray(res.positions, "<f4").tofile(f)
        np.ascontiguousarray(res.visibility, "<f4").tofile(f)
        np.ascontiguousarray(valid, np.uint8).tofile(f)
        np.ascontiguousarray(point_groups, "<i4").tofile(f)
        np.ascontiguousarray(dep, "<f4").tofile(f)
        np.ascontiguousarray(aff.reshape(T, Q, 4), "<f4").tofile(f)
    remember_last_tapfx(path)
    return path


def last_tapfx_file(name: str = "last_tapfx.txt") -> str:
    """Fichier où l'on note le dernier export (lu par l'effet OFX quand son
    champ « Fichier de suivi » est vide)."""
    import os
    import sys
    if sys.platform.startswith("win"):
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "TAPNext")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support/TAPNext")
    else:
        base = os.path.expanduser("~/.config/TAPNext")
    return os.path.join(base, name)


def remember_last_tapfx(path: str, name: str = "last_tapfx.txt") -> None:
    import os
    try:
        f = last_tapfx_file(name)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(os.path.abspath(path))
    except OSError:
        pass


# =============================================================================
# Export vidéo
# =============================================================================

def write_shapes_video(out_base: str, info: "eng.VideoInfo", codec: str,
                       layers: Sequence[tuple], invert: bool,
                       preview: bool = False, per_group_names: Optional[List[str]] = None,
                       progress=None, cancel=None) -> List[str]:
    """Matte combinée (+ option : une matte par groupe, vidéo de contrôle)."""
    T = len(layers[0][1].alpha) if layers else 0
    ren = ShapeRenderer(info.width, info.height, info.width, info.height,
                        work_max_side=max(info.width, info.height))
    writer = eng.MatteWriter(out_base, info, codec)
    outputs = [writer.path]
    group_writers = []
    if per_group_names:
        for (st, gd, on), name in zip(layers, per_group_names):
            if on and gd is not None and len(gd.ids):
                ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
                safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in ascii_name) or "groupe"
                gw = eng.MatteWriter(out_base, info, codec, suffix=f"_matte_{safe}")
                group_writers.append((st, gd, gw))
                outputs.append(gw.path)
    pv = None
    src = None
    if preview:
        pv = eng.MatteWriter(out_base, info, "h264", suffix="_preview", gray=False)
        outputs.append(pv.path)
        src = eng.iter_frames(info.path)
    try:
        for t in range(T):
            if cancel is not None and cancel.is_set():
                raise eng.Cancelled()
            m = ren.render(layers, t, invert)
            writer.write(m)
            for st, gd, gw in group_writers:
                gm = ren.render([(st.__class__(**{**st.to_dict(), "mode": "add"}), gd, True)], t)
                gw.write(gm)
            if pv is not None:
                nxt = next(src, None)
                frame = nxt[1] if nxt else np.zeros((info.height, info.width, 3), np.uint8)
                a = (m.astype(np.float32) / 255.0 * 0.5)[..., None]
                tint = np.zeros_like(frame)
                tint[..., 2] = 255
                tint[..., 1] = 70
                pv.write((frame * (1 - a) + tint * a).astype(np.uint8))
            if progress is not None and (t % 4 == 0 or t == T - 1):
                progress("Rendu des formes", (t + 1) / max(1, T))
    finally:
        writer.close()
        for _, _, gw in group_writers:
            gw.close()
        if pv is not None:
            pv.close()
    return outputs
