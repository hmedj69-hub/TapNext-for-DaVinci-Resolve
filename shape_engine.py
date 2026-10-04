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
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

import tap_resolve_tool as eng

SHAPES = ["circle", "square", "rounded", "diamond", "triangle", "hexagon",
          "star", "cross", "ring", "image"]
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
    rotation: float = 0.0             # degrés
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
                  vis_threshold: float = 0.5) -> GroupData:
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
    return GroupData(pos, alpha.astype(np.float32), speed, direction,
                     jit[cols].astype(np.float32), cols)


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
              sx: float, sy: float, ang: float, value: int) -> None:
        if st.shape == "image":
            stamp = self._stamp(st.image_path)
            if stamp is None:
                return self._draw_poly(canvas, "circle", cx, cy, sx, sy, ang, value)
            h, w = stamp.shape
            k = 2.0 / max(h, w)
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            A = np.array([[ca * sx, -sa * sy], [sa * sx, ca * sy]]) * k
            t = np.array([cx, cy]) - A @ np.array([w / 2, h / 2])
            M = np.hstack([A, t[:, None]]).astype(np.float32)
            r = int(math.ceil(max(sx, sy) * 1.5)) + 2
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
        self._draw_poly(canvas, st.shape, cx, cy, sx, sy, ang, value, ring=st.shape == "ring")

    def _draw_poly(self, canvas, shape, cx, cy, sx, sy, ang, value, ring=False):
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        R = np.array([[ca, -sa], [sa, ca]])
        f = float(1 << self.SHIFT)
        for poly in UNIT.get(shape, UNIT["circle"]):
            pts = (poly * (sx, sy)) @ R.T + (cx, cy)
            p = np.round(pts * f).astype(np.int32)
            if ring:
                th = max(1, int(round(0.3 * min(sx, sy))))
                cv2.polylines(canvas, [p], True, value, th, cv2.LINE_AA, self.SHIFT)
            else:
                cv2.fillPoly(canvas, [p], value, cv2.LINE_AA, self.SHIFT)

    def _stamps_for(self, st: ShapeStyle, gd: GroupData, t: int):
        """Liste (cx, cy, sx, sy, angle, alpha) en pixels de travail."""
        out = []
        trail = max(0, int(st.trail))
        base = st.size * self.rs
        for k in range(0, trail + 1):
            tk = t - k
            if tk < 0:
                break
            fade = 1.0 - k / (trail + 1.0)
            pos, al, sp, dv = gd.pos[tk], gd.alpha[tk], gd.speed[tk], gd.direction[tk]
            for j in np.nonzero((al > 1.0 / 255) & np.isfinite(pos[:, 0]))[0]:
                v = float(sp[j]) * self.vnorm
                g = min(st.max_scale, 1.0 + st.grow * v)
                s = min(st.max_scale, 1.0 + st.stretch * v)
                r = base * float(gd.jitter[j]) * g * (fade if k else 1.0)
                ang = st.rotation
                if (st.follow_motion or st.stretch > 0) and sp[j] > 0.05:
                    ang += math.degrees(math.atan2(dv[j, 1], dv[j, 0]))
                out.append((pos[j, 0] * self.rs, pos[j, 1] * self.rs, max(0.5, r * s), max(0.5, r),
                            ang, float(al[j]) * (fade if k else 1.0)))
        out.sort(key=lambda e: e[5])
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

    def render_group(self, st: ShapeStyle, gd: GroupData, t: int) -> np.ndarray:
        """Matte float32 [h, w] (résolution de travail) d'un groupe."""
        stamps = self._stamps_for(st, gd, t)
        if not stamps:
            return np.zeros((self.h, self.w), np.float32)
        base = max(0.5, st.size * self.rs)
        if st.merge > 0:
            shape = np.zeros((self.h, self.w), np.uint8)
            amap = np.zeros_like(shape)
            cover = np.zeros_like(shape)
            for cx, cy, sx, sy, ang, al in stamps:
                self._draw(shape, st, cx, cy, sx, sy, ang, 255)
                self._draw(amap, st, cx, cy, sx * 1.5, sy * 1.5, ang, int(round(al * 255)))
                self._draw(cover, st, cx, cy, sx * 1.5, sy * 1.5, ang, 255)
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
            for cx, cy, sx, sy, ang, al in stamps:
                self._draw(canvas, st, cx, cy, sx, sy, ang, int(round(al * 255)))
            m = self._blur(canvas, st.softness * base * 2.0)
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


# =============================================================================
# Fichier de suivi pour l'effet OFX « TAPNext Shapes » (.tapfx)
# =============================================================================

def export_tapfx(path: str, res: "eng.TrackResult", point_groups: np.ndarray,
                 width: int, height: int, fps: float) -> str:
    """Écrit le fichier binaire lu par le plugin OFX (voir TAPNextShapes.cpp).

    point_groups : [Q] numéro de groupe (0, 1, 2… dans l'ordre de Studio).
    """
    T, Q = res.visibility.shape
    valid = valid_mask(res, np.arange(Q))
    with open(path, "wb") as f:
        f.write(b"TAPFX01\0")
        np.array([width, height, T, Q, 0, 0, 0, 0], "<i4").tofile(f)
        np.array([fps], "<f4").tofile(f)
        np.ascontiguousarray(res.positions, "<f4").tofile(f)
        np.ascontiguousarray(res.visibility, "<f4").tofile(f)
        np.ascontiguousarray(valid, np.uint8).tofile(f)
        np.ascontiguousarray(point_groups, "<i4").tofile(f)
    remember_last_tapfx(path)
    return path


def last_tapfx_file() -> str:
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
    return os.path.join(base, "last_tapfx.txt")


def remember_last_tapfx(path: str) -> None:
    import os
    try:
        f = last_tapfx_file()
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
