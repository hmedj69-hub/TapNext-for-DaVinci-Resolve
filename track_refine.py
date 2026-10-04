# -*- coding: utf-8 -*-
"""
track_refine.py — Affinage sous-pixel des trajectoires TAPNext++.

TAPNext++ est très robuste (suivi long, occultations, re-détection) mais sa
précision est limitée : il prédit dans un espace 256×256, soit ±4 px environ
en 1080p (±8 px en 4K). Pour stabiliser ou faire du match-move, il faut
mieux que le pixel.

Méthode hybride (inspirée des trackers « coarse-to-fine ») :
  • TAPNext++ fournit l'ancrage : position approximative, visibilité, et
    surtout l'identité du point sur toute la durée du plan.
  • Un flux optique local Lucas-Kanade pyramidal, calculé image par image sur
    l'image haute résolution, donne la précision sous-pixel.
  • Garde-fous : contrôle aller-retour du flux, et si le résultat s'écarte de
    l'ancrage TAPNext++ de plus de ~2 « pixels modèle », on se recale sur
    TAPNext++.
  • Fusion par filtre complémentaire : hautes fréquences (mouvement fin) de
    Lucas-Kanade, basses fréquences (position absolue) de TAPNext++. On
    obtient la stabilité image-à-image du flux optique sans sa dérive.
  • La position cliquée par l'utilisateur sur son image de pose est exacte :
    le segment de trajectoire qui la contient est recalé dessus.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import tap_resolve_tool as eng


def _conv_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convolution centrée de même longueur que x (même si k est plus long)."""
    r = len(k) // 2
    return np.convolve(x, k, mode="full")[r:r + len(x)]


def refine_tracks(path: str, res: "eng.TrackResult", seg_in: int, seg_out: int,
                  width: int, height: int, query_xy: Optional[np.ndarray] = None,
                  progress=None, cancel=None, max_side: int = 1920,
                  fusion_sigma: float = 10.0) -> dict:
    """Affine res.positions en place (les positions brutes TAPNext++ sont
    gardées dans res.raw_positions). Renvoie quelques statistiques."""
    T, Q = res.visibility.shape
    s = min(1.0, max_side / float(max(width, height)))
    raw = res.positions.copy()
    anchor = eng.smooth_tracks(raw, res.visibility, 2.0)       # ancrage lissé
    vis = (res.visibility >= 0.5) & np.isfinite(raw[..., 0])
    gate = max(3.0, 2.0 * max(width, height) / 256.0)           # ~2 pixels modèle
    out = raw.copy()
    chain = np.full((T, Q), -1, np.int64)
    cur_chain = np.full(Q, -1, np.int64)
    next_id = 0
    prev_gray = None
    prev_pts = np.full((Q, 2), np.nan, np.float32)
    prev_ok = np.zeros(Q, bool)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    n_lk = n_reset = 0
    total = max(1, seg_out - seg_in + 1)
    for t, frame in eng.iter_frames(path, seg_in, seg_out):
        if cancel is not None and cancel.is_set():
            raise eng.Cancelled()
        if s < 1.0:
            frame = cv2.resize(frame, (int(round(width * s)), int(round(height * s))),
                               interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok_t = vis[t]
        new = np.full((Q, 2), np.nan, np.float32)
        if prev_gray is not None:
            idx = np.nonzero(prev_ok & ok_t)[0]
            if idx.size:
                p0 = (prev_pts[idx] * s).astype(np.float32).reshape(-1, 1, 2)
                # Prédiction : mouvement de l'ancrage TAPNext++ entre t-1 et t.
                d = (anchor[t, idx] - anchor[t - 1, idx]) * s
                d = np.where(np.isfinite(d), d, 0.0)
                init = (p0[:, 0] + d).astype(np.float32).reshape(-1, 1, 2)
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, g, p0, init.copy(), **lk)
                p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g, prev_gray, p1, p0.copy(), **lk)
                fb = np.linalg.norm((p0b - p0).reshape(-1, 2), axis=1)
                p1s = p1.reshape(-1, 2) / s
                dist = np.linalg.norm(p1s - anchor[t, idx], axis=1)
                good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.7) & (dist < gate)
                new[idx[good]] = p1s[good]
                n_lk += int(good.sum())
                n_reset += int((~good).sum())
        # Points visibles sans continuité : nouveau segment, départ sur l'ancrage.
        start = ok_t & ~np.isfinite(new[:, 0])
        if start.any():
            new[start] = anchor[t, start]
            k = int(start.sum())
            cur_chain[start] = np.arange(next_id, next_id + k)
            next_id += k
        cur_chain[~ok_t] = -1
        out[t, ok_t] = new[ok_t]
        chain[t] = cur_chain
        prev_gray, prev_pts, prev_ok = g, new, ok_t
        if progress is not None and (t - seg_in) % 8 == 0:
            progress("Affinage sous-pixel", (t - seg_in + 1) / total)

    # Filtre complémentaire, segment par segment : on retire à la chaîne LK la
    # composante basse fréquence de son écart à TAPNext++.
    if fusion_sigma > 0:
        k = np.exp(-0.5 * (np.arange(-int(3 * fusion_sigma), int(3 * fusion_sigma) + 1)
                           / fusion_sigma) ** 2)
        w_all = np.clip(res.visibility, 1e-3, 1.0)
        for i in range(Q):
            for c in np.unique(chain[:, i][chain[:, i] >= 0]):
                idx = np.nonzero(chain[:, i] == c)[0]
                if idx.size < 3:
                    continue
                r = out[idx, i] - raw[idx, i]
                w = w_all[idx, i]
                den = _conv_same(w, k)
                lp = np.stack([_conv_same(r[:, d] * w, k) for d in (0, 1)], -1)
                out[idx, i] -= lp / np.maximum(den, 1e-6)[:, None]

    # Recalage sur la position exacte posée par l'utilisateur.
    if query_xy is not None and res.query_frames is not None:
        for i in range(Q):
            q = int(res.query_frames[i])
            c = chain[q, i] if 0 <= q < T else -1
            if c < 0:
                continue
            delta = query_xy[i] - out[q, i]
            if np.all(np.isfinite(delta)) and np.linalg.norm(delta) < gate * 2:
                m = chain[:, i] == c
                out[m, i] += delta
    res.raw_positions = raw
    res.positions = out
    moved = np.linalg.norm(out - raw, axis=-1)[vis]
    return dict(lk=n_lk, resets=n_reset,
                mean_shift=float(np.nanmean(moved)) if moved.size else 0.0)
