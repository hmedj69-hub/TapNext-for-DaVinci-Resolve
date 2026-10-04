# -*- coding: utf-8 -*-
"""
camera_solver.py — Tracker 3D : caméra + points 3D à partir des trajectoires.

« Structure from motion » incrémental, comme les camera trackers (Fusion
Camera Tracker, SynthEyes, 3DEqualizer, Blender) :

  1. Couple initial d'images avec assez de parallaxe (matrice essentielle).
  2. Triangulation des premiers points 3D.
  3. Images clés suivantes : pose par PnP + RANSAC, triangulation des
     nouveaux points (angle de vue suffisant, erreur de reprojection faible).
  4. Ajustement de faisceaux (bundle adjustment, moindres carrés robustes
     creux) sur les images clés : poses, points 3D et focale.
  5. Pose de chaque image par PnP sur les points 3D finaux.

Les trajectoires longues de TAPNext++ (un même point vu pendant des
centaines d'images, re-détecté après occultation) sont idéales : beaucoup
d'observations par point = géométrie très contrainte.

Sans parallaxe (caméra sur pied qui pivote, zoom), la profondeur n'est pas
mesurable : on le détecte et on résout une caméra en rotation pure.

Repère : OpenCV (x droite, y bas, z devant la caméra). R, t : monde → caméra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraSolve:
    mode: str                    # "3d" | "rotation" | "échec"
    K: np.ndarray                # [3, 3]
    R: np.ndarray                # [T, 3, 3] monde → caméra (NaN si inconnue)
    t: np.ndarray                # [T, 3]
    X: np.ndarray                # [Q, 3] points 3D (NaN si non triangulés)
    rms: float                   # erreur de reprojection moyenne (px)
    frames: Tuple[int, int]
    width: int
    height: int
    message: str = ""

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def ok(self, t: int) -> bool:
        return 0 <= t < len(self.R) and bool(np.isfinite(self.R[t, 0, 0]))

    def depth(self) -> np.ndarray:
        """Profondeur caméra z de chaque point à chaque image [T, Q]."""
        T, Q = len(self.R), len(self.X)
        out = np.full((T, Q), np.nan)
        okX = np.isfinite(self.X[:, 0])
        for t in range(T):
            if self.ok(t):
                out[t, okX] = (self.X[okX] @ self.R[t].T + self.t[t])[:, 2]
        return out

    def project(self, t: int, X: np.ndarray) -> np.ndarray:
        Xc = X @ self.R[t].T + self.t[t]
        p = Xc @ self.K.T
        return p[:, :2] / p[:, 2:3]

    def center(self, t: int) -> np.ndarray:
        return -self.R[t].T @ self.t[t]


# =============================================================================
# Outils
# =============================================================================

def _K(f: float, W: int, H: int) -> np.ndarray:
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], float)


def _rodrigues_batch(r: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(r, axis=1, keepdims=True)
    k = r / np.maximum(th, 1e-12)
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, :, None], np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _triangulate(K, R1, t1, R2, t2, p1, p2):
    P1 = K @ np.hstack([R1, t1[:, None]])
    P2 = K @ np.hstack([R2, t2[:, None]])
    Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(float), p2.T.astype(float))
    return (Xh[:3] / Xh[3]).T


def _reproj_err(K, R, t, X, p):
    Xc = X @ R.T + t
    q = Xc @ K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        q = q[:, :2] / q[:, 2:3]
    return np.linalg.norm(q - p, axis=1), Xc[:, 2]


def _angle(C1, C2, X):
    a = X - C1
    b = X - C2
    cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# =============================================================================
# Ajustement de faisceaux
# =============================================================================

def bundle_adjust(K, kf, Rs, ts, X, obs, refine_focal=True, fix_first=True, max_nfev=60):
    """obs : liste (indice image clé, indice point, x, y)."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    obs = np.asarray(obs, float)
    ci = obs[:, 0].astype(int)
    pi = obs[:, 1].astype(int)
    uv = obs[:, 2:4]
    nc, npnt = len(kf), len(X)
    rv = np.array([cv2.Rodrigues(R)[0].ravel() for R in Rs])
    x0 = np.concatenate([rv.ravel(), np.asarray(ts).ravel(), X.ravel(), [K[0, 0]]])
    cx, cy = K[0, 2], K[1, 2]

    def unpack(x):
        r = x[:nc * 3].reshape(nc, 3)
        tt = x[nc * 3:nc * 6].reshape(nc, 3)
        P = x[nc * 6:nc * 6 + npnt * 3].reshape(npnt, 3)
        f = x[-1] if refine_focal else K[0, 0]
        if fix_first:
            r = r.copy(); tt = tt.copy()
            r[0], tt[0] = rv[0], np.asarray(ts)[0]
        return r, tt, P, f

    def resid(x):
        r, tt, P, f = unpack(x)
        Rm = _rodrigues_batch(r)
        Xc = np.einsum("nij,nj->ni", Rm[ci], P[pi]) + tt[ci]
        z = np.where(np.abs(Xc[:, 2]) < 1e-6, 1e-6, Xc[:, 2])
        u = f * Xc[:, 0] / z + cx
        v = f * Xc[:, 1] / z + cy
        return np.concatenate([u - uv[:, 0], v - uv[:, 1]])

    m = len(obs)
    A = lil_matrix((2 * m, len(x0)), dtype=int)
    rows = np.arange(m)
    for k in range(3):
        A[rows, ci * 3 + k] = 1
        A[m + rows, ci * 3 + k] = 1
        A[rows, nc * 3 + ci * 3 + k] = 1
        A[m + rows, nc * 3 + ci * 3 + k] = 1
        A[rows, nc * 6 + pi * 3 + k] = 1
        A[m + rows, nc * 6 + pi * 3 + k] = 1
    if refine_focal:
        A[:, -1] = 1
    res = least_squares(resid, x0, jac_sparsity=A, loss="huber", f_scale=2.0,
                        x_scale="jac", method="trf", max_nfev=max_nfev, verbose=0)
    r, tt, P, f = unpack(res.x)
    Rm = _rodrigues_batch(r)
    Kn = _K(f, int(round(cx * 2)), int(round(cy * 2)))
    rr = resid(res.x)
    err = np.sqrt(rr[:m] ** 2 + rr[m:] ** 2)
    return Kn, list(Rm), list(tt), P, err


# =============================================================================
# Solveur
# =============================================================================

def solve_camera(pos: np.ndarray, valid: np.ndarray, W: int, H: int, t0: int, t1: int,
                 fov_deg: Optional[float] = None, cols: Optional[np.ndarray] = None,
                 progress=None, max_keyframes: int = 60) -> CameraSolve:
    """pos [T, Q, 2], valid [T, Q]. fov_deg : angle de champ horizontal, ou
    None pour l'estimer (essais + ajustement de la focale)."""
    T, Qall = valid.shape
    if cols is None:
        cols = np.arange(Qall)
    P = pos[:, cols].astype(float)
    V = valid[:, cols] & np.isfinite(P[..., 0])

    def say(msg, f):
        if progress is not None:
            progress(msg, f)

    def attempt(fov, kfs, nfev):
        f = (W / 2.0) / math.tan(math.radians(fov) / 2)
        return _solve_once(P, V, W, H, t0, t1, f, kfs, nfev)

    if fov_deg:
        best = attempt(fov_deg, max_keyframes, 80)
    else:
        # Recherche de la focale (l'ajustement de faisceaux seul la corrige mal :
        # elle est très corrélée à la profondeur). Essais rapides sur peu
        # d'images clés, affinage par section dorée, puis résolution finale.
        scores = {}

        def score(fov):
            if fov not in scores:
                sol = attempt(fov, 20, 30)
                scores[fov] = sol.rms if sol is not None else float("inf")
            return scores[fov]

        grid = [30.0, 42.0, 55.0, 68.0, 85.0]
        for k, fov in enumerate(grid):
            say("Caméra 3D : recherche de la focale", 0.6 * k / len(grid))
            score(fov)
        g0 = min(scores, key=scores.get)
        if math.isfinite(scores[g0]):
            i = grid.index(g0)
            lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
            gr = (math.sqrt(5) - 1) / 2
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
            for it in range(6):
                say("Caméra 3D : affinage de la focale", 0.6 + 0.25 * it / 6)
                if score(c) < score(d):
                    hi, d = d, c
                    c = hi - gr * (hi - lo)
                else:
                    lo, c = c, d
                    d = lo + gr * (hi - lo)
            fov_best = min(scores, key=scores.get)
            say("Caméra 3D : résolution finale", 0.9)
            best = attempt(fov_best, max_keyframes, 80)
        else:
            best = None
    if best is None:
        say("Caméra 3D : rotation pure", 0.9)
        f = (W / 2.0) / math.tan(math.radians(fov_deg or 55.0) / 2)
        best = _solve_rotation(P, V, W, H, t0, t1, f)
    # points 3D dans la numérotation complète
    Xfull = np.full((Qall, 3), np.nan)
    Xfull[cols] = best.X
    best.X = Xfull
    say("Caméra 3D : terminé", 1.0)
    return best


def _solve_once(P, V, W, H, t0, t1, f, max_keyframes, nfev=80):
    T, Q = V.shape
    K = _K(f, W, H)
    step = max(1, (t1 - t0) // max_keyframes)
    kfs = list(range(t0, t1 + 1, step))
    if kfs[-1] != t1:
        kfs.append(t1)
    # 1) couple initial : parallaxe suffisante, beaucoup de points communs
    pair = None
    for ia in range(min(len(kfs), 8)):
        a = kfs[ia]
        for b in kfs[ia + 1:]:
            c = V[a] & V[b]
            if c.sum() < 25:
                break
            pa, pb = P[a, c], P[b, c]
            if np.median(np.linalg.norm(pa - pb, axis=1)) < 0.01 * W:
                continue
            E, me = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.0)
            if E is None or E.shape != (3, 3):
                continue
            Hm, mh = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
            n_e, n_h = int(me.sum()), int(mh.sum()) if mh is not None else 0
            if n_e < 20 or n_h > 0.85 * n_e:       # trop « plat » / rotation pure
                continue
            _, R, tv, mp = cv2.recoverPose(E, pa, pb, K, mask=me.copy())
            idx = np.nonzero(c)[0][mp.ravel() > 0]
            if len(idx) < 20:
                continue
            Xi = _triangulate(K, np.eye(3), np.zeros(3), R, tv.ravel(), P[a, idx], P[b, idx])
            ang = _angle(np.zeros(3), -R.T @ tv.ravel(), Xi)
            if np.median(ang) < 1.0:
                continue
            score = len(idx) * min(np.median(ang), 6.0)
            if pair is None or score > pair[0]:
                pair = (score, a, b, R, tv.ravel(), idx, Xi)
        if pair is not None and pair[0] > 200:
            break
    if pair is None:
        return None
    _, a, b, R, tv, idx, Xi = pair
    X = np.full((Q, 3), np.nan)
    e1, z1 = _reproj_err(K, np.eye(3), np.zeros(3), Xi, P[a, idx])
    e2, z2 = _reproj_err(K, R, tv, Xi, P[b, idx])
    good = (e1 < 2) & (e2 < 2) & (z1 > 0) & (z2 > 0)
    X[idx[good]] = Xi[good]
    poses = {a: (np.eye(3), np.zeros(3)), b: (R, tv)}

    # 2) images clés : PnP + triangulation incrémentale
    order = [k for k in kfs if k > a and k != b] + [k for k in reversed(kfs) if k < a]
    for k in order:
        have = V[k] & np.isfinite(X[:, 0])
        if have.sum() < 8:
            continue
        ok, rv, tvk, inl = cv2.solvePnPRansac(X[have], P[k, have], K, None,
                                              reprojectionError=2.0, iterationsCount=200,
                                              flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 8:
            continue
        hi = np.nonzero(have)[0][inl.ravel()]
        ok, rv, tvk = cv2.solvePnP(X[hi], P[k, hi], K, None, rv, tvk, True, cv2.SOLVEPNP_ITERATIVE)
        Rk = cv2.Rodrigues(rv)[0]
        poses[k] = (Rk, tvk.ravel())
        # nouveaux points avec l'image clé déjà posée la plus éloignée
        new = V[k] & ~np.isfinite(X[:, 0])
        if new.sum() < 1:
            continue
        Ck = -Rk.T @ tvk.ravel()
        best_j, best_d = None, 0
        for j, (Rj, tj) in poses.items():
            if j == k:
                continue
            d = np.linalg.norm(Ck - (-Rj.T @ tj))
            if (V[j] & new).sum() >= 4 and d > best_d:
                best_j, best_d = j, d
        if best_j is None:
            continue
        Rj, tj = poses[best_j]
        m = np.nonzero(new & V[best_j])[0]
        Xn = _triangulate(K, Rj, tj, Rk, tvk.ravel(), P[best_j, m], P[k, m])
        ej, zj = _reproj_err(K, Rj, tj, Xn, P[best_j, m])
        ek, zk = _reproj_err(K, Rk, tvk.ravel(), Xn, P[k, m])
        ang = _angle(-Rj.T @ tj, Ck, Xn)
        gm = (ej < 2) & (ek < 2) & (zj > 0) & (zk > 0) & (ang > 1.0)
        X[m[gm]] = Xn[gm]

    if len(poses) < 3:
        return None
    # 3) ajustement de faisceaux sur les images clés
    kf_list = sorted(poses)
    kf_index = {k: i for i, k in enumerate(kf_list)}
    pts = np.nonzero(np.isfinite(X[:, 0]))[0]
    p_index = {p: i for i, p in enumerate(pts)}
    obs = [(kf_index[k], p_index[p], P[k, p, 0], P[k, p, 1])
           for k in kf_list for p in pts if V[k, p]]
    first = kf_list.index(a)
    if first != 0:   # la 1re caméra (fixée) doit être celle de référence
        kf_list.insert(0, kf_list.pop(first))
        kf_index = {k: i for i, k in enumerate(kf_list)}
        obs = [(kf_index[k], p_index[p], P[k, p, 0], P[k, p, 1])
               for k in kf_list for p in pts if V[k, p]]
    K2, Rs, ts, Xb, err = bundle_adjust(K, kf_list, [poses[k][0] for k in kf_list],
                                        [poses[k][1] for k in kf_list], X[pts], obs,
                                        max_nfev=nfev)
    if not np.isfinite(K2[0, 0]) or K2[0, 0] <= 0:
        return None
    X[pts] = Xb
    # points aberrants après ajustement
    o = np.asarray(obs)
    bad_pts = set(o[err > 4.0, 1].astype(int))
    for bp in bad_pts:
        if (err[o[:, 1].astype(int) == bp] > 4.0).mean() > 0.3:
            X[pts[bp]] = np.nan
    K = K2
    # 4) toutes les images
    R_all = np.full((T, 3, 3), np.nan)
    t_all = np.full((T, 3), np.nan)
    errs = []
    prev = None
    for k in range(t0, t1 + 1):
        have = V[k] & np.isfinite(X[:, 0])
        if have.sum() < 6:
            continue
        init = poses.get(k) or prev
        if init is not None:
            rv0 = cv2.Rodrigues(init[0])[0]
            ok, rv, tvk, inl = cv2.solvePnPRansac(X[have], P[k, have], K, None, rv0,
                                                  init[1].reshape(3, 1).copy(), True,
                                                  reprojectionError=3.0, iterationsCount=100)
        else:
            ok, rv, tvk, inl = cv2.solvePnPRansac(X[have], P[k, have], K, None,
                                                  reprojectionError=3.0, iterationsCount=200)
        if not ok or inl is None or len(inl) < 6:
            continue
        hi = np.nonzero(have)[0][inl.ravel()]
        ok, rv, tvk = cv2.solvePnP(X[hi], P[k, hi], K, None, rv, tvk, True, cv2.SOLVEPNP_ITERATIVE)
        R_all[k] = cv2.Rodrigues(rv)[0]
        t_all[k] = tvk.ravel()
        e, _ = _reproj_err(K, R_all[k], t_all[k], X[hi], P[k, hi])
        errs.append(np.mean(e))
        prev = (R_all[k], t_all[k])
    if not errs:
        return None
    # Échelle : profondeur médiane des points = 10 unités (pratique dans Fusion).
    d = []
    for k in range(t0, t1 + 1, max(1, (t1 - t0) // 20)):
        if np.isfinite(R_all[k, 0, 0]):
            ok = np.isfinite(X[:, 0])
            d.append(np.median((X[ok] @ R_all[k].T + t_all[k])[:, 2]))
    s = 10.0 / max(np.median(d), 1e-9) if d else 1.0
    X *= s
    t_all *= s
    n_ok = int(np.isfinite(X[:, 0]).sum())
    return CameraSolve("3d", K, R_all, t_all, X, float(np.mean(errs)), (t0, t1), W, H,
                       f"Caméra 3D résolue : {n_ok} points 3D, erreur {np.mean(errs):.2f} px, "
                       f"focale {K[0, 0]:.0f} px")


def _solve_rotation(P, V, W, H, t0, t1, f) -> CameraSolve:
    """Caméra en rotation pure : H = K·R·K⁻¹ entre images (pas de profondeur)."""
    import motion_solver as ms

    T, Q = V.shape
    K = _K(f, W, H)
    Ki = np.linalg.inv(K)
    sol = ms.solve_motion(P, V, t0, t0, t1, "perspective")
    R_all = np.full((T, 3, 3), np.nan)
    t_all = np.full((T, 3), np.nan)
    errs = []
    for k in range(t0, t1 + 1):
        if not sol.ok(k):
            continue
        M = Ki @ sol.H[k] @ K
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        R_all[k], t_all[k] = R, np.zeros(3)
        if np.isfinite(sol.rms[k]):
            errs.append(sol.rms[k])
    # Points « à l'infini » : direction de vue sur l'image de référence, à 10 unités.
    X = np.full((Q, 3), np.nan)
    ok = V[t0]
    if ok.any():
        rays = np.c_[P[t0, ok], np.ones(ok.sum())] @ Ki.T
        X[ok] = rays / np.linalg.norm(rays, axis=1, keepdims=True) * 10.0
    rms = float(np.mean(errs)) if errs else float("nan")
    return CameraSolve("rotation", K, R_all, t_all, X, rms, (t0, t1), W, H,
                       "Pas assez de parallaxe (caméra qui pivote sans se déplacer) : "
                       "caméra en rotation pure, profondeur non mesurable géométriquement.")


# =============================================================================
# Exports
# =============================================================================

def _euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """Angles (degrés) tels que R = Rz·Ry·Rx (ordre XYZ de Fusion)."""
    sy = -R[2, 0]
    y = math.asin(max(-1.0, min(1.0, sy)))
    if abs(sy) < 0.99999:
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


S_FLIP = np.diag([1.0, -1.0, -1.0])   # OpenCV (y bas, z devant) → Fusion (y haut, z arrière)


def fusion_camera_keys(cs: CameraSolve, frames, offset: int = 0):
    keys = {k: {} for k in ("tx", "ty", "tz", "rx", "ry", "rz")}
    for t in frames:
        if not cs.ok(t):
            continue
        Rc2w = cs.R[t].T
        C = cs.center(t)
        Rf = S_FLIP @ Rc2w @ S_FLIP
        Cf = S_FLIP @ C
        rx, ry, rz = _euler_xyz(Rf)
        k = t + offset
        keys["tx"][k], keys["ty"][k], keys["tz"][k] = (float(v) for v in Cf)
        keys["rx"][k], keys["ry"][k], keys["rz"][k] = rx, ry, rz
    return keys


def write_fusion_camera(path: str, cs: CameraSolve, frames, offset: int = 0,
                        locators: Optional[np.ndarray] = None) -> str:
    """Nœud Camera3D animé (+ repères Locator3D sur des points 3D)."""
    import fusion_export as fx

    keys = fusion_camera_keys(cs, frames, offset)
    W, H = cs.width, cs.height
    ap_w = 1.0                                   # ouverture en pouces (arbitraire)
    ap_h = ap_w * H / W
    flen = cs.focal / W * ap_w * 25.4            # focale en mm pour cette ouverture
    names = {"tx": "Transform3DOp.Translate.X", "ty": "Transform3DOp.Translate.Y",
             "tz": "Transform3DOp.Translate.Z", "rx": "Transform3DOp.Rotate.X",
             "ry": "Transform3DOp.Rotate.Y", "rz": "Transform3DOp.Rotate.Z"}
    cols = {"tx": (255, 0, 0), "ty": (0, 255, 0), "tz": (0, 0, 255),
            "rx": (255, 128, 128), "ry": (128, 255, 128), "rz": (128, 128, 255)}
    inputs, splines = [], []
    for k, inp in names.items():
        sp = "TAP_Camera3D_" + k
        inputs.append('\t\t\t\t["%s"] = Input { SourceOp = "%s", Source = "Value", },' % (inp, sp))
        splines.append(fx._spline(sp, keys[k], cols[k]))
    inputs += [
        '\t\t\t\tFilmGate = Input { Value = FuID { "User" }, },',
        '\t\t\t\tApertureW = Input { Value = %s, },' % fx._fmt(ap_w),
        '\t\t\t\tApertureH = Input { Value = %s, },' % fx._fmt(ap_h),
        '\t\t\t\tFLength = Input { Value = %s, },' % fx._fmt(flen),
    ]
    blocks = ["\n".join(["\t\tTAP_Camera3D = Camera3D {", "\t\t\tInputs = {"] + inputs +
                        ["\t\t\t},", "\t\t\tViewInfo = OperatorInfo { Pos = { 0, 0 } },", "\t\t},"])]
    blocks += splines
    if locators is not None:
        for i, Xw in enumerate(locators):
            Xf = S_FLIP @ Xw
            blocks.append("\n".join([
                "\t\tTAP_Point3D_%d = Locator3D {" % (i + 1), "\t\t\tInputs = {",
                '\t\t\t\t["Transform3DOp.Translate.X"] = Input { Value = %s, },' % fx._fmt(Xf[0]),
                '\t\t\t\t["Transform3DOp.Translate.Y"] = Input { Value = %s, },' % fx._fmt(Xf[1]),
                '\t\t\t\t["Transform3DOp.Translate.Z"] = Input { Value = %s, },' % fx._fmt(Xf[2]),
                "\t\t\t},",
                "\t\t\tViewInfo = OperatorInfo { Pos = { %d, 66 } }," % (110 * i),
                "\t\t},"]))
    text = "{\n\tTools = ordered() {\n" + "\n".join(blocks) + '\n\t},\n\tActiveTool = "TAP_Camera3D"\n}\n'
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def pick_locators(cs: CameraSolve, valid: np.ndarray, n: int = 12) -> np.ndarray:
    """Points 3D bien observés et bien répartis (pour accrocher des objets)."""
    ok = np.nonzero(np.isfinite(cs.X[:, 0]))[0]
    if len(ok) == 0:
        return np.zeros((0, 3))
    seen = valid[:, ok].sum(0)
    order = ok[np.argsort(-seen)]
    chosen = []
    for i in order:
        if all(np.linalg.norm(cs.X[i] - cs.X[j]) > 0.5 for j in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    return cs.X[chosen]


def write_ply(path: str, cs: CameraSolve) -> None:
    ok = np.isfinite(cs.X[:, 0])
    X = cs.X[ok] @ S_FLIP.T
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(X))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in X:
            f.write("%.5f %.5f %.5f\n" % tuple(p))


def write_json(path: str, cs: CameraSolve, frames) -> None:
    import json

    data = dict(mode=cs.mode, width=cs.width, height=cs.height, focal_px=cs.focal,
                rms_px=cs.rms, convention="OpenCV : x droite, y bas, z devant ; R,t monde→caméra",
                frames={})
    for t in frames:
        if cs.ok(t):
            data["frames"][str(t)] = dict(R=cs.R[t].round(7).tolist(), t=cs.t[t].round(6).tolist())
    data["points"] = [None if not np.isfinite(p[0]) else [round(float(v), 5) for v in p] for p in cs.X]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
