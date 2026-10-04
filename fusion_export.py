#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion_export.py — Conversion des trajectoires TAPNext++ en nœuds Fusion.

Ce module est volontairement écrit en Python « stdlib only » (aucune
dépendance : ni numpy, ni OpenCV) pour pouvoir être exécuté :

  1. comme bibliothèque, importée par ``tap_resolve_tool.py`` ;
  2. en ligne de commande :
         python fusion_export.py tracks.csv --mode stabilize -o stab.setting
  3. directement DANS DaVinci Resolve (page Fusion) : copiez ce fichier dans
     le dossier « Scripts/Comp » de Fusion puis lancez-le via
     Workspace > Scripts > Comp > fusion_export. Le script demande le CSV
     et colle directement les nœuds dans la composition active.

Le résultat est un « .setting » Fusion (texte Lua) contenant un nœud
``Transform`` dont Center / Pivot / Angle / Size sont animés image par image.
Un ``.setting`` peut être glissé-déposé dans le Node Editor ou copié-collé
(Ctrl+V) comme n'importe quel nœud Fusion.

Conventions de coordonnées
--------------------------
* CSV : pixels, origine en haut à gauche, Y vers le bas (convention OpenCV).
* Fusion : coordonnées normalisées [0..1], origine en bas à gauche, Y vers le
  haut. Conversion : ``xn = x / W`` ; ``yn = 1 - y / H``.
* Angle Fusion : degrés, positif = sens anti-horaire.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

# tracks[frame][point_id] = (x, y, visibility)
Tracks = Dict[int, Dict[int, Tuple[float, float, float]]]


def load_tracks_csv(path: str) -> Tracks:
    """Lit le CSV produit par tap_resolve_tool.py.

    Colonnes attendues : frame_index, point_id, x_pixels, y_pixels,
    visibility (velocity est optionnelle et ignorée ici).
    """
    tracks: Tracks = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fr = int(row["frame_index"])
            pid = int(row["point_id"])
            tracks.setdefault(fr, {})[pid] = (
                float(row["x_pixels"]),
                float(row["y_pixels"]),
                float(row["visibility"]),
            )
    return tracks


def load_metadata(csv_path: str) -> dict:
    """Charge le JSON compagnon (même nom que le CSV, extension .json)."""
    path = os.path.splitext(csv_path)[0] + ".json"
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", data)


# ---------------------------------------------------------------------------
# Estimation de mouvement (similitude 2D pondérée, pur Python)
# ---------------------------------------------------------------------------

def _fit_similarity(
    src: Sequence[Tuple[float, float]],
    dst: Sequence[Tuple[float, float]],
    w: Sequence[float],
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float]]:
    """Ajuste dst ≈ s·R(θ)·(src − cs) + cd  (moindres carrés pondérés).

    Retourne (scale, theta_rad, centroid_src, centroid_dst).
    θ est exprimé dans le repère image (Y vers le bas) : θ > 0 = rotation
    horaire à l'écran.
    """
    sw = sum(w)
    csx = sum(wi * p[0] for wi, p in zip(w, src)) / sw
    csy = sum(wi * p[1] for wi, p in zip(w, src)) / sw
    cdx = sum(wi * p[0] for wi, p in zip(w, dst)) / sw
    cdy = sum(wi * p[1] for wi, p in zip(w, dst)) / sw
    a_num = b_num = den = 0.0
    for wi, (sx, sy), (dx, dy) in zip(w, src, dst):
        sx -= csx
        sy -= csy
        dx -= cdx
        dy -= cdy
        a_num += wi * (sx * dx + sy * dy)
        b_num += wi * (sx * dy - sy * dx)
        den += wi * (sx * sx + sy * sy)
    if den < 1e-9:
        return 1.0, 0.0, (csx, csy), (cdx, cdy)
    a = a_num / den
    b = b_num / den
    return math.hypot(a, b), math.atan2(b, a), (csx, csy), (cdx, cdy)


def estimate_motion(
    tracks: Tracks,
    ref_frame: Optional[int] = None,
    point_ids: Optional[Iterable[int]] = None,
    model: str = "similarity",
    min_visibility: float = 0.5,
    outlier_px: float = 0.0,
) -> Tuple[Tuple[float, float], Dict[int, Tuple[float, float, float, float]]]:
    """Calcule, pour chaque image, le mouvement global par rapport à ref_frame.

    Args:
        tracks: données issues de load_tracks_csv().
        ref_frame: image de référence (défaut : première image du CSV).
        point_ids: sous-ensemble de points à utiliser (défaut : tous).
        model: "translation" ou "similarity" (translation+rotation+échelle).
        min_visibility: seuil en dessous duquel un point est ignoré.
        outlier_px: si > 0, rejette (2 passes) les points dont le résidu
            dépasse max(outlier_px, 3 × médiane).

    Returns:
        anchor: point d'ancrage (centroïde de référence, pixels).
        motion[frame] = (cx, cy, scale, theta) : position de l'ancre dans
        l'image courante + échelle + rotation (radians, repère image).
    """
    frames = sorted(tracks)
    if not frames:
        raise ValueError("Aucune donnée de tracking.")
    ref_frame = frames[0] if ref_frame is None else ref_frame
    ref = tracks[ref_frame]
    ids = sorted(ref) if point_ids is None else [p for p in point_ids if p in ref]
    ids = [p for p in ids if ref[p][2] >= min_visibility] or ids
    if not ids:
        raise ValueError("Aucun point valide dans l'image de référence.")
    anchor = (
        sum(ref[p][0] for p in ids) / len(ids),
        sum(ref[p][1] for p in ids) / len(ids),
    )

    motion: Dict[int, Tuple[float, float, float, float]] = {}
    last = (anchor[0], anchor[1], 1.0, 0.0)
    for fr in frames:
        cur = tracks[fr]
        sel = [p for p in ids if p in cur and cur[p][2] >= min_visibility]
        if not sel:
            motion[fr] = last  # aucun point visible : on maintient
            continue
        src = [(ref[p][0], ref[p][1]) for p in sel]
        dst = [(cur[p][0], cur[p][1]) for p in sel]
        w = [cur[p][2] for p in sel]

        for _ in range(2 if outlier_px > 0 else 1):
            if model == "translation" or len(sel) < 2:
                s, th = 1.0, 0.0
                sw = sum(w)
                tx = sum(wi * (d[0] - r[0]) for wi, r, d in zip(w, src, dst)) / sw
                ty = sum(wi * (d[1] - r[1]) for wi, r, d in zip(w, src, dst)) / sw
                cs, cd = anchor, (anchor[0] + tx, anchor[1] + ty)
            else:
                s, th, cs, cd = _fit_similarity(src, dst, w)
            if outlier_px <= 0 or len(sel) < 4:
                break
            c, sn = math.cos(th), math.sin(th)
            res = []
            for (sx, sy), (dx, dy) in zip(src, dst):
                px = cd[0] + s * (c * (sx - cs[0]) - sn * (sy - cs[1]))
                py = cd[1] + s * (sn * (sx - cs[0]) + c * (sy - cs[1]))
                res.append(math.hypot(px - dx, py - dy))
            med = sorted(res)[len(res) // 2]
            lim = max(outlier_px, 3.0 * med)
            keep = [i for i, r in enumerate(res) if r <= lim]
            if len(keep) == len(res) or len(keep) < 2:
                break
            src = [src[i] for i in keep]
            dst = [dst[i] for i in keep]
            w = [w[i] for i in keep]

        # Position de l'ancre de référence dans l'image courante.
        c, sn = math.cos(th), math.sin(th)
        ax, ay = anchor[0] - cs[0], anchor[1] - cs[1]
        cx = cd[0] + s * (c * ax - sn * ay)
        cy = cd[1] + s * (sn * ax + c * ay)
        last = (cx, cy, s, th)
        motion[fr] = last
    return anchor, motion


def _smooth_series(values: List[float], radius: int) -> List[float]:
    """Lissage gaussien 1D (bords « reflect »), pur Python."""
    if radius <= 0 or len(values) < 3:
        return list(values)
    sigma = radius / 2.0
    k = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    n = len(values)
    out = []
    for i in range(n):
        acc = wsum = 0.0
        for j, kj in enumerate(k):
            idx = i + j - radius
            if idx < 0:
                idx = -idx
            if idx >= n:
                idx = 2 * (n - 1) - idx
            idx = min(max(idx, 0), n - 1)
            acc += kj * values[idx]
            wsum += kj
        out.append(acc / wsum)
    return out


# ---------------------------------------------------------------------------
# Calcul des keyframes Fusion
# ---------------------------------------------------------------------------

def compute_transform_keys(
    tracks: Tracks,
    width: int,
    height: int,
    mode: str = "stabilize",
    model: str = "similarity",
    point_ids: Optional[Iterable[int]] = None,
    smooth_radius: int = 0,
    min_visibility: float = 0.5,
    frame_offset: int = 0,
    outlier_px: float = 0.0,
    ref_frame: Optional[int] = None,
) -> Dict[str, Dict[int, object]]:
    """Calcule les keyframes Center / Pivot / Angle / Size d'un Transform.

    mode = "stabilize"  : annule le mouvement de caméra.
        smooth_radius = 0 → plan verrouillé (lock-off) ;
        smooth_radius > 0 → on ne retire que les vibrations (stabilisation
        « douce » : le mouvement lissé est conservé).
    mode = "matchmove"  : applique le mouvement suivi à un élément
        (texte, logo, incrustation) positionné sur l'image de référence.

    Formules (repère image, A = ancre de référence, C_t = ancre courante) :
        stabilize : out = C_s + (s_s/s)·R(θ_s − θ)·(in − C_t)
                    → Pivot = C_t, Center = C_s, Size = s_s/s,
                      Angle = deg(θ − θ_s)   (C_s = A si lock-off)
        matchmove : out = C_t + s·R(θ)·(in − A)
                    → Pivot = A, Center = C_t, Size = s, Angle = −deg(θ)
    """
    if ref_frame not in tracks:
        ref_frame = None
    anchor, motion = estimate_motion(
        tracks,
        ref_frame=ref_frame,
        point_ids=point_ids,
        model=model,
        min_visibility=min_visibility,
        outlier_px=outlier_px,
    )
    frames = sorted(motion)
    cx = [motion[f][0] for f in frames]
    cy = [motion[f][1] for f in frames]
    ls = [math.log(max(motion[f][2], 1e-6)) for f in frames]
    # Déroulement de l'angle pour éviter les sauts ±π avant lissage.
    th = []
    prev = 0.0
    for f in frames:
        t = motion[f][3]
        while t - prev > math.pi:
            t -= 2 * math.pi
        while t - prev < -math.pi:
            t += 2 * math.pi
        th.append(t)
        prev = t

    def norm(px: float, py: float) -> Tuple[float, float]:
        return px / float(width), 1.0 - py / float(height)

    keys: Dict[str, Dict[int, object]] = {
        "Center": {},
        "Pivot": {},
        "Angle": {},
        "Size": {},
    }
    if mode == "stabilize":
        if smooth_radius > 0:
            scx = _smooth_series(cx, smooth_radius)
            scy = _smooth_series(cy, smooth_radius)
            sls = _smooth_series(ls, smooth_radius)
            sth = _smooth_series(th, smooth_radius)
        else:
            scx = [anchor[0]] * len(frames)
            scy = [anchor[1]] * len(frames)
            sls = [0.0] * len(frames)
            sth = [0.0] * len(frames)
        for i, f in enumerate(frames):
            k = f + frame_offset
            keys["Pivot"][k] = norm(cx[i], cy[i])
            keys["Center"][k] = norm(scx[i], scy[i])
            keys["Size"][k] = math.exp(sls[i] - ls[i])
            keys["Angle"][k] = math.degrees(th[i] - sth[i])
    elif mode == "matchmove":
        if smooth_radius > 0:
            cx = _smooth_series(cx, smooth_radius)
            cy = _smooth_series(cy, smooth_radius)
            ls = _smooth_series(ls, smooth_radius)
            th = _smooth_series(th, smooth_radius)
        a = norm(*anchor)
        for i, f in enumerate(frames):
            k = f + frame_offset
            keys["Pivot"][k] = a
            keys["Center"][k] = norm(cx[i], cy[i])
            keys["Size"][k] = math.exp(ls[i])
            keys["Angle"][k] = -math.degrees(th[i])
    else:
        raise ValueError("mode doit être 'stabilize' ou 'matchmove'")

    if model == "translation":
        keys.pop("Angle")
        keys.pop("Size")
    return keys


def compute_point_paths(
    tracks: Tracks,
    width: int,
    height: int,
    point_ids: Iterable[int],
    frame_offset: int = 0,
    min_visibility: float = 0.0,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Trajectoire normalisée Fusion (x, y) de points individuels."""
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for pid in point_ids:
        path = {}
        for f in sorted(tracks):
            p = tracks[f].get(pid)
            if p is None or p[2] < min_visibility:
                continue
            path[f + frame_offset] = (p[0] / width, 1.0 - p[1] / height)
        if path:
            out[pid] = path
    return out


# ---------------------------------------------------------------------------
# Génération du texte .setting (format Lua de Fusion)
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return ("%.7f" % v).rstrip("0").rstrip(".") if v == v else "0"


def _spline(name: str, keys: Dict[int, float], color: Tuple[int, int, int]) -> str:
    lines = [
        "\t\t%s = BezierSpline {" % name,
        "\t\t\tSplineColor = { Red = %d, Green = %d, Blue = %d }," % color,
        "\t\t\tNameSet = true,",
        "\t\t\tKeyFrames = {",
    ]
    for f in sorted(keys):
        lines.append(
            "\t\t\t\t[%d] = { %s, Flags = { Linear = true } }," % (f, _fmt(keys[f]))
        )
    lines += ["\t\t\t}", "\t\t},"]
    return "\n".join(lines)


def _xypath(prefix: str, keys: Dict[int, Tuple[float, float]]) -> List[str]:
    """Un XYPath = deux BezierSpline (X et Y) reliées à un input Point."""
    xs = {f: v[0] for f, v in keys.items()}
    ys = {f: v[1] for f, v in keys.items()}
    path = "\n".join(
        [
            "\t\t%sXYPath = XYPath {" % prefix,
            "\t\t\tShowKeyPoints = false,",
            '\t\t\tDrawMode = "ModifyOnly",',
            "\t\t\tInputs = {",
            '\t\t\t\tX = Input { SourceOp = "%sX", Source = "Value", },' % prefix,
            '\t\t\t\tY = Input { SourceOp = "%sY", Source = "Value", },' % prefix,
            "\t\t\t},",
            "\t\t},",
        ]
    )
    return [
        path,
        _spline(prefix + "X", xs, (255, 0, 0)),
        _spline(prefix + "Y", ys, (0, 255, 0)),
    ]


def _transform_tool(
    name: str, keys: Dict[str, Dict[int, object]], pos: Tuple[int, int]
) -> List[str]:
    inputs, extra = [], []
    for inp in ("Center", "Pivot"):
        if inp in keys and keys[inp]:
            prefix = "%s%s" % (name, inp)
            inputs.append(
                '\t\t\t\t%s = Input { SourceOp = "%sXYPath", Source = "Value", },'
                % (inp, prefix)
            )
            extra += _xypath(prefix, keys[inp])  # type: ignore[arg-type]
    for inp, col in (("Angle", (255, 128, 0)), ("Size", (0, 128, 255))):
        if inp in keys and keys[inp]:
            inputs.append(
                '\t\t\t\t%s = Input { SourceOp = "%s%s", Source = "Value", },'
                % (inp, name, inp)
            )
            extra.append(_spline(name + inp, keys[inp], col))  # type: ignore[arg-type]
    tool = "\n".join(
        ["\t\t%s = Transform {" % name, "\t\t\tInputs = {"]
        + inputs
        + [
            "\t\t\t},",
            "\t\t\tViewInfo = OperatorInfo { Pos = { %d, %d } }," % pos,
            "\t\t},",
        ]
    )
    return [tool] + extra


def build_setting(
    transforms: Dict[str, Dict[str, Dict[int, object]]],
    point_paths: Optional[Dict[int, Dict[int, Tuple[float, float]]]] = None,
) -> str:
    """Assemble un .setting Fusion à partir de plusieurs Transform animés.

    transforms : {nom_du_noeud: keys} (keys issues de compute_transform_keys)
    point_paths : {point_id: {frame: (xn, yn)}} → un Transform par point
        (Center animé), pratique pour accrocher un élément à un point précis.
    """
    blocks: List[str] = []
    x = 0
    active = None
    for name, keys in transforms.items():
        blocks += _transform_tool(name, keys, (x, 0))
        active = active or name
        x += 110
    for pid, path in (point_paths or {}).items():
        name = "TAP_Point_%d" % pid
        blocks += _transform_tool(name, {"Center": path}, (x, 66))  # type: ignore[dict-item]
        active = active or name
        x += 110
    return (
        "{\n\tTools = ordered() {\n"
        + "\n".join(blocks)
        + '\n\t},\n\tActiveTool = "%s"\n}\n' % (active or "")
    )


def export_setting(
    csv_path: str,
    out_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "stabilize",
    model: str = "similarity",
    point_ids: Optional[Iterable[int]] = None,
    smooth_radius: int = 0,
    frame_offset: int = 0,
    min_visibility: float = 0.5,
    point_paths_ids: Optional[Iterable[int]] = None,
    outlier_px: float = 0.0,
    ref_frame: Optional[int] = None,
) -> str:
    """Raccourci : CSV → fichier .setting. Retourne le texte généré."""
    text = setting_from_csv(
        csv_path, width, height, mode, model, point_ids, smooth_radius,
        frame_offset, min_visibility, point_paths_ids, outlier_px, ref_frame,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def setting_from_csv(
    csv_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    mode: str = "stabilize",
    model: str = "similarity",
    point_ids: Optional[Iterable[int]] = None,
    smooth_radius: int = 0,
    frame_offset: int = 0,
    min_visibility: float = 0.5,
    point_paths_ids: Optional[Iterable[int]] = None,
    outlier_px: float = 0.0,
    ref_frame: Optional[int] = None,
) -> str:
    """CSV → texte .setting. L'image de référence par défaut est celle où les
    points ont été définis (``query_frame`` du JSON), sinon la première."""
    tracks = load_tracks_csv(csv_path)
    meta = load_metadata(csv_path)
    if ref_frame is None:
        ref_frame = meta.get("query_frame")
    width = int(width or meta.get("width") or 0)
    height = int(height or meta.get("height") or 0)
    if not width or not height:
        raise ValueError(
            "Résolution inconnue : fournissez --width/--height ou le JSON "
            "compagnon généré par tap_resolve_tool.py."
        )
    keys = compute_transform_keys(
        tracks, width, height, mode=mode, model=model, point_ids=point_ids,
        smooth_radius=smooth_radius, min_visibility=min_visibility,
        frame_offset=frame_offset, outlier_px=outlier_px, ref_frame=ref_frame,
    )
    name = "TAP_Stabilize" if mode == "stabilize" else "TAP_MatchMove"
    paths = None
    if point_paths_ids:
        paths = compute_point_paths(
            tracks, width, height, point_paths_ids, frame_offset, min_visibility
        )
    return build_setting({name: keys}, paths)


# ---------------------------------------------------------------------------
# Exécution dans DaVinci Resolve / Fusion
# ---------------------------------------------------------------------------

def _run_inside_fusion(fu_obj, comp) -> None:  # pragma: no cover (Resolve)
    """Boîte de dialogue + collage des nœuds dans la composition active."""
    try:
        fmt = comp.GetPrefs("Comp.FrameFormat")
        comp_w, comp_h = int(fmt["Width"]), int(fmt["Height"])
    except Exception:
        comp_w, comp_h = 1920, 1080

    dlg = comp.AskUser(
        "TAPNext++ → Fusion",
        {
            1: {1: "csv", "Name": "CSV de tracking", 2: "FileBrowse", "Save": False},
            2: {1: "mode", "Name": "Mode", 2: "Dropdown",
                "Options": {1: "Stabilize", 2: "Match-move"}},
            3: {1: "model", "Name": "Modèle", 2: "Dropdown",
                "Options": {1: "Similarity (pos+rot+échelle)", 2: "Translation"}},
            4: {1: "smooth", "Name": "Lissage (0 = lock-off)", 2: "Slider",
                "Min": 0, "Max": 100, "Integer": True, "Default": 0},
            5: {1: "offset", "Name": "Décalage d'image", 2: "Slider",
                "Min": -1000, "Max": 1000, "Integer": True, "Default": 0},
            6: {1: "points", "Name": "IDs (vide = tous)", 2: "Text",
                "Lines": 1, "Default": ""},
        },
    )
    if not dlg:
        return
    pts = [int(p) for p in str(dlg.get("points", "")).replace(";", ",").split(",")
           if p.strip().isdigit()] or None
    has_meta = bool(load_metadata(dlg["csv"]))
    text = setting_from_csv(
        dlg["csv"],
        width=None if has_meta else comp_w,
        height=None if has_meta else comp_h,
        mode="stabilize" if int(dlg.get("mode", 0)) == 0 else "matchmove",
        model="similarity" if int(dlg.get("model", 0)) == 0 else "translation",
        point_ids=pts,
        smooth_radius=int(dlg.get("smooth", 0)),
        frame_offset=int(dlg.get("offset", 0)),
    )
    comp.Lock()
    comp.StartUndo("TAPNext++ import")
    try:
        bmd = globals().get("bmd")
        if bmd is None:
            import BlackmagicFusion as bmd  # type: ignore
        comp.Paste(bmd.readstring(text))
    finally:
        comp.EndUndo(True)
        comp.Unlock()
    print("TAPNext++ : nœuds Transform collés dans la composition.")


def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Convertit un CSV TAPNext++ en nœud Transform Fusion (.setting)."
    )
    ap.add_argument("csv", help="CSV produit par tap_resolve_tool.py")
    ap.add_argument("-o", "--output", help="Fichier .setting de sortie")
    ap.add_argument("--mode", choices=["stabilize", "matchmove"], default="stabilize")
    ap.add_argument("--model", choices=["similarity", "translation"], default="similarity")
    ap.add_argument("--points", default="", help="IDs de points à utiliser, ex: 0,4,12")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Rayon de lissage (images). 0 = verrouillage total.")
    ap.add_argument("--frame-offset", type=int, default=0,
                    help="Décalage ajouté aux numéros d'image (ex: 1001).")
    ap.add_argument("--min-visibility", type=float, default=0.5)
    ap.add_argument("--outlier-px", type=float, default=0.0,
                    help="Rejet des points aberrants (px). 0 = désactivé.")
    ap.add_argument("--point-paths", default="",
                    help="IDs de points à exporter chacun dans un Transform dédié.")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    a = ap.parse_args(argv)

    def ids(s: str) -> Optional[List[int]]:
        v = [int(x) for x in s.replace(";", ",").split(",") if x.strip()]
        return v or None

    out = a.output or os.path.splitext(a.csv)[0] + "_%s.setting" % a.mode
    export_setting(
        a.csv, out, a.width, a.height, a.mode, a.model, ids(a.points),
        a.smooth, a.frame_offset, a.min_visibility, ids(a.point_paths), a.outlier_px,
    )
    print("Écrit :", out)


if __name__ == "__main__":
    _fu = globals().get("fu") or globals().get("fusion")
    _comp = globals().get("comp")
    if _comp is None and _fu is not None:
        _comp = _fu.GetCurrentComp()
    if _comp is not None and len(sys.argv) <= 1:
        _run_inside_fusion(_fu, _comp)
    else:
        _cli()
