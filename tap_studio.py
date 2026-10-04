#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAPNext Studio — interface visuelle de tracking TAPNext++ pour DaVinci Resolve.

  • Visionneuse vidéo (zoom molette, déplacement clic-milieu, timeline, lecture).
  • Placement des points : dessinez une zone autour du sujet (elle se remplit de
    points sur les détails texturés) ou cliquez des points précis, sur
    n'importe quelle image du plan. Chaque zone devient un groupe de formes.
  • Suivi TAPNext++ bidirectionnel avec contrôle aller-retour, en arrière-plan.
  • Groupes de formes : forme (cercle, étoile, image PNG…), révéler/découper,
    taille, rotation, réaction au mouvement, fondus activables, toujours
    visible, fusion, traînée — rendu en direct, sans relancer le suivi.
  • Projets .tapnext (points, suivi et formes enregistrés).
  • Export : matte N&B (ProRes/DNxHR/H.264/PNG), une matte par groupe, CSV/JSON,
    nœuds Fusion. Depuis Resolve, les résultats y repartent automatiquement.

Raccourcis : Espace lecture · ←/→ image · I/O début/fin · F cadrer · Z zone ·
R rectangle · A point · S sélection · Échap désélectionner · Suppr supprimer ·
Ctrl+A tout sélectionner · Ctrl+Z annuler · Ctrl+O ouvrir · Ctrl+S enregistrer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage, QKeySequence,
                           QPainter, QPainterPath, QPalette, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fusion_export as fx  # noqa: E402
import shape_engine as se  # noqa: E402
import tap_resolve_tool as eng  # noqa: E402

APP_NAME = "TAPNext Studio"
PREVIEW_MAX_SIDE = 1280      # résolution des images gardées en mémoire (JPEG)
ACCENT = "#e8833a"

STYLE = f"""
QWidget {{ background: #1c1d21; color: #d8d8dc; font-size: 12px; }}
QMainWindow::separator {{ background: #111; width: 1px; }}
QGroupBox {{ border: 1px solid #2c2d33; border-radius: 6px; margin-top: 14px;
            padding: 10px 8px 8px 8px; background: #212227; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {ACCENT}; font-weight: bold; font-size: 13px; }}
QLabel#help {{ color: #8b8c94; font-size: 11px; }}
QLabel#value {{ color: #f0f0f2; min-width: 44px; }}
QPushButton, QToolButton {{ background: #2d2e35; border: 1px solid #3a3b43; border-radius: 4px;
                            padding: 5px 10px; }}
QPushButton:hover, QToolButton:hover {{ background: #383a42; }}
QPushButton:disabled {{ color: #66676e; background: #26272c; }}
QToolButton:checked {{ background: {ACCENT}; color: #111; border-color: {ACCENT}; }}
QPushButton#primary {{ background: {ACCENT}; color: #141414; font-weight: bold;
                       border: none; padding: 8px; font-size: 13px; }}
QPushButton#primary:hover {{ background: #f39a55; }}
QPushButton#primary:disabled {{ background: #5b4130; color: #9a8c80; }}
QComboBox, QSpinBox, QLineEdit {{ background: #15161a; border: 1px solid #34353c;
                                   border-radius: 3px; padding: 3px 5px; }}
QSlider::groove:horizontal {{ height: 4px; background: #34353c; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e6e6ea; width: 12px; margin: -5px 0; border-radius: 6px; }}
QProgressBar {{ background: #15161a; border: 1px solid #34353c; border-radius: 3px;
                text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QScrollArea {{ border: none; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
QStatusBar {{ background: #141518; color: #9a9ba3; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: #26272c; color: #b8b9c0; padding: 7px 12px; border: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: #2f3037; color: {ACCENT}; font-weight: bold; }}
QListWidget {{ background: #15161a; border: 1px solid #34353c; border-radius: 3px; }}
QListWidget::item {{ padding: 4px; }}
QListWidget::item:selected {{ background: #3a3b43; color: #ffffff; }}
"""


# =============================================================================
# Données vidéo : images d'aperçu gardées en mémoire (JPEG)
# =============================================================================

class VideoStore:
    def __init__(self, path: str):
        self.path = path
        self.info = eng.probe_video(path)
        W, H = self.info.width, self.info.height
        self.scale = min(1.0, PREVIEW_MAX_SIDE / float(max(W, H)))
        self.pw, self.ph = max(1, int(round(W * self.scale))), max(1, int(round(H * self.scale)))
        self.total = max(1, self.info.frame_count)
        self.jpegs: List[Optional[bytes]] = [None] * self.total
        self.loaded = 0
        self.complete = False
        self._lru: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def put(self, i: int, frame_bgr: np.ndarray) -> None:
        small = frame_bgr if self.scale >= 1 else cv2.resize(
            frame_bgr, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i >= len(self.jpegs):
            self.jpegs.extend([None] * (i + 1 - len(self.jpegs)))
        self.jpegs[i] = buf.tobytes() if ok else None
        self.loaded = i + 1

    def finish(self, n: int) -> None:
        self.total = max(1, n)
        del self.jpegs[self.total:]
        self.complete = True

    def frame(self, i: int) -> Optional[np.ndarray]:
        if i in self._lru:
            self._lru.move_to_end(i)
            return self._lru[i]
        if not (0 <= i < len(self.jpegs)) or self.jpegs[i] is None:
            return None
        img = cv2.imdecode(np.frombuffer(self.jpegs[i], np.uint8), cv2.IMREAD_COLOR)
        self._lru[i] = img
        if len(self._lru) > 48:
            self._lru.popitem(last=False)
        return img


# =============================================================================
# Threads de travail
# =============================================================================

class LoadThread(QThread):
    progress = Signal(int)
    done = Signal(int)
    failed = Signal(str)

    def __init__(self, store: VideoStore):
        super().__init__()
        self.store = store
        self.stop_flag = threading.Event()

    def run(self):
        try:
            n = 0
            for i, frame in eng.iter_frames(self.store.path):
                if self.stop_flag.is_set():
                    return
                self.store.put(i, frame)
                n = i + 1
                if n % 8 == 0:
                    self.progress.emit(n)
            self.store.finish(n)
            self.done.emit(n)
        except Exception as e:  # pragma: no cover
            self.failed.emit(str(e))


_TRACKER_CACHE: dict = {}


class TrackThread(QThread):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, store, pts, qf, seg_in, seg_out, input_res, verify, fp16):
        super().__init__()
        self.store, self.pts, self.qf = store, pts, qf
        self.seg_in, self.seg_out = seg_in, seg_out
        self.input_res, self.verify, self.fp16 = input_res, verify, fp16
        self.cancel = threading.Event()

    def run(self):
        try:
            name = eng.CHECKPOINT_URLS[self.input_res][0]
            ck_path = os.path.join(HERE, "checkpoints", name)
            if not os.path.isfile(ck_path):
                self.progress.emit("Téléchargement du modèle TAPNext++ (~2,5 Go, une seule fois)…", 0)
            ckpt = eng.ensure_checkpoint(None, self.input_res)
            key = (self.input_res, self.fp16)
            tracker = _TRACKER_CACHE.get(key)
            if tracker is None:
                self.progress.emit("Chargement du modèle sur le GPU…", 0)
                _TRACKER_CACHE.clear()
                eng._free_gpu()
                tracker = eng.TAPNextPPTracker(ckpt, self.input_res, "cuda", self.fp16)
                _TRACKER_CACHE[key] = tracker
            n = self.seg_out - self.seg_in + 1
            frames = []
            for k, i in enumerate(range(self.seg_in, self.seg_out + 1)):
                if self.cancel.is_set():
                    raise eng.Cancelled()
                img = self.store.frame(i)
                if img is None:
                    raise RuntimeError(f"Image {i} non chargée.")
                frames.append(tracker.preprocess(img))
                if k % 16 == 0:
                    self.progress.emit("Préparation des images", k / n)
            info = self.store.info
            t0 = time.time()
            res = eng.track_segment(
                tracker, frames, self.seg_in, self.pts, self.qf, info.width, info.height,
                self.store.total, self.verify, progress=self.progress.emit, cancel=self.cancel)
            res.elapsed = time.time() - t0
            res.device = str(tracker.device)
            self.done.emit(res)
        except eng.Cancelled:
            self.failed.emit("Suivi annulé.")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


class ValueSlider(QWidget):
    """Curseur flottant avec libellé et valeur affichée."""
    changed = Signal(float)

    def __init__(self, label, lo, hi, value, step, fmt="{:.2f}", tip=""):
        super().__init__()
        self.lo, self.step, self.fmt = lo, step, fmt
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setHorizontalSpacing(6)
        name = QLabel(label)
        name.setToolTip(tip)
        self.val = QLabel()
        self.val.setObjectName("value")
        self.val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.s = QSlider(Qt.Horizontal)
        self.s.setRange(0, int(round((hi - lo) / step)))
        self.s.setToolTip(tip)
        lay.addWidget(name, 0, 0)
        lay.addWidget(self.val, 0, 1)
        lay.addWidget(self.s, 1, 0, 1, 2)
        self.s.valueChanged.connect(self._on)
        self.set(value)

    def value(self) -> float:
        return self.lo + self.s.value() * self.step

    def set(self, v: float) -> None:
        self.s.setValue(int(round((v - self.lo) / self.step)))
        self.val.setText(self.fmt.format(self.value()))

    def _on(self, _):
        self.val.setText(self.fmt.format(self.value()))
        self.changed.emit(self.value())


def help_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("help")
    lab.setWordWrap(True)
    return lab


class Timeline(QWidget):
    seek = Signal(int)

    def __init__(self, win: "StudioWindow"):
        super().__init__()
        self.win = win
        self.setMinimumHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def _x(self, f: int) -> float:
        n = max(1, self.win.n_frames() - 1)
        return 8 + (self.width() - 16) * f / n

    def _f(self, x: float) -> int:
        n = max(1, self.win.n_frames() - 1)
        return int(round(min(max((x - 8) / max(1, self.width() - 16), 0), 1) * n))

    def paintEvent(self, _):
        w = self.win
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#141518"))
        if not w.store:
            return
        y0, h = 12, 14
        p.fillRect(QRectF(self._x(0), y0, self._x(w.n_frames() - 1) - self._x(0), h), QColor("#2a2b31"))
        if not w.store.complete:
            p.fillRect(QRectF(self._x(0), y0, self._x(w.store.loaded) - self._x(0), h), QColor("#33343b"))
        # plage début/fin
        p.fillRect(QRectF(self._x(w.seg_in), y0, self._x(w.seg_out) - self._x(w.seg_in) + 1, h),
                   QColor(70, 110, 170, 150))
        # images suivies
        if w.res is not None:
            idx = np.nonzero(w.res.tracked)[0]
            if idx.size:
                p.fillRect(QRectF(self._x(idx[0]), y0 + h - 3, self._x(idx[-1]) - self._x(idx[0]) + 1, 3),
                           QColor("#5fd38a"))
        # images de pose des points
        if len(w.qf):
            p.setPen(QPen(QColor("#ffd34d"), 1))
            for f in np.unique(w.qf):
                x = self._x(int(f))
                p.drawLine(QPointF(x, 4), QPointF(x, 10))
        # tête de lecture
        x = self._x(w.cur)
        p.setPen(QPen(QColor(ACCENT), 2))
        p.drawLine(QPointF(x, 2), QPointF(x, self.height() - 2))
        p.setPen(QColor("#9a9ba3"))
        p.setFont(QFont(self.font().family(), 8))
        p.drawText(QRectF(self._x(w.seg_in) + 2, y0 + h, 80, 12), Qt.AlignLeft, f"début {w.seg_in}")
        p.drawText(QRectF(self._x(w.seg_out) - 82, y0 + h, 80, 12), Qt.AlignRight, f"fin {w.seg_out}")

    def mousePressEvent(self, e):
        self.seek.emit(self._f(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.seek.emit(self._f(e.position().x()))


class ExportThread(QThread):
    progress = Signal(str, float)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, job: dict):
        super().__init__()
        self.job = job
        self.cancel = threading.Event()

    def run(self):
        j = self.job
        try:
            info, res, rd = j["info"], j["res"], j["rd"]
            os.makedirs(j["out_dir"], exist_ok=True)
            base = os.path.join(j["out_dir"], j["name"])
            outputs = []
            csv_path = base + "_tracks.csv"
            if j["data"] or j["fusion"] != "none" or j.get("job_mode"):
                self.progress.emit("Export des trajectoires", 0)
                eng.export_csv(csv_path, rd.smoothed, res.visibility, rd.velocity, res.tracked)
                params = dict(j["params"], query_frames=[int(v) for v in res.query_frames])
                eng.export_json(base + "_tracks.json", info, j["pts"], params, len(res.tracked))
                outputs += [csv_path, base + "_tracks.json"]
            if j["fusion"] != "none":
                modes = ["stabilize", "matchmove"] if j["fusion"] == "both" else [j["fusion"]]
                for mode in modes:
                    p = f"{base}_fusion_{mode}.setting"
                    fx.export_setting(csv_path, p, info.width, info.height, mode=mode,
                                      min_visibility=0.5, ref_frame=j["ref_frame"],
                                      smooth_radius=j["fusion_smooth"],
                                      outlier_px=max(2.0, 0.002 * info.width))
                    outputs.append(p)
            if j["matte"]:
                outputs += se.write_shapes_video(
                    base, info, j["codec"], j["layers"], j["invert"], j["preview"],
                    j["group_names"] if j["per_group"] else None,
                    progress=self.progress.emit, cancel=self.cancel)
            self.done.emit(outputs)
        except eng.Cancelled:
            self.failed.emit("Export annulé.")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


# =============================================================================
# Groupes de formes
# =============================================================================

GROUP_COLORS = ["#ff8a3d", "#4fc3f7", "#9ccc65", "#f06292", "#ffd54f", "#ba68c8",
                "#4db6ac", "#e57373", "#90a4ae", "#aed581"]


class ShapeGroup:
    def __init__(self, gid: int, name: str, style: "se.ShapeStyle"):
        self.id = gid
        self.name = name
        self.style = style
        self.enabled = True
        self.color = QColor(GROUP_COLORS[gid % len(GROUP_COLORS)])


def np_to_qimage_gray(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    return QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_Grayscale8).copy()


def shape_qicon(shape: str) -> QIcon:
    g = se.shape_icon(shape, 22)
    rgba = np.zeros((22, 22, 4), np.uint8)
    rgba[..., :3] = 235
    rgba[..., 3] = g
    img = QImage(rgba.data, 22, 22, 88, QImage.Format_RGBA8888).copy()
    return QIcon(QPixmap.fromImage(img))


def color_icon(c: QColor) -> QIcon:
    pm = QPixmap(14, 14)
    pm.fill(c)
    return QIcon(pm)


class StyleEditor(QWidget):
    """Édite le ShapeStyle du groupe courant. Signal changed(nom du champ)."""
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.st = se.ShapeStyle()
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        def section(title):
            g = QGroupBox(title)
            v = QVBoxLayout(g)
            v.setSpacing(4)
            lay.addWidget(g)
            return v

        v = section("Forme")
        self.shape = QComboBox()
        for sh in se.SHAPES:
            self.shape.addItem(shape_qicon(sh), se.SHAPE_LABELS[sh], sh)
        self.shape.currentIndexChanged.connect(self._on_shape)
        v.addWidget(self.shape)
        self.mode = QComboBox()
        self.mode.addItems(["Révéler : forme blanche dans la matte",
                            "Découper : trou dans la matte"])
        self.mode.currentIndexChanged.connect(lambda i: self._set("mode", ["add", "subtract"][i]))
        v.addWidget(self.mode)
        self.size = self._slider(v, "size", "Taille (px)", 1, 400, 0.5, "{:.1f}",
                                 "Rayon de chaque forme, en pixels de la vidéo source")
        self.opacity = self._slider(v, "opacity", "Opacité", 0, 1, 0.01, "{:.2f}")
        self.rotation = self._slider(v, "rotation", "Rotation (°)", -180, 180, 1, "{:.0f}")
        self.follow = self._check(v, "follow_motion", "Orienter dans le sens du mouvement")
        self.jitter = self._slider(v, "size_jitter", "Variation aléatoire de taille", 0, 1, 0.01, "{:.2f}")

        v = section("Réaction au mouvement")
        self.grow = self._slider(v, "grow", "Grossir avec la vitesse", 0, 1, 0.005, "{:.3f}")
        self.stretch = self._slider(v, "stretch", "Étirer dans la direction", 0, 1, 0.005, "{:.3f}")
        self.max_scale = self._slider(v, "max_scale", "Agrandissement maximal (×)", 1, 8, 0.1, "{:.1f}")

        v = section("Apparition")
        self.always = self._check(v, "always_visible",
                                  "Toujours visible (ignorer les occultations)")
        row = QHBoxLayout()
        self.fin_on = QCheckBox("Fondu d'apparition")
        self.fin_on.toggled.connect(lambda b: self._set("fade_in_on", b))
        self.fin = QSpinBox()
        self.fin.setRange(1, 120)
        self.fin.setSuffix(" img")
        self.fin.valueChanged.connect(lambda x: self._set("fade_in", int(x)))
        row.addWidget(self.fin_on, 1)
        row.addWidget(self.fin)
        v.addLayout(row)
        row = QHBoxLayout()
        self.fout_on = QCheckBox("Fondu de disparition")
        self.fout_on.toggled.connect(lambda b: self._set("fade_out_on", b))
        self.fout = QSpinBox()
        self.fout.setRange(1, 120)
        self.fout.setSuffix(" img")
        self.fout.valueChanged.connect(lambda x: self._set("fade_out", int(x)))
        row.addWidget(self.fout_on, 1)
        row.addWidget(self.fout)
        v.addLayout(row)

        v = section("Fusion et bords")
        v.addWidget(help_label("Fusion à 0 = formes nettes et séparées. Au-dessus, les formes "
                               "proches se rejoignent (effet « metaball »)."))
        self.merge = self._slider(v, "merge", "Fusion des formes", 0, 2, 0.01, "{:.2f}")
        self.thr = self._slider(v, "threshold", "Seuil de fusion", 0.05, 0.95, 0.01, "{:.2f}")
        self.soft = self._slider(v, "softness", "Douceur du bord", 0, 0.5, 0.01, "{:.2f}")

        v = section("Effets")
        self.trail = self._slider(v, "trail", "Traînée dans la matte (images)", 0, 30, 1, "{:.0f}")
        self.smooth = self._slider(v, "smooth", "Lissage des trajectoires", 0, 5, 0.1, "{:.1f}")
        self.set_style(self.st)

    # ------------------------------------------------------------ helpers
    def _slider(self, v, key, label, lo, hi, step, fmt, tip=""):
        s = ValueSlider(label, lo, hi, getattr(self.st, key), step, fmt, tip)
        s.changed.connect(lambda x, k=key: self._set(k, int(round(x)) if k == "trail" else x))
        v.addWidget(s)
        return s

    def _check(self, v, key, label):
        c = QCheckBox(label)
        c.toggled.connect(lambda b, k=key: self._set(k, b))
        v.addWidget(c)
        return c

    def _set(self, key, value):
        if self._loading:
            return
        setattr(self.st, key, value)
        self._sync_enabled()
        self.changed.emit(key)

    def _on_shape(self, i):
        sh = self.shape.itemData(i)
        if self._loading:
            return
        if sh == "image":
            p, _ = QFileDialog.getOpenFileName(self, "Image de la forme (PNG avec transparence)",
                                               self.st.image_path or "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
            if not p:
                self._loading = True
                self.shape.setCurrentIndex(se.SHAPES.index(self.st.shape))
                self._loading = False
                return
            self.st.image_path = p
        self._set("shape", sh)

    def _sync_enabled(self):
        a = self.st.always_visible
        for w in (self.fin_on, self.fout_on):
            w.setEnabled(not a)
        self.fin.setEnabled(not a and self.st.fade_in_on)
        self.fout.setEnabled(not a and self.st.fade_out_on)
        self.thr.setEnabled(self.st.merge > 0)

    def set_style(self, st: "se.ShapeStyle"):
        self.st = st
        self._loading = True
        self.shape.setCurrentIndex(se.SHAPES.index(st.shape) if st.shape in se.SHAPES else 0)
        self.mode.setCurrentIndex(1 if st.mode == "subtract" else 0)
        for w, k in ((self.size, "size"), (self.opacity, "opacity"), (self.rotation, "rotation"),
                     (self.jitter, "size_jitter"), (self.grow, "grow"), (self.stretch, "stretch"),
                     (self.max_scale, "max_scale"), (self.merge, "merge"), (self.thr, "threshold"),
                     (self.soft, "softness"), (self.trail, "trail"), (self.smooth, "smooth")):
            w.s.blockSignals(True)
            w.set(getattr(st, k))
            w.s.blockSignals(False)
        self.follow.setChecked(st.follow_motion)
        self.always.setChecked(st.always_visible)
        self.fin_on.setChecked(st.fade_in_on)
        self.fout_on.setChecked(st.fade_out_on)
        self.fin.setValue(st.fade_in)
        self.fout.setValue(st.fade_out)
        self._loading = False
        self._sync_enabled()


# =============================================================================
# Visionneuse
# =============================================================================

class Viewer(QWidget):
    """Affiche l'image courante + calques (points, trajectoires, matte, zones)."""

    def __init__(self, win: "StudioWindow"):
        super().__init__()
        self.win = win
        self.setMinimumSize(480, 300)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.qimg: Optional[QImage] = None
        self.overlay: Optional[QImage] = None
        self.s = 1.0                      # pixels écran par pixel source
        self.off = QPointF(0, 0)
        self.fit_pending = True
        self.drag_poly: List[QPointF] = []   # en coordonnées source
        self.drag_rect_start: Optional[QPointF] = None
        self._rect_end = QPointF(0, 0)
        self._rect_tool = "rect"
        self.pan_start = None
        self.hover = -1

    # ---------------------------------------------------- transformations
    def fit(self):
        st = self.win.store
        if not st:
            return
        W, H = st.info.width, st.info.height
        self.s = min(self.width() / W, self.height() / H) * 0.97
        self.off = QPointF((self.width() - W * self.s) / 2, (self.height() - H * self.s) / 2)
        self.fit_pending = True
        self.update()

    def to_screen(self, x, y) -> QPointF:
        return QPointF(self.off.x() + x * self.s, self.off.y() + y * self.s)

    def to_src(self, p: QPointF):
        return (p.x() - self.off.x()) / self.s, (p.y() - self.off.y()) / self.s

    def resizeEvent(self, _):
        if self.fit_pending:
            self.fit()

    # ------------------------------------------------------------- dessin
    def paintEvent(self, _):
        w = self.win
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0e0f11"))
        if not w.store:
            p.setPen(QColor("#77787f"))
            f = p.font()
            f.setPointSize(15)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Glissez une vidéo ici\nou cliquez sur « Ouvrir une vidéo… »")
            return
        info = w.store.info
        target = QRectF(self.off.x(), self.off.y(), info.width * self.s, info.height * self.s)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        mode = w.view_mode()
        if mode == "matte":
            p.fillRect(target, QColor("black"))
        elif self.qimg is not None:
            p.drawImage(target, self.qimg)
        else:
            p.setPen(QColor("#77787f"))
            p.drawText(target, Qt.AlignCenter, "Chargement…")
        if self.overlay is not None and mode in ("matte", "overlay"):
            p.drawImage(target, self.overlay)
        p.setRenderHint(QPainter.Antialiasing)
        if mode != "matte" or w.show_points_on_matte():
            self._draw_tracks(p)
        self._draw_zones(p)
        p.setPen(QColor(220, 220, 225, 200))
        p.drawText(QRectF(10, 8, 600, 18), Qt.AlignLeft,
                   f"Image {w.cur} / {w.n_frames() - 1}    {info.width}×{info.height}"
                   + (f"    {int(w.sel.sum())} point(s) sélectionné(s)" if w.sel.any() else ""))

    def _draw_tracks(self, p: QPainter):
        w = self.win
        n = len(w.pts)
        if n == 0:
            return
        size = w.disp_size.value()
        trail = int(w.disp_trail.value())
        t = w.cur
        rd, res = w.rd, w.res
        nt = w.n_tracked
        cols = w.point_colors()
        sel = w.sel
        if rd is not None and nt and t < len(rd.smoothed):
            pos, alpha, vis = rd.smoothed, rd.alpha, rd.visible
            if trail > 0:
                t0 = max(0, t - trail)
                for i in range(nt):
                    if alpha[t, i] < 0.05:
                        continue
                    seg = pos[t0:t + 1, i]
                    ok = vis[t0:t + 1, i] & np.isfinite(seg[:, 0])
                    path = QPainterPath()
                    started = False
                    for k in range(len(seg)):
                        if not ok[k]:
                            started = False
                            continue
                        sp = self.to_screen(seg[k, 0], seg[k, 1])
                        if started:
                            path.lineTo(sp)
                        else:
                            path.moveTo(sp)
                            started = True
                    col = QColor(cols[i])
                    col.setAlphaF(0.5)
                    p.setPen(QPen(col, max(1.0, size * 0.4)))
                    p.drawPath(path)
            for i in range(nt):
                x, y = pos[t, i]
                if not np.isfinite(x) or not res.tracked[t]:
                    continue
                sp = self.to_screen(x, y)
                r = size * (1.6 if i == self.hover else 1.0)
                if vis[t, i]:
                    p.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
                    p.setBrush(QBrush(cols[i]))
                    p.drawEllipse(sp, r, r)
                elif alpha[t, i] > 0.02:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(170, 170, 175, int(220 * alpha[t, i])), 1.2))
                    p.drawEllipse(sp, r * 0.8, r * 0.8)
                if sel[i]:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor("white"), 1.5))
                    p.drawEllipse(sp, r + 3, r + 3)
        # points pas encore suivis : visibles sur leur image de pose
        for i in range(nt, n):
            sp = self.to_screen(*w.pts[i])
            col = QColor(cols[i]) if w.qf[i] == t else QColor(200, 200, 200, 90)
            r = size + 3
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 180), 3))
            p.drawLine(QPointF(sp.x() - r, sp.y()), QPointF(sp.x() + r, sp.y()))
            p.drawLine(QPointF(sp.x(), sp.y() - r), QPointF(sp.x(), sp.y() + r))
            p.setPen(QPen(col, 1.5))
            p.drawLine(QPointF(sp.x() - r, sp.y()), QPointF(sp.x() + r, sp.y()))
            p.drawLine(QPointF(sp.x(), sp.y() - r), QPointF(sp.x(), sp.y() + r))
            if sel[i]:
                p.setPen(QPen(QColor("white"), 1.5))
                p.drawEllipse(sp, r + 2, r + 2)

    def _draw_zones(self, p: QPainter):
        sel = self._rect_tool == "select"
        pen = QPen(QColor("white" if sel else ACCENT), 1.4, Qt.DashLine)
        p.setBrush(QColor(255, 255, 255, 18) if sel else QColor(232, 131, 58, 30))
        p.setPen(pen)
        if self.drag_poly:
            p.drawPolyline(QPolygonF([self.to_screen(q.x(), q.y()) for q in self.drag_poly]))
        if self.drag_rect_start is not None:
            a = self.to_screen(self.drag_rect_start.x(), self.drag_rect_start.y())
            b = self.to_screen(self._rect_end.x(), self._rect_end.y())
            p.drawRect(QRectF(a, b).normalized())

    # ------------------------------------------------------------- souris
    def wheelEvent(self, e):
        if not self.win.store:
            return
        k = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        pos = e.position()
        sx, sy = self.to_src(pos)
        self.s *= k
        self.off = QPointF(pos.x() - sx * self.s, pos.y() - sy * self.s)
        self.fit_pending = False
        self.update()

    def mouseDoubleClickEvent(self, _):
        self.fit()

    def mousePressEvent(self, e):
        w = self.win
        if not w.store:
            return
        pos = e.position()
        tool = w.tool()
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and tool == "pan"):
            self.pan_start = (pos, QPointF(self.off))
            return
        if e.button() == Qt.RightButton:
            i = self._nearest(pos)
            if i >= 0:
                w.delete_points(np.array([i]))
            return
        if e.button() != Qt.LeftButton:
            return
        x, y = self.to_src(pos)
        if tool == "point":
            w.add_points(np.array([[x, y]], np.float32), new_group=False)
        elif tool == "lasso":
            self.drag_poly = [QPointF(x, y)]
        elif tool in ("rect", "select"):
            self._rect_tool = tool
            self.drag_rect_start = QPointF(x, y)
            self._rect_end = QPointF(x, y)
            self._shift = bool(e.modifiers() & Qt.ShiftModifier)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self.pan_start is not None:
            p0, off0 = self.pan_start
            self.off = off0 + (pos - p0)
            self.fit_pending = False
            self.update()
            return
        if self.drag_poly:
            x, y = self.to_src(pos)
            self.drag_poly.append(QPointF(x, y))
            self.update()
            return
        if self.drag_rect_start is not None:
            self._rect_end = QPointF(*self.to_src(pos))
            self.update()
            return
        h = self._nearest(pos)
        if h != self.hover:
            self.hover = h
            self.update()

    def mouseReleaseEvent(self, e):
        if self.pan_start is not None:
            self.pan_start = None
            return
        if self.drag_poly:
            poly = [(q.x(), q.y()) for q in self.drag_poly]
            self.drag_poly = []
            if len(poly) >= 3:
                self.win.fill_zone(np.array(poly, np.float32))
            self.update()
        elif self.drag_rect_start is not None:
            a, b = self.drag_rect_start, self._rect_end
            self.drag_rect_start = None
            big = abs(a.x() - b.x()) * self.s > 4 and abs(a.y() - b.y()) * self.s > 4
            if self._rect_tool == "select":
                if big:
                    self.win.select_rect(min(a.x(), b.x()), min(a.y(), b.y()),
                                         max(a.x(), b.x()), max(a.y(), b.y()), self._shift)
                else:
                    self.win.select_point(self._nearest(self.to_screen(a.x(), a.y())), self._shift)
            elif big:
                poly = np.array([[a.x(), a.y()], [b.x(), a.y()], [b.x(), b.y()], [a.x(), b.y()]],
                                np.float32)
                self.win.fill_zone(poly)
            self.update()

    def _nearest(self, pos: QPointF) -> int:
        w = self.win
        xy = w.display_positions()
        if xy is None or not len(xy):
            return -1
        sx = self.off.x() + xy[:, 0] * self.s
        sy = self.off.y() + xy[:, 1] * self.s
        d = np.hypot(sx - pos.x(), sy - pos.y())
        d[~np.isfinite(d)] = 1e9
        i = int(np.argmin(d))
        return i if d[i] < 14 else -1

    # ------------------------------------------------------- glisser-déposer
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if p.lower().endswith(".tapnext"):
                self.win.load_project(p)
            else:
                self.win.open_video(p)


# =============================================================================
# Fenêtre principale
# =============================================================================

SETTINGS_FILE = os.path.join(HERE, "studio_settings.json")


def load_default_style() -> "se.ShapeStyle":
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return se.ShapeStyle.from_dict(json.load(f).get("default_style"))
    except (OSError, ValueError):
        return se.ShapeStyle()


class StudioWindow(QMainWindow):
    def __init__(self, video: str = "", job: Optional[dict] = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 940)
        self.job = job
        self.job_sent = False
        self.store: Optional[VideoStore] = None
        self.loader: Optional[LoadThread] = None
        self.tracker_thread: Optional[TrackThread] = None
        self.export_thread: Optional[ExportThread] = None
        self.cur = 0
        self.seg_in, self.seg_out = 0, 0
        self._range_user = False
        self.pts = np.zeros((0, 2), np.float32)
        self.qf = np.zeros(0, int)
        self.pgroup = np.zeros(0, int)
        self.sel = np.zeros(0, bool)
        self.groups: List[ShapeGroup] = []
        self._next_gid = 0
        self.res = None
        self.rd = None
        self.n_tracked = 0
        self.tracked_seg = None
        self.res_version = 0
        self._gcache: dict = {}
        self.undo: List[int] = []
        self.default_style = load_default_style()
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._refresh_overlay)
        self._build()
        if video:
            QTimer.singleShot(50, lambda: self.open_video(video))

    # ------------------------------------------------------------ interface
    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)
        top = QHBoxLayout()
        b_open = QPushButton("Ouvrir…")
        b_open.setToolTip("Ouvrir une vidéo ou un projet .tapnext (Ctrl+O)")
        b_open.clicked.connect(self._ask_open)
        top.addWidget(b_open)
        b_save = QPushButton("Enregistrer le projet")
        b_save.setToolTip("Sauvegarde points, suivi et formes (Ctrl+S) : rien à recalculer")
        b_save.clicked.connect(self.save_project)
        top.addWidget(b_save)
        top.addSpacing(12)
        self.tool_group = QButtonGroup(self)
        self.tool_btns = {}
        for key, text, tip in [
            ("lasso", "◯  Zone", "Entourez le sujet : la zone se remplit de points (Z)"),
            ("rect", "▭  Rectangle", "Rectangle rempli de points (R)"),
            ("point", "✚  Point", "Ajoute un point précis au groupe sélectionné (A)"),
            ("select", "⬚  Sélection", "Sélectionnez des points (glisser ; Maj = ajouter) (S)"),
            ("pan", "✋", "Déplacer la vue (ou clic milieu)"),
        ]:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            self.tool_group.addButton(b)
            self.tool_btns[key] = b
            top.addWidget(b)
        self.tool_btns["lasso"].setChecked(True)
        top.addStretch(1)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Image + points", "Image + matte", "Matte seule"])
        self.view_combo.setCurrentIndex(1)
        self.view_combo.currentIndexChanged.connect(lambda _: self._refresh_overlay())
        top.addWidget(QLabel("Vue :"))
        top.addWidget(self.view_combo)
        b_fit = QPushButton("Cadrer")
        b_fit.clicked.connect(lambda: self.viewer.fit())
        top.addWidget(b_fit)
        lv.addLayout(top)

        self.viewer = Viewer(self)
        lv.addWidget(self.viewer, 1)

        trans = QHBoxLayout()
        self.b_play = QPushButton("▶")
        self.b_play.setFixedWidth(40)
        self.b_play.clicked.connect(self.toggle_play)
        b_prev = QPushButton("◀|")
        b_prev.setFixedWidth(40)
        b_prev.clicked.connect(lambda: self.set_frame(self.cur - 1))
        b_next = QPushButton("|▶")
        b_next.setFixedWidth(40)
        b_next.clicked.connect(lambda: self.set_frame(self.cur + 1))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self.set_frame)
        trans.addWidget(b_prev)
        trans.addWidget(self.b_play)
        trans.addWidget(b_next)
        trans.addWidget(self.frame_spin)
        self.timeline = Timeline(self)
        self.timeline.seek.connect(self.set_frame)
        trans.addWidget(self.timeline, 1)
        lv.addLayout(trans)
        root.addWidget(left, 1)

        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(410)
        self.tabs.addTab(self._scroll(self._tab_tracking()), "① Points et suivi")
        self.tabs.addTab(self._scroll(self._tab_shapes()), "② Formes")
        self.tabs.addTab(self._scroll(self._tab_export()), "③ Export")
        root.addWidget(self.tabs)
        self.statusBar().showMessage("Ouvrez une vidéo pour commencer.")

        def sc(key, fn):
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.setShortcutContext(Qt.ApplicationShortcut)
            a.triggered.connect(fn)
            self.addAction(a)
        sc("Space", self.toggle_play)
        sc("Left", lambda: self.set_frame(self.cur - 1))
        sc("Right", lambda: self.set_frame(self.cur + 1))
        sc("Home", lambda: self.set_frame(self.seg_in))
        sc("End", lambda: self.set_frame(self.seg_out))
        sc("I", lambda: self.in_spin.setValue(self.cur))
        sc("O", lambda: self.out_spin.setValue(self.cur))
        sc("F", lambda: self.viewer.fit())
        sc("A", lambda: self.tool_btns["point"].setChecked(True))
        sc("Z", lambda: self.tool_btns["lasso"].setChecked(True))
        sc("R", lambda: self.tool_btns["rect"].setChecked(True))
        sc("S", lambda: self.tool_btns["select"].setChecked(True))
        sc("Esc", self.clear_selection)
        sc("Delete", self.delete_selection)
        sc("Ctrl+Z", self.undo_last)
        sc("Ctrl+O", self._ask_open)
        sc("Ctrl+S", self.save_project)
        sc("Ctrl+A", self.select_all)

    def _scroll(self, w: QWidget) -> QScrollArea:
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        s.setWidget(w)
        return s

    def _tab_tracking(self) -> QWidget:
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 4, 10, 10)
        g = QGroupBox("Placer les points")
        gl = QVBoxLayout(g)
        gl.addWidget(help_label(
            "Allez sur une image où le sujet est bien visible, puis <b>entourez-le</b> avec "
            "l'outil Zone : il se remplit de points sur les détails texturés. Chaque zone "
            "devient un <b>groupe de formes</b> (onglet ②). <b>Clic droit</b> = supprimer un point."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Points par zone"))
        self.n_per_zone = QSpinBox()
        self.n_per_zone.setRange(1, 5000)
        self.n_per_zone.setValue(500)
        row.addWidget(self.n_per_zone)
        gl.addLayout(row)
        self.seed_mode = QComboBox()
        self.seed_mode.addItems(["Détails texturés (recommandé)", "Grille régulière"])
        gl.addWidget(self.seed_mode)
        row = QHBoxLayout()
        b_all = QPushButton("Remplir toute l'image")
        b_all.clicked.connect(self.fill_full)
        b_clear = QPushButton("Tout effacer")
        b_clear.clicked.connect(self.clear_points)
        row.addWidget(b_all)
        row.addWidget(b_clear)
        gl.addLayout(row)
        self.lbl_points = QLabel("Aucun point")
        gl.addWidget(self.lbl_points)
        pv.addWidget(g)

        g = QGroupBox("Suivre")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.in_spin, self.out_spin = QSpinBox(), QSpinBox()
        for sp in (self.in_spin, self.out_spin):
            sp.setRange(0, 0)
        self.in_spin.valueChanged.connect(self._range_changed)
        self.out_spin.valueChanged.connect(self._range_changed)
        b_in = QPushButton("[")
        b_in.setToolTip("Début = image courante (I)")
        b_in.setFixedWidth(28)
        b_in.clicked.connect(lambda: self.in_spin.setValue(self.cur))
        b_out = QPushButton("]")
        b_out.setToolTip("Fin = image courante (O)")
        b_out.setFixedWidth(28)
        b_out.clicked.connect(lambda: self.out_spin.setValue(self.cur))
        row.addWidget(QLabel("Début"))
        row.addWidget(self.in_spin)
        row.addWidget(b_in)
        row.addWidget(QLabel("Fin"))
        row.addWidget(self.out_spin)
        row.addWidget(b_out)
        gl.addLayout(row)
        self.quality = QComboBox()
        self.quality.addItems(["Précision maximale (512 px)", "Rapide (256 px)"])
        gl.addWidget(self.quality)
        self.cb_verify = QCheckBox("Contrôle aller-retour (recommandé)")
        self.cb_verify.setChecked(True)
        gl.addWidget(self.cb_verify)
        gl.addWidget(help_label("Chaque point est re-suivi à l'envers : s'il ne revient pas à "
                                "son départ, il est coupé là où il a décroché (≈ ×2 temps)."))
        self.cb_fp16 = QCheckBox("Économie de VRAM (FP16)")
        gl.addWidget(self.cb_fp16)
        self.b_track = QPushButton("▶  Lancer le suivi")
        self.b_track.setObjectName("primary")
        self.b_track.clicked.connect(self.start_tracking)
        gl.addWidget(self.b_track)
        self.b_cancel = QPushButton("Annuler")
        self.b_cancel.clicked.connect(self.cancel_work)
        self.b_cancel.setEnabled(False)
        gl.addWidget(self.b_cancel)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        gl.addWidget(self.prog)
        self.lbl_prog = help_label("")
        gl.addWidget(self.lbl_prog)
        pv.addWidget(g)

        g = QGroupBox("Affichage")
        gl = QVBoxLayout(g)
        self.disp_size = ValueSlider("Taille des points", 1, 14, 2, 0.5, "{:.1f}")
        self.disp_size.changed.connect(lambda _: self.viewer.update())
        self.disp_trail = ValueSlider("Traînées affichées (images)", 0, 60, 35, 1, "{:.0f}")
        self.disp_trail.changed.connect(lambda _: self.viewer.update())
        self.cb_points_on_matte = QCheckBox("Afficher les points sur la vue « Matte seule »")
        self.cb_points_on_matte.toggled.connect(lambda _: self.viewer.update())
        gl.addWidget(self.disp_size)
        gl.addWidget(self.disp_trail)
        gl.addWidget(self.cb_points_on_matte)
        pv.addWidget(g)
        pv.addStretch(1)
        return panel

    def _tab_shapes(self) -> QWidget:
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 4, 10, 10)
        g = QGroupBox("Groupes de formes")
        gl = QVBoxLayout(g)
        gl.addWidget(help_label(
            "Chaque groupe de points porte ses formes. Cochez/décochez pour l'activer, "
            "double-cliquez pour le renommer. Avec l'outil <b>Sélection</b> (S), prenez des "
            "points et créez-en un nouveau groupe pour les régler à part."))
        self.group_list = QListWidget()
        self.group_list.setMinimumHeight(110)
        self.group_list.setMaximumHeight(170)
        self.group_list.currentRowChanged.connect(self._on_group_row)
        self.group_list.itemChanged.connect(self._on_group_item_changed)
        gl.addWidget(self.group_list)
        row = QHBoxLayout()
        b = QPushButton("Sélectionner ses points")
        b.clicked.connect(self.select_group_points)
        row.addWidget(b)
        b = QPushButton("Supprimer le groupe")
        b.clicked.connect(self.delete_group)
        row.addWidget(b)
        gl.addLayout(row)
        self.lbl_sel = QLabel("Aucun point sélectionné")
        gl.addWidget(self.lbl_sel)
        row = QHBoxLayout()
        self.b_new_group = QPushButton("Nouveau groupe avec la sélection")
        self.b_new_group.clicked.connect(self.group_from_selection)
        row.addWidget(self.b_new_group)
        self.b_del_sel = QPushButton("Supprimer")
        self.b_del_sel.clicked.connect(self.delete_selection)
        row.addWidget(self.b_del_sel)
        gl.addLayout(row)
        pv.addWidget(g)

        g = QGroupBox("Style du groupe")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItem("Préréglage…")
        for name in se.PRESETS:
            self.preset.addItem(name)
        self.preset.activated.connect(self._apply_preset)
        row.addWidget(self.preset, 1)
        gl.addLayout(row)
        self.editor = StyleEditor()
        self.editor.changed.connect(self._on_style_changed)
        gl.addWidget(self.editor)
        row = QHBoxLayout()
        b = QPushButton("Appliquer à tous les groupes")
        b.clicked.connect(self._style_to_all)
        row.addWidget(b)
        b = QPushButton("Style par défaut")
        b.setToolTip("Les nouveaux groupes utiliseront ce style")
        b.clicked.connect(self._save_default_style)
        row.addWidget(b)
        gl.addLayout(row)
        pv.addWidget(g)

        g = QGroupBox("Matte finale")
        gl = QVBoxLayout(g)
        self.cb_invert = QCheckBox("Inverser la matte finale")
        self.cb_invert.setChecked(True)
        self.cb_invert.toggled.connect(self._schedule_overlay)
        gl.addWidget(self.cb_invert)
        pv.addWidget(g)
        pv.addStretch(1)
        self._group_widgets_enabled(False)
        return panel

    def _tab_export(self) -> QWidget:
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 4, 10, 10)
        g = QGroupBox("Exporter" + (" vers DaVinci Resolve" if self.job else ""))
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Dossier de sortie")
        b_dir = QPushButton("…")
        b_dir.setFixedWidth(28)
        b_dir.clicked.connect(self._ask_outdir)
        row.addWidget(self.out_edit)
        row.addWidget(b_dir)
        gl.addLayout(row)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nom des fichiers")
        gl.addWidget(self.name_edit)
        row = QHBoxLayout()
        self.cb_matte = QCheckBox("Matte N&&B")
        self.cb_matte.setChecked(True)
        self.codec = QComboBox()
        self.codec.addItems(["ProRes 422 HQ (.mov)", "DNxHR HQ (.mov)", "H.264 (.mp4)", "PNG (séquence)"])
        row.addWidget(self.cb_matte)
        row.addWidget(self.codec, 1)
        gl.addLayout(row)
        self.cb_per_group = QCheckBox("Une matte par groupe en plus")
        self.cb_per_group.setToolTip("Utile pour corriger chaque groupe avec un nœud différent")
        gl.addWidget(self.cb_per_group)
        self.cb_data = QCheckBox("Trajectoires CSV + JSON")
        self.cb_data.setChecked(True)
        gl.addWidget(self.cb_data)
        row = QHBoxLayout()
        row.addWidget(QLabel("Fusion"))
        self.fusion_mode = QComboBox()
        self.fusion_mode.addItems(["Aucun nœud", "Stabilisation", "Match-move", "Les deux"])
        self.fusion_mode.setCurrentIndex(0 if self.job else 3)
        row.addWidget(self.fusion_mode, 1)
        gl.addLayout(row)
        self.fusion_smooth = ValueSlider("Lissage de la stabilisation (0 = plan verrouillé)",
                                         0, 100, 0, 1, "{:.0f}")
        gl.addWidget(self.fusion_smooth)
        self.cb_attach = QCheckBox("Attacher la matte au clip dans Resolve")
        self.cb_attach.setChecked(True)
        self.cb_attach.setVisible(bool(self.job))
        gl.addWidget(self.cb_attach)
        if self.job:
            gl.addWidget(help_label("Le nœud Fusion choisi est inséré directement dans la comp "
                                    "du clip, dans Resolve."))
        self.cb_preview = QCheckBox("Vidéo de contrôle (preview)")
        self.cb_preview.setChecked(True)
        gl.addWidget(self.cb_preview)
        self.b_export = QPushButton("⇪  Exporter et envoyer à Resolve" if self.job else "⇪  Exporter")
        self.b_export.setObjectName("primary")
        self.b_export.clicked.connect(self.start_export)
        self.b_export.setEnabled(False)
        gl.addWidget(self.b_export)
        self.prog_exp = QProgressBar()
        self.prog_exp.setRange(0, 1000)
        gl.addWidget(self.prog_exp)
        self.lbl_exp = help_label("")
        gl.addWidget(self.lbl_exp)
        pv.addWidget(g)
        pv.addStretch(1)
        return panel

    # --------------------------------------------------------------- état
    def n_frames(self) -> int:
        return self.store.total if self.store else 1

    def tool(self) -> str:
        for k, b in self.tool_btns.items():
            if b.isChecked():
                return k
        return "lasso"

    def view_mode(self) -> str:
        return ["points", "overlay", "matte"][self.view_combo.currentIndex()]

    def show_points_on_matte(self) -> bool:
        return self.cb_points_on_matte.isChecked()

    def group_by_id(self, gid: int) -> Optional[ShapeGroup]:
        for g in self.groups:
            if g.id == gid:
                return g
        return None

    def point_colors(self) -> List[QColor]:
        cmap = {g.id: g.color for g in self.groups}
        grey = QColor(180, 180, 180)
        return [cmap.get(int(gid), grey) if self.group_by_id(int(gid)) and
                self.group_by_id(int(gid)).enabled else grey for gid in self.pgroup]

    def display_positions(self) -> Optional[np.ndarray]:
        n = len(self.pts)
        if n == 0:
            return None
        xy = self.pts.astype(np.float64).copy()
        if self.rd is not None and self.n_tracked:
            t = self.cur
            cur = self.rd.smoothed[t, :self.n_tracked].astype(np.float64)
            if not self.res.tracked[t]:
                cur[:] = np.nan
            xy[:self.n_tracked] = cur
        return xy

    # --------------------------------------------------------- ouverture
    def _ask_open(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une vidéo ou un projet", "",
            "Vidéos et projets (*.mp4 *.mov *.mxf *.mkv *.avi *.tapnext);;Tous (*.*)")
        if p:
            if p.lower().endswith(".tapnext"):
                self.load_project(p)
            else:
                self.open_video(p)

    def _reset_points(self):
        self.pts = np.zeros((0, 2), np.float32)
        self.qf = np.zeros(0, int)
        self.pgroup = np.zeros(0, int)
        self.sel = np.zeros(0, bool)
        self.groups = []
        self.res = self.rd = None
        self.n_tracked = 0
        self.tracked_seg = None
        self.res_version += 1
        self._gcache.clear()
        self.undo.clear()
        self._refresh_group_list()

    def open_video(self, path: str) -> bool:
        if self.loader is not None:
            self.loader.stop_flag.set()
            self.loader.wait()
        try:
            store = VideoStore(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Impossible d'ouvrir la vidéo :\n{e}")
            return False
        self.store = store
        self._reset_points()
        n = store.total
        for sp in (self.frame_spin, self.in_spin, self.out_spin):
            sp.blockSignals(True)
            sp.setRange(0, n - 1)
            sp.blockSignals(False)
        self.seg_in, self.seg_out = 0, n - 1
        if self.job:
            self.seg_in = int(min(max(self.job.get("start", 0), 0), n - 1))
            self.seg_out = int(min(max(self.job.get("end", n - 1), self.seg_in), n - 1))
        self._set_range(self.seg_in, self.seg_out)
        self._range_user = bool(self.job)
        stem = os.path.splitext(os.path.basename(path))[0]
        self.out_edit.setText((self.job or {}).get("out_dir")
                              or os.path.join(os.path.dirname(os.path.abspath(path)), "TAPNext"))
        self.name_edit.setText((self.job or {}).get("name") or stem)
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(path)}")
        self.loader = LoadThread(store)
        self.loader.progress.connect(self._on_load_progress)
        self.loader.done.connect(self._on_load_done)
        self.loader.failed.connect(lambda m: QMessageBox.critical(self, APP_NAME, m))
        self.loader.start()
        self.cur = -1
        self.viewer.fit()
        QTimer.singleShot(200, lambda: self.set_frame(self.seg_in))
        self._update_labels()
        return True

    def _on_load_progress(self, n):
        self.statusBar().showMessage(f"Chargement de la vidéo : {n} / {self.store.total} images…")
        if self.viewer.qimg is None:
            self.set_frame(max(self.cur, 0), force=True)
        self.timeline.update()

    def _on_load_done(self, n):
        st = self.store
        for sp in (self.frame_spin, self.in_spin, self.out_spin):
            sp.blockSignals(True)
            sp.setRange(0, n - 1)
            sp.blockSignals(False)
        self.seg_out = min(self.seg_out, n - 1) if self._range_user else n - 1
        self._set_range(min(self.seg_in, self.seg_out), self.seg_out)
        self.statusBar().showMessage(
            f"Vidéo chargée : {n} images, {st.info.width}×{st.info.height} @ {st.info.fps:.3f} i/s. "
            "Entourez le sujet avec l'outil Zone.", 15000)
        self.set_frame(self.cur, force=True)
        self.timeline.update()

    def _ask_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Dossier de sortie", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    # --------------------------------------------------------- navigation
    def set_frame(self, f: int, force: bool = False):
        if not self.store:
            return
        f = int(min(max(f, 0), self.n_frames() - 1))
        if f == self.cur and not force:
            return
        self.cur = f
        img = self.store.frame(f)
        if img is not None:
            rgb = np.ascontiguousarray(img)
            self._img_ref = rgb
            self.viewer.qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                                      QImage.Format_BGR888)
        else:
            self.viewer.qimg = None
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(f)
        self.frame_spin.blockSignals(False)
        self._refresh_overlay()
        self.timeline.update()

    def toggle_play(self):
        if not self.store:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.b_play.setText("▶")
        else:
            if self.cur >= self.seg_out:
                self.set_frame(self.seg_in)
            self.play_timer.start(int(1000 / max(1.0, self.store.info.fps)))
            self.b_play.setText("❚❚")

    def _play_tick(self):
        nxt = self.cur + 1
        if nxt > self.seg_out or nxt >= self.store.loaded:
            nxt = self.seg_in
        self.set_frame(nxt)

    def _set_range(self, a: int, b: int):
        """Fixe début/fin sans que l'un écrase l'autre (signaux bloqués)."""
        for sp, v in ((self.in_spin, a), (self.out_spin, b)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self.seg_in, self.seg_out = self.in_spin.value(), max(self.out_spin.value(), self.in_spin.value())
        self.timeline.update()

    def _range_changed(self, _=None):
        self._range_user = True
        self.seg_in = self.in_spin.value()
        self.seg_out = max(self.out_spin.value(), self.seg_in)
        self.timeline.update()

    # ------------------------------------------------------------- groupes
    def new_group(self, name: Optional[str] = None, style: Optional["se.ShapeStyle"] = None) -> ShapeGroup:
        gid = self._next_gid
        self._next_gid += 1
        g = ShapeGroup(gid, name or f"Groupe {gid + 1}",
                       style or se.ShapeStyle.from_dict(self.default_style.to_dict()))
        self.groups.append(g)
        self._refresh_group_list(select=gid)
        return g

    def current_group(self) -> Optional[ShapeGroup]:
        row = self.group_list.currentRow()
        return self.groups[row] if 0 <= row < len(self.groups) else None

    def _refresh_group_list(self, select: Optional[int] = None):
        cur = self.current_group()
        sel_id = select if select is not None else (cur.id if cur else None)
        self.group_list.blockSignals(True)
        self.group_list.clear()
        row_sel = -1
        for i, g in enumerate(self.groups):
            n = int((self.pgroup == g.id).sum())
            mode = "découpe" if g.style.mode == "subtract" else se.SHAPE_LABELS[g.style.shape].lower()
            it = QListWidgetItem(color_icon(g.color), f"{g.name}   ·  {n} pts  ·  {mode}")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if g.enabled else Qt.Unchecked)
            it.setData(Qt.UserRole, g.id)
            self.group_list.addItem(it)
            if g.id == sel_id:
                row_sel = i
        if row_sel < 0 and self.groups:
            row_sel = len(self.groups) - 1
        self.group_list.setCurrentRow(row_sel)
        self.group_list.blockSignals(False)
        self._on_group_row(row_sel)

    def _on_group_row(self, row: int):
        g = self.groups[row] if 0 <= row < len(self.groups) else None
        self._group_widgets_enabled(g is not None)
        if g is not None:
            self.editor.set_style(g.style)

    def _group_widgets_enabled(self, on: bool):
        if hasattr(self, "editor"):
            self.editor.setEnabled(on)
            self.preset.setEnabled(on)

    def _on_group_item_changed(self, it: QListWidgetItem):
        g = self.group_by_id(it.data(Qt.UserRole))
        if g is None:
            return
        g.enabled = it.checkState() == Qt.Checked
        text = it.text().split("   ·  ")[0].strip()
        if text and text != g.name:
            g.name = text
            QTimer.singleShot(0, self._refresh_group_list)
        self.viewer.update()
        self._schedule_overlay()

    def _on_style_changed(self, key: str):
        if key in ("shape", "mode"):
            self._refresh_group_list()
        self._schedule_overlay()

    def _apply_preset(self, idx: int):
        g = self.current_group()
        if g is None or idx <= 0:
            return
        name = self.preset.itemText(idx)
        d = se.ShapeStyle().to_dict()
        d.update(se.PRESETS[name])
        g.style = se.ShapeStyle.from_dict(d)
        self.editor.set_style(g.style)
        self.preset.setCurrentIndex(0)
        self._refresh_group_list()
        self._schedule_overlay()

    def _style_to_all(self):
        g = self.current_group()
        if g is None:
            return
        for o in self.groups:
            if o is not g:
                o.style = se.ShapeStyle.from_dict(g.style.to_dict())
        self._refresh_group_list()
        self._schedule_overlay()

    def _save_default_style(self):
        g = self.current_group()
        if g is None:
            return
        self.default_style = se.ShapeStyle.from_dict(g.style.to_dict())
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({"default_style": self.default_style.to_dict()}, f, indent=1)
            self.statusBar().showMessage("Style enregistré comme style par défaut.", 5000)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Impossible d'enregistrer : {e}")

    def delete_group(self):
        g = self.current_group()
        if g is None or self._busy():
            return
        self.delete_points(np.nonzero(self.pgroup == g.id)[0], drop_group=g.id)

    def select_group_points(self):
        g = self.current_group()
        if g is not None:
            self.sel = self.pgroup == g.id
            self._sel_changed()

    # ----------------------------------------------------------- sélection
    def _sel_changed(self):
        n = int(self.sel.sum())
        self.lbl_sel.setText(f"{n} point(s) sélectionné(s)" if n else "Aucun point sélectionné")
        self.b_new_group.setEnabled(n > 0)
        self.b_del_sel.setEnabled(n > 0)
        self.viewer.update()

    def select_rect(self, x0, y0, x1, y1, add=False):
        xy = self.display_positions()
        if xy is None:
            return
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        self.sel = (self.sel | inside) if add else inside
        self._sel_changed()
        if self.sel.any():
            self.tabs.setCurrentIndex(1)

    def select_point(self, i: int, add=False):
        if not add:
            self.sel[:] = False
        if i >= 0:
            self.sel[i] = not self.sel[i] if add else True
            g = self.group_by_id(int(self.pgroup[i]))
            if g is not None:
                self.group_list.setCurrentRow(self.groups.index(g))
        self._sel_changed()

    def select_all(self):
        self.sel[:] = True
        self._sel_changed()

    def clear_selection(self):
        self.sel[:] = False
        self._sel_changed()

    def group_from_selection(self):
        if not self.sel.any():
            return
        src = self.group_by_id(int(self.pgroup[np.argmax(self.sel)]))
        style = se.ShapeStyle.from_dict(src.style.to_dict()) if src else None
        g = self.new_group(f"Sélection {self._next_gid + 1}", style)
        self.pgroup[self.sel] = g.id
        self.sel[:] = False
        self._drop_empty_groups()
        self._gcache.clear()
        self._refresh_group_list(select=g.id)
        self._sel_changed()
        self._schedule_overlay()
        self.statusBar().showMessage(f"Groupe « {g.name} » créé : réglez son style ci-dessous.", 6000)

    def delete_selection(self):
        if self.sel.any():
            self.delete_points(np.nonzero(self.sel)[0])

    def _drop_empty_groups(self):
        used = set(int(v) for v in self.pgroup)
        self.groups = [g for g in self.groups if g.id in used]

    # ------------------------------------------------------------- points
    def _busy(self) -> bool:
        if self.tracker_thread is not None:
            self.statusBar().showMessage("Suivi en cours : attendez la fin pour modifier les points.", 4000)
            return True
        return False

    def add_points(self, pts: np.ndarray, frame: Optional[int] = None, new_group: bool = True,
                   name: Optional[str] = None):
        if not len(pts) or not self.store or self._busy():
            return
        W, H = self.store.info.width, self.store.info.height
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        f = self.cur if frame is None else frame
        if not (self.seg_in <= f <= self.seg_out):
            self._set_range(min(self.seg_in, f), max(self.seg_out, f))
            self._range_user = True
        g = None if new_group else self.current_group()
        if g is None:
            g = self.new_group(name or (f"Zone {self._next_gid + 1}" if new_group
                                        else f"Points {self._next_gid + 1}"))
        self.pts = np.concatenate([self.pts, pts.astype(np.float32)])
        self.qf = np.concatenate([self.qf, np.full(len(pts), f, int)])
        self.pgroup = np.concatenate([self.pgroup, np.full(len(pts), g.id, int)])
        self.sel = np.concatenate([self.sel, np.zeros(len(pts), bool)])
        self.undo.append(len(pts))
        self._refresh_group_list(select=g.id)
        self._update_labels()
        self.viewer.update()
        self.timeline.update()

    def fill_zone(self, poly_src: np.ndarray):
        if self._busy():
            return
        st = self.store
        img = st.frame(self.cur) if st else None
        if img is None:
            self.statusBar().showMessage("Image pas encore chargée.", 4000)
            return
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly_src * st.scale).astype(np.int32)], 255)
        mode = "features" if self.seed_mode.currentIndex() == 0 else "grid"
        pts = eng.seed_points(img, self.n_per_zone.value(), mask, mode) / st.scale
        if not len(pts):
            self.statusBar().showMessage("Zone trop petite : aucun point placé.", 4000)
            return
        self.add_points(pts)
        self.statusBar().showMessage(
            f"{len(pts)} points placés sur l'image {self.cur} (nouveau groupe). "
            "Lancez le suivi.", 8000)

    def fill_full(self):
        st = self.store
        if not st:
            return
        W, H = st.info.width, st.info.height
        m = 0.02
        self.fill_zone(np.array([[W * m, H * m], [W * (1 - m), H * m],
                                 [W * (1 - m), H * (1 - m)], [W * m, H * (1 - m)]], np.float32))

    def delete_points(self, idx: np.ndarray, drop_group: Optional[int] = None):
        if self._busy():
            return
        keep = np.ones(len(self.pts), bool)
        keep[np.asarray(idx, int)] = False
        self._keep_points(keep)
        if drop_group is not None:
            self.groups = [g for g in self.groups if g.id != drop_group]
            self._refresh_group_list()

    def _keep_points(self, keep: np.ndarray):
        nt = self.n_tracked
        self.pts, self.qf = self.pts[keep], self.qf[keep]
        self.pgroup, self.sel = self.pgroup[keep], self.sel[keep]
        if self.res is not None and nt:
            kt = keep[:nt]
            r = self.res
            r.positions, r.visibility = r.positions[:, kt], r.visibility[:, kt]
            r.query_frames = r.query_frames[kt]
            r.cut_frames = r.cut_frames[kt]
            self.n_tracked = int(kt.sum())
            if self.n_tracked == 0:
                self.res = self.rd = None
            self._tracking_changed()
        self._drop_empty_groups()
        self.undo.clear()
        self._refresh_group_list()
        self._update_labels()
        self._sel_changed()
        self.viewer.hover = -1
        self.timeline.update()

    def undo_last(self):
        if not self.undo or self._busy():
            return
        k = self.undo.pop()
        n = len(self.pts)
        if n - k < self.n_tracked:
            return
        keep = np.ones(n, bool)
        keep[n - k:] = False
        undo = self.undo
        self._keep_points(keep)
        self.undo = undo

    def clear_points(self):
        if self._busy():
            return
        self._reset_points()
        self._update_labels()
        self._sel_changed()
        self._refresh_overlay()
        self.timeline.update()

    def _update_labels(self):
        n, nt = len(self.pts), self.n_tracked
        if n == 0:
            txt = "Aucun point"
        elif nt == n:
            cut = int((self.res.cut_frames >= 0).sum()) if self.res is not None else 0
            txt = f"{n} points suivis" + (f" · {cut} coupé(s) au décrochage" if cut else "")
        else:
            txt = f"{n} points · {n - nt} à suivre"
        self.lbl_points.setText(txt)
        self.b_export.setEnabled(self.res is not None and self.export_thread is None)
        self.b_track.setEnabled(n > nt and self.tracker_thread is None)
        if n > nt and nt:
            self.b_track.setText(f"▶  Suivre les {n - nt} nouveaux points")
        else:
            self.b_track.setText("▶  Lancer le suivi")

    # ------------------------------------------------------------- suivi
    def start_tracking(self):
        if not self.store or self.tracker_thread is not None:
            return
        if not self.store.complete and self.store.loaded <= self.seg_out:
            QMessageBox.information(self, APP_NAME, "La vidéo est encore en cours de chargement.")
            return
        seg = (self.seg_in, self.seg_out)
        if self.res is not None and self.tracked_seg == seg and self.n_tracked:
            first = self.n_tracked
        else:
            first = 0
            self.res = self.rd = None
            self.n_tracked = 0
        pts, qf = self.pts[first:], np.clip(self.qf[first:], *seg)
        res_px = 512 if self.quality.currentIndex() == 0 else 256
        self.tracker_thread = TrackThread(self.store, pts, qf, seg[0], seg[1], res_px,
                                          self.cb_verify.isChecked(), self.cb_fp16.isChecked())
        self.tracker_thread.progress.connect(self._on_progress)
        self.tracker_thread.done.connect(lambda r: self._on_tracked(r, first, seg))
        self.tracker_thread.failed.connect(self._on_failed)
        self.tracker_thread.finished.connect(self._thread_finished)
        self.b_track.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.prog.setValue(0)
        self.tracker_thread.start()

    def _on_progress(self, desc: str, frac: float):
        bar, lbl = (self.prog_exp, self.lbl_exp) if self.export_thread is not None else (self.prog, self.lbl_prog)
        bar.setValue(int(frac * 1000))
        lbl.setText(desc)
        self.statusBar().showMessage(f"{desc}… {frac * 100:.0f} %")

    def _on_tracked(self, r, first: int, seg):
        if first and self.res is not None:
            old = self.res
            old.positions = np.concatenate([old.positions, r.positions], axis=1)
            old.visibility = np.concatenate([old.visibility, r.visibility], axis=1)
            old.query_frames = np.concatenate([old.query_frames, r.query_frames])
            old.cut_frames = np.concatenate([old.cut_frames, r.cut_frames])
        else:
            self.res = r
        self.n_tracked = first + r.positions.shape[1]
        self.tracked_seg = seg
        self._tracking_changed()
        self.prog.setValue(1000)
        n_cut = int((r.cut_frames >= 0).sum())
        msg = (f"Suivi terminé en {getattr(r, 'elapsed', 0):.0f} s sur {getattr(r, 'device', '?')}"
               + (f" · {n_cut} point(s) coupé(s) au décrochage" if n_cut else ""))
        self.lbl_prog.setText(msg)
        self.statusBar().showMessage(msg + ". Réglez les formes dans l'onglet ②.", 15000)
        self._update_labels()

    def _on_failed(self, msg: str):
        for lbl in (self.lbl_prog, self.lbl_exp):
            lbl.setText(msg)
        self.statusBar().showMessage(msg, 15000)
        if "annulé" not in msg:
            QMessageBox.warning(self, APP_NAME, msg)

    def _thread_finished(self):
        self.tracker_thread = None
        self.export_thread = None
        self.b_cancel.setEnabled(False)
        self._update_labels()

    def cancel_work(self):
        for t in (self.tracker_thread, self.export_thread):
            if t is not None:
                t.cancel.set()

    # ------------------------------------------------------------ rendu
    def _tracking_changed(self):
        self.res_version += 1
        self._gcache.clear()
        if self.res is None:
            self.rd = None
        else:
            self.rd = eng.prepare_render_data(self.res, 1.0, 0.5, 6, 8)
        self._refresh_group_list()
        self._refresh_overlay()

    def _group_data(self, g: ShapeGroup) -> Optional["se.GroupData"]:
        if self.res is None or not self.n_tracked:
            return None
        cols = np.nonzero(self.pgroup[:self.n_tracked] == g.id)[0]
        if not len(cols):
            return None
        st = g.style
        key = (self.res_version, tuple(cols[:8]), len(cols), st.smooth, st.always_visible,
               st.fade_in_on, st.fade_in, st.fade_out_on, st.fade_out, st.size_jitter)
        hit = self._gcache.get(g.id)
        if hit is not None and hit[0] == key:
            return hit[1]
        gd = se.prepare_group(self.res, cols, st)
        self._gcache[g.id] = (key, gd)
        return gd

    def layers(self) -> list:
        return [(g.style, self._group_data(g), g.enabled) for g in self.groups]

    def _schedule_overlay(self, *_):
        self.render_timer.start(25)

    def _refresh_overlay(self):
        v = self.viewer
        if self.res is None or self.view_mode() == "points" or not self.store:
            v.overlay = None
            v.update()
            return
        st, t = self.store, self.cur
        key = (st.pw, st.ph)
        if getattr(self, "_rkey", None) != key:
            self._renderer = se.ShapeRenderer(st.info.width, st.info.height, st.pw, st.ph, 960)
            self._rkey = key
        layers = [L for L in self.layers() if L[1] is not None]
        m = self._renderer.render(layers, t, self.cb_invert.isChecked()) if layers else \
            np.full((st.ph, st.pw), 255 if self.cb_invert.isChecked() else 0, np.uint8)
        if self.view_mode() == "matte":
            v.overlay = np_to_qimage_gray(m)
        else:
            rgba = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 70, 60
            rgba[..., 3] = (m.astype(np.uint16) * 140 // 255).astype(np.uint8)
            self._ov_ref = rgba
            v.overlay = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                               QImage.Format_RGBA8888)
        v.update()

    # ------------------------------------------------------------ projet
    def save_project(self):
        if not self.store:
            return
        base = os.path.splitext(self.store.path)[0] + ".tapnext"
        p, _ = QFileDialog.getSaveFileName(self, "Enregistrer le projet", base,
                                           "Projet TAPNext (*.tapnext)")
        if not p:
            return
        meta = dict(video=self.store.path, seg=[self.seg_in, self.seg_out], cur=self.cur,
                    n_tracked=self.n_tracked, tracked_seg=self.tracked_seg,
                    invert=self.cb_invert.isChecked(), next_gid=self._next_gid,
                    groups=[dict(id=g.id, name=g.name, enabled=g.enabled,
                                 style=g.style.to_dict()) for g in self.groups])
        arrays = dict(pts=self.pts, qf=self.qf, pgroup=self.pgroup,
                      meta=np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8))
        if self.res is not None:
            r = self.res
            arrays.update(positions=r.positions, visibility=r.visibility, tracked=r.tracked,
                          query_frames=r.query_frames, cut_frames=r.cut_frames)
        try:
            with open(p, "wb") as f:
                np.savez_compressed(f, **arrays)
            self.statusBar().showMessage(f"Projet enregistré : {p}", 8000)
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Impossible d'enregistrer : {e}")

    def load_project(self, p: str):
        try:
            z = np.load(p, allow_pickle=False)
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Projet illisible :\n{e}")
            return
        video = meta.get("video", "")
        if not os.path.isfile(video):
            video, _ = QFileDialog.getOpenFileName(self, "Vidéo du projet introuvable : où est-elle ?",
                                                   os.path.dirname(p))
            if not video:
                return
        if not self.open_video(video):
            return
        self.pts, self.qf, self.pgroup = z["pts"], z["qf"], z["pgroup"]
        self.sel = np.zeros(len(self.pts), bool)
        self.groups = []
        for d in meta.get("groups", []):
            g = ShapeGroup(int(d["id"]), d["name"], se.ShapeStyle.from_dict(d["style"]))
            g.enabled = bool(d.get("enabled", True))
            self.groups.append(g)
        self._next_gid = int(meta.get("next_gid", len(self.groups)))
        self.cb_invert.setChecked(bool(meta.get("invert", True)))
        a, b = meta.get("seg", [self.seg_in, self.seg_out])
        self._set_range(int(a), int(b))
        self._range_user = True
        if "positions" in z.files:
            self.res = eng.TrackResult(z["positions"], z["visibility"], z["tracked"],
                                       z["query_frames"], z["cut_frames"])
            self.n_tracked = int(meta.get("n_tracked", self.res.positions.shape[1]))
            ts = meta.get("tracked_seg")
            self.tracked_seg = tuple(ts) if ts else None
        self._tracking_changed()
        self._update_labels()
        self._sel_changed()
        QTimer.singleShot(250, lambda: self.set_frame(int(meta.get("cur", self.seg_in)), force=True))
        self.statusBar().showMessage(f"Projet ouvert : {os.path.basename(p)}", 8000)

    # ------------------------------------------------------------ export
    def start_export(self):
        if self.res is None or self.export_thread is not None:
            return
        out_dir = self.out_edit.text().strip()
        name = self.name_edit.text().strip() or "tapnext"
        if not out_dir:
            QMessageBox.warning(self, APP_NAME, "Choisissez un dossier de sortie.")
            return
        info = self.store.info
        if len(self.res.tracked) != self.store.total:
            self._pad_result(self.store.total)
        vals, counts = np.unique(self.res.query_frames, return_counts=True)
        ref = int(vals[np.argmax(counts)])
        groups = [g for g in self.groups]
        layers = [(se.ShapeStyle.from_dict(g.style.to_dict()), self._group_data(g), g.enabled)
                  for g in groups]
        keep = [i for i, L in enumerate(layers) if L[1] is not None]
        job = dict(
            info=info, res=self.res, rd=self.rd, pts=self.pts[:self.n_tracked],
            out_dir=out_dir, name=name,
            matte=self.cb_matte.isChecked() or bool(self.job),
            codec=["prores", "dnxhr", "h264", "png"][self.codec.currentIndex()],
            data=self.cb_data.isChecked(),
            fusion="none" if self.job else
            ["none", "stabilize", "matchmove", "both"][self.fusion_mode.currentIndex()],
            fusion_to_resolve=["none", "stabilize", "matchmove", "both"][self.fusion_mode.currentIndex()],
            fusion_smooth=int(self.fusion_smooth.value()),
            preview=self.cb_preview.isChecked(),
            layers=[layers[i] for i in keep], group_names=[groups[i].name for i in keep],
            per_group=self.cb_per_group.isChecked(), invert=self.cb_invert.isChecked(),
            ref_frame=ref, job_mode=bool(self.job),
            params=dict(start_frame=ref, segment=[self.seg_in, self.seg_out],
                        groups=[dict(name=g.name, enabled=g.enabled, style=g.style.to_dict(),
                                     points=[int(i) for i in np.nonzero(self.pgroup[:self.n_tracked] == g.id)[0]])
                                for g in groups]),
        )
        if not job["layers"]:
            QMessageBox.warning(self, APP_NAME, "Aucun groupe de points suivis à exporter.")
            return
        self.export_thread = ExportThread(job)
        self.export_thread.progress.connect(self._on_progress)
        self.export_thread.done.connect(lambda outs: self._on_exported(outs, job))
        self.export_thread.failed.connect(self._on_failed)
        self.export_thread.finished.connect(self._thread_finished)
        self.b_export.setEnabled(False)
        self.b_cancel.setEnabled(True)
        self.prog_exp.setValue(0)
        self.export_thread.start()

    def _pad_result(self, n: int):
        r = self.res
        T, Q = r.visibility.shape
        if n > T:
            r.positions = np.concatenate([r.positions, np.full((n - T, Q, 2), np.nan, np.float32)])
            r.visibility = np.concatenate([r.visibility, np.zeros((n - T, Q), np.float32)])
            r.tracked = np.concatenate([r.tracked, np.zeros(n - T, bool)])
        else:
            r.positions, r.visibility, r.tracked = r.positions[:n], r.visibility[:n], r.tracked[:n]
        self._tracking_changed()

    def _on_exported(self, outputs: list, job: dict):
        self.prog_exp.setValue(1000)
        self.lbl_exp.setText("Export terminé.")
        if self.job:
            base = os.path.join(job["out_dir"], job["name"])
            mode = job["fusion_to_resolve"]
            done = dict(status="ok", matte=next((o for o in outputs if o.endswith(("_matte.mov", "_matte.mp4"))
                                                 or "_matte_png" in o), ""),
                        csv=base + "_tracks.csv", outputs=outputs,
                        query_frame=job["ref_frame"], job_id=str(self.job.get("job_id", "")),
                        fusion_mode="stabilize" if mode == "both" else mode,
                        fusion_smooth=job["fusion_smooth"], attach=self.cb_attach.isChecked())
            with open(self.job["done"], "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=1)
            self.job_sent = True
            self.lbl_exp.setText("Export terminé — en attente de DaVinci Resolve…")
            self._ack_deadline = time.time() + 20
            self._ack_timer = QTimer(self)
            self._ack_timer.timeout.connect(self._poll_ack)
            self._ack_timer.start(500)
        else:
            QMessageBox.information(self, APP_NAME, "Export terminé :\n\n" + "\n".join(
                os.path.basename(o) for o in outputs) + f"\n\nDossier : {job['out_dir']}")

    def _poll_ack(self):
        """Attend la confirmation d'import écrite par le script Resolve."""
        path = self.job.get("ack", "")
        ack = None
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    ack = json.load(f)
            except (OSError, ValueError):
                ack = None
        if ack is not None and str(ack.get("job_id", "")) in ("", str(self.job.get("job_id", ""))):
            self._ack_timer.stop()
            ok = ack.get("status") == "ok"
            self.lbl_exp.setText("Importé dans DaVinci Resolve." if ok else "Import incomplet.")
            (QMessageBox.information if ok else QMessageBox.warning)(
                self, APP_NAME, "DaVinci Resolve :\n\n" + ack.get("report", "")
                + ("\n\nVous pouvez fermer TAPNext Studio." if ok else ""))
            return
        if time.time() > self._ack_deadline:
            self._ack_timer.stop()
            self.lbl_exp.setText("Export terminé. Import : relancez le script dans Resolve.")
            QMessageBox.information(
                self, APP_NAME,
                "Export terminé.\n\nPour importer dans DaVinci Resolve :\n"
                "retournez dans Resolve et relancez\n"
                "Workspace → Scripts → TAPNext_Tracker.\n\n"
                "La matte sera attachée au clip et le nœud Fusion ajouté.")

    def closeEvent(self, e):
        for t in (self.tracker_thread, self.export_thread):
            if t is not None:
                t.cancel.set()
                t.wait(3000)
        if self.loader is not None:
            self.loader.stop_flag.set()
            self.loader.wait(2000)
        if self.job and not self.job_sent:
            try:
                with open(self.job["done"], "w", encoding="utf-8") as f:
                    json.dump({"status": "cancelled"}, f)
            except OSError:
                pass
        super().closeEvent(e)


def dark_palette(app: QApplication):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, "#1c1d21"), (QPalette.WindowText, "#d8d8dc"),
                      (QPalette.Base, "#15161a"), (QPalette.AlternateBase, "#212227"),
                      (QPalette.Text, "#d8d8dc"), (QPalette.Button, "#2d2e35"),
                      (QPalette.ButtonText, "#d8d8dc"), (QPalette.Highlight, ACCENT),
                      (QPalette.HighlightedText, "#111111"), (QPalette.ToolTipBase, "#2d2e35"),
                      (QPalette.ToolTipText, "#e8e8ec")]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def _setup_logging() -> None:
    """Journal dans logs/studio.log. Sous Windows (pythonw), il n'y a pas de
    console : stdout/stderr sont redirigés vers ce fichier."""
    import logging

    log_dir = os.path.join(HERE, "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        f = open(os.path.join(log_dir, "studio.log"), "w", encoding="utf-8", buffering=1)
    except OSError:
        return
    if sys.stdout is None or sys.stderr is None or IS_PYTHONW:
        sys.stdout = sys.stderr = f
    logging.basicConfig(level=logging.INFO, stream=f,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


IS_PYTHONW = os.path.basename(sys.executable).lower().startswith("pythonw")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("video", nargs="?", default="", help="Vidéo ou projet .tapnext")
    ap.add_argument("--job", help="Fichier de tâche envoyé par DaVinci Resolve (JSON)")
    a = ap.parse_args(argv)
    _setup_logging()
    job = None
    if a.job:
        with open(a.job, encoding="utf-8") as f:
            job = json.load(f)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    dark_palette(app)
    target = (job or {}).get("video") or a.video
    is_project = target.lower().endswith(".tapnext")
    win = StudioWindow("" if is_project else target, job)
    if is_project:
        QTimer.singleShot(50, lambda: win.load_project(target))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
