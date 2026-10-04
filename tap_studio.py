#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAPNext Studio — interface visuelle de tracking TAPNext++ pour DaVinci Resolve.

  • Visionneuse vidéo (zoom molette, déplacement clic-milieu, timeline, lecture).
  • Placement des points : dessinez une zone autour du sujet (elle se remplit de
    points sur les détails texturés) ou cliquez des points précis, sur
    n'importe quelle image du plan.
  • Suivi TAPNext++ bidirectionnel avec contrôle aller-retour, en arrière-plan.
  • Trajectoires et matte affichées en direct ; les réglages de matte se
    voient immédiatement, sans relancer le suivi.
  • Export : matte N&B (ProRes/DNxHR/H.264/PNG), CSV/JSON, nœuds Fusion.
  • Lancé depuis Resolve (Workspace > Scripts > TAPNext_Tracker), les
    résultats repartent automatiquement dans Resolve.

Raccourcis : Espace lecture · ←/→ image · I/O début/fin · F cadrer ·
A outil points · Z outil zone · R outil rectangle · Ctrl+Z annuler · Suppr effacer.
"""

from __future__ import annotations

import argparse
import json
import math
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
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QImage, QKeySequence, QPainter,
                           QPainterPath, QPalette, QPen, QPolygonF)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fusion_export as fx  # noqa: E402
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
"""


def point_color(i: int, n: int) -> QColor:
    return QColor.fromHsvF(((i * 0.61803398875) % 1.0), 0.75, 1.0)


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
                                      outlier_px=max(2.0, 0.002 * info.width))
                    outputs.append(p)
            if j["matte"]:
                outputs += eng.write_matte_video(base, info, j["codec"], j["mparams"], rd,
                                                 j["preview"], progress=self.progress.emit,
                                                 cancel=self.cancel)
            self.done.emit(outputs)
        except eng.Cancelled:
            self.failed.emit("Export annulé.")
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__} : {e}")


# =============================================================================
# Widgets
# =============================================================================

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
        if self.qimg is not None and mode != "matte":
            p.drawImage(target, self.qimg)
        elif mode != "matte":
            p.setPen(QColor("#77787f"))
            p.drawText(target, Qt.AlignCenter, "Chargement…")
        if self.overlay is not None and mode in ("matte", "overlay"):
            p.drawImage(target, self.overlay)
        p.setRenderHint(QPainter.Antialiasing)
        if mode != "matte":
            self._draw_tracks(p)
        self._draw_zones(p)
        p.setPen(QColor(220, 220, 225, 200))
        p.drawText(QRectF(10, 8, 400, 18), Qt.AlignLeft,
                   f"Image {w.cur} / {w.n_frames() - 1}    {info.width}×{info.height}")

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
        if rd is not None and nt and t < len(rd.smoothed):
            pos, alpha, vis = rd.smoothed, rd.alpha, rd.visible
            if trail > 0:
                t0 = max(0, t - trail)
                for i in range(nt):
                    if alpha[t, i] < 0.05:
                        continue
                    seg = pos[t0:t + 1, i]
                    ok = vis[t0:t + 1, i] & np.isfinite(seg[:, 0])
                    col = point_color(i, n)
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
                    col.setAlphaF(0.55)
                    p.setPen(QPen(col, max(1.0, size * 0.35)))
                    p.drawPath(path)
            for i in range(nt):
                x, y = pos[t, i]
                if not np.isfinite(x) or not res.tracked[t]:
                    continue
                sp = self.to_screen(x, y)
                col = point_color(i, n)
                r = size * (1.5 if i == self.hover else 1.0)
                if vis[t, i]:
                    p.setPen(QPen(QColor(0, 0, 0, 200), 1.2))
                    p.setBrush(QBrush(col))
                    p.drawEllipse(sp, r, r)
                elif alpha[t, i] > 0.02:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(170, 170, 175, int(220 * alpha[t, i])), 1.2))
                    p.drawEllipse(sp, r * 0.8, r * 0.8)
        # points pas encore suivis : visibles sur leur image de pose
        for i in range(nt, n):
            sp = self.to_screen(*w.pts[i])
            on_frame = w.qf[i] == t
            col = point_color(i, n) if on_frame else QColor(200, 200, 200, 90)
            r = size + 2
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 180), 3))
            p.drawLine(QPointF(sp.x() - r, sp.y()), QPointF(sp.x() + r, sp.y()))
            p.drawLine(QPointF(sp.x(), sp.y() - r), QPointF(sp.x(), sp.y() + r))
            p.setPen(QPen(col, 1.5))
            p.drawLine(QPointF(sp.x() - r, sp.y()), QPointF(sp.x() + r, sp.y()))
            p.drawLine(QPointF(sp.x(), sp.y() - r), QPointF(sp.x(), sp.y() + r))

    def _draw_zones(self, p: QPainter):
        pen = QPen(QColor(ACCENT), 1.6, Qt.DashLine)
        p.setBrush(QColor(232, 131, 58, 30))
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
                w.delete_point(i)
            return
        if e.button() != Qt.LeftButton:
            return
        x, y = self.to_src(pos)
        if tool == "point":
            w.add_points(np.array([[x, y]], np.float32))
        elif tool == "lasso":
            self.drag_poly = [QPointF(x, y)]
        elif tool == "rect":
            self.drag_rect_start = QPointF(x, y)
            self._rect_end = QPointF(x, y)

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
            if abs(a.x() - b.x()) > 4 and abs(a.y() - b.y()) > 4:
                poly = np.array([[a.x(), a.y()], [b.x(), a.y()], [b.x(), b.y()], [a.x(), b.y()]],
                                np.float32)
                self.win.fill_zone(poly)
            self.update()

    def _nearest(self, pos: QPointF) -> int:
        w = self.win
        best, bd = -1, 14.0
        for i in range(len(w.pts)):
            xy = w.display_pos(i)
            if xy is None:
                continue
            sp = self.to_screen(*xy)
            d = math.hypot(sp.x() - pos.x(), sp.y() - pos.y())
            if d < bd:
                best, bd = i, d
        return best

    # ------------------------------------------------------- glisser-déposer
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            self.win.open_video(urls[0].toLocalFile())


# =============================================================================
# Fenêtre principale
# =============================================================================

class StudioWindow(QMainWindow):
    def __init__(self, video: str = "", job: Optional[dict] = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self.job = job
        self.job_sent = False
        self.store: Optional[VideoStore] = None
        self.loader: Optional[LoadThread] = None
        self.tracker_thread: Optional[TrackThread] = None
        self.export_thread: Optional[ExportThread] = None
        self.cur = 0
        self.seg_in, self.seg_out = 0, 0
        self.pts = np.zeros((0, 2), np.float32)
        self.qf = np.zeros(0, int)
        self.res = None
        self.rd = None
        self.n_tracked = 0
        self.tracked_seg = None
        self.undo: List[int] = []
        self.out_dir = ""
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._refresh_overlay)
        self.rd_timer = QTimer(self)
        self.rd_timer.setSingleShot(True)
        self.rd_timer.timeout.connect(self._recompute_rd)
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
        b_open = QPushButton("Ouvrir une vidéo…")
        b_open.clicked.connect(self._ask_open)
        top.addWidget(b_open)
        top.addSpacing(12)
        self.tool_group = QButtonGroup(self)
        self.tool_btns = {}
        for key, text, tip in [
            ("lasso", "◯  Zone (lasso)", "Entourez le sujet : la zone se remplit de points (Z)"),
            ("rect", "▭  Zone (rectangle)", "Rectangle rempli de points (R)"),
            ("point", "✚  Point", "Cliquez pour poser un point précis (A)"),
            ("pan", "✋  Déplacer", "Déplacer la vue (ou clic milieu)"),
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

        # ---- panneau de droite
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(10, 4, 12, 10)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        # ① Points
        g = QGroupBox("① Placer les points")
        gl = QVBoxLayout(g)
        gl.addWidget(help_label(
            "Allez sur une image où le sujet est bien visible, puis <b>entourez-le</b> avec "
            "l'outil Zone : il se remplit de points sur les détails texturés (ce que TAPNext++ "
            "suit le mieux). Outil Point pour un point précis. <b>Clic droit</b> = supprimer."))
        row = QHBoxLayout()
        row.addWidget(QLabel("Points par zone"))
        self.n_per_zone = QSpinBox()
        self.n_per_zone.setRange(1, 3000)
        self.n_per_zone.setValue(80)
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

        # ② Suivi
        g = QGroupBox("② Suivre")
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
        self.cb_verify.setToolTip("Chaque piste est re-suivie en sens inverse ; si elle ne revient "
                                  "pas à son point de départ, elle est coupée là où elle a décroché.")
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

        # ③ Affichage
        g = QGroupBox("③ Affichage")
        gl = QVBoxLayout(g)
        self.disp_size = ValueSlider("Taille des points", 2, 14, 5, 0.5, "{:.1f}")
        self.disp_size.changed.connect(lambda _: self.viewer.update())
        self.disp_trail = ValueSlider("Traînées (images)", 0, 60, 12, 1, "{:.0f}")
        self.disp_trail.changed.connect(lambda _: self.viewer.update())
        gl.addWidget(self.disp_size)
        gl.addWidget(self.disp_trail)
        pv.addWidget(g)

        # ④ Matte
        g = QGroupBox("④ Matte (page Color)")
        gl = QVBoxLayout(g)
        gl.addWidget(help_label("Réglages visibles en direct (vue « Image + matte »), "
                                "sans relancer le suivi."))
        self.m_radius = ValueSlider("Rayon des blobs (px)", 4, 400, 40, 1, "{:.0f}",
                                    "Rayon de chaque point, en pixels de la vidéo source")
        self.m_merge = ValueSlider("Fusion des blobs", 0.1, 2.0, 0.6, 0.05, "{:.2f}",
                                   "Flou de fusion (× rayon) : plus haut = masque plus continu")
        self.m_thr = ValueSlider("Seuil", 0.1, 0.9, 0.5, 0.01, "{:.2f}",
                                 "Plus bas = masque plus gros et plus lié")
        self.m_soft = ValueSlider("Douceur du bord", 0.0, 0.3, 0.08, 0.01, "{:.2f}")
        self.m_motion = QComboBox()
        self.m_motion.addItems(["Mouvement : aucun effet", "Mouvement : agrandir",
                                "Mouvement : étirer", "Mouvement : agrandir + étirer"])
        self.m_sens = ValueSlider("Sensibilité au mouvement", 0.0, 0.5, 0.05, 0.005, "{:.3f}")
        self.m_fin = ValueSlider("Fondu d'apparition (images)", 0, 60, 6, 1, "{:.0f}")
        self.m_fout = ValueSlider("Fondu de disparition (images)", 0, 60, 8, 1, "{:.0f}")
        self.m_smooth = ValueSlider("Lissage des trajectoires", 0.0, 5.0, 1.0, 0.1, "{:.1f}")
        self.m_invert = QCheckBox("Inverser la matte")
        for wdg in (self.m_radius, self.m_merge, self.m_thr, self.m_soft, self.m_motion,
                    self.m_sens, self.m_fin, self.m_fout, self.m_smooth, self.m_invert):
            gl.addWidget(wdg)
        for wdg in (self.m_radius, self.m_merge, self.m_thr, self.m_soft, self.m_sens):
            wdg.changed.connect(self._schedule_overlay)
        self.m_motion.currentIndexChanged.connect(self._schedule_overlay)
        self.m_invert.toggled.connect(self._schedule_overlay)
        for wdg in (self.m_fin, self.m_fout, self.m_smooth):
            wdg.changed.connect(lambda _: self.rd_timer.start(150))
        pv.addWidget(g)

        # ⑤ Export
        g = QGroupBox("⑤ Exporter" + (" vers DaVinci Resolve" if self.job else ""))
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
        self.cb_matte = QCheckBox("Matte N&B")
        self.cb_matte.setChecked(True)
        self.codec = QComboBox()
        self.codec.addItems(["ProRes 422 HQ (.mov)", "DNxHR HQ (.mov)", "H.264 (.mp4)", "PNG (séquence)"])
        row.addWidget(self.cb_matte)
        row.addWidget(self.codec, 1)
        gl.addLayout(row)
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
        if self.job:
            gl.addWidget(help_label("Le nœud Fusion se règle dans la fenêtre de Resolve "
                                    "(il y est inséré automatiquement)."))
        self.cb_preview = QCheckBox("Vidéo de contrôle (preview)")
        gl.addWidget(self.cb_preview)
        self.b_export = QPushButton("⇪  Exporter et envoyer à Resolve" if self.job else "⇪  Exporter")
        self.b_export.setObjectName("primary")
        self.b_export.clicked.connect(self.start_export)
        self.b_export.setEnabled(False)
        gl.addWidget(self.b_export)
        pv.addWidget(g)
        pv.addStretch(1)

        self.statusBar().showMessage("Ouvrez une vidéo pour commencer.")

        # raccourcis
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
        sc("Ctrl+Z", self.undo_last)
        sc("Ctrl+O", self._ask_open)

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

    def display_pos(self, i: int):
        if i < self.n_tracked and self.rd is not None:
            x, y = self.rd.smoothed[self.cur, i]
            if np.isfinite(x) and self.res.tracked[self.cur]:
                return float(x), float(y)
            return None
        if i < len(self.pts):
            return float(self.pts[i, 0]), float(self.pts[i, 1])
        return None

    def matte_params(self, scale: float = 1.0, max_side: int = 1920) -> eng.MatteParams:
        return eng.MatteParams(
            radius=self.m_radius.value() * scale, merge=self.m_merge.value(),
            threshold=self.m_thr.value(), edge_softness=self.m_soft.value(),
            motion_mode=["none", "scale", "stretch", "both"][self.m_motion.currentIndex()],
            motion_sensitivity=self.m_sens.value(), render_max_side=max_side,
            invert=self.m_invert.isChecked())

    # --------------------------------------------------------- ouverture
    def _ask_open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Ouvrir une vidéo", "",
                                           "Vidéos (*.mp4 *.mov *.mxf *.mkv *.avi);;Tous (*.*)")
        if p:
            self.open_video(p)

    def open_video(self, path: str):
        if self.loader is not None:
            self.loader.stop_flag.set()
            self.loader.wait()
        try:
            store = VideoStore(path)
        except Exception as e:
            QMessageBox.critical(self, APP_NAME, f"Impossible d'ouvrir la vidéo :\n{e}")
            return
        self.store = store
        self.pts = np.zeros((0, 2), np.float32)
        self.qf = np.zeros(0, int)
        self.res = self.rd = None
        self.n_tracked = 0
        self.undo.clear()
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
        self.viewer.fit_pending = True
        self.viewer.fit()
        QTimer.singleShot(200, lambda: self.set_frame(self.seg_in))
        self._update_labels()

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

    # ------------------------------------------------------------- points
    def _busy(self) -> bool:
        if self.tracker_thread is not None:
            self.statusBar().showMessage("Suivi en cours : attendez la fin pour modifier les points.", 4000)
            return True
        return False

    def add_points(self, pts: np.ndarray, frame: Optional[int] = None):
        if not len(pts) or not self.store or self._busy():
            return
        W, H = self.store.info.width, self.store.info.height
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        f = self.cur if frame is None else frame
        if not (self.seg_in <= f <= self.seg_out):
            # Poser un point hors plage élargit la plage de suivi.
            self.in_spin.setValue(min(self.seg_in, f))
            self.out_spin.setValue(max(self.seg_out, f))
        self.pts = np.concatenate([self.pts, pts.astype(np.float32)])
        self.qf = np.concatenate([self.qf, np.full(len(pts), f, int)])
        self.undo.append(len(pts))
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
            f"{len(pts)} points placés sur l'image {self.cur}. Lancez le suivi (②).", 8000)

    def fill_full(self):
        st = self.store
        if not st:
            return
        W, H = st.info.width, st.info.height
        m = 0.02
        self.fill_zone(np.array([[W * m, H * m], [W * (1 - m), H * m],
                                 [W * (1 - m), H * (1 - m)], [W * m, H * (1 - m)]], np.float32))

    def delete_point(self, i: int):
        if self._busy():
            return
        keep = np.ones(len(self.pts), bool)
        keep[i] = False
        self._keep_points(keep)

    def _keep_points(self, keep: np.ndarray):
        nt = self.n_tracked
        self.pts, self.qf = self.pts[keep], self.qf[keep]
        if self.res is not None and nt:
            kt = keep[:nt]
            r = self.res
            r.positions, r.visibility = r.positions[:, kt], r.visibility[:, kt]
            r.query_frames = r.query_frames[kt]
            r.cut_frames = r.cut_frames[kt]
            self.n_tracked = int(kt.sum())
            if self.n_tracked == 0:
                self.res = self.rd = None
            else:
                self._recompute_rd()
        self.undo.clear()
        self._update_labels()
        self.viewer.hover = -1
        self.viewer.update()
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
        self.pts = np.zeros((0, 2), np.float32)
        self.qf = np.zeros(0, int)
        self.res = self.rd = None
        self.n_tracked = 0
        self.undo.clear()
        self._update_labels()
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
        # Les nouveaux points seuls si la plage n'a pas changé, sinon tout.
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
        self.prog.setValue(int(frac * 1000))
        self.lbl_prog.setText(desc)
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
        self._recompute_rd()
        self.prog.setValue(1000)
        n_cut = int((r.cut_frames >= 0).sum())
        msg = (f"Suivi terminé en {getattr(r, 'elapsed', 0):.0f} s sur {getattr(r, 'device', '?')}"
               + (f" · {n_cut} point(s) coupé(s) au décrochage" if n_cut else ""))
        self.lbl_prog.setText(msg)
        self.statusBar().showMessage(msg + ". Lecture : Espace.", 15000)
        self._update_labels()

    def _on_failed(self, msg: str):
        self.lbl_prog.setText(msg)
        self.statusBar().showMessage(msg, 15000)
        if not msg.endswith("annulé.") and not msg.endswith("annulé"):
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
    def _recompute_rd(self):
        if self.res is None:
            self.rd = None
        else:
            self.rd = eng.prepare_render_data(self.res, self.m_smooth.value(), 0.5,
                                              int(self.m_fin.value()), int(self.m_fout.value()))
        self._refresh_overlay()

    def _schedule_overlay(self, *_):
        self.render_timer.start(30)

    def _refresh_overlay(self):
        v = self.viewer
        if self.rd is None or self.view_mode() == "points" or not self.store:
            v.overlay = None
            if self.store and self.view_mode() == "matte":
                black = QImage(self.store.pw, self.store.ph, QImage.Format_RGB888)
                black.fill(QColor("black"))
                v.overlay = black
            v.update()
            return
        st, t = self.store, self.cur
        rd = self.rd
        key = (st.pw, st.ph, self.m_radius.value(), self.m_merge.value(), self.m_thr.value(),
               self.m_soft.value(), self.m_motion.currentIndex(), self.m_sens.value(),
               self.m_invert.isChecked())
        if getattr(self, "_rkey", None) != key:
            self._renderer = eng.MatteRenderer(st.pw, st.ph, self.matte_params(st.scale, 960))
            self._rkey = key
        s = st.scale
        m = self._renderer.render(rd.draw_pos[t] * s, rd.alpha[t], rd.speed[t] * s, rd.direction[t])
        if self.view_mode() == "matte":
            g = np.ascontiguousarray(m)
            self._ov_ref = g
            v.overlay = QImage(g.data, g.shape[1], g.shape[0], g.strides[0], QImage.Format_Grayscale8)
        else:
            rgba = np.zeros((m.shape[0], m.shape[1], 4), np.uint8)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 70, 60
            rgba[..., 3] = (m.astype(np.uint16) * 140 // 255).astype(np.uint8)
            self._ov_ref = rgba
            v.overlay = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0],
                               QImage.Format_RGBA8888)
        v.update()

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
        n_full = len(self.res.tracked)
        if n_full != self.store.total:   # aligne sur le nombre réel d'images
            self._pad_result(self.store.total)
        vals, counts = np.unique(self.res.query_frames, return_counts=True)
        ref = int(vals[np.argmax(counts)])
        job = dict(
            info=info, res=self.res, rd=self.rd, pts=self.pts[:self.n_tracked],
            out_dir=out_dir, name=name,
            matte=self.cb_matte.isChecked() or bool(self.job),
            codec=["prores", "dnxhr", "h264", "png"][self.codec.currentIndex()],
            data=self.cb_data.isChecked(),
            fusion=["none", "stabilize", "matchmove", "both"][self.fusion_mode.currentIndex()],
            preview=self.cb_preview.isChecked(), mparams=self.matte_params(),
            ref_frame=ref, job_mode=bool(self.job),
            params=dict(start_frame=ref, segment=[self.seg_in, self.seg_out],
                        radius=self.m_radius.value(), merge=self.m_merge.value(),
                        threshold=self.m_thr.value()),
        )
        self.export_thread = ExportThread(job)
        self.export_thread.progress.connect(self._on_progress)
        self.export_thread.done.connect(lambda outs: self._on_exported(outs, job))
        self.export_thread.failed.connect(self._on_failed)
        self.export_thread.finished.connect(self._thread_finished)
        self.b_export.setEnabled(False)
        self.b_cancel.setEnabled(True)
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
        self._recompute_rd()

    def _on_exported(self, outputs: list, job: dict):
        self.prog.setValue(1000)
        self.lbl_prog.setText("Export terminé.")
        if self.job:
            base = os.path.join(job["out_dir"], job["name"])
            done = dict(status="ok", matte=next((o for o in outputs if "_matte" in o), ""),
                        csv=base + "_tracks.csv", outputs=outputs,
                        query_frame=job["ref_frame"])
            with open(self.job["done"], "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=1)
            self.job_sent = True
            QMessageBox.information(
                self, APP_NAME, "Résultats envoyés à DaVinci Resolve.\n\n"
                "La matte est attachée au clip (page Color → clic droit → Add Matte).\n"
                "Vous pouvez fermer TAPNext Studio.")
        else:
            QMessageBox.information(self, APP_NAME, "Export terminé :\n\n" + "\n".join(
                os.path.basename(o) for o in outputs) + f"\n\nDossier : {job['out_dir']}")

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
    console : stdout/stderr sont redirigés vers ce fichier (barres de
    progression du téléchargement, erreurs…)."""
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
    ap.add_argument("video", nargs="?", default="")
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
    win = StudioWindow((job or {}).get("video") or a.video, job)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
