# -*- coding: utf-8 -*-
"""太湖蓝藻监测桌面应用 (PySide6)

功能:
- 打开影像 (卫星 TIFF / 任意 JPG/PNG), 或直接把文件拖进窗口
- 左侧原图, 右侧 ExG 热力图 (ExG = 2G - R - B, p10 基线校正)
- 在任意一侧按住左键拖出矩形, 实时统计框内 ExG 平均值/最大值/标准差/蓝藻占比
- 两侧框选同步显示

用法: python app.py
"""
import os
import sys

import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QSplitter,
                               QFileDialog, QMessageBox, QToolBar, QLabel, QStatusBar)

from src import analysis as A
from src.loader import load_full

EXG_MIN, EXG_MAX = -500, 3500
BG = QColor(13, 13, 22)
NONWATER = (18, 18, 26)
CLOUD_C = (222, 222, 230)   # 云: 浅灰白
SHADOW_C = (110, 110, 122)  # 云影: 中灰

_STOPS = [(-500, (20, 25, 45)), (300, (45, 90, 180)), (500, (80, 205, 170)),
          (900, (255, 230, 40)), (1400, (245, 95, 20)), (3500, (205, 15, 15))]

QSS = """
QMainWindow, QWidget { background: #0d0d16; color: #e8e8ee; font-family: 'Microsoft YaHei'; }
QToolBar { background: #15151f; border-bottom: 1px solid #26263a; padding: 4px; }
QToolButton { background: #2a3a55; color: #fff; border: 0; border-radius: 5px; padding: 7px 14px; }
QToolButton:hover { background: #354a6e; }
QStatusBar { background: #15151f; border-top: 1px solid #26263a; color: #9aa; }
QStatusBar QLabel { padding: 0 10px; }
QSplitter::handle { background: #26263a; width: 3px; }
"""


def colormap_lut():
    xs = np.arange(EXG_MIN, EXG_MAX + 1)
    lut = np.zeros((len(xs), 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, [s[0] for s in _STOPS], [s[1][ch] for s in _STOPS])
    return lut


LUT = colormap_lut()


def exg_to_rgb(exg, water):
    """ExG 数组 -> 热力图 RGB (非水体为深色)"""
    idx = np.clip(np.round(exg).astype(np.int32) - EXG_MIN, 0, len(LUT) - 1)
    heat = LUT[idx]
    heat[~water] = NONWATER
    return heat


def to_qimage(rgb):
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return img.copy()


class ImagePanel(QWidget):
    """影像显示面板: 适应窗口缩放显示, 支持鼠标拖拽画矩形"""
    rectChanged = Signal(float, float, float, float)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.base = None          # QImage
        self.sel = None           # (x0,y0,x1,y1) 图像坐标 (避免与 QWidget.rect 方法重名)
        self._t = None            # 显示变换 (target QRectF, scale)
        self._dragging = False
        self._anchor = None
        self.setMinimumSize(320, 320)

    def setBase(self, qimg):
        self.base = qimg
        self.update()

    def setRect(self, r):
        self.sel = r
        self.update()

    def _fit(self):
        if self.base is None:
            return None
        iw, ih = self.base.width(), self.base.height()
        w, h = self.width(), self.height()
        s = min(w / iw, h / ih)
        return QRectF((w - iw * s) / 2, (h - ih * s) / 2, iw * s, ih * s), s

    def _toImage(self, pos):
        t, s = self._t
        x = (pos.x() - t.x()) / s
        y = (pos.y() - t.y()) / s
        iw, ih = self.base.width(), self.base.height()
        return max(0.0, min(iw, x)), max(0.0, min(ih, y))

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        if self.base is None:
            p.setPen(QColor(120, 120, 140))
            p.drawText(self.rect(), Qt.AlignCenter,
                       f"{self.title}\n(拖入影像文件或点「打开影像」)")
            return
        self._t = self._fit()
        t, s = self._t
        p.drawImage(t, self.base)
        p.setPen(QColor(110, 110, 130))
        p.drawText(QRectF(6, 6, 300, 24), Qt.AlignLeft, self.title)
        if self.sel is not None:
            x0, y0, x1, y1 = self.sel
            r = QRectF(t.x() + x0 * s, t.y() + y0 * s, (x1 - x0) * s, (y1 - y0) * s)
            p.fillRect(r, QColor(255, 230, 0, 40))
            p.setPen(QPen(QColor(255, 230, 0), 2))
            p.drawRect(r)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.base is not None and self._t:
            self._dragging = True
            x, y = self._toImage(e.position())
            self._anchor = (x, y)
            self.sel = (x, y, x, y)
            self.rectChanged.emit(x, y, x, y)

    def mouseMoveEvent(self, e):
        if self._dragging:
            x, y = self._toImage(e.position())
            x0, y0 = self._anchor
            self.sel = (x0, y0, x, y)
            self.rectChanged.emit(x0, y0, x, y)
            self.update()

    def mouseReleaseEvent(self, e):
        self._dragging = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("太湖蓝藻监测 - ExG 热力图分析")
        self.resize(1500, 850)
        self.setAcceptDrops(True)
        self.setStyleSheet(QSS)

        self.exg_masked = None   # NaN=非水体
        self.rect = None
        mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "taihu_mask.npy")
        self.lake_mask = np.load(mask_path) if os.path.exists(mask_path) else None

        self.left = ImagePanel("原图")
        self.right = ImagePanel("ExG 热力图 (浅灰=云/影, 深色=非水体)")
        self.left.rectChanged.connect(self.on_rect)
        self.right.rectChanged.connect(self.on_rect)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.left)
        split.addWidget(self.right)
        split.setSizes([750, 750])
        self.setCentralWidget(split)

        tb = QToolBar("工具栏")
        tb.setMovable(False)
        tb.addAction("打开影像", self.open_dialog)
        tb.addAction("清除框选", self.clear_rect)
        tb.addAction("退出", self.close)
        self.addToolBar(tb)

        sb = QStatusBar()
        self.lbl_badge = QLabel("就绪")
        self.lbl_file = QLabel("拖入影像文件或点「打开影像」")
        self.lbl_reason = QLabel("")
        self.lbl_valid = QLabel("有效占比: -")
        self.lbl_n = QLabel("像素: -")
        self.lbl_mean = QLabel("均值: -")
        self.lbl_max = QLabel("最大: -")
        self.lbl_std = QLabel("标准差: -")
        self.lbl_pct = QLabel("蓝藻占比: -")
        for w in (self.lbl_n, self.lbl_mean, self.lbl_max, self.lbl_std, self.lbl_pct, self.lbl_valid):
            w.setStyleSheet("color:#8df0a0; font-weight:bold;")
        self.lbl_badge.setStyleSheet("font-size:15px; font-weight:bold; padding:0 8px;")
        self.lbl_file.setStyleSheet("color:#ffd76e;")
        self.lbl_reason.setStyleSheet("color:#c8a0f0;")
        sb.addWidget(self.lbl_badge)
        sb.addWidget(self.lbl_file)
        sb.addWidget(self.lbl_reason)
        sb.addPermanentWidget(self.lbl_pct)
        sb.addPermanentWidget(self.lbl_std)
        sb.addPermanentWidget(self.lbl_max)
        sb.addPermanentWidget(self.lbl_mean)
        sb.addPermanentWidget(self.lbl_valid)
        sb.addPermanentWidget(self.lbl_n)
        self.setStatusBar(sb)

    # ---------- 加载 ----------
    def load_image(self, path):
        try:
            r, g, b, nir, b5, b11, qa, note, gray = load_full(path)
        except Exception as e:
            QMessageBox.warning(self, "无法打开", f"{path}\n{e}")
            return

        lake = None
        if self.lake_mask is not None and self.lake_mask.shape == r.shape:
            lake = self.lake_mask > 0

        is_l1c = "_l1c" in os.path.basename(path)
        water, valid, bad = A.build_water(r, g, b, nir, qa, lake, is_l1c)
        seam = A.detect_seam(r, g, b, lake) if lake is not None else None
        blocks = A.split_blocks(water, seam, r.shape[1])
        exg = A.anchored_exg(r, g, b, water, blocks)
        p10_off = None
        if len(blocks) > 1:
            p10_off = f"分块锚定(拼缝列{seam})"
        elif water.sum() >= 1000:
            p10 = float(np.percentile(exg[water], 10))
            p10_off = round(A.EXG_P10_ANCHOR - p10, 1)

        self.exg_masked = np.where(water, exg, np.nan).astype(np.float32)
        self.water = water
        self.bad = bad

        orig = np.clip(np.stack([r, g, b], axis=2) / 3000.0 * 255, 0, 255).astype(np.uint8)
        self.left.setBase(to_qimage(orig))
        heat = exg_to_rgb(exg, water)
        if lake is not None:
            heat[bad & lake] = CLOUD_C             # 所有被排除的(云/雾/亮斑)统一白色
            if qa is not None:
                heat[(qa == 2) & lake] = SHADOW_C  # 云影叠灰
        else:
            if qa is not None:
                heat[qa == 1] = CLOUD_C
                heat[qa == 2] = SHADOW_C
            else:
                heat[bad & water] = CLOUD_C
        self.right.setBase(to_qimage(heat))

        badge, reason = A.rate_quality(water, valid, bad, lake, nir, exg)
        name = os.path.basename(path)
        p10_txt = f" (基线校正 {p10_off:+})" if p10_off is not None else ""
        gray_txt = " [灰度图, 颜色法不适用]" if gray else ""
        self.lbl_badge.setText(badge)
        self.lbl_file.setText(f"{name}  |  {note}{p10_txt}{gray_txt}")
        self.lbl_reason.setText(reason)
        self.clear_rect()

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开影像", "", "影像文件 (*.tif *.tiff *.jpg *.jpeg *.png *.bmp)")
        if path:
            self.load_image(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isfile(p):
                self.load_image(p)
                return

    # ---------- 框选统计 ----------
    def on_rect(self, x0, y0, x1, y1):
        self.rect = (x0, y0, x1, y1)
        self.left.setRect(self.rect)
        self.right.setRect(self.rect)
        self.update_stats()

    def update_stats(self):
        if self.exg_masked is None or self.rect is None:
            return
        h, w = self.exg_masked.shape
        x0 = max(0, int(min(self.rect[0], self.rect[2])))
        x1 = min(w, int(max(self.rect[0], self.rect[2])) + 1)
        y0 = max(0, int(min(self.rect[1], self.rect[3])))
        y1 = min(h, int(max(self.rect[1], self.rect[3])) + 1)
        area = (y1 - y0) * (x1 - x0)
        vals = self.exg_masked[y0:y1, x0:x1]
        vals = vals[np.isfinite(vals)]
        labels = (self.lbl_valid, self.lbl_n, self.lbl_mean, self.lbl_max, self.lbl_std, self.lbl_pct)
        if vals.size == 0:
            for w_ in labels:
                w_.setText(w_.text().split(":")[0] + ": -")
            return
        n = vals.size
        self.lbl_valid.setText(f"有效占比: {100.0 * n / area:.0f}%")
        self.lbl_n.setText(f"像素: {n}")
        self.lbl_mean.setText(f"均值: {vals.mean():.1f}")
        self.lbl_max.setText(f"最大: {vals.max():.0f}")
        self.lbl_std.setText(f"标准差: {vals.std():.1f}")
        self.lbl_pct.setText(f"蓝藻占比: {100.0 * (vals > A.BLOOM_THRESH).sum() / n:.1f}%")

    def clear_rect(self):
        self.rect = None
        self.left.setRect(None)
        self.right.setRect(None)
        for w_ in (self.lbl_valid, self.lbl_n, self.lbl_mean, self.lbl_max, self.lbl_std, self.lbl_pct):
            w_.setText(w_.text().split(":")[0] + ": -")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        win.load_image(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
