# -*- coding: utf-8 -*-
"""阈值分割: Otsu 大津法自适应阈值 (无需人工设定, 对不同光照自适应)"""
import numpy as np


def otsu_threshold(values):
    """一维数组的 Otsu 最优阈值, 直方图按 min/max 均匀分 512 箱"""
    values = np.asarray(values, dtype=np.float32)
    if values.size < 100:
        return float(np.median(values)) if values.size else 0.0
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return lo
    hist, edges = np.histogram(values, bins=512, range=(lo, hi))
    total = values.size
    sum_all = np.dot(hist, (edges[:-1] + edges[1:]) / 2.0)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2.0)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros_like(sum_bg, dtype=np.float64), where=weight_bg > 0)
    mean_fg = np.divide(sum_all - sum_bg, weight_fg, out=np.zeros_like(sum_bg, dtype=np.float64), where=weight_fg > 0)
    var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    k = int(np.argmax(var_between))
    return float((edges[k] + edges[k + 1]) / 2.0)


def threshold_index(index, mask, thresh=None):
    """在 mask 内对 index 做阈值分割, 返回二值图; thresh=None 时用 Otsu"""
    if thresh is None:
        thresh = otsu_threshold(index[mask])
    return mask & (index > thresh), thresh
