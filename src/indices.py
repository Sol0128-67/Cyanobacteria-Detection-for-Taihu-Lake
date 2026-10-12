# -*- coding: utf-8 -*-
"""颜色指数计算 (输入均为 float32 数组, SR 地表反射率 x10000)"""
import numpy as np

_EPS = 1e-6


def excess_green(r, g, b):
    """ExG 过绿指数: 2G - R - B, 蓝藻叶绿素反射绿光 -> ExG 高"""
    return 2.0 * g - r - b


def g_minus_r(r, g):
    """G-R 差值"""
    return g - r


def g_over_b(g, b):
    """G/B 比值: 区分绿水与蓝清水"""
    return np.divide(g, b, out=np.zeros_like(g, dtype=np.float32), where=b > _EPS)


def vari(r, g, b):
    """VARI = (G-R)/(G+R-B), 比值形式对光照有归一化"""
    denom = g + r - b
    return np.divide(g - r, denom, out=np.zeros_like(g, dtype=np.float32), where=denom > 50)


def hue(r, g, b):
    """HSV 色相角 (度), 蓝藻约在 60~170 (绿到青)"""
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    h = np.zeros_like(r, dtype=np.float32)
    mask = delta > _EPS
    rr = r[mask]; gg = g[mask]; bb = b[mask]
    dm = delta[mask]
    hg = np.where(mx[mask] == rr, (gg - bb) / dm % 6, 0)
    hb = np.where(mx[mask] == gg, (bb - rr) / dm + 2, 0)
    hr = np.where(mx[mask] == bb, (rr - gg) / dm + 4, 0)
    h[mask] = hg + hb + hr
    return h * 60.0
