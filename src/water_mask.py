# -*- coding: utf-8 -*-
"""水体提取: 先在水中找蓝藻, 避免岸上植被误检

有近红外时用 NDWI=(G-NIR)/(G+NIR): 任何颜色的水(清水/绿水/黄泥水/藻水)
都吸收近红外, 陆地植被反射近红外 -> 与水的颜色无关, 泥沙水不再误杀。
无近红外时退回 NDWI=(B-R)/(B+R) (蓝光优势, 清水适用, 泥沙水会被误杀)。
QA 质量层可逐像素剔除云/云影。
"""
import numpy as np

from .segment import otsu_threshold

_BRIGHT = 3000.0  # 均值超过此值视为云/太阳耀斑, 剔除


def water_mask(r, g, b):
    """旧接口 (无 NIR/QA 时): 返回 (water, valid)"""
    water, valid, _ = water_mask_nir(r, g, b)
    return water, valid


def water_mask_nir(r, g, b, nir=None, qa=None):
    """返回 (water, valid, bad):
    water=水体, valid=有效像素(非nodata/非云影/非亮斑), bad=云/影/亮斑"""
    if nir is not None:
        ndwi = np.divide(g - nir, g + nir + 1e-6, out=np.zeros_like(g, dtype=np.float32), where=(g + nir) > 0)
    else:
        ndwi = np.divide(b - r, b + r + 1e-6, out=np.zeros_like(b, dtype=np.float32), where=(b + r) > 0)
    valid = (r > 0) & (g > 0) & (b > 0)
    t = otsu_threshold(ndwi[valid])
    water = valid & (ndwi > t)
    bright = valid & ((r + g + b) / 3.0 > _BRIGHT)
    if qa is not None:
        # QA: 1=云/卷云, 2=云影, 3=水体参考(SCL分类), 0=其他 -> 只剔除 1 和 2
        bad = bright | (qa == 1) | (qa == 2)
    else:
        bad = bright
    valid = valid & ~bad
    water = water & ~bad
    return water, valid, bad
