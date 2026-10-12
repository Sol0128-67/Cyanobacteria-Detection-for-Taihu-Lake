# -*- coding: utf-8 -*-
"""任意 RGB 影像读取: 自动检测格式/位数/通道顺序, 统一换算到 0-10000 尺度"""
import os

import cv2
import numpy as np


def load_any_image(path):
    """读取任意 RGB 图 -> (r, g, b) float32 + 说明文字 + 是否灰度图"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        # TIFF 一律用 tifffile (通道顺序可靠; cv2 对特殊压缩 TIFF 解码顺序不可靠)
        import tifffile
        a = tifffile.imread(path)
        gray = a.ndim == 2
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=2)
        elif a.ndim == 3 and a.shape[2] >= 3:
            a = a[:, :, :3]
        else:
            raise ValueError("不支持的 TIFF 结构: " + str(a.shape))
        rgb_order = True                     # tifffile 是 RGB 序
    else:
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError("无法读取 (格式不支持或文件损坏): " + path)
        gray = a.ndim == 2
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=2)
        elif a.ndim == 3 and a.shape[2] == 4:
            a = a[:, :, :3]
        rgb_order = False                    # cv2 是 BGR 序

    # 数值尺度换算 -> 目标: 16位反射率尺度 (水体约 500-2000, 阈值 T=500 有效)
    pos = a[a > 0]
    p98 = float(np.percentile(pos, 98)) if pos.size else 1.0
    if a.dtype == np.uint8:
        # 8位图没有原始辐射尺度, 按 98 百分位归一化到 3000 (自动曝光校正)
        scale = 3000.0 / max(p98, 1.0)
        note = f"8位图, 自动归一化 x{scale:.1f}"
    elif a.dtype == np.uint16:
        if float(a.max()) > 30000:
            scale = 3000.0 / max(p98, 1.0)
            note = f"16位图(原始DN), 归一化 x{scale:.1f}"
        else:
            scale = 1.0
            note = "16位图(反射率尺度), 原值"
    elif a.dtype in (np.float32, np.float64):
        if float(a.max()) <= 1.5:
            scale = 10000.0
            note = "浮点反射率(0-1), x10000"
        else:
            scale = 3000.0 / max(p98, 1.0)
            note = f"浮点图, 归一化 x{scale:.1f}"
    else:
        raise ValueError("不支持的数据类型: " + str(a.dtype))

    f = np.clip(a.astype(np.float32) * scale, 0, 12000)
    if rgb_order:
        r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    else:                                     # cv2 的 BGR -> RGB
        b, g, r = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    return r, g, b, note, gray


def load_full(path):
    """读任意影像, 返回全部可用层:
    (r, g, b, nir, b5, b11, qa, note, gray)
    7层 TIFF 布局: [B4红, B3绿, B2蓝, B8近红外, B5红边, B11短波红外, QA]
    普通 3 波段图片: nir/b5/b11/qa 均为 None
    """
    a = tifffile_read(path) if os.path.splitext(path)[1].lower() in (".tif", ".tiff") else None
    if a is None:
        r, g, b, note, gray = load_any_image(path)
        return r, g, b, None, None, None, None, note, gray
    gray = a.ndim == 2
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :a.shape[2]]
    else:
        raise ValueError("不支持的 TIFF 结构: " + str(a.shape))
    f = a.astype(np.float32)
    n_bands = f.shape[2]
    # 7层 = 我们导出的卫星影像 (反射率尺度, 保持原值); 其他层数按曝光归一化
    if n_bands >= 7:
        note = "7层卫星影像 (含近红外/QA)"
    else:
        k = 3000.0 / max(float(np.percentile(f[f > 0], 98)), 1.0)
        f = np.clip(f * k, 0, 12000)
        note = f"非标准影像, 归一化 x{k:.1f}"
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    nir = f[:, :, 3] if n_bands >= 4 else None
    b5 = f[:, :, 4] if n_bands >= 5 else None
    b11 = f[:, :, 5] if n_bands >= 6 else None
    qa = f[:, :, 6] if n_bands >= 7 else None
    return r, g, b, nir, b5, b11, qa, note, gray


def tifffile_read(path):
    import tifffile
    a = tifffile.imread(path)
    if a.ndim == 3 and a.shape[2] >= 3:
        return a
    return None
