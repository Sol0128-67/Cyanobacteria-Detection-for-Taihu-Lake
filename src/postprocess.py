# -*- coding: utf-8 -*-
"""后处理: 形态学去噪 + 连通域分析"""
import cv2
import numpy as np


def clean(binary, open_k=3, close_k=7, min_area=100):
    """开运算去孤立噪点, 闭运算填斑块孔洞, 剔除小斑块(面积<min_area 像素)"""
    b = binary.astype(np.uint8)
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel_o)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel_c)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = np.zeros_like(b)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out > 0


def component_stats(binary):
    """返回斑块列表 [(面积, 中心x, 中心y, bbox), ...] 按面积降序"""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    patches = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        patches.append({"area": int(area), "cx": float(cx), "cy": float(cy),
                        "bbox": (int(x), int(y), int(w), int(h))})
    patches.sort(key=lambda p: -p["area"])
    return patches
