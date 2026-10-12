# -*- coding: utf-8 -*-
"""结果可视化: 伪彩色叠加图、二值掩膜图、ExG 直方图"""
import os

import cv2
import numpy as np

# 伪彩色: 轻度=黄, 中度=橙, 重度=红
_SEV_COLOR = {1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 220)}  # BGR


def to_uint8(rgb):
    """SR float 数组 -> 8bit BGR (固定拉伸 0-3000), 供显示用"""
    disp = np.clip(np.asarray(rgb, dtype=np.float32) / 3000.0 * 255, 0, 255).astype(np.uint8)
    return disp[:, :, ::-1].copy()  # RGB -> BGR


def save_overlay(rgb, severity, path, alpha=0.55):
    """原始真彩色底图 + 半透明伪彩色蓝藻分级叠加"""
    base = to_uint8(rgb)
    overlay = base.copy()
    for level, color in _SEV_COLOR.items():
        overlay[severity == level] = color
    blended = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)
    cv2.imwrite(path, blended)


def save_binary_mask(mask, path):
    cv2.imwrite(path, (mask.astype(np.uint8) * 255))


def save_histogram(values, thresh, path, title="ExG distribution in water"):
    """水体内 ExG 直方图 + 分割阈值标记线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
        plt.rcParams["font.family"] = "Microsoft YaHei"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=200, color="#2ca02c", alpha=0.8)
    ax.axvline(thresh, color="red", linestyle="--", label=f"Otsu threshold = {thresh:.0f}")
    ax.axvline(thresh + 400, color="orange", linestyle=":", label="mild/moderate")
    ax.axvline(thresh + 900, color="darkred", linestyle=":", label="moderate/severe")
    ax.set_xlabel("ExG (SR x10000)")
    ax.set_ylabel("pixel count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
