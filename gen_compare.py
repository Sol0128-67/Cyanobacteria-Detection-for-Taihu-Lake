# -*- coding: utf-8 -*-
"""为全部 722 张影像生成对比图 (原图|ExG热力图 并排, 标注评级)

输出: output/compare/<年份>/taihu_YYYYMMDD.jpg
断点续跑: 已生成的自动跳过
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.loader import load_full
from src import analysis as A

OUT = "output/compare"
EXG_MIN, EXG_MAX = -500, 3500
_STOPS = [(-500, (20, 25, 45)), (300, (45, 90, 180)), (500, (80, 205, 170)),
          (900, (255, 230, 40)), (1400, (245, 95, 20)), (3500, (205, 15, 15))]
NONWATER = (18, 18, 26)
CLOUD_C = (222, 222, 230)
SHADOW_C = (110, 110, 122)


def make_lut():
    xs = np.arange(EXG_MIN, EXG_MAX + 1)
    lut = np.zeros((len(xs), 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, [s[0] for s in _STOPS], [s[1][ch] for s in _STOPS])
    return lut


LUT = make_lut()
FONT = None


def get_font(size):
    global FONT
    if FONT is None:
        for p in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
            if os.path.exists(p):
                FONT = p
                break
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def render_heat(exg, water, bad, qa, lake):
    idx = np.clip(np.round(exg).astype(np.int32) - EXG_MIN, 0, len(LUT) - 1)
    heat = LUT[idx]
    heat[~water] = NONWATER
    if lake is not None:
        heat[bad & lake] = CLOUD_C
        if qa is not None:
            heat[(qa == 2) & lake] = SHADOW_C
    else:
        heat[bad & water] = CLOUD_C
    return heat


def process_one(path, mask):
    name = os.path.basename(path)
    d8 = name[6:14]
    year = d8[:4]
    out_path = os.path.join(OUT, year, f"taihu_{d8}.jpg")
    if os.path.exists(out_path):
        return None
    is_l1c = "_l1c" in name
    r, g, b, nir, b5, b11, qa, note, gray = load_full(path)
    lake = mask if mask.shape == r.shape else None
    water, valid, bad = A.build_water(r, g, b, nir, qa, lake, is_l1c)
    seam = A.detect_seam(r, g, b, lake) if lake is not None else None
    blocks = A.split_blocks(water, seam, r.shape[1])
    exg = A.anchored_exg(r, g, b, water, blocks)
    badge, reason = A.rate_quality(water, valid, bad, lake, nir, exg)
    n_water = int(water.sum())
    bpct = 100.0 * (water & (exg > A.BLOOM_THRESH)).sum() / n_water if n_water else 0

    h, w = r.shape
    scale = 900 / w
    nw, nh = 900, int(h * scale)
    orig = np.clip(np.stack([r, g, b], 2) / 3000.0 * 255, 0, 255).astype(np.uint8)
    orig_s = cv2.resize(orig, (nw, nh), interpolation=cv2.INTER_AREA)
    heat = render_heat(exg, water, bad, qa, lake)
    heat_s = cv2.resize(heat, (nw, nh), interpolation=cv2.INTER_AREA)

    head = np.full((44, nw * 2, 3), 22, dtype=np.uint8)
    comp = np.vstack([head, np.hstack([orig_s, heat_s])])
    pil = Image.fromarray(comp[:, :, ::-1])
    draw = ImageDraw.Draw(pil)
    txt = f"{d8} [{note}]   {badge} {reason}   蓝藻覆盖率 {bpct:.1f}%"
    draw.text((10, 8), txt, fill=(255, 215, 110), font=get_font(22))
    draw.text((nw + 14, 8), "原图(左) | ExG热力图(右)  浅灰=云/影", fill=(180, 180, 200), font=get_font(18))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil.save(out_path, quality=86)
    return out_path


def main():
    files = sorted(glob.glob(os.path.join("data", "taihu_*.tif")))
    mask = np.load("data/taihu_mask.npy") > 0
    print(f"共 {len(files)} 张, 输出到 {OUT}/<年份>/")
    t0 = time.time()
    done = 0
    for f in files:
        try:
            out = process_one(f, mask)
            if out:
                done += 1
                if done % 30 == 0:
                    el = (time.time() - t0) / done
                    print(f"  已生成 {done} 张, 剩余约 {el*(len(files)-done)/60:.0f} 分钟")
        except Exception as e:
            print(f"  [失败] {os.path.basename(f)}: {type(e).__name__} {e}")
    print(f"完成: 本次生成 {done} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
