# -*- coding: utf-8 -*-
"""全量批量分析: 722 张影像逐个识别 + 质量标注 + 常年重灾区频率累积

输出:
- output/report_722.csv        每张影像的完整统计与标注 (❌/⚠️/✅ 及原因)
- output/bloom_frequency.png   多年蓝藻爆发频率图 (哪块区域常年严重)
- output/bloom_frequency.npy   频率数据 (供软件/后续分析)
断点续跑: 已分析过的日期自动跳过。
"""
import csv
import glob
import os
import sys
import time

import numpy as np

from src.loader import load_full
from src import analysis as A

DATA = "data"
OUT = "output"
FIELDS = ["date", "product", "badge", "reason", "coverage_pct", "clear_pct",
          "exg_span", "bloom_pct", "mild_pct", "moderate_pct", "severe_pct",
          "water_km2", "seam"]


def load_mask():
    return np.load(os.path.join(DATA, "taihu_mask.npy")) > 0


def freq_to_rgb(freq):
    """频率 0-1 -> RGB (深蓝->青->绿->黄->红)"""
    stops = [(0.0, (20, 30, 60)), (0.2, (40, 150, 200)), (0.4, (60, 200, 120)),
             (0.6, (240, 220, 40)), (0.8, (250, 120, 20)), (1.0, (200, 10, 10))]
    v = np.clip(freq, 0, 1)
    out = np.zeros(v.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(v, [s[0] for s in stops], [s[1][ch] for s in stops])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA, "taihu_*.tif")))
    mask = load_mask()
    M2 = 400.0 * np.cos(np.radians(31.2)) ** 2

    # 断点续跑: 已分析日期
    done_dates = set()
    csv_path = os.path.join(OUT, "report_722.csv")
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                done_dates.add(row["date"])
        print(f"断点续跑: 已完成 {len(done_dates)} 个日期")

    # 累积器
    bloom_acc = np.zeros(mask.shape, dtype=np.uint16)
    valid_acc = np.zeros(mask.shape, dtype=np.uint16)
    if os.path.exists(os.path.join(OUT, "bloom_acc.npy")):
        bloom_acc = np.load(os.path.join(OUT, "bloom_acc.npy"))
        valid_acc = np.load(os.path.join(OUT, "valid_acc.npy"))

    new_file = not os.path.exists(csv_path)
    f_out = open(csv_path, "a", newline="", encoding="utf-8-sig")
    wcsv = csv.DictWriter(f_out, fieldnames=FIELDS)
    if new_file:
        wcsv.writeheader()

    t0 = time.time()
    n_done = 0
    for i, f in enumerate(files):
        name = os.path.basename(f)
        d8 = name[6:14]
        if d8 in done_dates:
            continue
        is_l1c = "_l1c" in name
        try:
            r, g, b, nir, b5, b11, qa, note, gray = load_full(f)
        except Exception as e:
            print(f"  [读取失败] {name}: {type(e).__name__}")
            continue
        lake = mask if mask.shape == r.shape else None
        water, valid, bad = A.build_water(r, g, b, nir, qa, lake, is_l1c)
        seam = A.detect_seam(r, g, b, lake) if lake is not None else None
        blocks = A.split_blocks(water, seam, r.shape[1])
        exg = A.anchored_exg(r, g, b, water, blocks)
        bloom = water & (exg > A.BLOOM_THRESH)
        badge, reason = A.rate_quality(water, valid, bad, lake, nir, exg)

        n_water = int(water.sum())
        coverage = float((valid | bad)[lake].sum()) / lake.sum() if lake is not None else np.nan
        clear = float(valid[lake].sum()) / lake.sum() if lake is not None else np.nan
        span = (float(np.percentile(exg[water], 90) - np.percentile(exg[water], 10))
                if n_water >= 1000 else np.nan)
        bpct = 100.0 * bloom.sum() / n_water if n_water else 0.0
        sev = A.grade_severity(exg, bloom)
        row = {
            "date": d8,
            "product": "L1C" if is_l1c else "SR",
            "badge": badge,
            "reason": reason,
            "coverage_pct": round(coverage * 100, 1) if coverage == coverage else "",
            "clear_pct": round(clear * 100, 1) if clear == clear else "",
            "exg_span": round(span, 0) if span == span else "",
            "bloom_pct": round(bpct, 2),
            "mild_pct": round(100.0 * (sev == 1).sum() / n_water, 2) if n_water else 0,
            "moderate_pct": round(100.0 * (sev == 2).sum() / n_water, 2) if n_water else 0,
            "severe_pct": round(100.0 * (sev == 3).sum() / n_water, 2) if n_water else 0,
            "water_km2": round(n_water * M2 / 1e6, 1),
            "seam": seam or "",
        }
        wcsv.writerow(row)
        f_out.flush()
        n_done += 1

        # 频率累积 (只统计可用水体像素)
        valid_acc[water] += 1
        bloom_acc[bloom] += 1
        if n_done % 20 == 0:
            np.save(os.path.join(OUT, "bloom_acc.npy"), bloom_acc)
            np.save(os.path.join(OUT, "valid_acc.npy"), valid_acc)
            el = (time.time() - t0) / n_done
            print(f"  进度 {n_done}: {d8} [{badge}] 剩余约 {el*(len(files)-len(done_dates)-n_done)/60:.0f} 分钟")

    f_out.close()
    np.save(os.path.join(OUT, "bloom_acc.npy"), bloom_acc)
    np.save(os.path.join(OUT, "valid_acc.npy"), valid_acc)

    # 频率图
    freq = np.where(valid_acc >= 3, bloom_acc.astype(np.float32) / np.maximum(valid_acc, 1), np.nan)
    np.save(os.path.join(OUT, "bloom_frequency.npy"), freq)
    rgb = freq_to_rgb(freq)
    rgb[~mask] = (15, 15, 22)
    import cv2
    cv2.imwrite(os.path.join(OUT, "bloom_frequency.png"), rgb[:, :, ::-1])
    print(f"\n完成: 本次分析 {n_done} 张, 频率图已保存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
