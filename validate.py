# -*- coding: utf-8 -*-
"""精度验证: 将我们的 ExG 提取结果与参考真值 (geodata.cn 数据集) 逐像素对比

用法:
  python validate.py <参考文件> <我方影像>
  例: python validate.py ref/20180419_bloom.tif data/taihu_20180419.tif

参考文件: 单波段 GeoTIFF, >0 为蓝藻 (任意投影/分辨率, 自动重投影对齐)
我方影像: data/ 下的 3 波段 TIFF, 复用 main.py 管线提取蓝藻
指标: 混淆矩阵、精确率、召回率、F1、IoU、覆盖率对比
"""
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src import analyze, indices, postprocess, segment, water_mask

EXG_THRESH = 500.0


def load_ours(path, thresh=EXG_THRESH):
    """我方影像 -> 蓝藻二值图 + 水体掩膜 (在自身网格上)"""
    import tifffile
    a = tifffile.imread(path)
    r = a[:, :, 0].astype(np.float32)
    g = a[:, :, 1].astype(np.float32)
    b = a[:, :, 2].astype(np.float32)
    water, valid = water_mask.water_mask(r, g, b)
    exg = indices.excess_green(r, g, b)
    bloom, _ = segment.threshold_index(exg, water, thresh=thresh)
    bloom = postprocess.clean(bloom)
    return bloom, water, valid


def load_ref(path):
    """参考真值 -> (array, transform, crs), array 中 >0 为蓝藻"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        return arr, src.transform, src.crs


def align_to(ref_arr, ref_transform, ref_crs, tgt_shape, tgt_transform, tgt_crs):
    """把参考图重投影/重采样到目标网格, 最近邻"""
    dst = np.zeros(tgt_shape, dtype=np.float32)
    reproject(ref_arr, dst, src_transform=ref_transform, src_crs=ref_crs,
              dst_transform=tgt_transform, dst_crs=tgt_crs,
              resampling=Resampling.nearest)
    return dst


def metrics(pred, truth):
    """pred/truth 为 0/1 二值图, 返回指标 dict"""
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(f1, 4), "iou": round(iou, 4),
        "ref_coverage": round(100 * truth.mean(), 2),
        "our_coverage": round(100 * pred.mean(), 2),
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    ref_path, our_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(ref_path) or not os.path.exists(our_path):
        print("文件不存在:", ref_path, our_path)
        return 1

    print(f"参考真值: {ref_path}")
    print(f"我方影像: {our_path}")

    ref_arr, ref_tf, ref_crs = load_ref(ref_path)
    print(f"参考图: {ref_arr.shape}, CRS={ref_crs}, 蓝藻像素占比 {100 * (ref_arr > 0).mean():.2f}%")

    with rasterio.open(our_path) as src:
        our_tf, our_crs, our_shape = src.transform, src.crs, (src.height, src.width)
    bloom, water, valid = load_ours(our_path)

    ref_aligned = align_to(ref_arr, ref_tf, ref_crs, our_shape, our_tf, our_crs)

    # 在"参考图有标注的区域 ∩ 我方水体掩膜"内对比 (参考图只标注了湖区)
    region = (ref_aligned > 0) | (bloom & water) | (water & (ref_aligned == 0))
    region = region & water
    m = metrics(bloom[region], (ref_aligned > 0)[region])
    print(f"\n对比区域: {int(region.sum())} 像素 (我方水体掩膜 ∩ 参考标注区)")
    print(f"  混淆矩阵: TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print(f"  精确率 Precision = {m['precision']:.3f}")
    print(f"  召回率 Recall    = {m['recall']:.3f}")
    print(f"  F1 分数          = {m['f1']:.3f}")
    print(f"  IoU              = {m['iou']:.3f}")
    print(f"  覆盖率: 参考 {m['ref_coverage']}% vs 我方 {m['our_coverage']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
