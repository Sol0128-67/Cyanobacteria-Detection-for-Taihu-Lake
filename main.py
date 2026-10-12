# -*- coding: utf-8 -*-
"""太湖蓝藻监测 - 像素级图像分析主流程

流程: 读 TIFF -> 水体掩膜 -> ExG 指数 -> Otsu 分割 -> 形态学去噪
      -> 浓度分级/统计 -> 伪彩色图/二值图/直方图/CSV 报表

用法: python main.py [data目录] [输出目录]
"""
import glob
import os
import sys

import numpy as np
import tifffile

from src import analyze, indices, postprocess, segment, visualize, water_mask


# 固定 ExG 阈值 (SR x10000): 蓝藻聚集区与正常浑浊水体的分界。
# 依据: 9 张影像水体 ExG 分布 p50=458~745, 已知水华事件影像(20180509/20190603/
# 20200801/20230806)的 Otsu 阈值中位数约 455; 取 500 保证跨影像可比。
EXG_THRESH = 500.0


# ExG 清水基线锚定值: SR 影像水体 ExG p10 的中位水平 (~360)。
# 大气散射偏蓝, L1C 的水体 ExG 系统性偏低; 每个时相把 p10 平移到 360,
# 可同时校正 L1C/SR 产品差异与雾霾影响 (假设蓝藻季至少 10% 水体为清水)。
EXG_P10_ANCHOR = 360.0


def load_bands(path):
    """读 TIFF -> (r,g,b,source), 波段保持原始值"""
    a = tifffile.imread(path)  # (h,w,3) uint16, GEE 导出顺序 R,G,B
    r = a[:, :, 0].astype(np.float32)
    g = a[:, :, 1].astype(np.float32)
    b = a[:, :, 2].astype(np.float32)
    source = "L1C" if "_l1c" in os.path.basename(path) else "SR"
    return r, g, b, source


def process_one(path, out_dir, exg_thresh=EXG_THRESH):
    date = os.path.basename(path).replace("taihu_", "").replace(".tif", "")
    r, g, b, source = load_bands(path)

    # 1. 水体提取 (剔除陆地/云/耀斑)
    water, valid = water_mask.water_mask(r, g, b)

    # 2. ExG 指数 + p10 基线锚定 (清水基线对齐, 跨产品/跨时相可比)
    exg = indices.excess_green(r, g, b)
    p10 = float(np.percentile(exg[water], 10))
    exg = exg + (EXG_P10_ANCHOR - p10)

    # 3. Otsu 自适应阈值分割 -> 蓝藻二值图
    bloom, thresh = segment.threshold_index(exg, water, thresh=exg_thresh)

    # 4. 形态学去噪 + 连通域
    bloom = postprocess.clean(bloom)
    patches = postprocess.component_stats(bloom)

    # 5. 浓度分级与统计
    severity = analyze.grade_bloom(exg, bloom, thresh)
    row = analyze.summarize(exg, bloom, severity, water, valid, patches, date)
    row["source"] = source

    # 6. 输出
    rgb = np.stack([r, g, b], axis=2)
    visualize.save_overlay(rgb, severity, os.path.join(out_dir, f"{date}_overlay.png"))
    visualize.save_binary_mask(bloom, os.path.join(out_dir, f"{date}_bloom_mask.png"))
    visualize.save_histogram(exg[water], thresh, os.path.join(out_dir, f"{date}_exg_hist.png"),
                             title=f"{date} ExG in water (Otsu={thresh:.0f})")
    print(f"  {date} [{source}]: p10偏移={EXG_P10_ANCHOR-p10:+.0f} 覆盖率={row['bloom_coverage_pct']}% 斑块={row['n_patches']}个")
    return row


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(data_dir, "taihu_*.tif")))
    if not files:
        print(f"在 {data_dir} 下没有找到 taihu_*.tif")
        return 1

    rows = []
    for f in files:
        rows.append(process_one(f, out_dir))

    print("\n========== 太湖蓝藻浓度统计 (按覆盖率降序) ==========")
    analyze.print_table(rows)
    report_path = os.path.join(out_dir, "report.csv")
    analyze.write_report(rows, report_path)
    print(f"\n报表已保存: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
