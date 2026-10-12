# -*- coding: utf-8 -*-
"""共享分析逻辑 (桌面软件与批量分析共用)

7 层影像布局: [B4红, B3绿, B2蓝, B8近红外, B5红边, B11短波红外, QA]
QA 层: SR=SCL重映射(1=云/卷云, 2=云影, 3=水体参考, 0=其他)
       L1C=不透明云+卷云掩膜(1=云, 0=其他)
"""
import numpy as np

from . import indices, water_mask

EXG_P10_ANCHOR = 360.0
BLOOM_THRESH = 500.0
MILD_UP = 900.0
MODERATE_UP = 1400.0

COVERAGE_MIN = 0.50
CLEAR_MIN = 0.70
HAZE_NIR_MAX = 2000.0
SIGNAL_SPAN_MIN = 150.0
BRIGHT_MAX = 3000.0       # 亮度>此值 = 亮斑/云, 剔除
CLOUD_BRIGHT_MIN = 2500.0  # SR 的 SCL 云分类需亮度>此值才采信


def build_water(r, g, b, nir, qa, lake, is_l1c=False):
    """水体判定 -> (water, valid, bad)
    有湖掩膜: 湖内即水(含泥沙/浮渣), 剔除云/影/亮斑/无数据
    无湖掩膜: NDWI 水体掩膜
    """
    valid_raw = (r > 0) & (g > 0) & (b > 0)
    mean3 = (r + g + b) / 3.0
    bright = valid_raw & (mean3 > BRIGHT_MAX)
    if qa is not None:
        if is_l1c:
            cloud = qa == 1
        else:
            cloud = (qa == 1) & (mean3 > CLOUD_BRIGHT_MIN)
        bad = bright | cloud | (qa == 2)
    else:
        bad = bright
    if lake is not None:
        water = valid_raw & lake & ~bad
        return water, valid_raw & ~bad, bad
    w0, v0, b0 = water_mask.water_mask_nir(r, g, b, nir, qa)
    return w0, v0, b0


def detect_seam(r, g, b, lake):
    """检测拼接缝: 找湖面平均亮度突变最大的列, 返回列号或 None
    (不同瓦片辐射定标差异会在拼接边界产生垂直亮度跳变)
    """
    bright = ((r + g + b) / 3.0) * lake
    colsum = bright.sum(axis=0)
    colcnt = lake.sum(axis=0)
    ok = colcnt > 500
    if ok.sum() < 1000:
        return None
    colmean = np.zeros_like(colsum, dtype=np.float32)
    colmean[ok] = colsum[ok] / colcnt[ok]
    # 只在湖体中间 30%-70% 列范围找 (避开边缘)
    cols = np.where(ok)[0]
    lo, hi = int(np.percentile(cols, 30)), int(np.percentile(cols, 70))
    seg = colmean[lo:hi]
    diff = np.abs(np.diff(seg))
    # 排除无数据裂缝 (无数据的列 diff 巨大, 但 colmean=0)
    valid_seg = seg[:-1] > 0
    if not valid_seg.any():
        return None
    d = diff.copy()
    d[~valid_seg] = 0
    k = int(np.argmax(d))
    if d[k] < max(25.0, 0.05 * float(seg[:-1][valid_seg].mean())):
        return None
    return lo + k + 1


def anchored_exg(r, g, b, water, blocks):
    """分块 p10 锚定: 每块水体 ExG 的 p10 平移到 360, 返回锚定后的 ExG"""
    exg = indices.excess_green(r, g, b)
    if len(blocks) == 1:
        m = water
        if m.sum() >= 1000:
            exg = exg + (EXG_P10_ANCHOR - float(np.percentile(exg[m], 10)))
        return exg
    for m in blocks:
        if m.sum() >= 1000:
            p10 = float(np.percentile(exg[m], 10))
            exg[m] += EXG_P10_ANCHOR - p10
    return exg


def split_blocks(water, seam, width):
    """按拼缝把水体分成两块 (西/东), 无拼缝返回 [water]"""
    if seam is None or seam <= 0 or seam >= width:
        return [water]
    west = water & (np.arange(width)[None, :] < seam)
    east = water & (np.arange(width)[None, :] >= seam)
    if west.sum() < 5000 or east.sum() < 5000:
        return [water]
    return [west, east]


def rate_quality(water, valid, bad, lake, nir, exg):
    """质量评级 -> (徽章, 说明)"""
    if lake is not None:
        lake_n = float(lake.sum())
        raw = (valid | bad) & lake
        coverage = float(raw.sum()) / lake_n
        clear_frac = float((valid & lake).sum()) / lake_n
        nodata_frac = 1.0 - coverage
        mask = valid & lake
        if coverage < COVERAGE_MIN:
            return "❌", f"影像覆盖不全, 不适合识别 (只拍到湖面 {coverage*100:.0f}%)"
        if clear_frac < CLEAR_MIN:
            bad_frac = 1.0 - clear_frac - nodata_frac
            if bad_frac > nodata_frac:
                return "❌", f"云/雾遮挡严重, 不适合识别 (可用湖面仅 {clear_frac*100:.0f}%)"
            return "❌", f"影像覆盖不全, 不适合识别 (只拍到湖面 {coverage*100:.0f}%)"
        if mask.sum() < 5000:
            return "❌", "有效水面像素过少, 不适合识别"
        span = float(np.percentile(exg[mask], 90) - np.percentile(exg[mask], 10))
        if nir is not None:
            nir_m = float(nir[mask].mean())
            if nir_m > HAZE_NIR_MAX:
                return "❌", f"雾霾严重, 不适合识别 (水体NIR {nir_m:.0f})"
        if span < SIGNAL_SPAN_MIN:
            return "⚠️", f"可识别但无显著藻华信号 (ExG跨度 {span:.0f})"
        return "✅", f"适合识别 (可用湖面 {clear_frac*100:.0f}%, ExG跨度 {span:.0f})"
    if water.sum() < 5000:
        return "❌", "未检测到足够水体"
    mask = water & ~bad
    if mask.sum() < 1000:
        return "❌", "几乎无有效水面 (云/雾/亮度过高)"
    cloud_pct = 100.0 * (bad & water).sum() / water.sum()
    span = float(np.percentile(exg[mask], 90) - np.percentile(exg[mask], 10))
    if cloud_pct > 30:
        return "❌", f"云量过大, 不适合识别 (湖面被云/影覆盖 {cloud_pct:.0f}%)"
    if nir is not None:
        nir_m = float(nir[mask].mean())
        if nir_m > HAZE_NIR_MAX:
            return "❌", f"雾霾严重, 不适合识别 (水体NIR {nir_m:.0f})"
    if span < SIGNAL_SPAN_MIN:
        return "⚠️", f"可识别但无显著藻华信号 (ExG跨度 {span:.0f})"
    return "✅", f"适合识别 (云覆盖 {cloud_pct:.0f}%, ExG跨度 {span:.0f})"


def grade_severity(exg, bloom):
    """浓度分级: 0=非蓝藻 1=轻度 2=中度 3=重度"""
    sev = np.zeros_like(exg, dtype=np.uint8)
    sev[bloom & (exg < MILD_UP)] = 1
    sev[bloom & (exg >= MILD_UP) & (exg < MODERATE_UP)] = 2
    sev[bloom & (exg >= MODERATE_UP)] = 3
    return sev
