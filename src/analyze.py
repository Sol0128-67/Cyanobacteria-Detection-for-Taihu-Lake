# -*- coding: utf-8 -*-
"""浓度统计: 覆盖率、浓度分级 (ExG 值越高蓝藻越聚集)

分级阈值相对分割阈值 T (SR 反射率 x10000 单位), 可配置:
  轻度 mild:      T       <= ExG < T + MILD
  中度 moderate:  T+MILD  <= ExG < T + MODERATE
  重度 severe:    T+MODERATE <= ExG
"""
import csv

import numpy as np

MILD = 400.0
MODERATE = 900.0


def grade_bloom(exg, bloom, thresh):
    """返回 severity 图: 0=非蓝藻 1=轻度 2=中度 3=重度"""
    sev = np.zeros_like(exg, dtype=np.uint8)
    b = bloom
    sev[b & (exg < thresh + MILD)] = 1
    sev[b & (exg >= thresh + MILD) & (exg < thresh + MODERATE)] = 2
    sev[b & (exg >= thresh + MODERATE)] = 3
    return sev


def summarize(exg, bloom, severity, water, valid, patches, date):
    """汇总统计, 返回 dict"""
    n_water = int(water.sum())
    n_valid = int(valid.sum())
    n_bloom = int(bloom.sum())
    n1 = int((severity == 1).sum())
    n2 = int((severity == 2).sum())
    n3 = int((severity == 3).sum())
    return {
        "date": date,
        "water_pixels": n_water,
        "valid_pixels": n_valid,
        "bloom_pixels": n_bloom,
        "bloom_coverage_pct": round(100.0 * n_bloom / n_water, 3) if n_water else 0.0,
        "mild_pct": round(100.0 * n1 / n_water, 3) if n_water else 0.0,
        "moderate_pct": round(100.0 * n2 / n_water, 3) if n_water else 0.0,
        "severe_pct": round(100.0 * n3 / n_water, 3) if n_water else 0.0,
        "mean_exg_bloom": round(float(exg[bloom].mean()), 1) if n_bloom else 0.0,
        "max_exg_bloom": round(float(exg[bloom].max()), 1) if n_bloom else 0.0,
        "n_patches": len(patches),
        "largest_patch_area_km2": round(patches[0]["area"] * 20 * 20 / 1e6, 3) if patches else 0.0,
    }


def write_report(rows, path):
    fields = ["date", "source", "water_pixels", "valid_pixels", "bloom_pixels", "bloom_coverage_pct",
              "mild_pct", "moderate_pct", "severe_pct", "mean_exg_bloom", "max_exg_bloom",
              "n_patches", "largest_patch_area_km2"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(rows):
    print(f"{'日期':<10}{'蓝藻覆盖率%':>10}{'轻度%':>8}{'中度%':>8}{'重度%':>8}{'斑块数':>7}{'最大斑块km2':>12}{'平均ExG':>9}")
    for r in sorted(rows, key=lambda x: -x["bloom_coverage_pct"]):
        print(f"{r['date']:<10}{r['bloom_coverage_pct']:>10.2f}{r['mild_pct']:>8.2f}{r['moderate_pct']:>8.2f}"
              f"{r['severe_pct']:>8.2f}{r['n_patches']:>7}{r['largest_patch_area_km2']:>12.2f}{r['mean_exg_bloom']:>9.1f}")
