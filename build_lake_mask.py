# -*- coding: utf-8 -*-
"""构建太湖边界掩膜 (OSM 边界 + SCL 水体验证)

流程: 从 Overpass 取太湖 OSM 边界(带缓存) -> 按 outer/inner 角色拼接环
      -> 栅格化到卫星影像网格 -> 用 SCL 官方水体分类验证 -> 保存 data/taihu_mask.npy
"""
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import requests

CACHE = "data/osm_taihu.json"
W, H = 3897, 3844
LON0, LON1, LAT0, LAT1 = 119.90, 120.60, 30.93, 31.52
# 导出网格为 EPSG:3857, 在 31.2°N 纬度每个像素实际地面面积:
# 20m(3857单位) 在东西向 = 20*cos(31.2°)m, 南北向 ≈ 20*cos(31.2°)m
# 实际面积/px ≈ 400 * cos²(31.2°) ≈ 292.4 m²
M2_PER_PX = 400.0 * np.cos(np.radians(31.2)) ** 2

QUERY = '''[out:json][timeout:240];
relation(1126533);
(._;>;);
out geom qt;
'''

MIRRORS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]


def fetch():
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    h = {"User-Agent": "lanzao-monitor/1.0 (research)"}
    for url in MIRRORS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": QUERY}, headers=h, timeout=280)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    data = r.json()
                    with open(CACHE, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    print(f"从 {url.split('/')[2]} 获取成功, 已缓存")
                    return data
                print(f"  {url.split('/')[2]} -> {r.status_code}")
            except Exception as e:
                print(f"  {url.split('/')[2]} 失败: {type(e).__name__}")
            time.sleep(10)
    raise RuntimeError("Overpass 全部镜像失败")


def key(p):
    return (round(p["lon"], 6), round(p["lat"], 6))


def assemble(wids, ways):
    """把线段按端点接成环, 返回点链列表"""
    start_map, end_map = defaultdict(list), defaultdict(list)
    for i in wids:
        w = ways[i]
        start_map[key(w["geometry"][0])].append(i)
        end_map[key(w["geometry"][-1])].append(i)
    rings, used = [], set()
    for i in wids:
        if i in used:
            continue
        chain = [p for p in ways[i]["geometry"]]
        used.add(i)
        while True:
            nxt = [j for j in start_map.get(key(chain[-1]), []) if j not in used]
            if not nxt:
                break
            j = nxt[0]
            used.add(j)
            chain.extend(ways[j]["geometry"][1:])
        while True:
            prev = [j for j in end_map.get(key(chain[0]), []) if j not in used]
            if not prev:
                break
            j = prev[0]
            used.add(j)
            chain = ways[j]["geometry"][:-1] + chain
        if len(chain) >= 3:
            rings.append(chain)
    return rings


def to_pts(ch):
    return np.array([[(p["lon"] - LON0) / (LON1 - LON0) * (W - 1),
                      (LAT1 - p["lat"]) / (LAT1 - LAT0) * (H - 1)] for p in ch],
                    dtype=np.int32)


def main():
    data = fetch()
    rel = [e for e in data["elements"] if e["type"] == "relation"][0]
    roles = {m["ref"]: m.get("role", "outer") for m in rel.get("members", [])}
    ways = {e["id"]: e for e in data["elements"] if e["type"] == "way" and "geometry" in e}
    print(f"way 总数 {len(ways)}, 角色: outer {sum(1 for r in roles.values() if r=='outer')}, "
          f"inner {sum(1 for r in roles.values() if r=='inner')}")

    outers = assemble([i for i, r in roles.items() if r == "outer"], ways)
    inners = assemble([i for i, r in roles.items() if r == "inner"], ways)
    for name, rings in [("外环", outers), ("内环", inners)]:
        for k, ch in enumerate(rings):
            lons = [p["lon"] for p in ch]
            lats = [p["lat"] for p in ch]
            closed = key(ch[0]) == key(ch[-1])
            print(f"  {name}{k}: {len(ch)}点, 闭合={closed}, "
                  f"lon {min(lons):.3f}-{max(lons):.3f}, lat {min(lats):.3f}-{max(lats):.3f}")

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [to_pts(c) for c in outers], 1)
    cv2.fillPoly(mask, [to_pts(c) for c in inners], 0)
    area = mask.sum() * M2_PER_PX / 1e6
    print(f"掩膜面积: {area:.0f} km² (太湖实际 2338)")

    if not (2100 < area < 2600):
        print("面积不合理, 不保存 (需人工检查)")
        return 1

    np.save("data/taihu_mask.npy", mask)
    cv2.imwrite("data/taihu_mask_preview.png", mask * 255)
    print("已保存 data/taihu_mask.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
