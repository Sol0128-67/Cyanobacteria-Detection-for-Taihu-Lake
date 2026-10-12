# -*- coding: utf-8 -*-
"""GEE 全量下载太湖 Sentinel-2 影像 (5天网格, 多日拼接, 完整覆盖)

流程:
1. 从 Sentinel-2 拍到太湖的第一张影像起, 每 5 天一个日期, 直到 2026-09-05
2. 每个网格日期取 ±3 天内所有影像做 mosaic (离目标日期越近越在上层, 云少的优先),
   补全单日瓦片覆盖不到的半边湖 -> 完整全湖
3. SR (地表反射率) 优先, ±3 天内无 SR 才用 L1C (文件名 _l1c 后缀)
4. 分批导出: 每批 15 张 -> 下载到 data/ -> 删除 Drive 文件 (Drive 15GB 限额)
5. 断点续传: 本地已有的日期自动跳过; --recover 恢复 Drive 遗留文件
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta

import requests
import ee

PROJECT = "plated-course-507311-s1"
ee.Initialize(project=PROJECT)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DRIVE_FOLDER = "lan-zao"
WAVE = 15
SCALE = 20
END_DATE = "2026-09-06"
WIN_DAYS = 3  # 拼接窗口 ±3 天

ROI = ee.Geometry.Polygon([[
    [119.90, 30.93],
    [120.60, 30.93],
    [120.60, 31.52],
    [119.90, 31.52],
]])
CENTER = ee.Geometry.Point(120.15, 31.25)


def build_date_grid():
    """最早的太湖 S2 影像日期 -> 每5天一格 -> END_DATE"""
    l1c = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(CENTER)
    first = l1c.sort("system:time_start").first()
    d0 = ee.Date(first.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    dates = []
    d = datetime.strptime(d0, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=5)
    return dates


def fetch_features():
    """一次取回全部影像 (完整资源id, 云量) -> {'SR': [...], 'L1C': [...]}"""
    feats = {}
    for coll, tag in [("COPERNICUS/S2_SR_HARMONIZED", "SR"),
                      ("COPERNICUS/S2_HARMONIZED", "L1C")]:
        col = ee.ImageCollection(coll).filterBounds(ROI).filterDate("2015-06-01", END_DATE)
        ids = col.aggregate_array("system:id").getInfo()   # 完整路径, 可直接 ee.Image()
        clouds = col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        feats[tag] = list(zip(ids, clouds))
    return feats


def local_exists(d8):
    return (os.path.exists(os.path.join(DATA_DIR, f"taihu_{d8}.tif")) or
            os.path.exists(os.path.join(DATA_DIR, f"taihu_{d8}_l1c.tif")))


def build_mosaic(g, feats, tag):
    """取 ±1 天内最接近目标日期的影像日, 当天全部瓦片拼接 (不做跨日补齐, 缺半边就缺半边)"""
    def d8_of(img_id):
        return img_id.split("/")[-1][:8]

    best_diff, best_date = 9, None
    for img_id, _ in feats[tag]:
        d = datetime.strptime(d8_of(img_id), "%Y%m%d")
        diff = abs((d - g).days)
        if diff <= 1 and diff < best_diff:
            best_diff, best_date = diff, d8_of(img_id)
    if best_date is None:
        return None
    day_imgs = [(i, c if c is not None else 99) for i, c in feats[tag]
                if d8_of(i) == best_date]
    day_imgs.sort(key=lambda x: -x[1])     # 云少的排最后 = 拼接时在最上层
    return ee.ImageCollection([ee.Image(i) for i, _ in day_imgs]).mosaic()


def export_image(mosaic, tag):
    """7 层影像: [B4红, B3绿, B2蓝, B8近红外, B5红边, B11短波红外, QA]
    QA 质量层: SR=SCL分类重映射 (1=云/卷云, 2=云影, 3=水体参考, 0=其他)
               L1C=不透明云+卷云掩膜 (1=云/卷云, 0=其他)
    """
    base = mosaic.select(["B4", "B3", "B2", "B8", "B5", "B11"])
    if tag == "SR":
        qa = mosaic.select("SCL").remap([3, 6, 8, 9, 10], [2, 3, 1, 1, 1], 0).rename("QA").toUint16()
    else:
        opaque = mosaic.select("MSK_CLASSI_OPAQUE")
        cirrus = mosaic.select("MSK_CLASSI_CIRRUS")
        qa = opaque.add(cirrus).gt(0).rename("QA").toUint16()
    return base.addBands(qa)


def submit_exports(grid_dates, feats):
    """提交导出任务, 返回 {name: task}"""
    tasks = {}
    for gd in grid_dates:
        g = datetime.strptime(gd, "%Y-%m-%d")
        d8 = g.strftime("%Y%m%d")
        if local_exists(d8):
            continue
        for tag in ("SR", "L1C"):
            mosaic = build_mosaic(g, feats, tag)
            if mosaic is None:
                continue
            suffix = "" if tag == "SR" else "_l1c"
            name = f"taihu_{d8}{suffix}"
            for attempt in range(3):
                try:
                    task = ee.batch.Export.image.toDrive(
                        image=export_image(mosaic, tag),
                        description=name,
                        folder=DRIVE_FOLDER,
                        fileNamePrefix=name,
                        scale=SCALE,
                        crs="EPSG:3857",
                        region=ROI,
                        maxPixels=1e13,
                    )
                    task.start()
                    tasks[name] = task
                    break
                except Exception as e:
                    print(f"    [重试 {attempt + 1}] {name} 提交失败: {type(e).__name__}, 60秒后重试")
                    time.sleep(60)
            break
    return tasks


def wait_tasks(tasks):
    ok, failed = [], []
    pending = dict(tasks)
    last_log = time.time()
    while pending:
        time.sleep(30)
        try:
            for name, task in list(pending.items()):
                state = task.status()["state"]
                if state == "COMPLETED":
                    ok.append(name)
                    del pending[name]
                elif state in ("FAILED", "CANCELLED"):
                    failed.append((name, task.status().get("error_message", "")[:100]))
                    del pending[name]
        except Exception as e:
            print(f"    [网络抖动] 状态查询失败, 60秒后重试: {type(e).__name__}")
            time.sleep(60)
        if time.time() - last_log > 600:
            print(f"    [等待] 剩余 {len(pending)} 个任务 (本批总量 {len(tasks)})")
            last_log = time.time()
    return ok, failed


def drive_token():
    with open(os.path.expanduser("~/.config/earthengine/credentials")) as f:
        creds = json.load(f)
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "grant_type": "refresh_token",
        "client_id": "517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com",
        "client_secret": "RUP0RZ6e0pPhDzsqIJ7KlNd1",
        "refresh_token": creds["refresh_token"],
    }, timeout=60)
    return {"Authorization": "Bearer " + r.json()["access_token"]}


def drive_folder_id(h):
    q = f"name = '{DRIVE_FOLDER}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    folders = requests.get("https://www.googleapis.com/drive/v3/files",
                           headers=h, params={"q": q, "fields": "files(id)"}, timeout=60).json().get("files", [])
    if folders:
        return folders[0]["id"]
    resp = requests.post("https://www.googleapis.com/drive/v3/files", headers=h,
                         json={"name": DRIVE_FOLDER, "mimeType": "application/vnd.google-apps.folder"},
                         timeout=60).json()
    return resp["id"]


def drive_download(h, folder_id, names):
    """下载到本地, 返回 {name: file_id}"""
    got = {}
    for name in names:
        q = f"'{folder_id}' in parents and name contains '{name}' and trashed = false"
        resp = requests.get("https://www.googleapis.com/drive/v3/files",
                            headers=h, params={"q": q, "fields": "files(id,name,modifiedTime)",
                                               "orderBy": "modifiedTime desc"}, timeout=60)
        if resp.status_code == 401:
            raise RuntimeError("Drive 令牌过期 (401)")
        resp.raise_for_status()
        files = resp.json().get("files", [])
        if not files:
            print(f"    [缺失] Drive 中找不到 {name}")
            continue
        fid = files[0]["id"]
        for attempt in range(3):
            try:
                content = requests.get(f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media",
                                       headers=h, timeout=600).content
                with open(os.path.join(DATA_DIR, name + ".tif"), "wb") as f:
                    f.write(content)
                got[name] = fid
                break
            except Exception as e:
                print(f"    [重试 {attempt + 1}] {name} 下载失败: {type(e).__name__}, 60秒后重试")
                time.sleep(60)
    return got


def drive_delete(h, file_ids):
    for fid in file_ids:
        requests.delete(f"https://www.googleapis.com/drive/v3/files/{fid}", headers=h, timeout=60)


def recover_drive():
    """把 Drive lan-zao 文件夹里已有的导出文件全部拉回本地并删除"""
    print("[恢复] 列出 Drive 中已有文件 ...")
    h = drive_token()
    folder_id = drive_folder_id(h)

    files, token = [], None
    while True:
        params = {"q": f"'{folder_id}' in parents and trashed = false",
                  "fields": "nextPageToken,files(id,name,modifiedTime)", "pageSize": 1000}
        if token:
            params["pageToken"] = token
        resp = requests.get("https://www.googleapis.com/drive/v3/files",
                            headers=h, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        token = data.get("nextPageToken")
        if not token:
            break

    files.sort(key=lambda x: (x["name"], x["modifiedTime"]), reverse=True)
    seen, uniq = set(), []
    for f in files:
        if f["name"] not in seen:
            seen.add(f["name"])
            uniq.append(f)
    print(f"[恢复] Drive 中有 {len(files)} 个文件 (去重后 {len(uniq)} 个), 开始下载 ...")

    got, skipped = 0, 0
    for f in uniq:
        local = os.path.join(DATA_DIR, f["name"])
        if os.path.exists(local):
            requests.delete(f"https://www.googleapis.com/drive/v3/files/{f['id']}",
                            headers=h, timeout=60)
            skipped += 1
            continue
        for attempt in range(3):
            try:
                content = requests.get(
                    f"https://www.googleapis.com/drive/v3/files/{f['id']}?alt=media",
                    headers=h, timeout=600).content
                with open(local, "wb") as fh:
                    fh.write(content)
                requests.delete(f"https://www.googleapis.com/drive/v3/files/{f['id']}",
                                headers=h, timeout=60)
                got += 1
                break
            except Exception as e:
                print(f"    [重试 {attempt + 1}] {f['name']}: {type(e).__name__}")
                h = drive_token()
                time.sleep(30)
        if got % 10 == 0 and got:
            print(f"    [恢复] 已下载 {got}, 跳过 {skipped}, 剩余 {len(uniq) - got - skipped}")
    print(f"[恢复] 完成: 下载 {got} 张到 data/, 跳过 {skipped} 张 (本地已有)")
    return 0


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if "--recover" in sys.argv:
        return recover_drive()

    print("[1/3] 构建 5 天日期网格 ...")
    grid = build_date_grid()
    print(f"  第一张影像: {grid[0]}, 共 {len(grid)} 个网格日期")

    print("[2/3] 获取全部影像特征 ...")
    feats = fetch_features()
    print(f"  SR 影像 {len(feats['SR'])} 景, L1C 影像 {len(feats['L1C'])} 景")

    todo = [gd for gd in grid if not local_exists(datetime.strptime(gd, "%Y-%m-%d").strftime("%Y%m%d"))]
    print(f"  待下载: {len(todo)} 个日期, 本地已有: {len(grid) - len(todo)}")

    if not todo:
        print("没有需要下载的")
        return 0

    print(f"[3/3] 分批下载 (每批 {WAVE} 张, 当天瓦片拼接) ...")
    done, failed_all = 0, []
    while todo:
        h = drive_token()          # 每批刷新令牌 (1小时过期)
        folder_id = drive_folder_id(h)
        wave = todo[:WAVE]
        tasks = submit_exports(wave, feats)
        if tasks:
            ok, failed = wait_tasks(tasks)
            h = drive_token()          # 下载前刷新 (等待导出可能超过1小时)
            folder_id = drive_folder_id(h)
            try:
                got = drive_download(h, folder_id, ok)
            except RuntimeError as e:
                print(f"    [令牌过期] {e}, 刷新后重试一次")
                time.sleep(30)
                h = drive_token()
                got = drive_download(h, folder_id, ok)
            if got:
                drive_delete(h, list(got.values()))
            done += len(got)
            failed_all.extend(failed)
            for name in ok:
                if name not in got:
                    print(f"    [警告] 导出完成但下载失败: {name}")
        else:
            print(f"    [跳过] 本批 {len(wave)} 个日期无可用数据")
        todo = todo[WAVE:]
        print(f"  进度: 已下载 {done}, 剩余 {len(todo)} 个日期, 失败 {len(failed_all)}")

    print(f"完成: 共下载 {done} 张到 {DATA_DIR}")
    if failed_all:
        print("失败清单:")
        for name, err in failed_all:
            print(f"  {name}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
