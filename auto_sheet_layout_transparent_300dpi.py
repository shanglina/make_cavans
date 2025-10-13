#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_canvases_from_sizefolders.py  (2025-09-14, MaxRects[BSSF+BAF]+Skyline, TIFF, grouped preview, count-check)

更新要点
- 大画布输出：TIFF（tiff_deflate）
- 利用率提升：高度二分搜索 + 多排序策略 + 随机重启；
  同时尝试 MaxRects 两种启发式：BSSF（Best Short Side Fit）、BAF（Best Area Fit）
  并保留 Skyline Bottom-Left（自研实现）作为补充
- 固定尺寸智能方向判定（“后者”方案）：
  * 对 A*B（cm）固定尺寸，读取原图宽高比；在 A×B 与 B×A 中选择误差更小的方向；
  * 可选“锁死方向”避免装箱旋转导致宽高颠倒（默认锁死）
  配置：
    "fixed_orient_by_image": true,
    "fixed_lock_rotate": true
- 标注图：仅用预览图，底部 1/5 高度标注带；按 SKU 分组尽量排在一起（无视利用率）
- 数量表：优先 Handle 键（若存在），再回退 SKU
- 日志结尾输出：表格 count 总和 / 磁盘可命中数量 / 实际装箱数量 + 缺失明细（前若干）
"""

import argparse
import json
import re
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# 允许超大图（防 Pillow 解压炸弹告警）
Image.MAX_IMAGE_PIXELS = None

from rectpack import newPacker, PackingMode, PackingBin
# MaxRects 两种启发式：BSSF / BAF（如 BAF 不可用则回退 BSSF）
try:
    from rectpack.maxrects import MaxRectsBssf, MaxRectsBaf
    HAS_BAF = True
except Exception:
    from rectpack.maxrects import MaxRectsBssf
    MaxRectsBaf = MaxRectsBssf  # 回退
    HAS_BAF = False

# ============ 日志 ============
LOG_FILE = Path("run.log")
def log(msg: str):
    line = str(msg)
    print(line, file=sys.stdout, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

# ============ 常量 ============
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}

# ============ 单位换算 ============
def cm_to_px(cm: float, dpi: int) -> int:
    return max(1, int(round(cm * dpi / 2.54)))

def mm_to_px(mm: float, dpi: int) -> int:
    return max(1, int(round(mm * dpi / 25.4)))

def px_to_cm(px: int, dpi: int) -> float:
    return float(px) * 2.54 / float(dpi)

def fmt_cm(v: float, decimals: int = 1) -> str:
    s = f"{v:.{decimals}f}"
    return s.rstrip("0").rstrip(".")

# ============ 配置 ============
def load_config_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============ 尺寸规范化 ============
def _to_int_token(x: str) -> Optional[int]:
    m = re.search(r"(\d+(?:\.\d+)?)", x)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None

def normalize_size_key(raw: str) -> str:
    """"宽12/12cm宽/w12"→w12；"高24/h24"→h24；"13*5/13x5/13×5"→13*5"""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    s = s.replace("×", "x").replace("：", ":").replace(" ", "")
    s = s.replace("厘米", "cm")

    s_pair = re.sub(r"(宽|高|width|height)", "", s).replace("x", "*")
    m_pair = re.fullmatch(r"(\d+(?:\.\d+)?)(?:cm)?\*(\d+(?:\.\d+)?)(?:cm)?", s_pair)
    if m_pair:
        a = int(float(m_pair.group(1))); b = int(float(m_pair.group(2)))
        if a > 0 and b > 0: return f"{a}*{b}"

    mw = re.fullmatch(r"(?:w|width|宽)\s*(\d+(?:\.\d+)?)(?:cm)?", s)
    if mw:
        v = int(float(mw.group(1)));  return f"w{v}" if v > 0 else ""
    mh = re.fullmatch(r"(?:h|height|高)\s*(\d+(?:\.\d+)?)(?:cm)?", s)
    if mh:
        v = int(float(mh.group(1)));  return f"h{v}" if v > 0 else ""

    if s.startswith(("宽", "w")):
        v = _to_int_token(s);  return f"w{v}" if v else ""
    if s.startswith(("高", "h")):
        v = _to_int_token(s);  return f"h{v}" if v else ""
    return ""

def parse_size_key_to_rule(key: str) -> Tuple[str, Optional[float], Optional[float]]:
    """返回 (mode, w_cm, h_cm) 其中 mode: 'w'/'h'/'fixed'/'raw'"""
    if not key:
        return "raw", None, None
    nm = key.strip().lower()
    if ("*" in nm) or ("x" in nm):
        sep = "*" if "*" in nm else "x"
        parts = nm.split(sep, 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return "fixed", float(parts[0]), float(parts[1])
    if nm.startswith("w") and nm[1:].isdigit():
        return "w", float(nm[1:]), None
    if nm.startswith("h") and nm[1:].isdigit():
        return "h", None, float(nm[1:])
    return "raw", None, None

# ============ 数量表（优先 Handle 键） ============
def load_qty_table(path: Path, *, csv_encoding: str = "utf-8") -> pd.DataFrame:
    """读取 handle/SKU, size, count；规范化 size 为 size_key，按 (sku,size_key) 聚合 count。
       当 Handle 与 SKU 同时存在时，优先 Handle（与图片 stem 一致）"""
    if not path.exists():
        raise FileNotFoundError(f"未找到数量表：{path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, encoding=csv_encoding)
    else:
        raise ValueError("数量表仅支持 .xlsx / .xls / .csv")

    cols_map = {str(c).strip().lower(): c for c in df.columns}

    # 优先顺序：handle > sku > 其他别名
    sku_key_candidates   = [k for k in ("handle", "sku", "款号", "商品编码", "图片名称", "图片名") if k in cols_map]
    size_key_candidates  = [k for k in ("size", "尺寸") if k in cols_map]
    count_key_candidates = [k for k in ("count", "数量", "qty", "quantity", "件数") if k in cols_map]

    sku_col  = cols_map[sku_key_candidates[0]]  if sku_key_candidates else None
    size_col = cols_map[size_key_candidates[0]] if size_key_candidates else None
    cnt_col  = cols_map[count_key_candidates[0]] if count_key_candidates else None

    if sku_col is None or size_col is None or cnt_col is None:
        raise KeyError(
            f"数量表缺少必要列。需包含：Handle 或 SKU、size/尺寸、count/数量/qty/quantity/件数；当前列：{list(df.columns)}"
        )

    def norm_key(x: str) -> str:
        s = str(x).strip()
        if "." in s:
            s = s.rsplit(".", 1)[0]
        return s.lower()

    def safe_int(x):
        try:
            return max(0, int(float(x)))
        except Exception:
            return 0

    out = pd.DataFrame({
        "sku": df[sku_col].map(norm_key),
        "size_key": df[size_col].map(lambda v: normalize_size_key(str(v)) if pd.notna(v) else ""),
        "count": df[cnt_col].map(lambda v: safe_int(v) if pd.notna(v) else 0)
    })
    out = out.groupby(["sku", "size_key"], dropna=False, as_index=False)["count"].sum()

    log(f"🧾 数量表键列使用：{sku_col}（候选顺序：Handle » SKU » 其他别名）")
    try:
        log(f"🔍 数量表预览前8：{out.head(8).to_dict(orient='records')}")
    except Exception:
        pass
    return out

def get_copies_for(sku: str, folder_size_key: str, qty_df: pd.DataFrame, default_count: int) -> int:
    sku_l = sku.lower()
    df_sku = qty_df[qty_df["sku"] == sku_l]
    if df_sku.empty:
        return default_count
    df_match = df_sku[df_sku["size_key"] == folder_size_key]
    if not df_match.empty:
        return int(df_match["count"].sum())
    return default_count

# ============ 数据结构 ============
@dataclass
class Item:
    path: Path
    w: int
    h: int
    allow_rotate: bool
    sku: str
    copies_label: Optional[int]
    folder_size_key: str
    order_key: Tuple[int, str]

# ============ 排序策略 ============
def sort_for_pack(items: List[Item], strategy: str) -> List[Item]:
    if strategy == "area_desc":
        return sorted(items, key=lambda it: it.w * it.h, reverse=True)
    if strategy == "maxside_desc":
        return sorted(items, key=lambda it: max(it.w, it.h), reverse=True)
    if strategy == "height_desc":
        return sorted(items, key=lambda it: it.h, reverse=True)
    if strategy == "width_desc":
        return sorted(items, key=lambda it: it.w, reverse=True)
    if strategy == "shortside_desc":
        return sorted(items, key=lambda it: min(it.w, it.h), reverse=True)
    if strategy == "name":
        return sorted(items, key=lambda it: it.sku.lower())
    return items[:]  # 'none' / unknown

# ============ MaxRects 工具 ============
def _get_maxrects_algo(algo: str):
    if algo.upper() == "BAF" and HAS_BAF:
        return MaxRectsBaf
    return MaxRectsBssf  # 默认/回退

def try_pack_maxrects_height(items: List[Item], sheet_w_px: int, sheet_h_px: int,
                             margin_px: int, gutter_px: int, *, algo: str = "BSSF"):
    usable_w = max(1, sheet_w_px - 2 * margin_px)
    usable_h = max(1, sheet_h_px - 2 * margin_px)

    pack_algo = _get_maxrects_algo(algo)
    packer = newPacker(mode=PackingMode.Offline, pack_algo=pack_algo, bin_algo=PackingBin.BFF, rotation=True)
    packer.add_bin(width=usable_w, height=usable_h, count=1)
    for i, it in enumerate(items):
        packer.add_rect(it.w + gutter_px, it.h + gutter_px, i)
    packer.pack()

    try:
        rects_out = packer.rect_list()
    except Exception:
        rects_out = packer[0].rect_list()

    if len(rects_out) < len(items):
        return None, False

    placements = []
    for tpl in rects_out:
        if len(tpl) == 6:
            _b, x, y, w, h, rid = tpl
        elif len(tpl) == 5:
            x, y, w, h, rid = tpl
        else:
            x, y, w, h = tpl[:4]; rid = tpl[4] if len(tpl) > 4 else 0

        it = items[int(rid)]
        rotated = False
        # 注意：固定尺寸若锁死方向，则 allow_rotate 为 False，不做旋转判断
        if it.allow_rotate and (w == it.h + gutter_px) and (h == it.w + gutter_px):
            rotated = True; place_w, place_h = it.h, it.w
        else:
            place_w, place_h = it.w, it.h
        place_x = margin_px + x + gutter_px // 2
        place_y = margin_px + y + gutter_px // 2
        placements.append({
            "path": it.path, "x": place_x, "y": place_y,
            "w": place_w, "h": place_h, "rotated": rotated,
            "sku": it.sku, "copies_label": it.copies_label,
            "folder_size_key": it.folder_size_key
        })
    return [{"placements": placements}], True

# ============ Skyline Bottom-Left ============
class Skyline:
    def __init__(self, usable_w: int, usable_h: int):
        self.usable_w = usable_w
        self.usable_h = usable_h
        self.nodes = [(0, 0, usable_w)]  # (x,y,width)

    def _fits_at(self, idx: int, w: int, h: int) -> Optional[int]:
        x, y, width = self.nodes[idx]
        if w > width:
            return None
        cur_x = x
        top_y = y
        i = idx
        remaining_w = w
        while remaining_w > 0:
            if i >= len(self.nodes):
                return None
            nx, ny, nw = self.nodes[i]
            if ny > top_y:
                top_y = ny
            if top_y + h > self.usable_h:
                return None
            take = min(nw, remaining_w)
            cur_x += take
            remaining_w -= take
            i += 1
        return top_y

    def _add_skyline_level(self, idx: int, x: int, y: int, w: int, h: int):
        new_node = (x, y + h, w)
        self.nodes.insert(idx, new_node)
        i = idx + 1
        while i < len(self.nodes):
            nx, ny, nw = self.nodes[i]
            px, py, pw = self.nodes[i - 1]
            if (py == ny) and (px + pw == nx):
                self.nodes[i - 1] = (px, py, pw + nw)
                self.nodes.pop(i)
            elif (px <= nx) and (nx < px + pw):
                overlap = px + pw - nx
                self.nodes[i] = (nx + overlap, ny, nw - overlap)
                if self.nodes[i][2] <= 0:
                    self.nodes.pop(i)
                else:
                    i += 1
            else:
                i += 1
        self.nodes = [n for n in self.nodes if n[2] > 0]

    def place(self, w: int, h: int) -> Optional[Tuple[int, int, int]]:
        best_y = None
        best_x = None
        best_i = None
        for i in range(len(self.nodes)):
            y = self._fits_at(i, w, h)
            if y is None:
                continue
            x = self.nodes[i][0]
            if (best_y is None) or (y < best_y) or (y == best_y and x < best_x):
                best_y, best_x, best_i = y, x, i
        if best_y is None:
            return None
        return best_x, best_y, best_i

def try_pack_skyline_height(items: List[Item], sheet_w_px: int, sheet_h_px: int, margin_px: int, gutter_px: int):
    usable_w = max(1, sheet_w_px - 2 * margin_px)
    usable_h = max(1, sheet_h_px - 2 * margin_px)

    sk = Skyline(usable_w, usable_h)
    placements = []
    for it in items:
        W = it.w + gutter_px
        H = it.h + gutter_px
        options = []
        res0 = sk.place(W, H)
        if res0 is not None:
            options.append((res0[0], res0[1], res0[2], False, it.w, it.h))
        if it.allow_rotate:
            res1 = sk.place(it.h + gutter_px, it.w + gutter_px)
            if res1 is not None:
                options.append((res1[0], res1[1], res1[2], True, it.h, it.w))

        if not options:
            return None, False

        options.sort(key=lambda t: (t[1], t[0]))
        x, y, at, rotated, pw, ph = options[0]
        sk._add_skyline_level(at, x, y, pw + gutter_px, ph + gutter_px)

        placements.append({
            "path": it.path,
            "x": margin_px + x + gutter_px // 2,
            "y": margin_px + y + gutter_px // 2,
            "w": pw, "h": ph,
            "rotated": rotated,
            "sku": it.sku,
            "copies_label": it.copies_label,
            "folder_size_key": it.folder_size_key
        })

    return [{"placements": placements}], True

# ============ 高度边界 ============
def lower_bound_height_px(items: List[Item], sheet_w_px: int, margin_px: int, gutter_px: int) -> int:
    usable_w = max(1, sheet_w_px - 2 * margin_px)
    total_area = sum((it.w + gutter_px) * (it.h + gutter_px) for it in items)
    area_lb = int((total_area / usable_w) + 0.9999)
    max_item_h = max(it.h for it in items)
    return max(area_lb, max_item_h) + 2 * margin_px

def infinite_height_upper_bound(items: List[Item], sheet_w_px: int, margin_px: int, gutter_px: int):
    usable_w = max(1, sheet_w_px - 2 * margin_px)
    packer = newPacker(mode=PackingMode.Offline, pack_algo=MaxRectsBssf, bin_algo=PackingBin.BFF, rotation=True)
    packer.add_bin(width=usable_w, height=10**9, count=1)
    for i, it in enumerate(items):
        packer.add_rect(it.w + gutter_px, it.h + gutter_px, i)
    packer.pack()
    try:
        rects_out = packer.rect_list()
    except Exception:
        rects_out = packer[0].rect_list()
    used_max_y = 0
    placements = []
    for tpl in rects_out:
        if len(tpl) == 6:
            _b, x, y, w, h, rid = tpl
        elif len(tpl) == 5:
            x, y, w, h, rid = tpl
        else:
            x, y, w, h = tpl[:4]; rid = tpl[4] if len(tpl) > 4 else 0
        it = items[int(rid)]
        rotated = False
        if it.allow_rotate and (w == it.h + gutter_px) and (h == it.w + gutter_px):
            rotated = True; place_w, place_h = it.h, it.w
        else:
            place_w, place_h = it.w, it.h
        place_x = margin_px + x + gutter_px // 2
        place_y = margin_px + y + gutter_px // 2
        used_max_y = max(used_max_y, place_y + place_h)
        placements.append({
            "path": it.path, "x": place_x, "y": place_y,
            "w": place_w, "h": place_h, "rotated": rotated,
            "sku": it.sku, "copies_label": it.copies_label,
            "folder_size_key": it.folder_size_key
        })
    h = used_max_y + margin_px
    return {"placements": placements}, h

# ============ 全局装箱：高度二分 + 多算法/启发式 ============
def pack_global_search(items: List[Item], sheet_w_px: int, margin_px: int, gutter_px: int,
                       allow_rotate: bool, strategies: List[str], try_no_rotate: bool,
                       random_restarts: int, height_iters: int, pack_scale: float):
    """全局高度搜索 + MaxRects(BSSF/BAF) + Skyline + 多策略 + 随机重启"""

    # 可选微缩
    if pack_scale != 1.0:
        scaled = []
        for it in items:
            scaled.append(Item(
                path=it.path,
                w=max(1, int(round(it.w * pack_scale))),
                h=max(1, int(round(it.h * pack_scale))),
                allow_rotate=it.allow_rotate,
                sku=it.sku, copies_label=it.copies_label,
                folder_size_key=it.folder_size_key, order_key=it.order_key
            ))
        items = scaled

    # 旋转/不旋转版本
    items_rot_on  = [Item(**{**it.__dict__, "allow_rotate": allow_rotate}) for it in items]
    items_rot_off = [Item(**{**it.__dict__, "allow_rotate": False}) for it in items]

    # 无限高上界
    sheet0, h_ub0 = infinite_height_upper_bound(items_rot_on, sheet_w_px, margin_px, gutter_px)
    h_lb = lower_bound_height_px(items_rot_on, sheet_w_px, margin_px, gutter_px)
    log(f"🔎 高度搜索区间：LB={h_lb}px  UB0={h_ub0}px  scale={pack_scale}")

    best_sheet = sheet0
    best_h     = h_ub0
    best_meta  = {
        "strategy": "maxrects_inf",
        "allow_rotate": True,
        "util": 0.0,
        "pack_scale": pack_scale,
        "height_lb": h_lb,
        "height_ub0": h_ub0
    }

    # 高度二分搜索
    lo, hi = h_lb, h_ub0
    for _ in range(height_iters):
        mid = (lo + hi) // 2
        success = False
        cand_h = 10**18
        cand_sheet = None
        cand_meta = {}

        def try_family(items_try: List[Item], rot_flag: bool, family_tag: str):
            nonlocal success, cand_h, cand_sheet, cand_meta

            for s in strategies:
                seq = sort_for_pack(items_try, s)

                # --- MaxRects: BSSF + BAF ---
                for algo_name in ("BSSF", "BAF"):
                    sh, ok = try_pack_maxrects_height(seq, sheet_w_px, mid, margin_px, gutter_px, algo=algo_name)
                    if ok and mid < cand_h:
                        success = True; cand_h = mid; cand_sheet = sh[0]
                        cand_meta = {"strategy": f"{family_tag}/maxrects-{algo_name}/{s}", "allow_rotate": rot_flag}

                    for _ in range(random_restarts):
                        ri = seq[:]; random.shuffle(ri)
                        sh, ok = try_pack_maxrects_height(ri, sheet_w_px, mid, margin_px, gutter_px, algo=algo_name)
                        if ok and mid < cand_h:
                            success = True; cand_h = mid; cand_sheet = sh[0]
                            cand_meta = {"strategy": f"{family_tag}/maxrects-{algo_name}/{s}#rnd", "allow_rotate": rot_flag}

                # --- Skyline 固序 + 随机重启 ---
                sh, ok = try_pack_skyline_height(seq, sheet_w_px, mid, margin_px, gutter_px)
                if ok and mid < cand_h:
                    success = True; cand_h = mid; cand_sheet = sh[0]
                    cand_meta = {"strategy": f"{family_tag}/skyline/{s}", "allow_rotate": rot_flag}

                for _ in range(random_restarts):
                    ri = seq[:]; random.shuffle(ri)
                    sh, ok = try_pack_skyline_height(ri, sheet_w_px, mid, margin_px, gutter_px)
                    if ok and mid < cand_h:
                        success = True; cand_h = mid; cand_sheet = sh[0]
                        cand_meta = {"strategy": f"{family_tag}/skyline/{s}#rnd", "allow_rotate": rot_flag}

        # 允许旋转
        try_family(items_rot_on, True, "rot")
        # 禁止旋转对照
        if try_no_rotate:
            try_family(items_rot_off, False, "no-rot")

        if success and cand_sheet is not None:
            hi = mid
            best_sheet = cand_sheet
            best_h = cand_h
            best_meta.update(cand_meta)
            total_area = sum(it.w * it.h for it in items)
            util = min(1.0, float(total_area) / float(max(1, sheet_w_px * best_h)))
            best_meta["util"] = util
            log(f"  ✅ mid={mid} OK → hi={hi} | {best_meta['strategy']} | util≈{util*100:.2f}%")
        else:
            lo = mid + 1
            log(f"  ❌ mid={mid} FAIL → lo={lo}")

        if lo >= hi:
            break

    return best_sheet, best_h, best_meta

# ============ 预览（SKU 分组、尽量相邻） ============
def group_preview_by_sku(items: List[Item]) -> List[Item]:
    buckets: Dict[str, List[Item]] = {}
    for it in items:
        buckets.setdefault(it.sku, []).append(it)
    ordered = []
    for sku in sorted(buckets.keys()):
        group = sorted(buckets[sku], key=lambda x: (x.folder_size_key, x.path.name.lower()))
        ordered.extend(group)
    return ordered

def layout_preview_flow(items: List[Item], sheet_w_px: int, margin_px: int,
                        gutter_px: int, preview_w_px: int, dpi: int) -> Tuple[List[dict], int]:
    usable_w = max(1, sheet_w_px - 2 * margin_px)
    x = y = 0
    row_max_h = 0
    placements: List[dict] = []

    for it in items:
        try:
            with Image.open(it.path) as im:
                ow, oh = im.size
        except Exception:
            ow, oh = it.w, it.h
        if ow <= 0 or oh <= 0:
            continue
        scale = preview_w_px / float(ow)
        img_w = preview_w_px
        img_h = max(1, int(round(oh * scale)))
        annot_h = max(1, int(round(img_h * 0.2)))  # 1/5
        tile_w = img_w
        tile_h = img_h + annot_h

        x_next = tile_w if x == 0 else (x + gutter_px + tile_w)
        if x_next > usable_w:
            y += row_max_h + gutter_px
            x = 0
            row_max_h = 0

        px = margin_px + x
        py = margin_px + y
        placements.append({
            "path": it.path, "x": px, "y": py,
            "img_w": img_w, "img_h": img_h, "annot_h": annot_h,
            "sku": it.sku, "copies_label": it.copies_label,
            "w_cm": fmt_cm(px_to_cm(it.w, dpi)), "h_cm": fmt_cm(px_to_cm(it.h, dpi))
        })

        x = tile_w if x == 0 else (x + gutter_px + tile_w)
        row_max_h = max(row_max_h, tile_h)

    sheet_h_px = margin_px + y + row_max_h + margin_px
    return placements, sheet_h_px

# ============ 渲染 ============
def render_big_tiff(sheet: dict, sheet_w_px: int, sheet_h_px: int, out_path: Path, dpi: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.new("RGBA", (sheet_w_px, sheet_h_px), (0, 0, 0, 0))
    for p in sheet["placements"]:
        with Image.open(p["path"]) as im:
            im = im.convert("RGBA")
            if p.get("rotated"):
                im = im.rotate(90, expand=True)
            im = im.resize((p["w"], p["h"]), Image.LANCZOS)
            base.paste(im, (p["x"], p["y"]), im)
    base.save(out_path, compression="tiff_deflate", dpi=(dpi, dpi))

def render_preview_annot_png(placements: List[dict], sheet_w_px: int, sheet_h_px: int,
                             out_path: Path, annot_bg_alpha: int,
                             label_mode: str, include_size: bool, decimals: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (sheet_w_px, sheet_h_px), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    for p in placements:
        try:
            with Image.open(p["path"]) as im:
                im = im.convert("RGBA")
                im = im.resize((p["img_w"], p["img_h"]), Image.LANCZOS)
        except Exception:
            im = Image.new("RGBA", (p["img_w"], p["img_h"]), (220, 220, 220, 255))
        img.paste(im, (p["x"], p["y"]), im)

        ax0 = p["x"]; ay0 = p["y"] + p["img_h"]
        band = Image.new("RGBA", (p["img_w"], p["annot_h"]), (255, 255, 0, max(0, min(255, annot_bg_alpha))))
        img.paste(band, (ax0, ay0), band)

        if label_mode == "sku_x_count" and p.get("copies_label") is not None:
            text = f"{p['sku']}×{p['copies_label']}"
        else:
            text = p["sku"]
        if include_size:
            text += f" • {p['w_cm']}×{p['h_cm']} cm"

        target_w = int(p["img_w"] * 0.9)
        target_h = int(p["annot_h"] * 0.8)
        for s in range(min(target_h, 64), 9, -2):
            try:
                font = ImageFont.truetype("Arial.ttf", s)
            except Exception:
                font = ImageFont.load_default()
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            except AttributeError:
                tw, th = draw.textsize(text, font=font)
            if tw <= target_w and th <= target_h:
                break
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except AttributeError:
            tw, th = draw.textsize(text, font=font)
        tx = p["x"] + (p["img_w"] - tw)//2
        ty = ay0 + (p["annot_h"] - th)//2
        draw.text((tx, ty), text, font=font, fill=(0, 0, 0, 255))

    img.save(out_path, format="PNG", optimize=True)

# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(description="MaxRects(BSSF+BAF)+Skyline 高度搜索装箱（TIFF）+ SKU 分组预览（PNG）")
    ap.add_argument("--config", default="layout_config.json", help="配置文件（默认 layout_config.json）")
    args = ap.parse_args()
    cfg = load_config_json(Path(args.config).expanduser())

    # 基础配置
    # 生成动态日期字符串，例如 20251007
    dateString = datetime.now().strftime("%Y%m%d")

    # 从配置读取根目录，并拼接日期/out/images
    root_dir = Path(f"{cfg.get('root_dir', './images')}/{dateString}/out/images").expanduser()
    qty_table = Path(f"{cfg.get('qty_table', './result_handles.csv')}/{dateString}/out/finded_handles.xlsx").expanduser()
    csv_encoding  = cfg.get("csv_encoding", "utf-8")
    dpi           = int(cfg.get("dpi", 300))
    canvas_w_cm   = float(cfg.get("canvas_w_cm", 58.0))
    margin_mm     = float(cfg.get("margin_mm", 5.0))
    gutter_mm     = float(cfg.get("gutter_mm", 2.0))
    allow_rotate  = bool(cfg.get("allow_rotate", True))
    default_cnt   = int(cfg.get("default_count", 1))
    # 固定尺寸智能方向配置
    fixed_orient_by_image = bool(cfg.get("fixed_orient_by_image", True))
    fixed_lock_rotate     = bool(cfg.get("fixed_lock_rotate", True))

    # 标注参数
    label_mode           = cfg.get("label_mode", "sku_x_count")
    label_include_size   = bool(cfg.get("label_include_size_cm", True))
    label_decimals       = int(cfg.get("label_decimals", 1))

    # 预览参数
    preview_w_px         = int(cfg.get("preview_w_px", 300))
    preview_gutter_mm    = float(cfg.get("preview_gutter_mm", 2.0))
    annot_bg_alpha       = int(cfg.get("annot_bg_alpha", 170))

    # 输出
    big_canvas           = bool(cfg.get("big_canvas", True))
    allow_huge_image     = bool(cfg.get("allow_huge_image", True))
    output_dir           = Path(cfg.get("output_dir", "_all_sizes_out")).expanduser()

    # 打包增强
    strategies           = cfg.get("pack_strategies", ["area_desc","maxside_desc","height_desc","width_desc","shortside_desc","name","none"])
    try_no_rotate        = bool(cfg.get("try_no_rotate_contrast", True))
    pack_scale           = float(cfg.get("pack_scale", 1.0))
    random_restarts      = int(cfg.get("random_restarts", 10))
    height_search_iters  = int(cfg.get("height_search_iters", 12))

    if not root_dir.exists(): raise SystemExit(f"根目录不存在：{root_dir}")
    if not qty_table.exists(): raise SystemExit(f"数量表不存在：{qty_table}")
    if allow_huge_image: Image.MAX_IMAGE_PIXELS = None

    # 日志头
    LOG_FILE.write_text("", encoding="utf-8")
    log("🚀 启动：加载配置完成")
    log(json.dumps({
        "root_dir": str(root_dir), "qty_table": str(qty_table),
        "dpi": dpi, "canvas_w_cm": canvas_w_cm, "margin_mm": margin_mm,
        "gutter_mm(pack)": gutter_mm, "allow_rotate": allow_rotate,
        "preview_w_px": preview_w_px, "preview_gutter_mm": preview_gutter_mm,
        "pack_scale": pack_scale, "random_restarts": random_restarts,
        "height_search_iters": height_search_iters,
        "strategies": strategies, "try_no_rotate": try_no_rotate,
        "fixed_orient_by_image": fixed_orient_by_image,
        "fixed_lock_rotate": fixed_lock_rotate,
        "has_maxrects_baf": HAS_BAF
    }, ensure_ascii=False))

    # 单位换算
    sheet_w_px     = cm_to_px(canvas_w_cm, dpi)
    margin_px      = mm_to_px(margin_mm, dpi)
    gutter_px_pack = mm_to_px(gutter_mm, dpi)
    gutter_px_prev = mm_to_px(preview_gutter_mm, dpi)
    log(f"📏 画布：{canvas_w_cm}cm → {sheet_w_px}px @ {dpi}DPI; margin={margin_mm}mm; gutter(pack)={gutter_mm}mm({gutter_px_pack}px)")

    # 读取数量表 + 目标数量
    qty_df = load_qty_table(qty_table, csv_encoding=csv_encoding)
    expected_total = int(qty_df["count"].sum())
    log(f"📄 数量表：{len(qty_df)} 行（sku+size 聚合），count 总和 = {expected_total}")

    # —— 建立磁盘索引（size_key -> stems） —— #
    size_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())
    size_dir_index = {d.name: idx for idx, d in enumerate(size_dirs)}
    disk_index: Dict[str, set] = {}
    for size_dir in size_dirs:
        s_key = normalize_size_key(size_dir.name)
        stems = set()
        for p in size_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                stems.add(p.stem.lower())
        disk_index[s_key] = stems

    # 汇总实际可渲染 items（仅磁盘存在的）
    all_items: List[Item] = []
    matched_on_disk_total = 0
    missing_list = []  # (sku,size_key,count)

    for size_dir in size_dirs:
        size_key = normalize_size_key(size_dir.name)
        mode, wcm, hcm = parse_size_key_to_rule(size_key)
        log(f"\n📂 目录: {size_dir.name} → 规范={size_key} 模式={mode} w={wcm} h={hcm}")

        imgs = [p for p in size_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not imgs:
            log("  ⚠️ 无图片，跳过")
            continue

        for p in sorted(imgs, key=lambda x: x.name.lower()):
            sku = p.stem
            try:
                with Image.open(p) as im:
                    im = im.convert("RGBA")
                    ow, oh = im.size
            except UnidentifiedImageError:
                log(f"  ⚠️ 无法识别：{p.name}")
                continue
            except Exception as e:
                log(f"  ⚠️ 读取失败：{p.name} -> {e}")
                continue

            copies = get_copies_for(sku, size_key, qty_df, default_cnt)
            if copies <= 0:
                log(f"  ⏭️ 跳过 SKU={sku} (数量=0)")
                continue

            matched_on_disk_total += copies

            # 目标像素（含固定尺寸智能方向）
            item_allow_rotate = allow_rotate
            if mode == "fixed" and wcm and hcm:
                w0 = cm_to_px(wcm, dpi); h0 = cm_to_px(hcm, dpi)
                if fixed_orient_by_image:
                    r_img = float(ow) / max(1.0, float(oh))
                    r0 = float(w0) / max(1.0, float(h0))
                    r1 = float(h0) / max(1.0, float(w0))  # 交换
                    e0 = abs(r_img - r0) / max(r_img, r0)
                    e1 = abs(r_img - r1) / max(r_img, r1)
                    if e1 < e0:
                        tw, th = h0, w0
                        chosen = "HxW"
                    else:
                        tw, th = w0, h0
                        chosen = "WxH"
                    log(f"    ↪ 固定尺寸定向：{size_key}  原比={r_img:.4f}  "
                        f"候选比(WxH={r0:.4f}, HxW={r1:.4f}) → 选择 {chosen}  "
                        f"误差({e0:.4f}/{e1:.4f})")
                else:
                    tw, th = w0, h0
                if fixed_lock_rotate:
                    item_allow_rotate = False  # 锁死方向
            elif mode == "w" and wcm:
                tw = cm_to_px(wcm, dpi); th = max(1, int(round(tw * (oh / max(1.0, float(ow))))))
            elif mode == "h" and hcm:
                th = cm_to_px(hcm, dpi); tw = max(1, int(round(th * (ow / max(1.0, float(oh))))))
            else:
                tw, th = ow, oh

            log(f"  ✅ {sku}  size={size_key}  count={copies}  target={tw}x{th}px "
                f"({fmt_cm(px_to_cm(tw, dpi))}×{fmt_cm(px_to_cm(th, dpi))}cm) "
                f"{'(lock-orient)' if (mode=='fixed' and fixed_lock_rotate) else ''}")

            for _ in range(copies):
                all_items.append(Item(
                    path=p, w=tw, h=th, allow_rotate=item_allow_rotate,
                    sku=sku, copies_label=copies, folder_size_key=size_key,
                    order_key=(size_dir_index[size_dir.name], p.name.lower())
                ))

    # 统计“表里有但磁盘没有”的缺失
    for _, row in qty_df.iterrows():
        sku_l = str(row["sku"]).lower()
        s_key = str(row["size_key"]).strip().lower()
        cnt = int(row["count"])
        disk_has = (s_key in disk_index) and (sku_l in disk_index.get(s_key, set()))
        if not disk_has and cnt > 0:
            missing_list.append((row["sku"], row["size_key"], cnt))

    if not all_items:
        log("❌ 没有可排布元素（磁盘上没有与数量表匹配的图片），结束")
        log("\n===== 数量对比 =====")
        log(f"表格 count 总和: {expected_total}")
        log(f"磁盘可命中数量: {matched_on_disk_total}")
        log(f"实际装箱数量  : 0")
        if missing_list:
            log(f"📉 缺失条目（表里有但磁盘找不到），示例前20：{missing_list[:20]}")
        return

    # —— 大画布：二分搜索 + 多算法 —— #
    packed_total = 0
    if big_canvas:
        best_sheet, H, meta = pack_global_search(
            all_items, sheet_w_px, margin_px, gutter_px_pack,
            allow_rotate=allow_rotate, strategies=strategies,
            try_no_rotate=try_no_rotate, random_restarts=random_restarts,
            height_iters=height_search_iters, pack_scale=pack_scale
        )
        util_pct = f"{meta.get('util', 0.0)*100:.2f}%"
        log(f"\n🧩 大画布最优：H={H}px  利用率≈{util_pct}  策略={meta.get('strategy')}  "
            f"旋转={meta.get('allow_rotate')}  scale={meta.get('pack_scale')}")
        log(f"    高度区间：LB={meta.get('height_lb')}  UB0={meta.get('height_ub0')}")

        out_dir = (root_dir / output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        big_tif = out_dir / "big_packed_transparent.tif"
        render_big_tiff(best_sheet, sheet_w_px, H, big_tif, dpi)
        log(f"  ✅ 大画布 TIFF：{big_tif}")

        packed_total = len(best_sheet.get("placements", []))

    # —— 标注预览：SKU 分组 + 流式 PNG（只用预览图） —— #
    grouped = group_preview_by_sku(all_items)
    placements, prev_H = layout_preview_flow(grouped, sheet_w_px, margin_px, gutter_px_prev, preview_w_px, dpi)
    out_dir = (root_dir / output_dir)
    annot_png = out_dir / "big_preview_grouped.png"
    render_preview_annot_png(placements, sheet_w_px, prev_H, annot_png,
                             annot_bg_alpha, label_mode, label_include_size, label_decimals)
    log(f"  ✅ 标注预览 PNG（按 SKU 分组）：{annot_png}  (H={prev_H}px，流式)")

    # —— 末尾数量对比 —— #
    log("\n===== 数量对比（严格核对）=====")
    log(f"表格 count 总和: {expected_total}")
    log(f"磁盘可命中数量: {matched_on_disk_total}")
    log(f"实际装箱数量  : {packed_total if big_canvas else 0}")

    if expected_total != matched_on_disk_total or (big_canvas and matched_on_disk_total != packed_total):
        log("⚠️ 数量不一致，请检查：")
        if missing_list:
            log(f"📉 缺失条目（表里有但磁盘找不到），示例前20：{missing_list[:20]}")
    else:
        log("✅ 数量一致（表格总数 = 磁盘可命中 = 实际装箱）")

    log("\n✅ 全部完成")

if __name__ == "__main__":
    main()