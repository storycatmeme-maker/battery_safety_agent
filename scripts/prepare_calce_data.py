"""
prepare_calce_data.py — CALCE（马里兰大学）电池数据 -> 项目标准长表
====================================================================
CALCE CS2 数据集与 NASA 完全不同：
  1. 每个电芯被拆成 20+ 个 Excel 文件（按测试日期分），内部是 Arbin 测试报告格式
  2. 时间序列在 'Channel_*' 工作表里，列名是 Test_Time(s)/Current(A)/Voltage(V)/Cycle_Index
  3. 没有单体温度通道（真实数据的常见缺失）

本脚本负责把这些差异全部吸收掉，输出项目统一的
['timestamp','cell_id','voltage','current','temp'] 长表，
后面 preprocessor / feature_engine / anomaly_model 一行都不用改。

用法：
  python scripts/prepare_calce_data.py --root data/raw/calce --cells CS2_35 CS2_36 \
      --out data/processed/calce_pack.csv --ambient-temp 25.0
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _find_channel_sheet(xl: pd.ExcelFile) -> str:
    for name in xl.sheet_names:
        if "channel" in name.lower():
            return name
    raise ValueError(f"未找到 Channel 工作表，实际有: {xl.sheet_names}")


def _cell_dir(root: Path, cell: str) -> Path:
    """兼容两种目录结构：root/CS2_35/*.xlsx 或 root/CS2_35/CS2_35/*.xlsx"""
    direct = root / cell
    if glob.glob(str(direct / "*.xlsx")):
        return direct
    nested = direct / cell
    if glob.glob(str(nested / "*.xlsx")):
        return nested
    # 兜底：递归找
    hits = glob.glob(str(direct / "**" / "*.xlsx"), recursive=True)
    if hits:
        return Path(hits[0]).parent
    raise FileNotFoundError(f"在 {direct} 下找不到 .xlsx 文件")


def load_cell(cell: str, root: Path, ambient_temp: float, grid_s: float = 0.0) -> pd.DataFrame:
    """把一颗电芯的多个 xlsx 合并成连续时间序列。"""
    cdir = _cell_dir(root, cell)
    files = sorted(glob.glob(str(cdir / "*.xlsx")))
    print(f"  {cell}: {len(files)} 个 Excel 文件")

    parts = []
    offset = 0.0
    cycle_offset = 0.0   # ★ 每个 Excel 文件的 Cycle_Index 都从 1 重新计数，需累计成全局循环数
    for f in files:
        xl = pd.ExcelFile(f)
        sheet = _find_channel_sheet(xl)
        x = xl.parse(sheet)
        x = x.dropna(subset=["Test_Time(s)", "Voltage(V)", "Current(A)"])
        if x.empty:
            continue
        t = pd.to_numeric(x["Test_Time(s)"], errors="coerce").to_numpy(dtype=float)
        # 每个文件的时间从 0/30 秒开始 -> 拼到全局连续时间轴上
        t0 = float(np.nanmin(t))
        t_rel = t - t0 + offset
        offset = float(np.nanmax(t_rel))
        cyc_local = pd.to_numeric(x.get("Cycle_Index"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        parts.append(pd.DataFrame({
            "timestamp": t_rel,
            "cell_id": cell,
            "voltage": pd.to_numeric(x["Voltage(V)"], errors="coerce").to_numpy(dtype=float),
            "current": pd.to_numeric(x["Current(A)"], errors="coerce").to_numpy(dtype=float),
            "temp": np.full(len(x), float(ambient_temp)),   # 无温度通道 -> 恒温占位
            "cycle": cyc_local + cycle_offset,             # 全局累计循环数
            "source_file": os.path.basename(f),
        }))
        cycle_offset += float(np.nanmax(cyc_local)) if np.isfinite(np.nanmax(cyc_local)) else 0.0
    if not parts:
        raise ValueError(f"{cell}: 没有解析到任何数据")
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out = out.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    # ★ 对齐到公共时间网格
    #   不同电芯的采样时刻存在微小差异（如 1020.484s vs 1020.520s），
    #   若不对齐，跨电芯的“同一时刻比较”会退化成单电芯、ΔV 恒为 0。
    ts_arr = out["timestamp"].to_numpy(dtype=float)
    dt_med = float(np.median(np.diff(ts_arr))) if len(ts_arr) > 1 else 1.0
    step = float(grid_s) if grid_s and grid_s > 0 else max(round(dt_med, 3), 0.001)
    grid = np.arange(0.0, float(ts_arr.max()) + step, step)
    cyc = pd.to_numeric(out["cycle"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    res = pd.DataFrame({
        "timestamp": np.round(grid, 3),
        "cell_id": cell,
        "voltage": np.interp(grid, ts_arr, out["voltage"].to_numpy(dtype=float)),
        "current": np.interp(grid, ts_arr, out["current"].to_numpy(dtype=float)),
        "temp": np.full(len(grid), float(ambient_temp)),
        "cycle": np.interp(grid, ts_arr, cyc),
    })
    print(f"    -> 原始 {len(out):,} 行，对齐到 {step:.1f}s 网格后 {len(res):,} 行；"
          f"时间跨度 {res['timestamp'].max()/3600:.1f} 小时；"
          f"电压 {res['voltage'].min():.3f}~{res['voltage'].max():.3f} V；"
          f"电流 {res['current'].min():.2f}~{res['current'].max():.2f} A")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="CALCE -> 标准长表")
    ap.add_argument("--root", default=str(ROOT / "data" / "raw" / "calce"))
    ap.add_argument("--cells", nargs="+", required=True, help="电芯编号，如 CS2_35 CS2_36")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "calce_pack.csv"))
    ap.add_argument("--grid-s", type=float, default=0.0,
                    help="公共时间网格步长[s]；0=自动按采样周期推断")
    ap.add_argument("--ambient-temp", type=float, default=25.0,
                    help="该数据集无温度通道，用此环境温度填充（默认 25°C）")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"读取 CALCE 数据（{len(args.cells)} 颗电芯）...")
    frames = []
    for cell in args.cells:
        try:
            frames.append(load_cell(cell, root, args.ambient_temp, grid_s=args.grid_s))
        except Exception as exc:
            print(f"  [warn] {cell} 读取失败: {exc}", file=sys.stderr)
    if not frames:
        print("❌ 没有读到任何电芯数据", file=sys.stderr)
        return 1

    pack = pd.concat(frames, ignore_index=True)
    pack = pack.sort_values(["timestamp", "cell_id"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pack[["timestamp", "cell_id", "voltage", "current", "temp", "cycle"]].to_csv(out_path, index=False)
    print(f"\n✅ 已写出标准长表: {out_path}")
    print(f"   总行数 {len(pack):,}；电芯 {sorted(pack['cell_id'].unique())}")
    if args.ambient_temp:
        print(f"   ⚠️ 注意：该数据集无单体温度通道，temp 列用恒定 {args.ambient_temp}°C 占位；")
        print("      温升类特征（dT/dt）将失效，一致性/压差分析仍然有效。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
