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


def load_cell(cell: str, root: Path, ambient_temp: float) -> pd.DataFrame:
    """把一颗电芯的多个 xlsx 合并成连续时间序列。"""
    cdir = _cell_dir(root, cell)
    files = sorted(glob.glob(str(cdir / "*.xlsx")))
    print(f"  {cell}: {len(files)} 个 Excel 文件")

    parts = []
    offset = 0.0
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
        parts.append(pd.DataFrame({
            "timestamp": t_rel,
            "cell_id": cell,
            "voltage": pd.to_numeric(x["Voltage(V)"], errors="coerce").to_numpy(dtype=float),
            "current": pd.to_numeric(x["Current(A)"], errors="coerce").to_numpy(dtype=float),
            "temp": np.full(len(x), float(ambient_temp)),   # 无温度通道 -> 恒温占位
            "cycle": pd.to_numeric(x.get("Cycle_Index"), errors="coerce").to_numpy(dtype=float),
            "source_file": os.path.basename(f),
        }))
    if not parts:
        raise ValueError(f"{cell}: 没有解析到任何数据")
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"    -> {len(out):,} 行，时间跨度 {out['timestamp'].max()/3600:.1f} 小时，"
          f"电压 {out['voltage'].min():.3f}~{out['voltage'].max():.3f} V，"
          f"电流 {out['current'].min():.2f}~{out['current'].max():.2f} A")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CALCE -> 标准长表")
    ap.add_argument("--root", default=str(ROOT / "data" / "raw" / "calce"))
    ap.add_argument("--cells", nargs="+", required=True, help="电芯编号，如 CS2_35 CS2_36")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "calce_pack.csv"))
    ap.add_argument("--ambient-temp", type=float, default=25.0,
                    help="该数据集无温度通道，用此环境温度填充（默认 25°C）")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"读取 CALCE 数据（{len(args.cells)} 颗电芯）...")
    frames = []
    for cell in args.cells:
        try:
            frames.append(load_cell(cell, root, args.ambient_temp))
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
