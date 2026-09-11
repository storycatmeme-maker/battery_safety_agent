"""
adapt_vendor_csv.py — 任意厂商 BMS 导出 CSV -> 项目标准长表
============================================================
真实项目里，换一个客户/换一套 BMS，导出的列名、单位、时间格式都不一样。
本项目把“格式差异”隔离在这一层：只要跑一次适配器，后面的
preprocessor / feature_engine / anomaly_model **一行都不用改**。

用法示例：
  python scripts/adapt_vendor_csv.py --input vendor.csv --output std.csv \
      --voltage-col Voltage_mV --voltage-scale 0.001 \
      --current-col Current_mA --current-scale 0.001

不带参数时会自动按关键词猜测列名，并打印猜到的映射供确认。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 关键词表：越靠前优先级越高
KEYWORDS = {
    "timestamp": ["timestamp", "time_utc", "datetime", "time", "时间", "日期"],
    "cell_id": ["cell_code", "cell_id", "cellid", "cell", "单体", "电芯"],
    "voltage": ["voltage_mv", "voltage", "volt", "v_mv", "电压"],
    "current": ["current_ma", "current", "curr", "i_ma", "电流"],
    "temp": ["temperature_c", "temperature", "temp", "温度"],
}
REQUIRED = ["timestamp", "cell_id", "voltage", "current", "temp"]


def _pick_column(columns, field):
    """按关键词在列名里找最匹配的一列。"""
    low = {c: str(c).strip().lower() for c in columns}
    for kw in KEYWORDS[field]:
        for c, name in low.items():
            if name == kw:
                return c
    for kw in KEYWORDS[field]:
        for c, name in low.items():
            if kw in name:
                return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="厂商 CSV 适配器")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timestamp-col")
    ap.add_argument("--cell-col")
    ap.add_argument("--voltage-col")
    ap.add_argument("--current-col")
    ap.add_argument("--temp-col")
    ap.add_argument("--voltage-scale", type=float, default=None, help="如 mV->V 填 0.001")
    ap.add_argument("--current-scale", type=float, default=None, help="如 mA->A 填 0.001")
    ap.add_argument("--time-is-absolute", action="store_true",
                    help="时间列是绝对时间；默认转成“相对第一个样本的秒数”")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"输入行数: {len(df):,}；原始列名: {list(df.columns)}")

    mapping = {
        "timestamp": args.timestamp_col or _pick_column(df.columns, "timestamp"),
        "cell_id": args.cell_col or _pick_column(df.columns, "cell_id"),
        "voltage": args.voltage_col or _pick_column(df.columns, "voltage"),
        "current": args.current_col or _pick_column(df.columns, "current"),
        "temp": args.temp_col or _pick_column(df.columns, "temp"),
    }
    print("自动识别到的列映射:", mapping)
    missing = [k for k, v in mapping.items() if v is None or v not in df.columns]
    if missing:
        print(f"❌ 无法识别这些字段: {missing}；请用 --xxx-col 显式指定", file=sys.stderr)
        return 2

    out = pd.DataFrame()
    # 时间：数值直接用；字符串按“相对首样本秒数”转换
    ts_raw = df[mapping["timestamp"]]
    if pd.api.types.is_numeric_dtype(ts_raw):
        out["timestamp"] = ts_raw.astype(float)
    else:
        ts = pd.to_datetime(ts_raw, errors="coerce")
        if args.time_is_absolute:
            out["timestamp"] = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
        else:
            out["timestamp"] = (ts - ts.min()).dt.total_seconds()

    out["cell_id"] = df[mapping["cell_id"]].astype(str).str.strip()
    out["voltage"] = pd.to_numeric(df[mapping["voltage"]], errors="coerce")
    out["current"] = pd.to_numeric(df[mapping["current"]], errors="coerce")
    out["temp"] = pd.to_numeric(df[mapping["temp"]], errors="coerce")

    # 单位自动换算（列名里带 mv/ma 时自动识别，也可用参数强制）
    vs = args.voltage_scale
    if vs is None:
        vs = 0.001 if "mv" in str(mapping["voltage"]).lower() else 1.0
    cs = args.current_scale
    if cs is None:
        cs = 0.001 if "ma" in str(mapping["current"]).lower() else 1.0
    out["voltage"] = out["voltage"] * vs
    out["current"] = out["current"] * cs
    print(f"单位换算: 电压×{vs}  电流×{cs}；输出行数 {len(out):,}")

    out = out.dropna(subset=["timestamp"]).sort_values(["cell_id", "timestamp"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"✅ 已写出标准长表: {args.output}")
    print(out.head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
