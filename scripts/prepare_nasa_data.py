"""
prepare_nasa_data.py — 把 NASA 电池老化原始 .mat 转换为“虚拟电池组”长表 CSV
============================================================================

数据来源：NASA PCoE Battery Aging Data (B. Saha & K. Goebel, 2007)

本脚本把 4 颗真实老化电芯（B0005/B0006/B0007/B0018）按同一循环协议对齐时间轴，
构造成一个“虚拟电池组”，输出字段满足 BatteryPreprocessor 要求的
['timestamp','cell_id','voltage','current','temp']，并附加 cycle / phase / capacity_ah
供 SoH 趋势与“失效提前量验证”使用。

构造说明（诚实披露，面试可讲）：
- 各电芯原始采样时刻不同步。对每个 (循环, 相) 段，取各电芯该段的最大时长，
  建立 5 s 公共时间网格，再把各电芯电压/电流/温度线性插值到该网格上，
  得到“同一时刻的多电芯同步读数”——这正是 BMS 对电池组同步采样的语义。
- 原始数据只记录充/放电过程，段与段之间未记录静置；本脚本不伪造静置数据。
- 公共时间轴从 0 秒单调递增，表示“电池组运行时长 [s]”。

输出：
  data/processed/nasa_pack_full.csv  全部循环（体量较大，供离线分析）
  data/processed/nasa_pack_demo.csv  精选循环子集（供 Streamlit 大屏演示）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

CELLS = ["B0005", "B0006", "B0007", "B0018"]
GRID_STEP_S = 5.0

# 演示子集：健康期 / 中期 / 衰退期 / 失效期 的代表循环
DEMO_CYCLES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    58, 60, 62, 64, 66, 68, 70, 72,
    110, 112, 114, 116, 118, 120, 122, 124,
    148, 152, 156, 160, 164, 168,
]


def _unwrap(a: object) -> np.ndarray:
    """把 MATLAB 嵌套数组解包为一维 float 数组。"""
    raw = np.asarray(a)
    while raw.ndim > 0 and raw.size == 1 and isinstance(raw.ravel()[0], np.ndarray):
        raw = np.asarray(raw.ravel()[0])
    if raw.ndim == 0:
        return np.array([float(raw)])
    return raw.ravel().astype(float)


def _phase_of(cyc: object) -> str:
    raw = np.asarray(cyc["type"]).ravel()
    if len(raw) == 0:
        return ""
    val = raw[0]
    return val.decode() if isinstance(val, bytes) else str(val)


def _data_field(cyc: object, *names: str) -> np.ndarray:
    """从 cyc['data'] 结构化数组里按候选字段名取值。"""
    d = cyc["data"]
    for nm in names:
        try:
            if nm in d.dtype.names:
                return _unwrap(d[nm])
        except Exception:
            continue
    return np.array([])


def _capacity_of_cell(cell: dict, cyc_idx: int) -> float:
    """取放电循环的容量 [Ah]；失败返回 NaN。"""
    try:
        summary = cell["summary"][0, 0]
        d = summary["discharge"][0, cyc_idx]
        cap = _unwrap(d["capacity"])
        return float(cap[0]) if cap.size else float("nan")
    except Exception:
        return float("nan")


def load_segments(cell_name: str) -> list:
    """返回 [(cell, cycle_1based, phase, t, v, i, temp, capacity)]。"""
    path = RAW_DIR / f"{cell_name}.mat"
    mat = sio.loadmat(path)
    key = [k for k in mat if not k.startswith("__")][0]
    cell = mat[key]
    n_cyc = cell["cycle"][0, 0].shape[1]
    segments = []
    cnt = {"charge": 0, "discharge": 0}
    for k in range(n_cyc):
        try:
            cyc = cell["cycle"][0, 0][0, k]
            phase = _phase_of(cyc)
            if phase not in ("charge", "discharge"):
                continue
            t = _data_field(cyc, "Time", "time")
            if t.size == 0:
                continue
            v = _data_field(cyc, "Voltage_measured", "Voltage_charge", "Voltage_load", "voltage")
            i = _data_field(cyc, "Current_measured", "Current_charge", "Current_load", "current")
            tmp = _data_field(cyc, "Temperature_measured", "temperature_measured", "temp")
            if v.size == 0 or i.size == 0 or tmp.size == 0:
                continue
            cap = _capacity_of_cell(cell, k) if phase == "discharge" else float("nan")
            cnt[phase] += 1
            segments.append((cell_name, cnt[phase], phase, t, v, i, tmp, cap))
        except Exception as exc:  # 单段异常不阻塞整体
            print(f"  [warn] {cell_name} cycle#{k} parse failed: {exc}", file=sys.stderr)
    return segments


def build_pack(segments: list, cycle_filter: set | None = None, step: float = GRID_STEP_S) -> pd.DataFrame:
    """把各电芯段对齐到公共时间网格，组装长表。"""
    seg_by_key: dict = {}
    for seg in segments:
        cell, cyc, phase = seg[0], seg[1], seg[2]
        if cycle_filter is not None and cyc not in cycle_filter:
            continue
        seg_by_key.setdefault((cyc, phase), []).append(seg)

    col_ts, col_cell, col_v, col_i, col_t, col_cyc, col_ph, col_cap = [], [], [], [], [], [], [], []
    pack_time = 0.0
    for (cyc, phase) in sorted(seg_by_key.keys()):
        segs = seg_by_key[(cyc, phase)]
        max_dur = max(float(s[3][-1] - s[3][0]) for s in segs if s[3].size)
        if max_dur <= 0:
            continue
        grid = np.arange(0.0, max_dur + step, step)
        for cell, c2, p2, t, v, i, tmp, cap in segs:
            dur = float(t[-1] - t[0])
            in_seg = grid <= dur + 1e-9
            vv = np.where(in_seg, np.interp(grid, t, v), np.nan)
            ii = np.where(in_seg, np.interp(grid, t, i), np.nan)
            tt = np.where(in_seg, np.interp(grid, t, tmp), np.nan)
            n = grid.size
            col_ts.append(pack_time + grid)
            col_cell.append(np.full(n, cell, dtype=object))
            col_v.append(vv)
            col_i.append(ii)
            col_t.append(tt)
            col_cyc.append(np.full(n, c2, dtype=np.int32))
            col_ph.append(np.full(n, p2, dtype=object))
            col_cap.append(np.full(n, cap, dtype=float))
        pack_time += max_dur

    df = pd.DataFrame(
        {
            "timestamp": np.concatenate(col_ts),
            "cell_id": np.concatenate(col_cell),
            "voltage": np.concatenate(col_v),
            "current": np.concatenate(col_i),
            "temp": np.concatenate(col_t),
            "cycle": np.concatenate(col_cyc),
            "phase": np.concatenate(col_ph),
            "capacity_ah": np.concatenate(col_cap),
        }
    )
    return df.sort_values(["timestamp", "cell_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="NASA .mat -> virtual pack long CSV")
    parser.add_argument("--step", type=float, default=GRID_STEP_S, help="common grid step [s]")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading raw .mat files ...")
    all_segments = []
    for cell in CELLS:
        segs = load_segments(cell)
        all_segments.extend(segs)
        print(f"  {cell}: {len(segs)} segments (charge+discharge)")
    if not all_segments:
        raise SystemExit("No data parsed. Check data/raw contains B0005/B0006/B0007/B0018.mat")

    print("Building full pack ...")
    full = build_pack(all_segments, step=args.step)
    full_path = OUT_DIR / "nasa_pack_full.csv"
    full.to_csv(full_path, index=False)
    print(f"  wrote {full_path}  rows={len(full):,}  size={full_path.stat().st_size/1e6:.1f} MB")

    print("Building demo pack ...")
    demo = build_pack(all_segments, cycle_filter=set(DEMO_CYCLES), step=args.step)
    demo_path = OUT_DIR / "nasa_pack_demo.csv"
    demo.to_csv(demo_path, index=False)
    print(f"  wrote {demo_path}  rows={len(demo):,}  size={demo_path.stat().st_size/1e6:.1f} MB")

    print("\nDemo pack preview:")
    print(demo.groupby("cell_id").agg(
        n=("voltage", "size"), v_min=("voltage", "min"), v_max=("voltage", "max"),
        t_min=("temp", "min"), t_max=("temp", "max"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
