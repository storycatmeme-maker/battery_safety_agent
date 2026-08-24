"""
fault_injection.py — 透明故障注入（演示与验证用）
==================================================
公开老化数据里几乎不存在“突发热失控”，为了验证检测器对热失控前兆的
灵敏度，本模块按电化学机理在真实数据上**可控、透明**地注入故障：

1. inject_thermal_precursor : 注入内短路/热失控前兆
   - 局部温升斜率叠加（默认 3.5 °C/min，持续 10 min）
   - 轻微电压跌落（微短路内阻效应）
   - 返回 ground-truth 元数据，可用于量化“检测提前量”
2. inject_comm_loss          : 模拟 BMS 通讯中断（置 NaN）
3. inject_voltage_spikes     : 注入电压毛刺（验证预处理滤波）

所有注入均保留原始数据副本，注入区间有明确标注，绝不混入真实数据。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

NUMERIC_COLS = ("voltage", "current", "temp")


def inject_thermal_precursor(
    df: pd.DataFrame,
    cell_id: str,
    t_start: float,
    duration_s: float = 600.0,
    dtdt_cmin: float = 3.5,
    voltage_drop_mv_per_min: float = 2.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """注入热失控前兆：线性温升叠加 + 轻微电压跌落。

    参数
    ----
    t_start                  : 注入起点 [s]
    duration_s               : 注入持续时长 [s]（默认 600 s = 10 min）
    dtdt_cmin                : 目标温升斜率 [°C/min]（默认 3.5）
    voltage_drop_mv_per_min  : 电压跌落速率 [mV/min]（微短路内阻效应）
    """
    df = df.copy()
    mask = (
        (df["cell_id"] == cell_id)
        & (df["timestamp"] >= t_start)
        & (df["timestamp"] < t_start + duration_s)
    )
    t = df.loc[mask, "timestamp"].to_numpy(dtype="float64")
    k = dtdt_cmin / 60.0  # °C/s
    df.loc[mask, "temp"] = df.loc[mask, "temp"] + k * (t - t_start)
    df.loc[mask, "voltage"] = df.loc[mask, "voltage"] - (
        voltage_drop_mv_per_min / 1000.0
    ) * ((t - t_start) / 60.0)

    meta = {
        "type": "thermal_precursor",
        "cell_id": str(cell_id),
        "t_start": float(t_start),
        "t_end": float(t_start + duration_s),
        "duration_s": float(duration_s),
        "dtdt_cmin": float(dtdt_cmin),
        "voltage_drop_mv_per_min": float(voltage_drop_mv_per_min),
        "injected": True,
    }
    return df, meta


def inject_comm_loss(
    df: pd.DataFrame, cell_id: str, t_start: float, duration_s: float = 30.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """模拟 BMS 通讯中断：把目标电芯一段时间内的读数置为 NaN。"""
    df = df.copy()
    mask = (
        (df["cell_id"] == cell_id)
        & (df["timestamp"] >= t_start)
        & (df["timestamp"] < t_start + duration_s)
    )
    for col in NUMERIC_COLS:
        df.loc[mask, col] = np.nan
    meta = {
        "type": "comm_loss",
        "cell_id": str(cell_id),
        "t_start": float(t_start),
        "t_end": float(t_start + duration_s),
        "duration_s": float(duration_s),
        "injected": True,
    }
    return df, meta


def inject_voltage_spikes(
    df: pd.DataFrame, cell_id: str, n: int = 5, magnitude_v: float = 0.3, seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """注入电压毛刺（验证滑动中值滤波的剔除能力）。"""
    df = df.copy()
    rng = np.random.default_rng(seed)
    cell_mask = (df["cell_id"] == cell_id).to_numpy()
    idx = np.flatnonzero(cell_mask)
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    df.loc[df.index[pick], "voltage"] = df.loc[df.index[pick], "voltage"] + magnitude_v
    meta = {
        "type": "voltage_spikes",
        "cell_id": str(cell_id),
        "n": int(len(pick)),
        "magnitude_v": float(magnitude_v),
        "injected": True,
    }
    return df, meta
