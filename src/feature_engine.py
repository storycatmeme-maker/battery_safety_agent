"""
feature_engine.py — 电化学机理特征提取
=======================================

职责
----
把预处理后的清洗数据（电压/电流/温度时序）转换为“有物理含义”的特征矩阵。
严禁直接拿原始电压/温度训练——本模块提取：

1. dT_dt         温升速率 [°C/min]：
   单体温升一阶差分（dT/dt）并做窗口平滑；温升斜率突变是内短路/热失控的
   早期前兆，比绝对温度阈值（如国标 60°C）提前数分钟到数十分钟。
2. delta_V       中位数电压差 [V]：V_i - median(V_pack)，体现单体微内阻发散。
3. delta_T       中位数温差 [°C]：T_i - median(T_pack)，体现散热/内阻差异。
4. pack_spread_T 组内温度极差 [°C]（滑动窗口内 Tmax-Tmin）。
5. pack_spread_V 组内电压极差 [V]（滑动窗口内 Vmax-Vmin）。
6. c_rate        电流倍率 [1/h] = I / Cn（工况上下文特征）。

输出
----
- 特征矩阵 DataFrame（每列带物理单位注释，见 FEATURE_UNITS）
- FEATURE_UNITS 元数据 {列名: (单位, 说明)}
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# 特征单位与物理含义元数据（面试/文档可直接引用）
FEATURE_UNITS: Dict[str, Tuple[str, str]] = {
    "dT_dt": ("°C/min", "单体温升速率（一阶差分+平滑），热失控前兆核心特征"),
    "delta_V": ("V", "单体电压相对组内中位数的偏离，体现微内阻/容量发散"),
    "delta_T": ("°C", "单体温度相对组内中位数的偏离，体现散热/内阻差异"),
    "pack_spread_T": ("°C", "组内温度极差（滑动窗口内 Tmax-Tmin），散热一致性"),
    "pack_spread_V": ("V", "组内电压极差（滑动窗口内 Vmax-Vmin），一致性发散"),
    "c_rate": ("1/h", "电流倍率 I/Cn，工况上下文"),
    "relax_dV_dt": ("V/min", "充电结束静置段内的端电压变化率（微短路/析锂敏感指标）"),
    "relax_slope_dev": ("V/min", "松弛斜率相对组内中位数的偏离度（一致性视角）"),
}


class BatteryFeatureExtractor:
    """电化学机理特征提取器。

    参数
    ----
    nominal_capacity_ah : 电芯标称容量 [Ah]（用于 C 倍率，默认 2.0）
    dtdt_smooth_window  : 温升速率平滑窗口（样本数，默认 5）
    spread_rolling      : 极差滑动窗口（样本数，默认 12 ≈ 1 min @5s）
    """

    def __init__(
        self,
        nominal_capacity_ah: float = 2.0,
        dtdt_smooth_window: int = 5,
        spread_rolling: int = 12,
        max_slope_gap_s: float = 30.0,
        align_timestamp_s: float = 1.0,
        relax_window_s: float = 600.0,
    ) -> None:
        self.nominal_capacity_ah = nominal_capacity_ah
        self.dtdt_smooth_window = dtdt_smooth_window
        self.spread_rolling = spread_rolling
        self.max_slope_gap_s = max_slope_gap_s
        self.align_timestamp_s = align_timestamp_s
        self.relax_window_s = relax_window_s

    def extract(
        self, cleaned: pd.DataFrame, use_valid_only: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
        """从预处理输出提取特征矩阵。"""
        if cleaned.empty:
            raise ValueError("输入为空，无法提取特征。")
        required = {"timestamp", "cell_id", "voltage", "current", "temp"}
        missing = required - set(cleaned.columns)
        if missing:
            raise ValueError(f"缺少必需列: {sorted(missing)}")

        df = cleaned.copy()
        if use_valid_only and "is_valid" in df.columns:
            df = df[df["is_valid"]].copy()
        if df.empty:
            raise ValueError("有效样本为空（全部被 data_loss/过渡区排除）。")

        df = df.sort_values(["timestamp", "cell_id"]).reset_index(drop=True)

        # 0) 时间戳离散化对齐（防 pivot 失效）
        #    不同电芯的浮点时间戳可能在末位相差毫秒级（如 10.001 与 10.000），
        #    直接 pivot 会被当成两个不同时刻，导致同一时刻只剩单颗电芯、
        #    ΔV/极差大面积失效。这里先对齐到 align_timestamp_s 网格。
        if self.align_timestamp_s and self.align_timestamp_s > 0:
            df["timestamp"] = (
                (df["timestamp"] / self.align_timestamp_s).round() * self.align_timestamp_s
            )

        # 1) C 倍率
        df["c_rate"] = df["current"] / self.nominal_capacity_ah

        # 2) 温升速率 dT/dt [°C/min]（同一电芯、同一循环、同一相内做一阶差分 + 平滑）
        #    ★ 必须在同段内差分：多段数据拼接时，段与段之间存在时间跳变，
        #      跨段差分会产生 ±7°C/min 的假温升（实测：三颗电芯在拼接点同时“跳温”1°C），
        #      从而造成“热失控前兆”误报。
        seg_keys = ["cell_id"] + [c for c in ("cycle", "phase") if c in df.columns]
        dT = df.groupby(seg_keys, sort=False)["temp"].diff()
        dt = df.groupby(seg_keys, sort=False)["timestamp"].diff()
        dT_dt = (dT / dt * 60.0).where(dt <= self.max_slope_gap_s).fillna(0.0)
        df["dT_dt"] = (
            dT_dt.groupby(df["cell_id"])
            .transform(lambda s: s.rolling(self.dtdt_smooth_window, min_periods=1).mean())
        )

        # 3) 组内中位数偏离（同一时间戳下跨电芯取中位数）
        pivot_v = df.pivot_table(index="timestamp", columns="cell_id", values="voltage", aggfunc="first")
        pivot_t = df.pivot_table(index="timestamp", columns="cell_id", values="temp", aggfunc="first")
        median_v = pivot_v.median(axis=1).rename("_med_v")
        median_t = pivot_t.median(axis=1).rename("_med_t")
        spread_v = (pivot_v.max(axis=1) - pivot_v.min(axis=1)).rename("_spread_v")
        spread_t = (pivot_t.max(axis=1) - pivot_t.min(axis=1)).rename("_spread_t")

        df = (
            df.merge(median_v.reset_index(), on="timestamp", how="left")
            .merge(median_t.reset_index(), on="timestamp", how="left")
            .merge(spread_v.reset_index(), on="timestamp", how="left")
            .merge(spread_t.reset_index(), on="timestamp", how="left")
        )
        df["delta_V"] = df["voltage"] - df["_med_v"]
        df["delta_T"] = df["temp"] - df["_med_t"]

        # 4) 静置自松弛斜率 [V/min]
        #    物理意义：电芯充满后静置，端电压因极化消除而缓慢回落趋于稳定；
        #    若存在微短路/析锂，端电压会异常快速跌落。
        #    只统计“充电 -> 静置”切换后 relax_window_s 内的静置样本；
        #    并给出相对组内中位数的偏离度（一致性视角）。
        if "regime" in df.columns:
            ts_arr = df["timestamp"].to_numpy(dtype=float)
            is_rest = (df["regime"] == "rest").to_numpy()
            prev_regime = df.groupby("cell_id", sort=False)["regime"].shift(1).to_numpy()
            starts = is_rest & (prev_regime == "charge")
            anchor = pd.Series(
                np.where(starts, ts_arr, np.nan), index=df.index
            ).groupby(df["cell_id"], sort=False).ffill().to_numpy()
            in_relax = (
                is_rest
                & np.isfinite(anchor)
                & ((ts_arr - anchor) <= self.relax_window_s)
            )
            seg_id = df["cell_id"].astype(str) + "#" + pd.Series(anchor, index=df.index).astype(str)
            dV = df.groupby(seg_id, sort=False)["voltage"].diff()
            dtv = df.groupby(seg_id, sort=False)["timestamp"].diff()
            slope = (dV / dtv * 60.0).where(dtv <= self.max_slope_gap_s)
            df["relax_dV_dt"] = np.where(in_relax, slope, np.nan)
            med = df.groupby("timestamp")["relax_dV_dt"].transform("median")
            df["relax_slope_dev"] = (df["relax_dV_dt"] - med).fillna(0.0)
        else:
            df["relax_dV_dt"] = np.nan
            df["relax_slope_dev"] = 0.0

        # 5) 滑动极差（组内发散度，窗口内取峰值）
        df["pack_spread_V"] = (
            df.groupby("cell_id")["_spread_v"]
            .transform(lambda s: s.rolling(self.spread_rolling, min_periods=1).max())
        )
        df["pack_spread_T"] = (
            df.groupby("cell_id")["_spread_t"]
            .transform(lambda s: s.rolling(self.spread_rolling, min_periods=1).max())
        )

        out_cols = [
            "timestamp", "cell_id", "cycle", "phase", "regime",
            "voltage", "current", "temp", "c_rate",
            "dT_dt", "delta_V", "delta_T", "pack_spread_T", "pack_spread_V",
            "relax_dV_dt", "relax_slope_dev",
        ]
        out_cols = [c for c in out_cols if c in df.columns]
        return df[out_cols].reset_index(drop=True), dict(FEATURE_UNITS)
