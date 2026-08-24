"""
preprocessor.py — 工业级电池时序数据预处理管道
================================================

模块定位
--------
把 BMS/测试台采集的原始长表时序清洗为可做特征工程与异常检测的标准数据，
同时输出“数据质量审计报告”。

设计原则（拒绝玩具代码）
------------------------
1. 绝不无条件填充缺失：
   - 断点 <= 3 s 的短缺失  -> 线性插值（BMS 正常丢点）
   - 断点 3~5 s            -> 插值但打上可疑告警（gap_3_5s）
   - 断点 > 5 s            -> 判定“通讯离线 (data_loss)”，置 NaN 且不伪造数据
2. 电压/温度毛刺：滑动中值滤波（窗口=5）平滑，并统计剔除的毛刺数用于审计；
   滤波结果与原始值同时保留（voltage_raw / temp_raw），保证可追溯。
3. 工况切分（按电池机理）：
   - I >  +0.05C  -> 充电 (charge)
   - I <  -0.05C  -> 放电 (discharge)
   - 其余          -> 静置 (rest)
   - 带滞回防抖：剔除时长过短的工况“毛刺”，避免阈值边界抖动。
4. 剔除工况切换瞬间 5 s 内的不稳定过渡区（标记 transition），
   避免把“大功率急停/工况切换的电流阶跃”误判为电芯异常。

输入字段：['timestamp', 'cell_id', 'voltage', 'current', 'temp']
  - timestamp: 数值秒（float）；若为 ISO 字符串/日期时间，自动转换为 epoch 秒
  - cell_id  : 电芯编号（str/int）
  - voltage  : 单体电压 [V]
  - current  : 回路电流 [A]（充电为正，放电为负）
  - temp     : 单体温度 [°C]

输出
----
- cleaned_df: 清洗后的长表 DataFrame，新增列：
    voltage_raw / temp_raw : 原始值（审计用）
    regime                  : charge / discharge / rest / transition
    data_loss               : 是否处于通讯离线（布尔）
    interpolated            : 是否经插值（布尔）
    gap_to_prev_s           : 与上一条样本的时间间隔 [s]
    loss_duration_s         : 本条样本所属离线段时长 [s]
    is_valid                : 是否可进入特征工程（非 data_loss 且非 transition）
- BatteryQualityReport: 数据质量审计（缺失率、丢包事件、毛刺数、工况分布等）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: List[str] = ["timestamp", "cell_id", "voltage", "current", "temp"]
TIMESTAMP_COL = "timestamp"
CELL_COL = "cell_id"

REGIME_CHARGE = "charge"
REGIME_DISCHARGE = "discharge"
REGIME_REST = "rest"
REGIME_TRANSITION = "transition"
REGIMES = (REGIME_CHARGE, REGIME_DISCHARGE, REGIME_REST, REGIME_TRANSITION)


@dataclass
class BatteryQualityReport:
    """数据质量审计报告（供运维数据分析岗位的汇报/面试使用）。"""

    n_cells: int
    n_samples_total: int
    per_cell_samples: Dict[str, int]
    per_cell_missing: Dict[str, Dict[str, int]]
    n_interpolated: int
    n_data_loss_events: int
    data_loss_total_seconds: float
    n_spikes_removed: Dict[str, int]
    regime_distribution: Dict[str, int]
    n_transition_samples: int
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_cells": int(self.n_cells),
            "n_samples_total": int(self.n_samples_total),
            "per_cell_samples": self.per_cell_samples,
            "per_cell_missing": self.per_cell_missing,
            "n_interpolated": int(self.n_interpolated),
            "n_data_loss_events": int(self.n_data_loss_events),
            "data_loss_total_seconds": round(float(self.data_loss_total_seconds), 3),
            "n_spikes_removed": self.n_spikes_removed,
            "regime_distribution": self.regime_distribution,
            "n_transition_samples": int(self.n_transition_samples),
            "warnings": self.warnings,
        }


def _fill_nan_small_gaps(
    values: np.ndarray,
    ts: np.ndarray,
    interp_limit_s: float,
    loss_min_s: float,
    warn_limit_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """按时间间隔策略处理单列 NaN。

    - 相邻有效点时间跨度 <= interp_limit_s 的 NaN 段：线性插值（3~5 s 段另计数告警）
    - 跨度 > loss_min_s 的 NaN 段：判定通讯离线，置 NaN 不插值
    - 序列首尾的 NaN 段：不做外推，保持 NaN

    返回 (filled, interp_mask, loss_mask, loss_duration, n_warn_gaps)
    """
    n = len(values)
    filled = values.copy().astype(float)
    interp_mask = np.zeros(n, dtype=bool)
    loss_mask = np.zeros(n, dtype=bool)
    loss_duration = np.zeros(n, dtype=float)
    n_warn_gaps = 0

    isnan = np.isnan(values)
    if not isnan.any():
        return filled, interp_mask, loss_mask, loss_duration, n_warn_gaps

    idx = np.flatnonzero(isnan)
    run_starts = [idx[0]]
    run_ends: List[int] = []
    for k in range(1, len(idx)):
        if idx[k] != idx[k - 1] + 1:
            run_ends.append(idx[k - 1])
            run_starts.append(idx[k])
    run_ends.append(idx[-1])

    for s, e in zip(run_starts, run_ends):
        prev_v, next_v = s - 1, e + 1
        if prev_v < 0 or next_v >= n:
            continue  # 首尾缺失，不插值不外推
        gap_dur = float(ts[next_v] - ts[prev_v])
        if gap_dur <= interp_limit_s:
            xp = [ts[prev_v], ts[next_v]]
            fp = [float(values[prev_v]), float(values[next_v])]
            filled[s : e + 1] = np.interp(ts[s : e + 1], xp, fp)
            interp_mask[s : e + 1] = True
            if gap_dur > warn_limit_s:
                n_warn_gaps += 1
        elif gap_dur > loss_min_s:
            loss_mask[s : e + 1] = True
            loss_duration[s : e + 1] = gap_dur
        else:
            # 配置异常兜底：interp_limit <= loss_min 时该分支不会触发
            interp_mask[s : e + 1] = True
    return filled, interp_mask, loss_mask, loss_duration, n_warn_gaps


def _median_filter(values: np.ndarray, window: int = 5) -> np.ndarray:
    """滑动中值滤波（窗口=5，忽略 NaN）。"""
    return (
        pd.Series(values, dtype="float64")
        .rolling(window, center=True, min_periods=1)
        .median()
        .to_numpy()
    )


def _smooth_regime(labels: np.ndarray, min_run: int = 2) -> np.ndarray:
    """滞回防抖：把时长 <= min_run 个采样的工况“毛刺”合并到前一段。"""
    codes, uniques = pd.factorize(labels)
    out = codes.copy()
    n = len(codes)
    i = 0
    while i < n:
        j = i
        while j < n and codes[j] == codes[i]:
            j += 1
        run_len = j - i
        if run_len <= min_run and 0 < i and j < n:
            out[i:j] = codes[i - 1]
        i = j
    return np.array([uniques[c] for c in out], dtype=object)


class BatteryPreprocessor:
    """电池时序数据预处理管道。

    参数
    ----
    gap_interp_max_s : 短缺失插值上限（默认 3.0 s）
    gap_warn_max_s   : 可疑断点告警上限（默认 5.0 s；3~5 s 之间插值并告警）
    data_loss_min_s  : 通讯离线判定阈值（默认 5.0 s）
    median_window    : 滑动中值滤波窗口（默认 5）
    charge_threshold_c  : 充电判定阈值（C 倍率，默认 0.05）
    discharge_threshold_c : 放电判定阈值（C 倍率，默认 0.05）
    nominal_capacity_ah  : 电芯标称容量 [Ah]，用于把电流换算为 C 倍率（默认 2.0）
    transition_window_s  : 工况切换过渡区时长（默认 5.0 s）
    voltage_spike_v  : 电压毛刺判定阈值 [V]（默认 0.15）
    temp_spike_c     : 温度毛刺判定阈值 [°C]（默认 2.0）
    """

    def __init__(
        self,
        gap_interp_max_s: float = 3.0,
        gap_warn_max_s: float = 5.0,
        data_loss_min_s: float = 5.0,
        median_window: int = 5,
        charge_threshold_c: float = 0.05,
        discharge_threshold_c: float = 0.05,
        nominal_capacity_ah: float = 2.0,
        transition_window_s: float = 5.0,
        voltage_spike_v: float = 0.15,
        temp_spike_c: float = 2.0,
    ) -> None:
        self.gap_interp_max_s = gap_interp_max_s
        self.gap_warn_max_s = gap_warn_max_s
        self.data_loss_min_s = data_loss_min_s
        self.median_window = median_window
        self.charge_threshold_c = charge_threshold_c
        self.discharge_threshold_c = discharge_threshold_c
        self.nominal_capacity_ah = nominal_capacity_ah
        self.transition_window_s = transition_window_s
        self.voltage_spike_v = voltage_spike_v
        self.temp_spike_c = temp_spike_c
        if gap_interp_max_s > data_loss_min_s:
            raise ValueError("gap_interp_max_s 必须 <= data_loss_min_s")

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    def process(
        self, data: Union[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, BatteryQualityReport]:
        """清洗数据。输入为 CSV 路径或长表 DataFrame，返回 (清洗后 df, 质量报告)。"""
        df = self._load(data)
        self._validate_schema(df)
        df = self._normalize(df)

        parts: List[pd.DataFrame] = []
        per_cell_samples: Dict[str, int] = {}
        per_cell_missing: Dict[str, Dict[str, int]] = {}
        warnings: List[str] = []

        for cell_id, group in df.groupby(CELL_COL, sort=True):
            processed, cell_warnings = self._process_cell(str(cell_id), group.copy())
            parts.append(processed)
            warnings.extend(cell_warnings)
            per_cell_samples[str(cell_id)] = int(len(group))
            per_cell_missing[str(cell_id)] = {
                col: int(group[col].isna().sum()) for col in ("voltage", "current", "temp")
            }

        if not parts:
            raise ValueError("输入数据为空，无法执行预处理。")

        cleaned = pd.concat(parts, ignore_index=True)
        cleaned = cleaned.sort_values([TIMESTAMP_COL, CELL_COL]).reset_index(drop=True)
        report = self._build_report(cleaned, per_cell_samples, per_cell_missing, warnings)
        return cleaned, report

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load(data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, str):
            df = pd.read_csv(data)
            logger.info("已读取 CSV: %s (rows=%d)", data, len(df))
            return df
        if isinstance(data, pd.DataFrame):
            return data.copy()
        raise TypeError("data 必须是 CSV 路径或 pandas.DataFrame")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必需字段: {missing}；请确保输入包含 {REQUIRED_COLUMNS}")
        if df.empty:
            raise ValueError("输入 DataFrame 为空。")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ts = df[TIMESTAMP_COL]
        if pd.api.types.is_numeric_dtype(ts):
            df[TIMESTAMP_COL] = ts.astype("float64")
        else:
            parsed = pd.to_datetime(ts, errors="coerce")
            if parsed.isna().all():
                raise ValueError("timestamp 既不是数值也不是可解析的时间格式。")
            df[TIMESTAMP_COL] = (parsed - pd.Timestamp("1970-01-01")).dt.total_seconds()
        df[CELL_COL] = df[CELL_COL].astype(str).str.strip()
        for col in ("voltage", "current", "temp"):
            coerced = pd.to_numeric(df[col], errors="coerce")
            dropped = int(coerced.isna().sum() - df[col].isna().sum())
            if dropped:
                logger.warning("字段 %s 有 %d 个非数值值被置为 NaN", col, dropped)
            df[col] = coerced
        return df

    def _process_cell(
        self, cell_id: str, group: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        warnings: List[str] = []
        g = group.sort_values(TIMESTAMP_COL).reset_index(drop=True)
        g = g.dropna(subset=[TIMESTAMP_COL]).reset_index(drop=True)
        ts = g[TIMESTAMP_COL].to_numpy(dtype="float64")
        n = len(g)

        # 1) 行级时间间隔与“通讯离线”（缺行）判定
        gap_to_prev = np.diff(ts, prepend=ts[0] if n else 0.0).astype("float64")
        gap_to_prev[0] = 0.0
        row_loss = gap_to_prev > self.data_loss_min_s
        loss_duration = np.where(row_loss, gap_to_prev, 0.0).astype("float64")

        # 2) 逐列处理 NaN 段（插值 / 离线）
        interp_total = np.zeros(n, dtype=bool)
        loss_total = row_loss.copy()
        n_warn_gaps = 0
        for col in ("voltage", "current", "temp"):
            raw = g[col].to_numpy(dtype="float64")
            filled, imask, lmask, ldur, nwarn = _fill_nan_small_gaps(
                raw,
                ts,
                interp_limit_s=self.gap_warn_max_s,
                loss_min_s=self.data_loss_min_s,
                warn_limit_s=self.gap_interp_max_s,
            )
            g[col] = filled
            g[f"{col}_raw"] = raw
            interp_total |= imask
            loss_total |= lmask
            loss_duration = np.maximum(loss_duration, ldur)
            n_warn_gaps += nwarn

        if n_warn_gaps:
            warnings.append(f"cell {cell_id}: 检测到 {n_warn_gaps} 次 3~5 s 可疑断点（已插值，建议复核）")
        n_loss_events = int(np.sum(loss_total & ~np.roll(loss_total, 1)))
        if n_loss_events:
            warnings.append(
                f"cell {cell_id}: 检测到 {n_loss_events} 次通讯离线（>={self.data_loss_min_s:g}s，未插值）"
            )

        # 3) 滑动中值滤波 + 毛刺剔除（电压/温度）：逐行布尔标记
        for col, tol, col_name in (
            ("voltage", self.voltage_spike_v, "voltage"),
            ("temp", self.temp_spike_c, "temp"),
        ):
            raw = g[f"{col}_raw"].to_numpy(dtype="float64")   # 原始值（含 NaN）
            cur = g[col].to_numpy(dtype="float64")            # 插值后当前值
            filtered = _median_filter(cur, self.median_window)
            ok = ~np.isnan(raw) & ~np.isnan(filtered)
            spike = ok & (np.abs(raw - filtered) > tol)
            g[f"_spike_{col_name}"] = spike
            fixed = np.where(spike, filtered, cur)            # 保留插值结果
            g[col] = fixed

        # 4) 工况切分（带滞回防抖）
        c_rate = g["current"].to_numpy(dtype="float64") / self.nominal_capacity_ah
        labels = np.where(
            c_rate > self.charge_threshold_c,
            REGIME_CHARGE,
            np.where(c_rate < -self.discharge_threshold_c, REGIME_DISCHARGE, REGIME_REST),
        )
        labels = _smooth_regime(labels)

        # 5) 工况切换过渡区（切换点前后 transition_window_s）
        change = np.zeros(n, dtype=bool)
        change[1:] = labels[1:] != labels[:-1]
        transition = np.zeros(n, dtype=bool)
        change_idx = np.flatnonzero(change)
        for ci in change_idx:
            t0 = ts[ci]
            transition |= np.abs(ts - t0) <= self.transition_window_s

        regime = np.where(transition, REGIME_TRANSITION, labels)
        is_valid = (~loss_total) & (~transition)

        g["gap_to_prev_s"] = gap_to_prev
        g["loss_duration_s"] = loss_duration
        g["data_loss"] = loss_total
        g["interpolated"] = interp_total
        g["regime"] = regime
        g["is_valid"] = is_valid
        return g, warnings

    def _build_report(
        self,
        cleaned: pd.DataFrame,
        per_cell_samples: Dict[str, int],
        per_cell_missing: Dict[str, Dict[str, int]],
        warnings: List[str],
    ) -> BatteryQualityReport:
        n_loss_events = int(
            cleaned.groupby(CELL_COL)["data_loss"]
            .apply(lambda s: int((s & ~s.shift(1, fill_value=False)).sum()))
            .sum()
        )
        regime_dist = (
            cleaned["regime"].value_counts().reindex(REGIMES, fill_value=0).to_dict()
        )
        return BatteryQualityReport(
            n_cells=int(cleaned[CELL_COL].nunique()),
            n_samples_total=int(len(cleaned)),
            per_cell_samples=per_cell_samples,
            per_cell_missing=per_cell_missing,
            n_interpolated=int(cleaned["interpolated"].sum()),
            n_data_loss_events=n_loss_events,
            data_loss_total_seconds=float(cleaned["loss_duration_s"].sum()),
            n_spikes_removed={
                "voltage": int(cleaned["_spike_voltage"].sum()),
                "temp": int(cleaned["_spike_temp"].sum()),
            },
            regime_distribution=regime_dist,
            n_transition_samples=int((cleaned["regime"] == REGIME_TRANSITION).sum()),
            warnings=warnings,
        )


def preprocess_file(path: str, **kwargs: Any) -> Tuple[pd.DataFrame, BatteryQualityReport]:
    """便捷函数：直接处理 CSV 文件。"""
    return BatteryPreprocessor(**kwargs).process(path)
