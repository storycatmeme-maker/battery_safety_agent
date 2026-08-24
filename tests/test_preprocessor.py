"""BatteryPreprocessor 单元测试：插值 / 离线 / 毛刺 / 工况切分。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import BatteryPreprocessor, REQUIRED_COLUMNS


def _grid_df(step: float = 1.0, n: int = 30) -> pd.DataFrame:
    ts = np.arange(0.0, n * step, step)
    rows = []
    for cid in ("A", "B"):
        for t in ts:
            rows.append({
                "timestamp": t, "cell_id": cid, "voltage": 3.8 + t / 1000.0,
                "current": 0.0, "temp": 25.0,
            })
    return pd.DataFrame(rows)


def test_schema_missing_column():
    df = _grid_df().drop(columns=["temp"])
    with pytest.raises(ValueError, match="缺少必需字段"):
        BatteryPreprocessor().process(df)


def test_empty_input_raises():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="为空"):
        BatteryPreprocessor().process(df)


def test_small_gap_interpolated():
    df = _grid_df(step=1.0)
    # A 电芯在 t=10 处电压置 NaN：两侧有效点跨度 2s <= 3s -> 插值
    df.loc[(df["cell_id"] == "A") & (df["timestamp"] == 10.0), "voltage"] = np.nan
    cleaned, report = BatteryPreprocessor().process(df)
    a = cleaned[cleaned["cell_id"] == "A"].set_index("timestamp")
    assert not np.isnan(a.loc[10.0, "voltage"])
    assert report.n_interpolated >= 1


def test_long_gap_is_data_loss():
    df = _grid_df(step=1.0)
    # A 电芯 t=10..15 连续 6 个点置 NaN：两侧跨度 7s > 5s -> 离线且不插值
    df.loc[(df["cell_id"] == "A") & (df["timestamp"] >= 10.0) & (df["timestamp"] < 16.0), "voltage"] = np.nan
    cleaned, report = BatteryPreprocessor().process(df)
    a = cleaned[cleaned["cell_id"] == "A"].set_index("timestamp")
    assert a.loc[12.0, "data_loss"]
    assert np.isnan(a.loc[12.0, "voltage"])          # 不伪造插值
    assert report.n_data_loss_events >= 1


def test_spike_removed():
    df = _grid_df(step=1.0)
    df.loc[(df["cell_id"] == "A") & (df["timestamp"] == 15.0), "voltage"] += 0.5  # 0.5V 毛刺
    cleaned, report = BatteryPreprocessor().process(df)
    a = cleaned[cleaned["cell_id"] == "A"].set_index("timestamp")
    assert abs(a.loc[15.0, "voltage"] - 3.815) < 0.02   # 被中值滤波拉回
    assert report.n_spikes_removed["voltage"] >= 1


def test_regime_segmentation(cleaned_pack):
    dist = cleaned_pack["regime"].value_counts()
    for reg in ("charge", "discharge", "rest", "transition"):
        assert dist.get(reg, 0) > 0, f"缺少工况 {reg}"
    # 过渡区样本不参与特征计算
    assert not cleaned_pack.loc[cleaned_pack["regime"] == "transition", "is_valid"].any()


def test_timestamp_datetime_normalization():
    df = _grid_df(step=1.0, n=5)
    base = pd.Timestamp("2024-01-01")
    df["timestamp"] = base + pd.to_timedelta(df["timestamp"], unit="s")
    cleaned, _ = BatteryPreprocessor().process(df)
    assert cleaned["timestamp"].dtype.kind == "f"
    assert cleaned["timestamp"].min() > 1.7e9  # epoch 秒
