"""BatteryFeatureExtractor 单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engine import FEATURE_UNITS, BatteryFeatureExtractor


def test_feature_units_metadata():
    assert "dT_dt" in FEATURE_UNITS and "delta_V" in FEATURE_UNITS
    assert FEATURE_UNITS["dT_dt"][0] == "°C/min"


def test_dT_dt_sign_by_regime(cleaned_pack):
    feats, _ = BatteryFeatureExtractor().extract(cleaned_pack)
    charge = feats[feats["regime"] == "charge"]
    discharge = feats[feats["regime"] == "discharge"]
    # 充电温升（斜率中位数 > 0），放电降温（< 0）
    assert charge["dT_dt"].median() > 0.0
    assert discharge["dT_dt"].median() < 0.0


def test_delta_v_median_deviation():
    # 构造 3 电芯：C2 电压整体偏高 0.1V
    ts = np.arange(0.0, 50.0, 5.0)
    rows = []
    for cid, bias in (("C1", 0.0), ("C2", 0.1), ("C3", 0.0)):
        for t in ts:
            rows.append({"timestamp": t, "cell_id": cid, "voltage": 3.8 + bias,
                         "current": 0.0, "temp": 25.0, "is_valid": True,
                         "regime": "rest"})
    df = pd.DataFrame(rows)
    feats, _ = BatteryFeatureExtractor().extract(df)
    c2 = feats[feats["cell_id"] == "C2"]
    c1 = feats[feats["cell_id"] == "C1"]
    assert abs(c2["delta_V"].mean() - 0.1) < 1e-9
    assert abs(c1["delta_V"].mean()) < 1e-9


def test_pack_spread_positive():
    ts = np.arange(0.0, 50.0, 5.0)
    rows = []
    for cid, bias in (("C1", 0.0), ("C2", 0.5)):
        for t in ts:
            rows.append({"timestamp": t, "cell_id": cid, "voltage": 3.8 + bias,
                         "current": 0.0, "temp": 25.0 + bias * 10, "is_valid": True,
                         "regime": "rest"})
    df = pd.DataFrame(rows)
    feats, _ = BatteryFeatureExtractor().extract(df)
    assert feats["pack_spread_V"].max() > 0.4
    assert feats["pack_spread_T"].max() > 4.0


def test_valid_only_filtering(cleaned_pack):
    feats_all, _ = BatteryFeatureExtractor().extract(cleaned_pack, use_valid_only=False)
    feats_valid, _ = BatteryFeatureExtractor().extract(cleaned_pack, use_valid_only=True)
    assert len(feats_valid) < len(feats_all)
