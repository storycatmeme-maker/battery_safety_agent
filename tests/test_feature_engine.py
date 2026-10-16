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

def test_relaxation_slope_feature_exists(cleaned_pack):
    from src.feature_engine import BatteryFeatureExtractor
    feats, units = BatteryFeatureExtractor().extract(cleaned_pack)
    assert "relax_dV_dt" in feats.columns
    assert "relax_slope_dev" in feats.columns
    assert "relax_dV_dt" in units
    # 充电后的静置段应当被识别为松弛期
    assert feats["relax_dV_dt"].notna().any()


def test_relaxation_slope_responds_to_micro_short():
    """注入静置期微短路后，该电芯的松弛斜率应明显变负，且不影响其他电芯。"""
    from src.fault_injection import inject_micro_short
    from src.preprocessor import BatteryPreprocessor
    from src.feature_engine import BatteryFeatureExtractor
    from conftest import make_synthetic_pack

    raw = make_synthetic_pack(n_cells=3, n_cycles=2)
    # 定位第 1 个“充电->静置”窗口
    base, _ = BatteryPreprocessor().process(raw)
    f0, _ = BatteryFeatureExtractor().extract(base)
    rest = f0[f0["relax_dV_dt"].notna()]
    t0 = float(rest["timestamp"].min())

    inj, _meta = inject_micro_short(raw, "C1", t0, duration_s=55.0, decay_mv_per_min=30.0)
    cleaned, _ = BatteryPreprocessor().process(inj)
    f1, _ = BatteryFeatureExtractor().extract(cleaned)

    def mean_slope(frame, cell):
        w = frame[(frame["cell_id"] == cell) & (frame["timestamp"] >= t0)
                  & frame["relax_dV_dt"].notna()]
        return float(w["relax_dV_dt"].mean())

    normal_c1, normal_c2 = mean_slope(f0, "C1"), mean_slope(f0, "C2")
    after_c1, after_c2 = mean_slope(f1, "C1"), mean_slope(f1, "C2")
    assert after_c1 < normal_c1 - 0.005, f"注入后 C1 松弛斜率应变负：{normal_c1:.4f} -> {after_c1:.4f}"
    assert abs(after_c2 - normal_c2) < 0.005, f"对照电芯 C2 不应受影响：{normal_c2:.4f} -> {after_c2:.4f}"
