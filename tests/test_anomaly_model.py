"""HybridAnomalyDetector 单元测试：无注入不误报 / 注入可检出。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_model import HybridAnomalyDetector, LEVEL_WARNING
from src.fault_injection import inject_thermal_precursor
from src.preprocessor import BatteryPreprocessor
from src.feature_engine import BatteryFeatureExtractor

from conftest import make_synthetic_pack


def _run(df: pd.DataFrame, warmup: int = 2000):
    cleaned, _ = BatteryPreprocessor().process(df)
    feats, _ = BatteryFeatureExtractor().extract(cleaned)
    det = HybridAnomalyDetector(sensitivity=0.95, warmup_samples=warmup)
    det.fit(feats)
    return det.score(feats), det


def test_no_injection_no_level2():
    df = make_synthetic_pack(n_cells=4, n_cycles=5)
    scored, _ = _run(df)
    assert int((scored["level"] >= LEVEL_WARNING).sum()) == 0


def test_thermal_injection_triggers_level2():
    df = make_synthetic_pack(n_cells=4, n_cycles=5)
    # 注入热失控前兆：C2, cycle 4 区间内 4°C/min
    t0 = 3 * 1260.0 + 100.0
    df2, meta = inject_thermal_precursor(df, "C2", t0, duration_s=300.0, dtdt_cmin=4.0)
    scored, det = _run(df2, warmup=2000)
    hit = scored[
        (scored["cell_id"] == "C2")
        & (scored["timestamp"] >= meta["t_start"])
        & (scored["level"] >= LEVEL_WARNING)
    ]
    assert len(hit) > 0, "注入后应触发二级预警"


def test_comm_loss_does_not_fabricate():
    from src.fault_injection import inject_comm_loss
    df = make_synthetic_pack(n_cells=4, n_cycles=3)
    t0 = 1260.0 + 100.0
    df2, meta = inject_comm_loss(df, "C1", t0, duration_s=10.0)
    cleaned, report = BatteryPreprocessor().process(df2)
    win = cleaned[(cleaned["cell_id"] == "C1") & (cleaned["timestamp"] >= t0 - 1) & (cleaned["timestamp"] <= t0 + 11)]
    assert win["data_loss"].any()
    assert report.n_data_loss_events >= 1
