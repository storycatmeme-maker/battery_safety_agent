"""pytest 共享夹具：合成多电芯电池组数据（不依赖 NASA 原始数据）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import BatteryPreprocessor
from src.feature_engine import BatteryFeatureExtractor


def make_synthetic_pack(
    n_cells: int = 4,
    n_cycles: int = 3,
    step_s: float = 5.0,
    nominal_v: float = 3.7,
    seed: int = 7,
) -> pd.DataFrame:
    """生成结构完整的合成电池组长表：
    - 每循环：充电(1.5A, 电压升, 温升) -> 静置(0A) -> 放电(-2A, 电压降)
    - 各电芯电压/温度带轻微个体差异（模拟一致性）
    """
    rng = np.random.default_rng(seed)
    cells = [f"C{i+1}" for i in range(n_cells)]
    bias_v = rng.normal(0, 0.01, n_cells)
    bias_t = rng.normal(0, 0.3, n_cells)

    rows = []
    t_seg = 0.0
    for cyc in range(1, n_cycles + 1):
        for c in range(n_cells):
            t_local = 0.0
            # 充电 600s
            for tt in np.arange(0, 600, step_s):
                rows.append({
                    "timestamp": t_seg + t_local + tt, "cell_id": cells[c], "cycle": cyc,
                    "phase": "charge",
                    "voltage": nominal_v + 0.4 * (1 - np.exp(-tt / 200.0)) + bias_v[c],
                    "current": 1.5,
                    "temp": 24.0 + 3.0 * (tt / 600.0) + bias_t[c] + rng.normal(0, 0.05),
                })
            t_local += 600
            # 静置 60s
            for tt in np.arange(0, 60, step_s):
                rows.append({
                    "timestamp": t_seg + t_local + tt, "cell_id": cells[c], "cycle": cyc,
                    "phase": "rest",
                    "voltage": nominal_v + 0.4 + bias_v[c],
                    "current": 0.0,
                    "temp": 27.0 + bias_t[c],
                })
            t_local += 60
            # 放电 600s
            for tt in np.arange(0, 600, step_s):
                rows.append({
                    "timestamp": t_seg + t_local + tt, "cell_id": cells[c], "cycle": cyc,
                    "phase": "discharge",
                    "voltage": nominal_v + 0.4 - 1.2 * (tt / 600.0) + bias_v[c],
                    "current": -2.0,
                    "temp": 27.0 - 1.0 * (tt / 600.0) + bias_t[c],
                })
        t_seg += 1260.0  # 一个循环时长

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def synthetic_pack() -> pd.DataFrame:
    return make_synthetic_pack(n_cells=4, n_cycles=3)


@pytest.fixture
def cleaned_pack(synthetic_pack: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = BatteryPreprocessor().process(synthetic_pack)
    return cleaned


@pytest.fixture
def feature_matrix(cleaned_pack: pd.DataFrame) -> pd.DataFrame:
    feats, _ = BatteryFeatureExtractor().extract(cleaned_pack)
    return feats
