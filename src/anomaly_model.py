"""
anomaly_model.py — 双重判定算法引擎
====================================

设计（拒绝黑盒、拒绝单点失效）
------------------------------
工业异常检测不能只靠一个算法。本模块实现“双引擎”：
1. AI 引擎  : sklearn IsolationForest 无监督异常评分
              输入 [delta_V, dT_dt, delta_T]，输出归一化 anomaly_score ∈ [0,1]
2. 物理规则引擎 : 动态 3-Sigma 包络
              - dT_dt 超过“该电芯滚动均值 + 3σ”           -> 温升异常
              - |delta_V| 超过“该电芯滚动包络 + 绝对下限” -> 压差异常
              包络随运行状态滚动更新，避免固定阈值失效。

综合判定
--------
- 二级预警（疑似内短路/热失控前兆）：AI 分数超标 且 物理规则持续 >= 3 个采样周期
- 一级关注：AI 或物理规则单侧触发（未持续）
- 0 级：正常
输出触发时间戳与电芯编号。

参数
----
sensitivity         : 灵敏度 [0,1]，越高越敏感（默认 0.95）
contamination       : IsolationForest 污染率（默认 0.05）
rule_sigma          : 物理规则 σ 倍数（默认 3.0）
min_sustain_samples : 持续触发判定窗口（默认 3）
warmup_samples      : 训练基线样本数（取最早的健康数据，默认 2000）
rule_window         : 滚动包络窗口（样本数，默认 600）
rule_min_period     : 滚动包络最小样本数（默认 120）
min_voltage_spread_v: 压差规则绝对下限 [V]（默认 0.02）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_FEATURES = ["delta_V", "dT_dt", "delta_T"]

LEVEL_NORMAL = 0
LEVEL_WATCH = 1
LEVEL_WARNING = 2


@dataclass
class DetectionEvent:
    """一次异常事件。"""

    cell_id: str
    timestamp: float
    cycle: int
    level: int
    anomaly_score: float
    dT_dt: float
    delta_V: float
    duration_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "timestamp": round(float(self.timestamp), 1),
            "cycle": int(self.cycle),
            "level": int(self.level),
            "anomaly_score": round(float(self.anomaly_score), 4),
            "dT_dt": round(float(self.dT_dt), 4),
            "delta_V": round(float(self.delta_V), 4),
            "duration_samples": int(self.duration_samples),
        }


class HybridAnomalyDetector:
    """IsolationForest + 动态 3-Sigma 规则 双引擎检测器。"""

    def __init__(
        self,
        sensitivity: float = 0.95,
        contamination: float = 0.05,
        rule_sigma: float = 3.0,
        min_sustain_samples: int = 3,
        warmup_samples: int = 12000,
        rule_window: int = 1200,
        rule_min_period: int = 240,
        min_voltage_spread_v: float = 0.05,
        dtdt_floor_cmin: float = 2.0,
        transition_suppress_s: float = 30.0,
        merge_gap_s: float = 120.0,
        random_state: int = 42,
    ) -> None:
        if not 0.0 < sensitivity <= 1.0:
            raise ValueError("sensitivity 必须在 (0, 1] 之间")
        self.sensitivity = sensitivity
        self.contamination = contamination
        self.rule_sigma = rule_sigma
        self.min_sustain_samples = min_sustain_samples
        self.warmup_samples = warmup_samples
        self.rule_window = rule_window
        self.rule_min_period = rule_min_period
        self.min_voltage_spread_v = min_voltage_spread_v
        self.dtdt_floor_cmin = dtdt_floor_cmin
        self.transition_suppress_s = transition_suppress_s
        self.merge_gap_s = merge_gap_s
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._model: IsolationForest | None = None
        self.ai_threshold_: float = 0.5

    # ------------------------------------------------------------------ #
    # 训练 / 评分
    # ------------------------------------------------------------------ #
    def fit(self, features: pd.DataFrame) -> "HybridAnomalyDetector":
        """用最早的健康窗口训练 IsolationForest 基线。"""
        base = features.sort_values("timestamp").head(self.warmup_samples)
        if len(base) < max(50, self.warmup_samples // 10):
            raise ValueError(f"基线样本不足（{len(base)}），无法训练无监督模型。")
        X = self._prepare_X(base)
        Xs = self._scaler.fit_transform(X)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(Xs)
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """对全量特征评分，输出带告警标记与风险等级的结果表。"""
        if self._model is None:
            raise RuntimeError("请先调用 fit() 训练基线模型。")

        df = features.sort_values(["timestamp", "cell_id"]).reset_index(drop=True).copy()
        X = self._prepare_X(df)
        Xs = self._scaler.transform(X)

        # ---- AI 引擎：异常分数（越大越异常，归一化到 [0,1]） ----
        raw = -self._model.decision_function(Xs)
        lo, hi = float(np.percentile(raw, 1)), float(np.percentile(raw, 99))
        score = np.clip((raw - lo) / (hi - lo) if hi > lo else 0.0, 0.0, 1.0)
        df["anomaly_score"] = score
        self.ai_threshold_ = float(np.quantile(score, self.sensitivity))
        df["ai_flag"] = score >= self.ai_threshold_

        # ---- 工况切换抑制窗：切换后 N 秒内不参与告警（防电流阶跃毛刺误报） ----
        df["suppressed"] = self._transition_suppression(df)
        ai_eff = df["ai_flag"].to_numpy() & ~df["suppressed"].to_numpy()

        # ---- 物理规则引擎：动态 3-Sigma 包络（逐电芯滚动） ----
        grp = df.groupby(["cell_id", "regime"], sort=False)
        dtdt_mean = grp["dT_dt"].transform(
            lambda s: s.rolling(self.rule_window, min_periods=self.rule_min_period).mean()
        )
        dtdt_std = grp["dT_dt"].transform(
            lambda s: s.rolling(self.rule_window, min_periods=self.rule_min_period).std()
        )
        dv_mean = grp["delta_V"].transform(
            lambda s: s.rolling(self.rule_window, min_periods=self.rule_min_period).mean()
        )
        dv_std = grp["delta_V"].transform(
            lambda s: s.rolling(self.rule_window, min_periods=self.rule_min_period).std()
        )

        std_ok = dtdt_std.fillna(0.0) > 1e-9
        # 温升异常：仅判正向（升温），且需超过物理下限（热失控前兆方向），
        # 消除放电降温/工况切换导致的负向误报
        rule_dtdt = (
            (df["dT_dt"] > dtdt_mean + self.rule_sigma * dtdt_std)
            & std_ok
            & (df["dT_dt"] > self.dtdt_floor_cmin)
        )
        dv_limit = np.maximum(self.min_voltage_spread_v, dv_std * self.rule_sigma)
        rule_dv = (df["delta_V"] > dv_mean + dv_limit) | (df["delta_V"] < dv_mean - dv_limit)
        rule_flag = rule_dtdt | rule_dv
        suppress_np = df["suppressed"].to_numpy()
        df["rule_dtdt_flag"] = (rule_dtdt.to_numpy() & ~suppress_np)
        df["rule_dv_flag"] = (rule_dv.to_numpy() & ~suppress_np)
        rule_eff = rule_flag.to_numpy() & ~df["suppressed"].to_numpy()
        df["rule_flag"] = rule_eff

        # ---- 持续触发判定（≥3 个采样周期，按电芯） ----
        grp_eff = df.groupby("cell_id", sort=False)
        sustain_any = grp_eff["rule_flag"].transform(
            lambda s: s.rolling(self.min_sustain_samples, min_periods=1).sum()
        )
        sustain_dtdt = grp_eff["rule_dtdt_flag"].transform(
            lambda s: s.rolling(self.min_sustain_samples, min_periods=1).sum()
        )
        df["rule_sustained"] = (sustain_any >= self.min_sustain_samples).to_numpy()
        df["dtdt_sustained"] = (sustain_dtdt >= self.min_sustain_samples).to_numpy()

        # ---- 综合等级 ----
        # 二级预警：AI 确认 + 温升异常持续（疑似内短路/热失控前兆）
        # 一级关注：AI 或任一物理规则触发（含一致性/内阻发散）
        level2 = ai_eff & df["dtdt_sustained"].to_numpy()
        level1 = (ai_eff | df["rule_sustained"].to_numpy()) & ~level2
        level = np.where(level2, LEVEL_WARNING, np.where(level1, LEVEL_WATCH, LEVEL_NORMAL))
        df["level"] = level
        df["is_alert"] = df["level"] >= LEVEL_WATCH
        return df

    def summarize(self, result: pd.DataFrame) -> Dict[str, Any]:
        """把评分结果压缩为结构化摘要（供 Skill 接口 / 报告使用）。"""
        if result.empty:
            return {
                "pack_health_status": "数据为空",
                "abnormal_cells_list": [],
                "max_dT_dt": 0.0,
                "max_abs_delta_v": 0.0,
                "max_pack_spread_T": 0.0,
                "events": [],
            }

        events = self._extract_events(result)
        cells = result.groupby("cell_id")
        abnormal = []
        for cid, g in result.groupby("cell_id"):
            max_lvl = int(g["level"].max())
            n_alert = int((g["level"] >= LEVEL_WATCH).sum())
            if max_lvl >= LEVEL_WATCH:
                first = g.loc[g["level"] >= LEVEL_WATCH, "timestamp"].min()
                abnormal.append(
                    {
                        "cell_id": str(cid),
                        "level": max_lvl,
                        "n_alert_samples": n_alert,
                        "first_trigger_ts": round(float(first), 1),
                    }
                )

        if any(e.level >= LEVEL_WARNING for e in events):
            status = "二级预警（疑似内短路/热失控前兆）"
        elif abnormal:
            status = "一级关注（需人工复核）"
        else:
            status = "正常"

        return {
            "pack_health_status": status,
            "abnormal_cells_list": abnormal,
            "max_dT_dt": round(float(result["dT_dt"].max()), 4),
            "max_abs_delta_v": round(float(result["delta_V"].abs().max()), 4),
            "max_pack_spread_T": round(float(result["pack_spread_T"].max()), 3),
            "events": [e.to_dict() for e in events],
            "ai_threshold": round(float(self.ai_threshold_), 4),
        }

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _transition_suppression(self, df: pd.DataFrame) -> np.ndarray:
        """标记工况切换后 transition_suppress_s 秒内的样本（该窗口不参与告警判定）。"""
        out = np.zeros(len(df), dtype=bool)
        for _, g in df.groupby("cell_id", sort=False):
            ts = g["timestamp"].to_numpy()
            regime = g["regime"].to_numpy()
            change = np.zeros(len(g), dtype=bool)
            change[1:] = regime[1:] != regime[:-1]
            idx = np.flatnonzero(change)
            mask = np.zeros(len(g), dtype=bool)
            for c in idx:
                mask |= (ts >= ts[c]) & (ts <= ts[c] + self.transition_suppress_s)
            out[g.index.to_numpy()] = mask
        return out

    @staticmethod
    def _prepare_X(features: pd.DataFrame) -> np.ndarray:
        """把模型特征组织为矩阵；NaN 以 0 填充（中性值，文档化）。"""
        missing = [c for c in MODEL_FEATURES if c not in features.columns]
        if missing:
            raise ValueError(f"特征矩阵缺少模型输入列: {missing}")
        return features[MODEL_FEATURES].fillna(0.0).to_numpy(dtype="float64")

    def _extract_events(self, result: pd.DataFrame) -> List[DetectionEvent]:
        """把连续告警段提取为事件（时间窗内相邻告警归并为一次事件，取段内峰值特征）。"""
        events: List[DetectionEvent] = []
        for cid, g in result.groupby("cell_id", sort=False):
            alert = (g["level"] >= LEVEL_WATCH).to_numpy()
            if not alert.any():
                continue
            ts = g["timestamp"].to_numpy()
            idx = np.flatnonzero(alert)
            runs: List[tuple] = []
            start = idx[0]
            for k in range(1, len(idx)):
                adjacent = idx[k] == idx[k - 1] + 1
                near = (ts[idx[k]] - ts[idx[k - 1]]) <= self.merge_gap_s
                if not (adjacent or near):
                    runs.append((start, idx[k - 1]))
                    start = idx[k]
            runs.append((start, idx[-1]))
            for s, e in runs:
                seg = g.iloc[s : e + 1]
                peak = seg.loc[seg["anomaly_score"].idxmax()]
                events.append(
                    DetectionEvent(
                        cell_id=str(cid),
                        timestamp=float(peak["timestamp"]),
                        cycle=int(peak.get("cycle", 0)),
                        level=int(seg["level"].max()),
                        anomaly_score=float(peak["anomaly_score"]),
                        dT_dt=float(peak["dT_dt"]),
                        delta_V=float(peak["delta_V"]),
                        duration_samples=int(e - s + 1),
                    )
                )
        events.sort(key=lambda ev: (-ev.level, ev.timestamp))
        return events


def run_hybrid_detection(
    features: pd.DataFrame,
    sensitivity: float = 0.95,
    warmup_samples: int = 2000,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """便捷函数：训练 + 评分 + 摘要，返回 (结果表, 摘要)。"""
    det = HybridAnomalyDetector(sensitivity=sensitivity, warmup_samples=warmup_samples, **kwargs)
    det.fit(features)
    result = det.score(features)
    return result, det.summarize(result)
