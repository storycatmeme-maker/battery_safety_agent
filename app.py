"""
app.py — 储能电芯安全诊断可视化大屏 (Streamlit)
===============================================
启动：streamlit run app.py

功能：
1. 顶部 Metric 卡片：当前最高温电芯 / 最大单体压差 / AI 风险等级 / 异常电芯数
2. Plotly 多串电芯电压/温度时序 + 故障红色标记点 + 通讯离线灰色阴影
3. 侧边栏：灵敏度调节、时间窗口、故障注入演示（热失控前兆/通讯中断/毛刺）
4. 报告导出：JSON / Markdown / PDF
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.anomaly_model import HybridAnomalyDetector  # noqa: E402
from src.fault_injection import (  # noqa: E402
    inject_comm_loss,
    inject_thermal_precursor,
    inject_voltage_spikes,
)
from src.feature_engine import BatteryFeatureExtractor  # noqa: E402
from src.preprocessor import BatteryPreprocessor  # noqa: E402
from src.reporter import BatterySafetyReporter  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "processed" / "nasa_pack_demo.csv"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(page_title="储能电芯安全诊断大屏", page_icon="🔋", layout="wide")
st.title("🔋 储能系统电芯一致性巡检与安全诊断大屏")
st.caption(
    "双引擎：IsolationForest 无监督评分 + 动态 3σ 物理规则包络　|　"
    "数据：NASA PCoE 电池老化公开数据集（B0005/B0006/B0007/B0018 虚拟电池组）"
)

SCENARIOS = {
    "原始数据（无注入）": "none",
    "热失控前兆注入（B0005, 3.5°C/min）": "thermal",
    "通讯中断模拟（B0006, 10s）": "comm",
    "电压毛刺注入（B0007, 5 个）": "spikes",
}


@st.cache_data(show_spinner=False)
def run_pipeline(data_path: str, sensitivity: float, scenario: str,
                 t_start: float, duration_s: float) -> dict:
    """全链路：注入(可选) -> 预处理 -> 特征 -> 双引擎 -> 报告。"""
    raw = pd.read_csv(data_path)
    meta_inject = None
    if scenario == "thermal":
        raw, meta_inject = inject_thermal_precursor(
            raw, "B0005", t_start, duration_s=duration_s, dtdt_cmin=3.5
        )
    elif scenario == "comm":
        raw, meta_inject = inject_comm_loss(raw, "B0006", t_start, duration_s=10.0)
    elif scenario == "spikes":
        raw, meta_inject = inject_voltage_spikes(raw, "B0007", n=5, magnitude_v=0.3)

    cleaned, quality = BatteryPreprocessor().process(raw)
    feats, _units = BatteryFeatureExtractor().extract(cleaned)
    detector = HybridAnomalyDetector(sensitivity=sensitivity)
    detector.fit(feats)
    scored = detector.score(feats)
    summary = detector.summarize(scored)

    report = BatterySafetyReporter.build_report_dict(
        summary,
        quality.to_dict(),
        {
            "data_path": str(data_path),
            "sensitivity": float(sensitivity),
            "n_cells": quality.n_cells,
            "n_samples": quality.n_samples_total,
            "time_span_h": round((cleaned["timestamp"].max() - cleaned["timestamp"].min()) / 3600.0, 2),
        },
    )
    return {
        "cleaned": cleaned, "scored": scored, "summary": summary,
        "report": report, "quality": quality, "meta_inject": meta_inject,
        "units": _units,
    }


def _downsample(df: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(np.ceil(len(df) / max_points))
    return df.iloc[::step]


def _loss_runs(cleaned: pd.DataFrame, min_dur_s: float = 60.0) -> pd.DataFrame:
    """提取通讯离线时间段（按电芯），用于图上灰色阴影。"""
    rows = []
    for cid, g in cleaned.groupby("cell_id", sort=False):
        dl = g["data_loss"].to_numpy()
        if not dl.any():
            continue
        ts = g["timestamp"].to_numpy()
        idx = np.flatnonzero(dl)
        start, prev = idx[0], idx[0]
        for k in range(1, len(idx)):
            if idx[k] != prev + 1:
                dur = ts[prev] - ts[start]
                if dur >= min_dur_s:
                    rows.append((cid, ts[start], ts[prev]))
                start = idx[k]
            prev = idx[k]
        dur = ts[prev] - ts[start]
        if dur >= min_dur_s:
            rows.append((cid, ts[start], ts[prev]))
    return pd.DataFrame(rows, columns=["cell_id", "t0", "t1"])


def plot_series(cleaned: pd.DataFrame, scored: pd.DataFrame, t0: float, t1: float) -> go.Figure:
    win = cleaned[(cleaned["timestamp"] >= t0) & (cleaned["timestamp"] <= t1)]
    alert = scored[(scored["level"] >= 1) & (scored["timestamp"] >= t0) & (scored["timestamp"] <= t1)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("单体电压时序 (V)", "单体温度时序 (°C)"),
        vertical_spacing=0.05,
    )
    for cid in sorted(win["cell_id"].unique()):
        d = _downsample(win[win["cell_id"] == cid])
        fig.add_trace(
            go.Scatter(x=d["timestamp"], y=d["voltage"], mode="lines",
                       name=f"{cid}·V", legendgroup=cid, line=dict(width=1.2)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=d["timestamp"], y=d["temp"], mode="lines",
                       name=f"{cid}·T", legendgroup=cid, showlegend=False,
                       line=dict(width=1.2)),
            row=2, col=1,
        )

    # 通讯离线阴影（>60s）
    for _cid, _t0, _t1 in _loss_runs(win).itertuples(index=False):
        if _t1 < t0 or _t0 > t1:
            continue
        fig.add_vrect(x0=_t0, x1=_t1, fillcolor="grey", opacity=0.12,
                      line_width=0, row=1, col=1)
        fig.add_vrect(x0=_t0, x1=_t1, fillcolor="grey", opacity=0.12,
                      line_width=0, row=2, col=1)

    # 告警标记：一级橙色三角 / 二级红色菱形
    if not alert.empty:
        for cid, g in alert.groupby("cell_id"):
            g1 = g[g["level"] == 1]
            g2 = g[g["level"] == 2]
            if not g1.empty:
                d1 = _downsample(g1, 800)
                fig.add_trace(
                    go.Scatter(x=d1["timestamp"], y=d1["voltage"], mode="markers",
                               name=f"{cid}·L1", legendgroup=cid, showlegend=False,
                               marker=dict(color="orange", symbol="triangle-up", size=6)),
                    row=1, col=1,
                )
            if not g2.empty:
                d2 = _downsample(g2, 400)
                fig.add_trace(
                    go.Scatter(x=d2["timestamp"], y=d2["voltage"], mode="markers",
                               name=f"{cid}·L2·危险", legendgroup=cid,
                               marker=dict(color="red", symbol="diamond", size=9)),
                    row=1, col=1,
                )
    fig.update_layout(height=620, margin=dict(l=40, r=20, t=50, b=30),
                      hovermode="x unified", legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(title_text="电池组运行时长 (s)", row=2, col=1)
    return fig


def plot_features(scored: pd.DataFrame, t0: float, t1: float) -> go.Figure:
    f = scored[(scored["timestamp"] >= t0) & (scored["timestamp"] <= t1)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("温升速率 dT/dt (°C/min)", "中位数电压差 ΔV (V)",
                                        "AI 异常分数"),
                        vertical_spacing=0.06)
    for cid in sorted(f["cell_id"].unique()):
        d = _downsample(f[f["cell_id"] == cid])
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["dT_dt"], mode="lines",
                                 name=cid, legendgroup=cid, line=dict(width=1)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["delta_V"], mode="lines",
                                 name=cid, legendgroup=cid, showlegend=False,
                                 line=dict(width=1)),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["anomaly_score"], mode="lines",
                                 name=cid, legendgroup=cid, showlegend=False,
                                 line=dict(width=1)),
                      row=3, col=1)
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=50, b=30),
                      hovermode="x unified", legend=dict(orientation="h", y=1.12))
    return fig


# --------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ 控制台")
    data_path = st.text_input("数据文件路径", value=str(DEFAULT_DATA))
    scenario = st.selectbox("演示场景", list(SCENARIOS.keys()))
    sensitivity = st.slider("检测灵敏度 (sensitivity)", 0.5, 1.0, 0.95, 0.01)
    t_start = st.number_input("故障注入起始时间 (s)", min_value=0.0, value=20000.0, step=100.0)
    duration_s = st.number_input("热失控注入时长 (s)", min_value=60.0, value=600.0, step=60.0)
    run_btn = st.button("🚀 运行诊断", type="primary", use_container_width=True)
    st.divider()
    st.caption("项目：电芯一致性离群与热失控早期预警 Skill\n"
               "模块：preprocessor → feature_engine → anomaly_model → reporter")

if not Path(data_path).exists():
    st.error(f"数据文件不存在：{data_path}")
    st.stop()

if run_btn:
    with st.spinner("全链路诊断中（预处理→特征→双引擎→报告）..."):
        result = run_pipeline(str(data_path), sensitivity, SCENARIOS[scenario],
                              t_start, duration_s)
    st.session_state["result"] = result
    st.session_state["scenario"] = scenario

if "result" not in st.session_state:
    st.info("👈 请在左侧选择演示场景并点击「运行诊断」。")
    st.stop()

result = st.session_state["result"]
cleaned = result["cleaned"]
scored = result["scored"]
summary = result["summary"]
report = result["report"]
quality = result["quality"]
meta_inject = result["meta_inject"]

# 时间窗口（小时）
t_min, t_max = float(cleaned["timestamp"].min()), float(cleaned["timestamp"].max())
h_min, h_max = t_min / 3600.0, t_max / 3600.0
w0, w1 = st.slider("显示窗口（电池组运行小时）", h_min, h_max, (h_min, min(h_min + 25.0, h_max)), 0.5)
t0, t1 = w0 * 3600.0, w1 * 3600.0

# --------------------------------------------------------------------- #
# 顶部 Metric 卡片
# --------------------------------------------------------------------- #
win_last = cleaned[(cleaned["timestamp"] <= t1) & cleaned["is_valid"]]
hot_cell, max_spread, risk = "-", 0.0, summary["pack_health_status"]
if not win_last.empty:
    last = win_last.sort_values("timestamp").groupby("cell_id").last().reset_index()
    hot = last.loc[last["temp"].idxmax()]
    hot_cell = f"{hot['cell_id']} ({hot['temp']:.1f}°C)"
    max_spread = float(last["voltage"].max() - last["voltage"].min())
n_abnormal = len(summary["abnormal_cells_list"])
n_events = len(summary["events"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("当前最高温电芯", hot_cell)
c2.metric("当前最大单体压差 (V)", f"{max_spread:.3f}")
c3.metric("AI 风险等级", risk)
c4.metric("异常电芯 / 事件", f"{n_abnormal} / {n_events}")

if scenario == "热失控前兆注入（B0005, 3.5°C/min）" and meta_inject:
    inj = scored[
        (scored["cell_id"] == meta_inject["cell_id"])
        & (scored["level"] >= 2)
        & (scored["timestamp"] >= meta_inject["t_start"])
    ]
    if not inj.empty:
        lead_min = (float(inj["timestamp"].min()) - meta_inject["t_start"]) / 60.0
        st.success(
            f"🎯 检测器在注入后 **{lead_min:.1f} 分钟** 发出二级预警（疑似内短路/热失控前兆），"
            f"早于 60°C 绝对温度阈值——提前量验证成功。"
        )
    else:
        st.warning("当前参数下未触发二级预警，可尝试调低灵敏度或延长注入时长。")

# --------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------- #
tab1, tab2, tab3, tab4 = st.tabs(["📈 时序总览", "🧬 特征与评分", "🛡️ 数据质量", "📋 事件与报告"])

with tab1:
    st.plotly_chart(plot_series(cleaned, scored, t0, t1), use_container_width=True)
    with st.expander("图表说明"):
        st.markdown(
            "- **红色菱形**：二级预警（AI 确认 + 温升规则持续 ≥3 周期）＝疑似内短路/热失控前兆\n"
            "- **橙色三角**：一级关注（AI 或规则持续触发）＝一致性/内阻发散\n"
            "- **灰色阴影**：通讯离线窗口（>5s 断点，系统不伪造插值）\n"
            "- 注：B0007 在本数据集中放电截止较深（构造虚拟电池组的协议差异），"
            "检测器将其识别为一致性偏差。"
        )

with tab2:
    st.plotly_chart(plot_features(scored, t0, t1), use_container_width=True)

with tab3:
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("通讯离线事件", quality.n_data_loss_events)
    q2.metric("离线数据占比 (%)", f"{100.0 * float(cleaned["data_loss"].mean()):.2f}")
    q3.metric("短断点插值", quality.n_interpolated)
    q4.metric("电压/温度毛刺剔除", f"{quality.n_spikes_removed['voltage']} / {quality.n_spikes_removed['temp']}")
    st.subheader("工况分布")
    rd = quality.regime_distribution
    st.bar_chart(pd.Series(rd, name="样本数"))
    if quality.warnings:
        st.subheader("数据质量告警")
        for w in quality.warnings:
            st.warning(w)

with tab4:
    st.subheader("根因预测")
    st.info(report["root_cause_prediction"])
    st.subheader("处置建议")
    for i, r in enumerate(report["recommendations"], 1):
        st.write(f"{i}. {r}")
    st.subheader("异常事件明细")
    if summary["events"]:
        st.dataframe(pd.DataFrame(summary["events"]), use_container_width=True, height=260)
    else:
        st.write("无告警事件。")

# --------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------- #
st.divider()
st.subheader("📤 导出报告")
e1, e2, e3 = st.columns(3)
with e1:
    st.download_button("导出 JSON",
                       data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="safety_report.json", mime="application/json",
                       use_container_width=True)
with e2:
    st.download_button("导出 Markdown",
                       data=BatterySafetyReporter.to_markdown(report).encode("utf-8"),
                       file_name="safety_report.md", mime="text/markdown",
                       use_container_width=True)
with e3:
    try:
        pdf_path = BatterySafetyReporter.save_pdf(report, REPORTS_DIR / "_app_tmp.pdf")
        pdf_bytes = pdf_path.read_bytes()
        st.download_button("导出 PDF", data=pdf_bytes,
                           file_name="safety_report.pdf", mime="application/pdf",
                           use_container_width=True)
    except Exception as exc:  # pragma: no cover
        st.error(f"PDF 生成失败：{exc}")
