"""
make_demo_gif.py — 生成演示动图（真实检测结果渲染）
====================================================
对 NASA 虚拟电池组注入热失控前兆，运行完整诊断链路，把“温升斜率突变 →
二级预警”的过程渲染为 GIF，用于 README / 简历 / 演示。

用法：python scripts/make_demo_gif.py
输出：docs/demo_thermal.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anomaly_model import HybridAnomalyDetector  # noqa: E402
from src.fault_injection import inject_thermal_precursor  # noqa: E402
from src.feature_engine import BatteryFeatureExtractor  # noqa: E402
from src.preprocessor import BatteryPreprocessor  # noqa: E402

# 中文字体
for font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    try:
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

T_START = 20000.0
DURATION = 600.0
VIEW_T0, VIEW_T1 = 18000.0, 26000.0
N_FRAMES = 40
FPS = 6


def main() -> None:
    data_path = ROOT / "data" / "processed" / "nasa_pack_demo.csv"
    raw = pd.read_csv(data_path)
    raw, meta = inject_thermal_precursor(raw, "B0005", T_START, duration_s=DURATION, dtdt_cmin=3.5)

    cleaned, _ = BatteryPreprocessor().process(raw)
    feats, _ = BatteryFeatureExtractor().extract(cleaned)
    det = HybridAnomalyDetector(sensitivity=0.95)
    det.fit(feats)
    scored = det.score(feats)
    summary = det.summarize(scored)

    inj = scored[(scored["cell_id"] == meta["cell_id"]) & (scored["level"] >= 2)
                 & (scored["timestamp"] >= meta["t_start"])]
    lead_min = (float(inj["timestamp"].min()) - meta["t_start"]) / 60.0 if not inj.empty else None

    win = cleaned[(cleaned["timestamp"] >= VIEW_T0) & (cleaned["timestamp"] <= VIEW_T1)]
    cells = sorted(win["cell_id"].unique())
    t_grid = np.linspace(VIEW_T0, VIEW_T1, N_FRAMES + 1)

    frames = []
    for fi in range(N_FRAMES):
        t_now = t_grid[fi + 1]
        view = win[win["timestamp"] <= t_now]
        alert = scored[(scored["level"] >= 1) & (scored["timestamp"] <= t_now)
                       & (scored["timestamp"] >= VIEW_T0)]

        fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
        fig.suptitle(
            "电芯一致性巡检与安全诊断  |  热失控前兆注入演示（B0005）",
            fontsize=13,
        )
        for ax, col, ylab in ((axes[0], "voltage", "单体电压 (V)"), (axes[1], "temp", "单体温度 (°C)")):
            for cid in cells:
                d = view[view["cell_id"] == cid]
                ax.plot(d["timestamp"], d[col], lw=1.0, label=cid)
            ax.set_ylabel(ylab, fontsize=10)
            ax.grid(alpha=0.3)
        # 告警标记
        for cid, g in alert.groupby("cell_id"):
            g1, g2 = g[g["level"] == 1], g[g["level"] == 2]
            if not g1.empty:
                axes[0].scatter(g1["timestamp"], g1["voltage"], s=8, c="orange", marker="^", alpha=0.7)
            if not g2.empty:
                axes[0].scatter(g2["timestamp"], g2["voltage"], s=18, c="red", marker="D", alpha=0.9)
                axes[1].scatter(g2["timestamp"], g2["temp"], s=18, c="red", marker="D", alpha=0.9)
        axes[1].set_xlabel("电池组运行时长 (s)", fontsize=10)
        axes[0].legend(ncol=4, fontsize=8, loc="upper right")
        axes[0].axvline(T_START, color="grey", ls="--", lw=0.8)
        axes[0].text(T_START + 40, axes[0].get_ylim()[1] * 0.98, "注入点", fontsize=8, color="grey")
        status = "二级预警（疑似内短路/热失控前兆）" if lead_min is not None else "运行中"
        axes[0].set_title(f"t={t_now:.0f}s　{status}"
                          + (f"　提前量≈{lead_min:.1f} min" if lead_min is not None else ""),
                          fontsize=9, loc="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(img))
        plt.close(fig)

    out_dir = ROOT / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_thermal.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"GIF written: {out_path}  ({len(frames)} frames, {out_path.stat().st_size/1024:.0f} KB)")
    print("status:", summary["pack_health_status"], "| lead_min:", lead_min)


if __name__ == "__main__":
    main()
