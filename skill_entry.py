"""
skill_entry.py — 标准化 Skill / Tool 调度入口
==============================================
任何 Agent 工作流系统都可以通过统一接口调用本技能：

    result = run_safety_inspection_skill(data_path="data/processed/nasa_pack_demo.csv",
                                         sensitivity=0.95,
                                         out_format="json")

返回标准 JSON 结构，包含：
  pack_health_status / abnormal_cells_list / max_dT_dt /
  root_cause_prediction / events / data_quality / meta

命令行用法：
    python skill_entry.py --data <csv> --sensitivity 0.95 --format json|md|pdf --outdir reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.preprocessor import BatteryPreprocessor
from src.feature_engine import BatteryFeatureExtractor
from src.anomaly_model import HybridAnomalyDetector
from src.reporter import BatterySafetyReporter

REQUIRED_KEYS = [
    "report_type", "schema_version", "generated_at", "meta",
    "data_quality", "pack_health_status", "abnormal_cells_list",
    "max_dT_dt", "max_abs_delta_v", "max_pack_spread_T",
    "events", "root_cause_prediction", "recommendations",
]


def validate_result_schema(report: Dict[str, Any]) -> None:
    """轻量 JSON Schema 校验：确认标准结构完整且类型正确。"""
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        raise ValueError(f"报告结构缺少必需字段: {missing}")
    if not isinstance(report["pack_health_status"], str):
        raise TypeError("pack_health_status 必须为字符串")
    if not isinstance(report["abnormal_cells_list"], list):
        raise TypeError("abnormal_cells_list 必须为列表")
    if not isinstance(report["events"], list):
        raise TypeError("events 必须为列表")
    if not isinstance(report["max_dT_dt"], (int, float)):
        raise TypeError("max_dT_dt 必须为数值")


def run_safety_inspection_skill(
    data_path: Union[str, Path, pd.DataFrame],
    sensitivity: float = 0.99,
    out_format: str = "json",
    out_dir: Optional[Union[str, Path]] = None,
    nominal_capacity_ah: float = 2.0,
    **preproc_kwargs: Any,
) -> Dict[str, Any]:
    """串联预处理 → 特征工程 → 双引擎检测 → 结构化报告。

    参数
    ----
    data_path      : 长表 CSV 路径（字段 timestamp/cell_id/voltage/current/temp）或 DataFrame
    sensitivity    : 检测灵敏度 [0,1]，默认 0.95
    out_format     : 输出形态，'json' | 'md' | 'pdf' | 'none'
    out_dir        : 输出目录（None 则不落盘）
    nominal_capacity_ah : 电芯标称容量 [Ah]
    """
    # ---- 1. 预处理 ----
    preprocessor = BatteryPreprocessor(nominal_capacity_ah=nominal_capacity_ah, **preproc_kwargs)
    cleaned, quality = preprocessor.process(data_path)

    # 护栏：一致性分析需要 >=2 颗电芯（组内中位数无定义时会产生无意义告警）
    if quality.n_cells < 2:
        quality.warnings.append(
            f"仅检测到 {quality.n_cells} 颗电芯：一致性分析需要 >=2 颗电芯，"
            "本次结果中的一致性类告警（ΔV/极差）不具物理意义，请补充多电芯数据后复核。"
        )

    # ---- 2. 特征工程 ----
    extractor = BatteryFeatureExtractor(nominal_capacity_ah=nominal_capacity_ah)
    features, units = extractor.extract(cleaned)

    # ---- 3. 双引擎检测 ----
    detector = HybridAnomalyDetector(sensitivity=sensitivity)
    detector.fit(features)
    scored = detector.score(features)
    summary = detector.summarize(scored)

    # ---- 4. 报告组装 ----
    if isinstance(data_path, pd.DataFrame):
        path_label = "<dataframe>"
    else:
        path_label = str(data_path)
    time_span_h = (
        float((cleaned["timestamp"].max() - cleaned["timestamp"].min()) / 3600.0)
        if len(cleaned)
        else 0.0
    )
    meta = {
        "data_path": path_label,
        "sensitivity": float(sensitivity),
        "n_cells": quality.n_cells,
        "n_samples": quality.n_samples_total,
        "time_span_h": round(time_span_h, 2),
    }
    report = BatterySafetyReporter.build_report_dict(summary, quality.to_dict(), meta)
    validate_result_schema(report)

    # ---- 5. 输出 ----
    if out_dir is not None and out_format != "none":
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_format == "json":
            BatterySafetyReporter.save_json(report, out_dir / "safety_report.json")
        elif out_format == "md":
            BatterySafetyReporter.save_markdown(report, out_dir / "safety_report.md")
        elif out_format == "pdf":
            BatterySafetyReporter.save_pdf(report, out_dir / "safety_report.pdf")
        else:
            raise ValueError(f"不支持的输出格式: {out_format}")
    return report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="储能电芯安全诊断 Skill 命令行入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", required=True, help="长表 CSV 路径")
    parser.add_argument("--sensitivity", type=float, default=0.99, help="检测灵敏度 [0,1]")
    parser.add_argument("--format", choices=["json", "md", "pdf", "none"], default="json")
    parser.add_argument("--outdir", default="reports", help="输出目录")
    parser.add_argument("--capacity", type=float, default=2.0, help="电芯标称容量 [Ah]")
    args = parser.parse_args(argv)

    report = run_safety_inspection_skill(
        data_path=args.data,
        sensitivity=args.sensitivity,
        out_format=args.format,
        out_dir=args.outdir,
        nominal_capacity_ah=args.capacity,
    )
    print(json.dumps({
        "pack_health_status": report["pack_health_status"],
        "abnormal_cells_list": report["abnormal_cells_list"],
        "max_dT_dt": report["max_dT_dt"],
        "root_cause_prediction": report["root_cause_prediction"],
        "n_events": len(report["events"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
