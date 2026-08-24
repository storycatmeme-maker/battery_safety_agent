"""
reporter.py — 诊断结论与结构化输出（JSON / Markdown / PDF）
=============================================================

把预处理质量报告 + 检测摘要组装为标准化诊断结果，并支持三种输出形态：
1. JSON ：机器可读，供任何 Agent 工作流 / 上位系统消费
2. Markdown ：人类可读巡检报告
3. PDF    ：可打印/归档的《储能系统电芯一致性巡检与安全诊断报告》
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORT_TITLE = "储能系统电芯一致性巡检与安全诊断报告"
SCHEMA_VERSION = "1.0"

# reportlab 中文字体（内置 CID 字体，无需外部字体文件）
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _PDF_AVAILABLE = True
except Exception:  # pragma: no cover - reportlab 未安装时降级
    _PDF_AVAILABLE = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def predict_root_cause(
    summary: Dict[str, Any], quality: Dict[str, Any]
) -> str:
    """基于检测结果的机理归因（规则化、可解释）。"""
    events = summary.get("events", [])
    level2 = [e for e in events if e.get("level", 0) >= 2]
    if level2:
        cells = ",".join(sorted({e["cell_id"] for e in level2}))
        return (
            f"疑似内短路/热失控前兆（电芯 {cells} 温升斜率持续异常，"
            "建议立即停机隔离并复核 BMS 告警）"
        )
    abnormal = summary.get("abnormal_cells_list", [])
    if abnormal:
        worst = max(abnormal, key=lambda c: (c.get("level", 0), c.get("n_alert_samples", 0)))
        cell = worst["cell_id"]
        max_dv = summary.get("max_abs_delta_v", 0.0)
        max_spread = summary.get("max_pack_spread_T", 0.0)
        causes = []
        if max_dv >= 0.10:
            causes.append("内阻异常偏高 / 容量衰减不一致")
        if max_spread >= 3.0:
            causes.append("局部散热失效风险")
        if not causes:
            causes.append("电芯一致性轻微发散（建议持续监测）")
        return f"电芯 {cell} 与组内其他电芯存在一致性偏差：{'；'.join(causes)}"
    return "数据质量正常，未检测到明显异常"


def build_recommendations(summary: Dict[str, Any]) -> List[str]:
    recs: List[str] = []
    abnormal = summary.get("abnormal_cells_list", [])
    level2 = [e for e in summary.get("events", []) if e.get("level", 0) >= 2]
    if level2:
        recs.append("立即对相关电芯所在模组执行停机隔离，并进行 HPPC 内阻复测与外观检查。")
        recs.append("将 BMS 温升斜率告警阈值下调并加入联锁保护逻辑。")
    if abnormal:
        cells = [c["cell_id"] for c in abnormal]
        recs.append(f"对 {','.join(cells)} 安排电芯一致性复核（容量/内阻），缩短巡检周期至原计划的一半。")
        if summary.get("max_pack_spread_T", 0.0) >= 3.0:
            recs.append("检查电池包散热风道与导热垫状态，排查局部散热失效。")
    recs.append("建议将本报告纳入运维档案，并与后续循环的容量衰减曲线交叉验证。")
    return recs


class BatterySafetyReporter:
    """把检测摘要 + 数据质量组装为标准诊断报告。"""

    @staticmethod
    def build_report_dict(
        summary: Dict[str, Any],
        quality: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = dict(meta or {})
        report = {
            "report_type": "battery_pack_safety_inspection",
            "schema_version": SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "meta": {
                "data_path": meta.get("data_path"),
                "sensitivity": meta.get("sensitivity", 0.95),
                "model": "IsolationForest + 动态3σ规则双引擎",
                "n_cells": meta.get("n_cells", quality.get("n_cells")),
                "n_samples": meta.get("n_samples", quality.get("n_samples_total")),
                "time_span_h": meta.get("time_span_h"),
            },
            "data_quality": quality,
            "pack_health_status": summary.get("pack_health_status", "未知"),
            "abnormal_cells_list": summary.get("abnormal_cells_list", []),
            "max_dT_dt": summary.get("max_dT_dt", 0.0),
            "max_abs_delta_v": summary.get("max_abs_delta_v", 0.0),
            "max_pack_spread_T": summary.get("max_pack_spread_T", 0.0),
            "events": summary.get("events", []),
            "root_cause_prediction": predict_root_cause(summary, quality),
            "recommendations": build_recommendations(summary),
        }
        return report

    # ------------------------------------------------------------ #
    @staticmethod
    def save_json(report: Dict[str, Any], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out_path

    @staticmethod
    def to_markdown(report: Dict[str, Any]) -> str:
        m = report.get("meta", {})
        q = report.get("data_quality", {})
        lines: List[str] = []
        lines.append(f"# {REPORT_TITLE}")
        lines.append("")
        lines.append(f"- 生成时间：{report['generated_at']}")
        lines.append(f"- 数据来源：{m.get('data_path', '-')}")
        lines.append(f"- 电芯数量：{m.get('n_cells', '-')}　样本数：{m.get('n_samples', '-')}")
        lines.append(f"- 灵敏度：{m.get('sensitivity', '-')}　模型：{m.get('model', '-')}")
        lines.append("")
        lines.append("## 一、总体结论")
        lines.append("")
        lines.append(f"**电池组健康状态：{report['pack_health_status']}**")
        lines.append("")
        lines.append("| 指标 | 数值 | 单位 |")
        lines.append("|---|---|---|")
        lines.append(f"| 最大温升速率 max dT/dt | {report['max_dT_dt']} | °C/min |")
        lines.append(f"| 最大单体压差 max ΔV | {report['max_abs_delta_v']} | V |")
        lines.append(f"| 组内最大温差极差 | {report['max_pack_spread_T']} | °C |")
        lines.append("")
        lines.append("## 二、异常电芯")
        lines.append("")
        abn = report.get("abnormal_cells_list", [])
        if abn:
            lines.append("| 电芯 | 风险等级 | 告警样本数 | 首次触发时间(s) |")
            lines.append("|---|---|---|---|")
            for c in abn:
                lines.append(
                    f"| {c['cell_id']} | {c['level']} | {c['n_alert_samples']} | {c['first_trigger_ts']} |"
                )
        else:
            lines.append("未发现异常电芯。")
        lines.append("")
        lines.append("## 三、异常事件明细")
        lines.append("")
        evs = report.get("events", [])[:20]
        if evs:
            lines.append("| 电芯 | 时间(s) | 循环 | 等级 | AI分数 | dT/dt(°C/min) | ΔV(V) | 持续样本 |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for e in evs:
                lines.append(
                    f"| {e['cell_id']} | {e['timestamp']} | {e['cycle']} | {e['level']} "
                    f"| {e['anomaly_score']} | {e['dT_dt']} | {e['delta_V']} | {e['duration_samples']} |"
                )
        else:
            lines.append("无告警事件。")
        lines.append("")
        lines.append("## 四、数据质量审计")
        lines.append("")
        lines.append(f"- 通讯离线事件：{q.get('n_data_loss_events', 0)} 次，累计 {q.get('data_loss_total_seconds', 0):.0f} s")
        lines.append(f"- 短断点插值：{q.get('n_interpolated', 0)} 条")
        lines.append(f"- 电压/温度毛刺剔除：{q.get('n_spikes_removed', {}).get('voltage', 0)} / {q.get('n_spikes_removed', {}).get('temp', 0)} 条")
        rd = q.get("regime_distribution", {})
        if rd:
            lines.append(f"- 工况分布：充电 {rd.get('charge', 0)} / 放电 {rd.get('discharge', 0)} / 静置 {rd.get('rest', 0)} / 过渡 {rd.get('transition', 0)}")
        for w in q.get("warnings", []):
            lines.append(f"- 告警：{w}")
        lines.append("")
        lines.append("## 五、根因预测")
        lines.append("")
        lines.append(report["root_cause_prediction"])
        lines.append("")
        lines.append("## 六、处置建议")
        lines.append("")
        for i, r in enumerate(report.get("recommendations", []), 1):
            lines.append(f"{i}. {r}")
        lines.append("")
        lines.append("---")
        lines.append("*本报告由 battery_safety_agent 自动生成，供运维人员复核。*")
        return "\n".join(lines)

    @staticmethod
    def save_markdown(report: Dict[str, Any], out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            BatterySafetyReporter.to_markdown(report), encoding="utf-8"
        )
        return out_path

    @staticmethod
    def to_pdf(report: Dict[str, Any], out_path: Path) -> Path:
        if not _PDF_AVAILABLE:
            raise RuntimeError("reportlab 不可用，无法生成 PDF。")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "CNTitle", parent=styles["Title"], fontName="STSong-Light", fontSize=16, leading=22
        )
        h2_style = ParagraphStyle(
            "CNH2", parent=styles["Heading2"], fontName="STSong-Light", fontSize=12, leading=18
        )
        body_style = ParagraphStyle(
            "CNBody", parent=styles["BodyText"], fontName="STSong-Light", fontSize=9, leading=14
        )
        cell_style = ParagraphStyle(
            "CNCell", parent=body_style, fontSize=7.5, leading=10
        )

        story = []
        story.append(Paragraph(REPORT_TITLE, title_style))
        story.append(Spacer(1, 6 * mm))
        m = report.get("meta", {})
        story.append(
            Paragraph(
                f"生成时间：{report['generated_at']}<br/>"
                f"数据来源：{m.get('data_path', '-')}<br/>"
                f"电芯数：{m.get('n_cells', '-')}　样本数：{m.get('n_samples', '-')}　"
                f"灵敏度：{m.get('sensitivity', '-')}",
                body_style,
            )
        )
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(f"一、总体结论：{report['pack_health_status']}", h2_style))
        story.append(Paragraph(
            f"最大温升速率 {report['max_dT_dt']} °C/min　|　最大单体压差 {report['max_abs_delta_v']} V　|　"
            f"组内最大温差极差 {report['max_pack_spread_T']} °C",
            body_style,
        ))
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph("二、异常电芯", h2_style))
        abn = report.get("abnormal_cells_list", [])
        if abn:
            data = [[Paragraph("电芯", cell_style), Paragraph("等级", cell_style),
                     Paragraph("告警样本", cell_style), Paragraph("首次触发(s)", cell_style)]]
            for c in abn:
                data.append([Paragraph(str(c["cell_id"]), cell_style),
                             Paragraph(str(c["level"]), cell_style),
                             Paragraph(str(c["n_alert_samples"]), cell_style),
                             Paragraph(str(c["first_trigger_ts"]), cell_style)])
            tbl = Table(data, colWidths=[40 * mm, 20 * mm, 30 * mm, 40 * mm])
            tbl.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.9, 0.9, 0.9)),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("未发现异常电芯。", body_style))
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph("三、根因预测与建议", h2_style))
        story.append(Paragraph(report["root_cause_prediction"], body_style))
        for i, r in enumerate(report.get("recommendations", []), 1):
            story.append(Paragraph(f"{i}. {r}", body_style))
        story.append(Spacer(1, 3 * mm))

        story.append(Paragraph("四、数据质量审计", h2_style))
        q = report.get("data_quality", {})
        story.append(Paragraph(
            f"通讯离线 {q.get('n_data_loss_events', 0)} 次 / {q.get('data_loss_total_seconds', 0):.0f} s；"
            f"插值 {q.get('n_interpolated', 0)} 条；毛刺剔除 "
            f"{q.get('n_spikes_removed', {}).get('voltage', 0)}/{q.get('n_spikes_removed', {}).get('temp', 0)} 条",
            body_style,
        ))

        doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                                rightMargin=18 * mm, leftMargin=18 * mm,
                                topMargin=15 * mm, bottomMargin=15 * mm,
                                title=REPORT_TITLE)
        doc.build(story)
        return out_path

    @staticmethod
    def save_pdf(report: Dict[str, Any], out_path: Path) -> Path:
        return BatterySafetyReporter.to_pdf(report, out_path)
